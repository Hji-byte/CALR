"""Text-only vLLM evaluation with immutable BF16 exports and resumable batches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def source_signature(path):
    files = sorted(set(path.glob('*.safetensors')) | set(path.glob('*.json')) | set(path.glob('*.jinja')))
    if not any(f.suffix == '.safetensors' for f in files):
        raise ValueError(f'No safetensors weights: {path}')
    return {f.name: file_hash(f) for f in files}


def export_model(model_path, adapter, destination):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import save_file
    signature = dict(base=source_signature(model_path),
        adapter=source_signature(adapter) if adapter else None, export_protocol='text_bf16_v1')
    marker = destination/'export_manifest.json'
    if destination.exists():
        if not marker.is_file():
            raise RuntimeError('Incomplete export directory; inspect it rather than overwrite')
        old = json.loads(marker.read_text())
        if old['source'] != signature or old['weights_sha256'] != file_hash(destination/'model.safetensors'):
            raise RuntimeError('Export mismatch; use a new output directory')
        return
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                device_map='cpu', local_files_only=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    # Export text-model names directly, avoiding save_pretrained's multimodal
    # namespace prefix. Work only on a CPU copy; never write source weights.
    state = {}
    for name, tensor in model.state_dict().items():
        key = name.replace('model.language_model.', 'model.', 1) if name.startswith('model.language_model.') else name
        if key in state or not key.startswith(('model.layers.', 'model.embed_tokens.', 'model.norm.', 'lm_head.')):
            raise RuntimeError(f'Unexpected/duplicate text weight {key}')
        state[key] = tensor.detach().contiguous().clone()
    config = model.config.get_text_config().to_dict()
    config.update(model_type='qwen3_5_text', architectures=['Qwen3_5ForCausalLM'],
                  eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    destination.mkdir(parents=True)
    (destination/'config.json').write_text(json.dumps(config, indent=2))
    tokenizer.save_pretrained(destination)
    save_file(state, str(destination/'model.safetensors'), metadata={'format':'pt'})
    marker.write_text(json.dumps(dict(source=signature,
        weights_sha256=file_hash(destination/'model.safetensors')), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--adapter', type=Path)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--max-model-len', type=int, default=6400, help='Prompt plus completion context budget')
    p.add_argument('--stop-at-chat-end', action='store_true', help='Also stop at the template im_end token (explicit Base chat protocol)')
    p.add_argument('--system-prompt', type=Path, default=Path('evaluation/prompt.yaml'))
    a = p.parse_args()
    if a.batch_size < 1 or a.max_new_tokens < 1:
        p.error('Positive limits required')
    if a.max_model_len <= a.max_new_tokens:
        p.error('max-model-len must leave room for the prompt')
    from evaluation.batched_transformers_reasoning import load_examples, load_system_prompt, collect_resume_records, build_messages
    from transformers import AutoTokenizer
    import importlib.metadata
    export = a.output.parent/('vllm_sft_export' if a.adapter else 'vllm_base_export')
    export_model(a.model, a.adapter, export)
    signature = dict(export=file_hash(export/'export_manifest.json'), dataset=file_hash(a.dataset),
        prompt=file_hash(a.system_prompt), batch_size=a.batch_size,
        max_new_tokens=a.max_new_tokens, temperature=0, thinking=False,
        vllm=importlib.metadata.version('vllm'))
    if a.stop_at_chat_end:
        signature['stop_at_chat_end'] = True
    # Preserve old 6400-context manifests; record every non-default budget.
    if a.max_model_len != 6400:
        signature['max_model_len'] = a.max_model_len
    manifest = a.output.with_suffix('.run.json')
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise RuntimeError('Resume configuration mismatch')
    elif a.output.exists():
        raise RuntimeError('Existing output without matching manifest')
    else:
        manifest.write_text(json.dumps(signature, indent=2))
    examples = load_examples(a.dataset)
    previous = collect_resume_records(examples, a.output, [])
    done = {r['question'] for r in previous}
    todo = [r for r in examples if r['question'] not in done]
    print(f'Resumed {len(previous)}/{len(examples)}; {len(todo)} remain.', flush=True)
    if not todo:
        return
    # Canonicalize valid saved rows atomically before appending. Malformed
    # JSON is rejected by the loader, not silently discarded.
    temporary = a.output.with_suffix('.resume.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        for row in previous:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    temporary.replace(a.output)
    tokenizer = AutoTokenizer.from_pretrained(export)
    system = load_system_prompt(a.system_prompt)
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(export), dtype='bfloat16', max_model_len=a.max_model_len,
        gpu_memory_utilization=0.6, max_num_seqs=a.batch_size,
        max_num_batched_tokens=4096, enable_prefix_caching=False, seed=42)
    stop_ids = None
    if a.stop_at_chat_end:
        chat_end = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if chat_end is None or tokenizer.convert_ids_to_tokens(chat_end) != '<|im_end|>':
            raise RuntimeError('Missing im_end token for explicit chat termination')
        stop_ids = [chat_end]
    params = SamplingParams(temperature=0, top_p=1.0, max_tokens=a.max_new_tokens, seed=42,
                            stop_token_ids=stop_ids)
    with a.output.open('a', encoding='utf-8') as f:
        for start in range(0, len(todo), a.batch_size):
            batch = todo[start:start+a.batch_size]
            messages = [build_messages(system, r['question']) for r in batch]
            prompts = [tokenizer.encode(tokenizer.apply_chat_template(m, tokenize=False,
                add_generation_prompt=True, enable_thinking=False), add_special_tokens=False) for m in messages]
            if any(len(ids)+a.max_new_tokens > a.max_model_len for ids in prompts):
                raise RuntimeError('Prompt exceeds context limit; no silent truncation')
            outputs = llm.generate([{'prompt_token_ids': ids} for ids in prompts], params, use_tqdm=False)
            for row, message, ids, out in zip(batch, messages, prompts, outputs, strict=True):
                answer = out.outputs[0]
                from evaluation.math_scoring import extract_final_answer
                extracted, _ = extract_final_answer(answer.text)
                record = dict(row, response=answer.text, explanation=answer.text, messages=message,
                    generated_answer=extracted if extracted is not None else '',
                    output_tokens=len(answer.token_ids), input_tokens=len(ids),
                    generation_backend='vllm', finish_reason=answer.finish_reason)
                f.write(json.dumps(record, ensure_ascii=False)+'\n')
            f.flush()
            print(f'Completed {len(previous)+min(start+a.batch_size,len(todo))}/{len(examples)}', flush=True)


if __name__ == '__main__':
    os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
    main()
