"""AIME2024 evaluation for original 2B or an explicitly selected adapter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'

def load_tests():
    examples=[]
    for year in (2024,):
        rows=json.loads((DATA/f'aime{year}.json').read_text(encoding='utf8'))['examples']
        expected={(year,p,n) for p in ('I','II') for n in range(1,16)}
        if len(rows)!=30 or {(x['year'],x['part'],x['problem_number']) for x in rows}!=expected:
            raise ValueError(f'Incomplete AIME{year}')
        examples.extend(rows)
    if len({x['id'] for x in examples})!=30:
        raise ValueError('Duplicate question IDs')
    return examples

def generation_command(model, output, adapter=None, max_new_tokens=8192):
    if max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')
    cmd = [sys.executable,'-u','-m','evaluation.vllm_cot_math500',
            '--model',str(model),'--dataset',str(output/'aime2024.json'),
            '--output',str(output/'responses.jsonl'),'--batch-size','16',
            '--max-new-tokens',str(max_new_tokens),'--max-model-len',str(max_new_tokens+2048),
            '--system-prompt',str(ROOT/'evaluation/prompt.yaml')]
    if adapter is not None:
        cmd += ['--adapter', str(adapter)]
    return cmd

def summarize(records, examples):
    from analysis.score_aime import evaluate
    scored,_=evaluate(records,examples)
    report={}
    years={str(x['id']):x['year'] for x in examples}
    for year in (2024,):
        rows=[x for x in scored if years[str(x['id'])]==year]
        if any(type(x.get('output_tokens')) is not int or x['output_tokens']<0 for x in rows):
            raise ValueError('Missing/invalid output token count')
        report[str(year)]=dict(total=len(rows),correct=sum(x['score'] for x in rows),
            accuracy=sum(x['score'] for x in rows)/len(rows),
            average_output_tokens=sum(x['output_tokens'] for x in rows)/len(rows),
            truncated=sum(x.get('finish_reason')=='length' for x in rows),
            invalid_final_answers=sum(x['generated_answer'] is None for x in rows))
    return scored,report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('/root/autodl-tmp/models/Qwen3.5-2B'))
    p.add_argument('--adapter',type=Path,help='Explicit SFT/RL adapter; omit for original model')
    p.add_argument('--max-new-tokens',type=int,default=8192)
    p.add_argument('--output',type=Path)
    p.add_argument('--run',action='store_true',help='Explicit GPU generation authorization; otherwise preview only')
    a=p.parse_args()
    if a.max_new_tokens <= 0:
        p.error('--max-new-tokens must be positive')
    if a.adapter is not None and a.output is None:
        p.error('--adapter requires a separate --output directory')
    if a.output is None:
        a.output=Path(f'/root/autodl-tmp/aime_2b_original_{a.max_new_tokens}_v1')
    rows=load_tests()
    cmd=generation_command(a.model,a.output,a.adapter,a.max_new_tokens)
    print(json.dumps(dict(tests=30,years=[2024],command=cmd,run=a.run),ensure_ascii=False),flush=True)
    if not a.run:
        return
    if a.model.name.lower()!='qwen3.5-2b':
        p.error('Expected original Qwen3.5-2B, not Base or an adapter')
    if a.adapter is not None:
        if not (a.adapter/'adapter_config.json').is_file() or not any(a.adapter.glob('adapter_model.*')):
            raise FileNotFoundError(f'Incomplete adapter: {a.adapter}')
    busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    if busy.stdout.strip():
        raise RuntimeError('GPU busy: refusing to start another experiment')
    a.output.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(dict(examples=rows),ensure_ascii=False,indent=2)+'\n'
    dataset=a.output/'aime2024.json'
    if dataset.exists():
        if dataset.read_text(encoding='utf8')!=payload:
            raise RuntimeError('Dataset changed; use a fresh output directory')
    else:
        with dataset.open('x',encoding='utf8') as f: f.write(payload)
    # One 30-question invocation shares the immutable export and loads vLLM once.
    # Underlying generator checks model/data/settings hashes before any resume.
    subprocess.run(cmd,cwd=ROOT,check=True)
    raw=a.output/'responses.jsonl'
    records=[json.loads(s) for s in raw.read_text(encoding='utf8').splitlines() if s.strip()]
    scored,report=summarize(records,rows)
    report=dict(model=str(a.model),adapter=str(a.adapter) if a.adapter else None,temperature=0,thinking=False,
        max_new_tokens=a.max_new_tokens,max_model_len=a.max_new_tokens+2048,
        policy='strict_aime_v1',input_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),years=report,
        note='One greedy answer per problem, not multi-sample pass@1. Raw generations are preserved.')
    # Derived summaries are safely reproducible after interruption; never alter raw responses.
    for name,text in [('scored.jsonl',''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in scored)),
                      ('summary.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')]:
        path=a.output/name
        temporary=path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(text,encoding='utf8')
        temporary.replace(path)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    main()
