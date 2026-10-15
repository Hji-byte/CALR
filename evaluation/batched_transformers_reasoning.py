#!/usr/bin/env python3
"""Run deterministic text-only reasoning with one batched Transformers model."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Dataset has no examples list: {path}")
    questions = [example.get("question") for example in examples]
    if any(not isinstance(question, str) or not question for question in questions):
        raise ValueError(f"Every example must have a non-empty question: {path}")
    if len(set(questions)) != len(questions):
        raise ValueError(f"Dataset contains duplicate questions: {path}")
    return examples


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} of {path} is not an object")
            records.append(record)
    return records


def is_successful_record(record: dict[str, Any]) -> bool:
    return bool(
        isinstance(record.get("question"), str)
        and record.get("generated_answer") is not None
        and not record.get("error")
    )


def collect_resume_records(
    examples: list[dict[str, Any]],
    output_path: Path,
    resume_paths: Iterable[Path],
) -> list[dict[str, Any]]:
    """Collect one successful record per dataset question in dataset order."""
    wanted = {example["question"] for example in examples}
    records_by_question: dict[str, dict[str, Any]] = {}
    for path in [output_path, *resume_paths]:
        for record in load_jsonl(path):
            question = record.get("question")
            if (
                question in wanted
                and question not in records_by_question
                and is_successful_record(record)
            ):
                records_by_question[question] = record
    return [
        records_by_question[example["question"]]
        for example in examples
        if example["question"] in records_by_question
    ]


def count_generated_tokens(token_ids: list[int], eos_token_ids: set[int]) -> int:
    """Match single-sample generate accounting by including the first EOS token."""
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return index + 1
    return len(token_ids)


def parse_math_answer(text: str) -> str:
    import re

    from evaluation.qwen_math_parser import (
        extract_answer,
    )

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else text
    if "\\boxed" not in answer and len(answer.split("\n\n")) == 1:
        answer = "\\boxed{" + answer + "}"
    return extract_answer(answer)


def build_messages(system_prompt: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    configured = getattr(model.generation_config, "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    if configured is None:
        return set()
    if isinstance(configured, int):
        return {configured}
    return {int(token_id) for token_id in configured}


def generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    system_prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    import torch

    messages = [
        build_messages(system_prompt, example["question"]) for example in examples
    ]
    rendered_prompts = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for conversation in messages
    ]
    inputs = tokenizer(
        rendered_prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    input_width = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[:, input_width:].detach().cpu()
    input_counts = inputs["attention_mask"].sum(dim=1).detach().cpu().tolist()
    eos_ids = _eos_token_ids(model, tokenizer)
    results: list[dict[str, Any]] = []
    for example, conversation, token_row, input_tokens in zip(
        examples, messages, generated.tolist(), input_counts, strict=True
    ):
        output_tokens = count_generated_tokens(token_row, eos_ids)
        response = tokenizer.decode(
            token_row[:output_tokens], skip_special_tokens=True
        )
        result = deepcopy(example)
        result.update(
            {
                "generated_answer": parse_math_answer(response),
                "explanation": response,
                "response": response,
                "messages": conversation,
                "input_tokens": int(input_tokens),
                "output_tokens": output_tokens,
                "cost": 0.0,
                "generation_backend": "transformers_batch",
            }
        )
        results.append(result)
    return results


def generate_with_oom_fallback(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    system_prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Split a batch after CUDA OOM while keeping each question independent."""
    import torch

    try:
        return generate_batch(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(examples) == 1:
            raise
        midpoint = len(examples) // 2
        print(
            f"CUDA OOM for batch of {len(examples)}; retrying as "
            f"{midpoint}+{len(examples) - midpoint}.",
            flush=True,
        )
        return generate_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            examples=examples[:midpoint],
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        ) + generate_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            examples=examples[midpoint:],
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )


def load_system_prompt(path: Path) -> str:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompt = payload.get("system_prompt") if isinstance(payload, dict) else None
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"Prompt file has no system_prompt: {path}")
    return prompt


def run(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this batched inference runner")
    if args.adapter is not None and not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter config missing: {args.adapter}")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    examples = load_examples(args.dataset)
    resumed = collect_resume_records(examples, args.output, args.resume_from)
    resumed_by_question = {record["question"]: record for record in resumed}
    todo = [
        example
        for example in examples
        if example["question"] not in resumed_by_question
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in resumed:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"Resumed {len(resumed)}/{len(examples)} questions; "
        f"{len(todo)} remain.",
        flush=True,
    )
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer defines neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    system_prompt = load_system_prompt(args.system_prompt)

    with args.output.open("a", encoding="utf-8") as handle:
        progress = tqdm(total=len(todo), desc="Batched generation")
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            results = generate_with_oom_fallback(
                model=model,
                tokenizer=tokenizer,
                examples=batch,
                system_prompt=system_prompt,
                max_new_tokens=args.max_new_tokens,
            )
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            progress.update(len(results))
        progress.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Optional SFT LoRA adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, action="append", default=[])
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path("evaluation/prompt.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
