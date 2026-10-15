"""Re-score existing CoT responses without model calls or overwriting inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.math_scoring import (
    POLICY, extract_final_answer, normalize_answer,
)
from evaluation.qwen_math_grader import math_equal


def score_record(record):
    answer, source = extract_final_answer(record.get("response", ""))
    question = record.get("question", "")
    pred = normalize_answer(answer, question) if answer is not None else None
    gold = normalize_answer(record.get("answer", record.get("true_answer")), question)
    correct = bool(pred) and math_equal(pred, gold, timeout=True)
    result = dict(record)
    result.update(legacy_score=record.get("score"),
                  legacy_generated_answer=record.get("generated_answer"),
                  generated_answer=pred, score=bool(correct),
                  extraction_source=source, scoring_policy=POLICY,
                  reasoning_verified=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    # Preflight every destination before scoring any input.
    destinations = []
    for path in args.inputs:
        output = path.with_name("scored.jsonl")
        summary = path.with_name("summary.json")
        if any(output == previous[1] for previous in destinations):
            raise ValueError('Each input must have its own result directory')
        if output.exists() or summary.exists():
            raise FileExistsError(f"Refusing to overwrite {output} or {summary}")
        destinations.append((path, output, summary))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path, output, summary_path in destinations:
            payload = path.read_bytes()
            records = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
            results = []
            for i, result in enumerate(pool.map(score_record, records), 1):
                results.append(result)
                if i % 50 == 0:
                    print(f"{path.name}: {i}/{len(records)}", flush=True)
            changes = [{"id":r.get("id"), "gold":r.get("answer"),
                        "old_answer":r["legacy_generated_answer"],
                        "new_answer":r["generated_answer"], "source":r["extraction_source"],
                        "old_score":r["legacy_score"], "new_score":r["score"]}
                       for r in results if bool(r["legacy_score"]) != r["score"]]
            correct = sum(r["score"] for r in results)
            summary = {"policy":POLICY, "input":str(path),
                       "input_sha256":hashlib.sha256(payload).hexdigest(),
                       "total":len(results), "correct":correct,
                       "accuracy":correct/len(results) if results else 0,
                       "legacy_correct":sum(bool(r["legacy_score"]) for r in results),
                       "missing_final_answers":sum(r["extraction_source"] == "missing_final_answer" for r in results),
                       "average_output_tokens":sum(r.get("output_tokens",0) for r in results)/len(results) if results else 0,
                       "changes":changes,
                       "note":"Answer-only score; not proof verification. Same existing generations, stricter extraction and comparison."}
            with output.open("x", encoding="utf-8") as stream:
                for result in results:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            with summary_path.open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, ensure_ascii=False, indent=2)
            print(json.dumps({k:v for k,v in summary.items() if k != "changes"},ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
