"""Score response JSONL against an authoritative AIME dataset, matched by ID."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.aime_scoring import POLICY, score_aime


def evaluate(records, examples, allow_partial=False):
    gold = {str(x['id']): x for x in examples}
    if len(gold) != len(examples) or not gold:
        raise ValueError('Empty dataset or duplicate gold IDs')
    results, seen = [], set()
    for row in records:
        key = str(row.get('id', row.get('question_id')))
        if key not in gold or key in seen:
            raise ValueError(f'Unknown or duplicate response ID: {key}')
        seen.add(key)
        item = gold[key]
        result = dict(row)
        result.update(score_aime(row['response'], item['answer'], item.get('accepted_answers')))
        result['answer'] = item['answer']
        results.append(result)
    if not allow_partial and seen != set(gold):
        raise ValueError('Incomplete evaluation; use --allow-partial for an explicitly partial report')
    correct = sum(x['score'] for x in results)
    summary = dict(policy=POLICY, expected=len(gold), completed=len(results),
                   missing=len(gold)-len(results), correct=correct,
                   accuracy=correct/len(results) if results else None,
                   partial=len(results)!=len(gold),
                   invalid_final_answers=sum(x['generated_answer'] is None for x in results))
    return results, summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args=parser.parse_args()
    output=args.input.with_name('scored.jsonl')
    summary_path=args.input.with_name('summary.json')
    if output.exists() or summary_path.exists():
        raise FileExistsError('Scored output already exists; refusing overwrite')
    records=[json.loads(s) for s in args.input.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    examples=json.loads(args.dataset.read_text(encoding='utf-8-sig'))['examples']
    results, summary=evaluate(records, examples, args.allow_partial)
    with output.open('x',encoding='utf8') as f:
        for row in results:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with summary_path.open('x',encoding='utf8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
