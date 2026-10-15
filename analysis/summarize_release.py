"""Verify released result completeness and produce separately labelled summaries."""
from pathlib import Path
import json
import re
from evaluation.math_scoring import normalize_answer
from evaluation.qwen_math_grader import math_equal

ROOT=Path(__file__).resolve().parents[1]

def main():
    report={'math500':{},'aime':{}}
    gold=None
    for mode in ('base','sft','grpo','calr'):
        folder=ROOT/'results'/mode/'math500'
        rows=[json.loads(s) for s in (folder/'scored.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
        questions={r['question']:r['answer'] for r in rows}
        assert len(rows)==len(questions)==500
        if gold is None: gold=questions
        assert questions==gold
        fixes=[]
        for r in rows:
            if r['extraction_source']!='missing_final_answer' or r.get('finish_reason')!='stop':continue
            m=re.search(r'<answer>((?:(?!<answer>).)*?)\\end\{answer\}\s*\}?\s*$',r['response'],re.S)
            if m and math_equal(normalize_answer(m.group(1),r['question']),normalize_answer(r['answer'],r['question']),timeout=True):fixes.append(r['id'])
        strict=sum(bool(r['score']) for r in rows)
        report['math500'][mode]=dict(total=500,strict_correct=strict,strict_accuracy=strict/500,
            format_compatible_correct=strict+len(fixes),format_compatible_accuracy=(strict+len(fixes))/500,
            format_rescued_ids=fixes,average_output_tokens=sum(r['output_tokens'] for r in rows)/500,
            truncated=sum(r.get('finish_reason')=='length' for r in rows))
        afolder=ROOT/'results'/mode/'aime'
        if mode=='sft' and not (afolder/'scored.jsonl').exists():
            report['aime'][mode]={'status':'not_evaluated', 'note':'No AIME2024 SFT result collected; not a zero accuracy.'}
            continue
        arows=[json.loads(s) for s in (afolder/'scored.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
        expected={r['id'] for r in json.loads((ROOT/'data/aime2024.json').read_text(encoding='utf8'))['examples']}
        assert len(arows)==len(expected)==30
        assert {r['id'] for r in arows}==expected
        asummary=json.loads((afolder/'summary.json').read_text(encoding='utf8'))
        assert set(asummary['years'])=={'2024'}
        assert asummary['years']['2024']['total']==30
        assert asummary['years']['2024']['correct']==sum(r['score'] for r in arows)
        report['aime'][mode]={k:asummary[k] for k in ('temperature','thinking','max_new_tokens','max_model_len','policy','years','note')}
    text=json.dumps(report,ensure_ascii=False,indent=2)
    (ROOT/'results/summary.json').write_text(text,encoding='utf8')
    print(json.dumps(report['math500'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
