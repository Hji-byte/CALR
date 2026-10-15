"""CALR main entry. Preview by default; --run explicitly executes GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def command(a):
    prefix=[sys.executable,'-u','-m']
    if a.action=='train':
        if a.method=='base': raise ValueError('Base is not trained')
        if a.adapter: raise ValueError('Training starts from original 2B, not an SFT/RL adapter')
        if not a.export: raise ValueError('--export is required for the immutable base vLLM export')
        mode='calr' if a.method=='calr' else 'correctness'
        return prefix+['training.train_cot_grpo','--model',str(a.model),'--from-base',
          '--reward-mode',mode,'--reward-alpha','0.1' if mode=='calr' else '1',
          '--calr-tau','256','--output',str(a.output),'--pilot','--max-steps','100',
          '--microbatch','4','--update-batch-size','128','--generation-batch-size','16',
          '--max-completion-length','4096','--lr','1e-6','--top-p','1','--kl-beta','0.01',
          '--eval-steps','10','--no-mask-truncated-completions',
          '--no-exclude-truncated-correct','--signal-audit','--vllm-model-copy',str(a.export)]
    if (a.method=='base') != (a.adapter is None):
        raise ValueError('Base must omit --adapter; grpo/calr must provide --adapter')
    if a.dataset=='aime':
        cmd=prefix+['evaluation.run_qwen35_2b_aime','--model',str(a.model),
          '--output',str(a.output),'--max-new-tokens','8192']
    else:
        cmd=prefix+['evaluation.vllm_cot_math500','--model',str(a.model),
          '--dataset',str(ROOT/'data/math500.json'),
          '--output',str(a.output/'responses.jsonl'),'--batch-size','16',
          '--max-new-tokens','4096','--max-model-len','6400']
    if a.adapter: cmd+=['--adapter',str(a.adapter)]
    return cmd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['train','evaluate'])
    p.add_argument('--method',choices=['base','grpo','calr'],required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adapter',type=Path)
    p.add_argument('--export',type=Path)
    p.add_argument('--dataset',choices=['math500','aime'],default='math500')
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    for key in ('model','output','adapter','export'):
        if getattr(a,key) is not None: setattr(a,key,getattr(a,key).resolve())
    cmd=command(a)
    print(json.dumps({'method':a.method,'command':cmd,'execute':a.run},indent=2))
    if not a.run: return
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU busy')
    if a.action=='train':
        from evaluation.vllm_cot_math500 import export_model
        export_model(a.model,None,a.export)
    elif a.dataset=='aime': cmd+=['--run']
    else: a.output.mkdir(parents=True,exist_ok=True)
    subprocess.run(cmd,cwd=ROOT,check=True)
    if a.action=='evaluate' and a.dataset=='math500':
        records=[json.loads(s) for s in (a.output/'responses.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
        gold={r['question']:r['answer'] for r in json.loads((ROOT/'data/math500.json').read_text(encoding='utf8'))['examples']}
        if len(records)!=500 or len({r['question'] for r in records})!=500 or {r['question'] for r in records}!=set(gold):
            raise RuntimeError('Expected exactly 500 unique matching questions')
        if any(r.get('answer',r.get('true_answer'))!=gold[r['question']] for r in records):
            raise RuntimeError('Reference answer mismatch')
        subprocess.run([sys.executable,'-m','analysis.score_math500',
                        str(a.output/'responses.jsonl'),'--workers','2'],cwd=ROOT,check=True)

if __name__=='__main__':
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER','0')
    os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
    main()
