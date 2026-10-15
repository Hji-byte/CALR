"""Single-GPU sample IDs survive TRL shuffling via an extra batch tensor.

Observer only: logs pre-PPO-clip signal, not gradient contribution. No extra scoring.
"""
from types import MethodType
import uuid
from training.grpo_monitor import append_json


def install_sample_audit(trainer, reward, output):
    import torch
    if trainer.accelerator.num_processes != 1 or trainer.args.num_iterations != 1:
        raise ValueError('Sample audit supports single-process num_iterations=1 only')
    session = uuid.uuid4().hex
    pending, metadata = [], {}
    score = reward.score
    def observe_score(*args, **kwargs):
        result = score(*args, **kwargs)
        pending.append(list(result))
        return result
    reward.score = observe_score
    calculate = trainer._calculate_rewards
    generated = []
    def calculate_rewards(self, inputs, prompts, completions, completion_ids, *args, **kwargs):
        pending.clear()
        result = calculate(inputs, prompts, completions, completion_ids, *args, **kwargs)
        if len(pending) != 1 or len(pending[0]) != len(inputs):
            raise RuntimeError('Sample audit scoring alignment failed')
        generated.clear()
        for i, (row, ids, c) in enumerate(zip(inputs, completion_ids, pending[0], strict=True)):
            generated.append(dict(question_id=row['question_id'], correct=c,
                output_tokens=len(ids), truncated=not ids or ids[-1] not in reward.stop_ids,
                candidate_index=i % 8, token_ids=list(ids)))
        return result
    trainer._calculate_rewards = MethodType(calculate_rewards, trainer)
    generate = trainer._generate_and_score_completions
    def generate_scored(self, inputs, *args, **kwargs):
        generated.clear()
        result = generate(inputs, *args, **kwargs)
        if len(generated) != len(result['completion_ids']):
            raise RuntimeError('Sample audit generation alignment failed')
        ids = []
        for row, tokens in zip(generated, result['completion_ids'], strict=True):
            if tokens[:row['output_tokens']].tolist() != row['token_ids']:
                raise RuntimeError('Sample audit token order mismatch')
            key = len(metadata)
            metadata[key] = {k:v for k,v in row.items() if k!='token_ids'}
            ids.append(key)
        result['audit_sample_id'] = torch.tensor(ids, device=result['completion_ids'].device)
        return result
    trainer._generate_and_score_completions = MethodType(generate_scored, trainer)
    compute = trainer.compute_loss
    def compute_loss(self, model, inputs, *args, **kwargs):
        if model.training:
            from training.grpo_signal_audit import signal_stats
            for i, key in enumerate(inputs['audit_sample_id'].tolist()):
                batch = {k:v[i:i+1] for k,v in inputs.items()
                         if k in ('completion_mask','tool_mask','advantages','importance_sampling_ratio')}
                stats = signal_stats(batch)
                append_json(output/'sample_signal_audit.jsonl', dict(session=session, sample_id=key,
                    step=self.state.global_step, **metadata[key], **stats,
                    advantage=float(inputs['advantages'][i].detach().reshape(-1)[0])))
        # Do not pass observer metadata to the actual loss implementation.
        clean = {k:v for k,v in inputs.items() if k!='audit_sample_id'}
        return compute(model, clean, *args, **kwargs)
    trainer.compute_loss = MethodType(compute_loss, trainer)
