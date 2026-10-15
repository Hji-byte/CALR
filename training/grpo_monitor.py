"""Rollout metrics, finite-value checks and pause state for GRPO."""
import json
import math
from pathlib import Path


def append_json(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


class LoggedReward:
    __name__ = 'strict_correctness'

    def __init__(self, score, output, tokenizer):
        self.score, self.output = score, output
        eos = tokenizer.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    def __call__(self, completions, answer, question, question_id,
                 completion_ids, trainer_state, **kwargs):
        rewards = self.score(completions, answer, question)
        if len(rewards) % 8 or any(len(set(question_id[i:i+8])) != 1 for i in range(0,len(rewards),8)):
            raise ValueError('Expected complete contiguous 8-answer question groups')
        lengths = [len(ids) for ids in completion_ids]
        truncated = [not ids or ids[-1] not in self.eos for ids in completion_ids]
        for i in range(0, len(rewards), 8):
            group = rewards[i:i+8]
            append_json(self.output/'rollout_groups.jsonl', dict(step=trainer_state.global_step,
                question_id=question_id[i], rewards=group, output_tokens=lengths[i:i+8],
                truncated=truncated[i:i+8], all_correct=all(group), all_wrong=not any(group),
                completions=completions[i:i+8]))
        return rewards


def make_monitor(trainer, args):
    import torch
    from transformers import TrainerCallback

    class Monitor(TrainerCallback):
        def __init__(self):
            self.best = -1
            self.best_step = None
            self.last_eval = None
            self.changed = False
            self.initial = {n: p.detach().cpu().clone() for n, p in trainer.model.named_parameters()
                            if p.requires_grad}
            if not self.initial:
                raise RuntimeError('No trainable adapter parameters')
            resume = getattr(args, 'resume_from_checkpoint', None)
            if resume:
                saved = json.loads((Path(resume)/'monitor_state.json').read_text())
                self.best = saved['best']
                self.best_step = saved['best_step']
                self.last_eval = saved['last_eval']
                self.changed = saved['changed']

        def on_save(self, config, state, control, **kwargs):
            checkpoint = Path(config.output_dir)/f'checkpoint-{state.global_step}'
            (checkpoint/'monitor_state.json').write_text(json.dumps(dict(
                best=self.best, best_step=self.best_step, last_eval=self.last_eval,
                changed=self.changed)), encoding='utf-8')

        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ('loss', 'grad_norm'):
                if logs and key in logs and not math.isfinite(float(logs[key])):
                    raise RuntimeError(f'Nonfinite {key}; stopping training')
            append_json(self_output/'training_metrics.jsonl', dict(step=state.global_step, **(logs or {})))

        def on_step_end(self, config, state, control, **kwargs):
            for name, param in trainer.model.named_parameters():
                if name in self.initial:
                    if not torch.isfinite(param).all():
                        raise RuntimeError('Nonfinite adapter weights')
                    if not torch.equal(param.detach().cpu(), self.initial[name]):
                        self.changed = True
            pause = getattr(args, 'pause_after_step', None)
            if pause is not None and state.global_step >= pause:
                control.should_save = True
                control.should_training_stop = True
            return control

    self_output = args.output
    return Monitor()
