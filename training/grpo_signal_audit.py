"""Observe loss inputs without changing rewards, masks or autograd."""
from pathlib import Path
from types import MethodType
from training.grpo_monitor import append_json


def signal_stats(inputs):
    import torch
    mask = inputs['completion_mask'].detach().bool()
    if 'tool_mask' in inputs:
        mask = mask & inputs['tool_mask'].detach().bool()
    advantages = inputs['advantages'].detach()
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    advantages = torch.broadcast_to(advantages, mask.shape)
    ratio = inputs.get('importance_sampling_ratio')
    ratio = torch.ones_like(advantages) if ratio is None else torch.broadcast_to(ratio.detach(), mask.shape)
    if not torch.isfinite(advantages).all() or not torch.isfinite(ratio).all():
        raise RuntimeError('Nonfinite advantages or sampling correction')
    eligible = mask & (advantages != 0) & (ratio != 0)
    active = mask.any(dim=1)
    return dict(answers=mask.shape[0], loss_tokens=int(mask.sum()),
        masked_answers=int((~active).sum()),
        zero_advantage_answers=int((active & ~((advantages != 0) & mask).any(dim=1)).sum()),
        correction_zero_answers=int((active & ~((ratio != 0) & mask).any(dim=1)).sum()),
        policy_signal_answers=int(eligible.any(dim=1).sum()),
        policy_signal_tokens=int(eligible.sum()),
        positive_signal_tokens=int((eligible & (advantages > 0)).sum()),
        negative_signal_tokens=int((eligible & (advantages < 0)).sum()),
        correction_mean_on_loss_tokens=float(ratio[mask].mean()) if mask.any() else None,
        note='Eligibility before PPO clipping, not measured gradient magnitude; KL is separate.')


def install_signal_audit(trainer, output):
    original = trainer.compute_loss

    def compute(self, model, inputs, *args, **kwargs):
        if model.training:
            append_json(Path(output)/'signal_audit.jsonl',
                        dict(step=self.state.global_step, **signal_stats(inputs)))
        return original(model, inputs, *args, **kwargs)

    trainer.compute_loss = MethodType(compute, trainer)
