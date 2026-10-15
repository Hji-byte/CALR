"""Single-GPU PEFT sync protection; does not change GRPO loss or rewards."""
import json
import time


def install_exact_sync(trainer, output):
    import torch
    from transformers import TrainerCallback
    if trainer.accelerator.num_processes != 1:
        raise RuntimeError('Exact sync profile supports one process only')
    generation = trainer.vllm_generation
    model = generation.model
    original = generation._iter_named_params

    def exact_iter():
        targets = [m for m in model.modules() if hasattr(m, 'lora_A') and hasattr(m, 'base_layer')]
        if not targets or any(m.merged for m in targets):
            raise RuntimeError('Expected unmerged LoRA targets before sync')
        snapshots = [(m.base_layer.weight, m.base_layer.weight.detach().clone()) for m in targets]
        iterator = original()
        started = time.monotonic()
        try:
            yield from iterator
        finally:
            iterator.close()
            if any(m.merged for m in targets):
                model.unmerge_adapter()
            with torch.no_grad():
                for param, saved in snapshots:
                    param.copy_(saved)
            if not all(torch.equal(param, saved) for param, saved in snapshots):
                raise RuntimeError('Base restoration failed')
            with (output/'exact_sync.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=trainer.state.global_step,
                    tensors=len(snapshots), exact_restore=True,
                    seconds=time.monotonic()-started))+'\n')
    generation._iter_named_params = exact_iter

    class SignalGuard(TrainerCallback):
        """Pilot safety heuristic, not an accuracy or convergence criterion."""
        def __init__(self):
            self.low_updates = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            value = (logs or {}).get('sampling/importance_sampling_ratio/mean')
            if value is None:
                return
            self.low_updates = self.low_updates + 1 if value < 0.001 else 0
            if self.low_updates >= 2:
                reason = dict(step=state.global_step, reason='Mean IS ratio below 0.001 for two updates',
                              note='Engineering guard only; review before any resume')
                (output/'signal_guard_stop.json').write_text(json.dumps(reason, indent=2))
                control.should_save = True
                control.should_training_stop = True
            return control
    trainer.add_callback(SignalGuard())
