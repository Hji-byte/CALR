"""Explicit, in-place PEFT checkpoint restore for root + reference adapters."""
import json
from pathlib import Path
from types import MethodType


def restore_adapters(model, checkpoint):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    identities = {n: id(p) for n, p in model.named_parameters()}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    expected = {}
    for adapter in model.peft_config:
        folder = checkpoint if adapter == 'default' else checkpoint / adapter
        expected[adapter] = load_file(str(folder / 'adapter_model.safetensors'))
        current = get_peft_model_state_dict(model, adapter_name=adapter)
        if current.keys() != expected[adapter].keys():
            raise RuntimeError(f'Checkpoint adapter keys mismatch: {adapter}')
        for key, tensor in current.items():
            if tensor.shape != expected[adapter][key].shape:
                raise RuntimeError(f'Checkpoint adapter shape mismatch: {adapter}/{key}')
    # Validate every file before changing any parameter. No adapter recreation.
    for adapter, state in expected.items():
        set_peft_model_state_dict(model, state, adapter_name=adapter)
    counts = {}
    for adapter, state in expected.items():
        current = get_peft_model_state_dict(model, adapter_name=adapter)
        for key, tensor in current.items():
            if not torch.equal(tensor.detach().cpu(), state[key].to(tensor.dtype)):
                raise RuntimeError(f'Checkpoint adapter value mismatch: {adapter}/{key}')
        counts[adapter] = len(state)
    if identities != {n: id(p) for n, p in model.named_parameters()}:
        raise RuntimeError('Resume replaced parameter objects; optimizer mapping unsafe')
    if trainable != {n for n, p in model.named_parameters() if p.requires_grad}:
        raise RuntimeError('Resume changed trainable parameters')
    if any('.ref.' in n for n in trainable):
        raise RuntimeError('Reference adapter is trainable')
    return counts


def install_adapter_resume(trainer, output):
    """Replace only PEFT weight loading; retain Trainer optimizer/RNG restoration."""
    def load(self, resume_from_checkpoint, model=None):
        counts = restore_adapters(self.model if model is None else model,
                                  resume_from_checkpoint)
        with (Path(output) / 'adapter_resume_audit.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'checkpoint': str(resume_from_checkpoint),
                                'verified_tensors': counts,
                                'parameter_identity_preserved': True}) + '\n')
    trainer._load_from_checkpoint = MethodType(load, trainer)
    original_optimizer_load = trainer._load_optimizer_and_scheduler
    def load_optimizer(self, checkpoint):
        original_optimizer_load(checkpoint)
        if checkpoint is None:
            return
        import torch
        saved = torch.load(Path(checkpoint)/'optimizer.pt', map_location='cpu', weights_only=True)
        actual = self.optimizer.state_dict()
        if actual['param_groups'] != saved['param_groups'] or actual['state'].keys() != saved['state'].keys():
            raise RuntimeError('Restored optimizer structure mismatch')
        for key, values in saved['state'].items():
            for name, value in values.items():
                got = actual['state'][key][name]
                equal = torch.equal(got.detach().cpu(), value.to(got.dtype)) if torch.is_tensor(value) else got == value
                if not equal:
                    raise RuntimeError(f'Restored optimizer value mismatch: {key}/{name}')
        step = json.loads((Path(checkpoint)/'trainer_state.json').read_text())['global_step']
        if self.lr_scheduler.state_dict()['last_epoch'] != step:
            raise RuntimeError('Restored scheduler step mismatch')
        with (Path(output)/'adapter_resume_audit.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'optimizer_states_verified': len(saved['state']),
                                'scheduler_step': step})+'\n')
    trainer._load_optimizer_and_scheduler = MethodType(load_optimizer, trainer)
