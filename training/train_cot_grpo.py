"""Single-GPU GRPO or CALR training with audited LoRA synchronization and resume.

Run as a module from the repository root. No teacher responses enter RL prompts.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def validate_resume(checkpoint, output, manifest):
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent != output.resolve():
        raise ValueError('Checkpoint must belong to this output directory')
    for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                 'monitor_state.json', 'adapter_model.safetensors'):
        if not (checkpoint/name).is_file():
            raise ValueError(f'Incomplete checkpoint: missing {name}')
    old = json.loads((output/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['arguments'].get('extend_training_budget', False):
        old_cfg, new_cfg = old['training_config'], manifest['training_config']
        if (old_cfg.get('lr_scheduler_type') != 'constant' or old_cfg.get('warmup_steps') != 0
                or new_cfg['max_steps'] < old_cfg['max_steps']):
            raise ValueError('Budget extension requires constant LR, zero warmup and nondecreasing steps')
        old = dict(old, training_config=dict(old_cfg, max_steps=new_cfg['max_steps']))
    for key in ('training_config', 'data_sha256', 'versions', 'reward'):
        if old[key] != manifest[key]:
            raise ValueError(f'Resume configuration mismatch: {key}')
    for key in ('model', 'adapter', 'smoke'):
        if old['arguments'][key] != manifest['arguments'][key]:
            raise ValueError(f'Resume argument mismatch: {key}')
    step = json.loads((checkpoint/'trainer_state.json').read_text())['global_step']
    if step < 1 or step >= manifest['training_config']['max_steps']:
        raise ValueError('Checkpoint step outside resumable training plan')
    return step


def config_kwargs(args):
    """Accumulate rollout groups without increasing simultaneous generation."""
    microbatch = getattr(args, 'microbatch', 1)
    if microbatch not in (1, 2, 4):
        raise ValueError('Microbatch must be 1, 2 or 4')
    update_batch = getattr(args, 'update_batch_size', 64)
    if update_batch not in (64, 128):
        raise ValueError('Update batch size must be 64 or 128 answers')
    config = dict(output_dir=str(args.output), per_device_train_batch_size=microbatch,
        gradient_accumulation_steps=(8 if args.smoke else update_batch)//microbatch,
        generation_batch_size=getattr(args,'generation_batch_size',8), num_generations=8,
        max_completion_length=args.max_completion_length,
        max_steps=2 if args.smoke else args.max_steps, learning_rate=args.lr,
        lr_scheduler_type='constant', warmup_steps=0, weight_decay=0.0, max_grad_norm=1.0,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, disable_dropout=True,
        temperature=0.6, top_p=getattr(args, 'top_p', 0.95), top_k=0,
        chat_template_kwargs={'enable_thinking': False},
        use_vllm=False, beta=args.kl_beta, loss_type='grpo', scale_rewards='group',
        mask_truncated_completions=getattr(args, 'mask_truncated_completions', True),
        num_iterations=1, epsilon=0.2, seed=42, data_seed=42,
        logging_steps=1, save_steps=args.eval_steps, save_total_limit=2, report_to='none')
    if getattr(args, 'vllm_model_copy', None):
        config.update(use_vllm=True, vllm_mode='colocate',
            vllm_gpu_memory_utilization=0.25, vllm_max_model_length=6400,
            vllm_enable_sleep_mode=True, vllm_importance_sampling_mode='sequence_mask',
            vllm_importance_sampling_correction=True)
    return config


def load_rows():
    rows = json.loads((DATA/'train.json').read_text(encoding='utf-8'))['examples']
    test = json.loads((ROOT/'data/math500.json').read_text(encoding='utf-8'))['examples']
    normalize = lambda s: ''.join(s.lower().split())
    forbidden = {normalize(r['question']) for r in test}
    for row in rows:
        prompt = row['prompt']
        if [m['role'] for m in prompt] != ['system','user'] or prompt[-1]['content'] != row['question']:
            raise ValueError('Question mismatch')
        if normalize(row['question']) in forbidden:
            raise ValueError('Training/heldout exact question overlap')
    if len(rows) != 1887 or len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError('Expected 1887 unique paired training questions')
    return rows


def correctness_reward(completions, answer, question, **kwargs):
    from evaluation.math_scoring import extract_final_answer, normalize_answer
    from evaluation.qwen_math_grader import math_equal
    rewards = []
    for completion, gold, q in zip(completions, answer, question, strict=True):
        text = completion if isinstance(completion, str) else completion[-1]['content']
        pred, _ = extract_final_answer(text)
        rewards.append(float(pred is not None and bool(pred.strip()) and math_equal(
            normalize_answer(pred, q), normalize_answer(gold, q), timeout=True)))
    return rewards


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    start = p.add_mutually_exclusive_group(required=True)
    start.add_argument('--adapter', help='Continue an existing SFT adapter (legacy profile)')
    start.add_argument('--from-base', action='store_true', help='Fresh LoRA on the unmodified post-trained model')
    p.add_argument('--reward-mode', choices=['correctness', 'calr'], default='correctness')
    p.add_argument('--calr-tau', type=float, default=256)
    p.add_argument('--reward-alpha', type=float, default=None,
                   help='Default: 1 for correctness, 0.1 for CALR')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--signal-audit', action='store_true', help='Log actual loss masks, advantages and correction eligibility')
    p.add_argument('--sample-signal-audit', action='store_true')
    p.add_argument('--extend-training-budget', action='store_true', help='Explicit constant-LR max_steps extension only')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--pilot', action='store_true', help='Explicitly authorize a short pilot after GPU smoke')
    p.add_argument('--max-steps', type=int, default=100)
    p.add_argument('--microbatch', type=int, choices=[1, 2, 4], default=1)
    p.add_argument('--update-batch-size', type=int, choices=[64, 128], default=64,
                   help='Answers per optimizer update on one GPU; smoke retains 8')
    p.add_argument('--max-completion-length', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-6)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--kl-beta', type=float, default=0.01)
    p.add_argument('--eval-steps', type=int, default=10)
    p.add_argument('--generation-batch-size', type=int, choices=[8,16,32], default=8)
    p.add_argument('--mask-truncated-completions', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--exclude-truncated-correct', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume-from-checkpoint', type=str)
    p.add_argument('--pause-after-step', type=int, help='Absolute update at which to save and pause; does not alter total schedule')
    p.add_argument('--vllm-model-copy', help='Compatible vLLM initialization copy; live weights are synchronized from SFT')
    args = p.parse_args()
    if args.extend_training_budget and not args.resume_from_checkpoint:
        p.error('--extend-training-budget requires a full checkpoint')
    if args.reward_alpha is None:
        args.reward_alpha = .1 if args.reward_mode == 'calr' else 1.0
    import math
    if args.reward_mode == 'calr' and (not 0 < args.reward_alpha < 1 or not math.isfinite(args.calr_tau) or args.calr_tau <= 0):
        p.error('CALR requires 0 < reward-alpha < 1 and finite calr-tau > 0')
    if not 0 < args.top_p <= 1:
        p.error('Require 0 < top-p <= 1')
    if not math.isfinite(args.reward_alpha) or args.reward_alpha <= 0:
        p.error('Require finite positive alpha')
    if args.lr <= 0 or args.kl_beta < 0 or args.eval_steps < 1:
        p.error('Require lr > 0, kl-beta >= 0, eval-steps >= 1')
    if not args.preflight and args.smoke == args.pilot:
        p.error('Choose exactly one of --smoke or --pilot; no implicit long run')
    if args.smoke and args.generation_batch_size != 8:
        p.error('Use --pilot for batch16; smoke uses one question/update')
    if args.pause_after_step is not None and (args.smoke or not 0 < args.pause_after_step < args.max_steps):
        p.error('Pause requires pilot and 0 < pause step < planned max steps')
    rows = load_rows()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lengths = [len(tokenizer.apply_chat_template(r['prompt'], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)) for r in rows]
    manifest = dict(arguments=vars(args) | {'output': str(args.output)}, training_rows=len(rows),
        max_prompt_tokens=max(lengths), enable_thinking=False, reward='strict_cot_v4 correctness only',
        data_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
    manifest['training_config'] = config_kwargs(args)
    if args.reward_mode in ('correctness', 'calr'):
        from evaluation.vllm_cot_math500 import source_signature
        manifest['reward'] = dict(mode=args.reward_mode, alpha=args.reward_alpha,
            group_size=8,
            truncation='exclude_from_correct_candidates; all_lengths_in_mean_fallback; mask_loss',
            protocol='direct_rl_group_v1')
        if not args.mask_truncated_completions or not args.exclude_truncated_correct:
            manifest['reward']['truncation'] = dict(mask_loss=args.mask_truncated_completions,
                exclude_from_correct_candidates=args.exclude_truncated_correct,
                all_lengths_in_mean_fallback=True)
        manifest['versions_pending_source'] = dict(
            model=source_signature(Path(args.model)),
            adapter=source_signature(Path(args.adapter)) if args.adapter else None)
        # Include immutable source identity in existing resume comparison keys.
        manifest['reward']['source'] = manifest.pop('versions_pending_source')
        manifest['reward']['from_base'] = args.from_base
        manifest['reward']['implementation_sha256'] = hashlib.sha256(
            (ROOT/'training/group_rewards.py').read_bytes()).hexdigest()
        if args.reward_mode == 'calr':
            manifest['reward'].update(tau=args.calr_tau, formula='c*(1-alpha*d/(1+d)); d=max(0,L-Lmin_correct)/max(Lmin_correct,tau); all_wrong=0',
                calr_sha256=hashlib.sha256(
                    (ROOT/'training/calr_reward.py').read_bytes()).hexdigest())
        if args.from_base:
            from training.lora_config import TARGETS
            manifest['reward']['fresh_lora'] = dict(r=64, alpha=128, dropout=0., targets=TARGETS)
    print(json.dumps(manifest, indent=2))
    if args.preflight:
        return
    import importlib.metadata as metadata
    if metadata.version('trl') != '1.12.0':
        raise RuntimeError('Use isolated TRL 1.12.0 environment; do not upgrade the SFT environment')
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, set_seed
    from trl import GRPOConfig, GRPOTrainer
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU required')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This initial profile requires BF16')
    manifest['versions'] = {k: metadata.version(k) for k in ['trl','torch','transformers','peft']}
    if args.vllm_model_copy:
        manifest['versions']['vllm'] = metadata.version('vllm')
        manifest['versions']['exact_sync_protocol'] = 'base_restore_v1'
        if args.resume_from_checkpoint:
            old = json.loads((args.output/'run_manifest.json').read_text())
            if old['arguments'].get('vllm_model_copy') != args.vllm_model_copy:
                raise ValueError('Resume vLLM model copy mismatch')
    if args.resume_from_checkpoint:
        resume_step = validate_resume(args.resume_from_checkpoint, args.output, manifest)
        if args.pause_after_step is not None and args.pause_after_step <= resume_step:
            raise ValueError('Pause must be after the checkpoint step')
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output is not empty; use an explicit full checkpoint to resume')
    if args.max_steps < 1 or args.max_completion_length < 1:
        raise ValueError('Step and completion limits must be positive')
    set_seed(42)
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        dtype=torch.bfloat16, device_map={'': 0})
    if args.from_base:
        from peft import LoraConfig, get_peft_model
        from training.lora_config import TARGETS
        model = get_peft_model(base, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.,
            target_modules=TARGETS, bias='none', task_type='CAUSAL_LM', init_lora_weights=True))
        if any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if '.lora_B.' in n):
            raise RuntimeError('Fresh LoRA must preserve starting model exactly')
    else:
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    if args.vllm_model_copy:
        model.name_or_path = args.vllm_model_copy
    config = GRPOConfig(**config_kwargs(args))
    from training.grpo_monitor import make_monitor
    args.output.mkdir(parents=True, exist_ok=True)
    if args.reward_mode == 'calr':
        from training.calr_reward import CALRReward
        reward = CALRReward(correctness_reward,args.output,tokenizer,args.reward_alpha,args.calr_tau,
                              exclude_truncated_correct=args.exclude_truncated_correct)
    else:
        from training.group_rewards import GroupReward
        reward = GroupReward(correctness_reward, args.output, tokenizer,
                             args.reward_alpha,
                             exclude_truncated_correct=args.exclude_truncated_correct)
    trainer = GRPOTrainer(model=model, args=config, processing_class=tokenizer,
        reward_funcs=reward, train_dataset=Dataset.from_list(rows[:16] if args.smoke else rows))
    from training.grpo_resume import install_adapter_resume
    install_adapter_resume(trainer, args.output)
    if args.sample_signal_audit:
        from training.grpo_sample_audit import install_sample_audit
        install_sample_audit(trainer, reward, args.output)
    if args.signal_audit:
        from training.grpo_signal_audit import install_signal_audit
        install_signal_audit(trainer, args.output)
    if args.vllm_model_copy:
        from training.grpo_exact_sync import install_exact_sync
        install_exact_sync(trainer, args.output)
    if args.kl_beta:
        if 'ref' not in model.peft_config:
            raise RuntimeError('Missing frozen SFT reference adapter')
        if any(p.requires_grad for n, p in model.named_parameters() if '.ref.' in n):
            raise RuntimeError('Reference adapter must be frozen')
    monitor = make_monitor(trainer, args)
    trainer.add_callback(monitor)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume_from_checkpoint:
        (args.output/'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    else:
        from training.grpo_monitor import append_json
        append_json(args.output/'resume_plans.jsonl', manifest)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    paused = trainer.state.global_step < config.max_steps
    if not paused:
        trainer.save_model(str(args.output/'final'))
        tokenizer.save_pretrained(args.output/'final')
    else:
        validate_resume(args.output/f'checkpoint-{trainer.state.global_step}', args.output, manifest)
    (args.output/'training_status.json').write_text(json.dumps({
        'steps': trainer.state.global_step, 'adapter_changed': monitor.changed,
        'paused': paused, 'resumed_from': args.resume_from_checkpoint,
        'gpu_peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
        'note': 'Smoke checks execution only; not evidence of accuracy preservation.'}, indent=2), encoding='utf-8')
    if not monitor.changed:
        raise RuntimeError('No adapter update detected; do not proceed to a longer run')


if __name__ == '__main__':
    main()
