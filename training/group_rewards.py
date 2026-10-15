"""Group rewards for the controlled direct-RL comparison; no torch dependency.

Truncated correct candidates can optionally be excluded.
The trainer separately configures whether truncated loss is masked.
"""
import math
from collections import defaultdict
from training.grpo_monitor import append_json


def group_rewards(correctness, lengths, question_ids, truncated, *,
                  alpha=1.0, group_size=8, exclude_truncated_correct=True):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Invalid reward coefficients')
    n = len(correctness)
    if not n or any(len(x) != n for x in (lengths, question_ids, truncated)):
        raise ValueError('Mismatched/empty reward batch')
    if any(c not in (0, 1) for c in correctness):
        raise ValueError('Correctness must be binary')
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 1 for x in lengths):
        raise ValueError('Nonempty token sequences required')
    groups = defaultdict(list)
    for i, qid in enumerate(question_ids):
        groups[qid].append(i)
    if group_size < 2 or any(len(g) != group_size for g in groups.values()):
        raise ValueError('Expected a complete group per question_id')
    eligible = [int(bool(c) and (not cut or not exclude_truncated_correct)) for c, cut in zip(correctness, truncated)]
    rewards, targets, penalties = [0.] * n, [0.] * n, [0.] * n
    for indices in groups.values():
        correct_lengths = [lengths[i] for i in indices if eligible[i]]
        target = min(correct_lengths) if correct_lengths else sum(lengths[i] for i in indices)/len(indices)
        for i in indices:
            targets[i] = target
            rewards[i] = alpha * eligible[i]
    return dict(rewards=rewards, targets=targets, penalties=penalties,
                correctness=eligible, groups=list(groups.values()))


class GroupReward:
    def __init__(self, score, output, tokenizer, alpha=1.0,
                 exclude_truncated_correct=True):
        self.score, self.output = score, output
        self.mode, self.alpha = 'correctness', alpha
        self.exclude_truncated_correct = exclude_truncated_correct
        self.__name__ = 'direct_rl_correctness'
        eos = tokenizer.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])
        # Match pinned TRL's end-of-generation detection.
        self.stop_ids.add(tokenizer.pad_token_id)
        self.stop_ids.discard(None)

    def __call__(self, completions, answer, question, question_id,
                 completion_ids, trainer_state, **kwargs):
        n = len(completions)
        if any(len(x) != n for x in (answer, question, question_id, completion_ids)):
            raise ValueError('Misaligned completion metadata')
        lengths = [len(ids) for ids in completion_ids]
        truncated = [not ids or ids[-1] not in self.stop_ids for ids in completion_ids]
        raw = self.score(completions, answer, question)
        result = group_rewards(raw, lengths, question_id, truncated,
                               alpha=self.alpha,
                               exclude_truncated_correct=self.exclude_truncated_correct)
        for indices in result['groups']:
            if len({question[i] for i in indices}) != 1 or len({answer[i] for i in indices}) != 1:
                raise ValueError('Conflicting question/answer within question_id')
        for indices in result['groups']:
            pick = lambda values: [values[i] for i in indices]
            correct = pick(result['correctness'])
            append_json(self.output/'rollout_groups.jsonl', dict(
                step=trainer_state.global_step, question_id=question_id[indices[0]],
                reward_mode=self.mode, alpha=self.alpha,
                rewards=pick(result['rewards']), correctness=correct,
                raw_answer_correctness=pick(raw), target_lengths=pick(result['targets']),
                length_penalties=pick(result['penalties']), output_tokens=pick(lengths),
                truncated=pick(truncated), all_correct=all(correct), all_wrong=not any(correct),
                exclude_truncated_correct=self.exclude_truncated_correct,
                completions=pick(completions)))
        return result['rewards']
