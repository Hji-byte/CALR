"""CALR: correct-only bounded excess-length reward, no advantage changes."""
import math
from training.group_rewards import group_rewards, GroupReward
from training.grpo_monitor import append_json


def calr_rewards(correctness, lengths, question_ids, truncated, *, alpha=.1, tau=256,
                   exclude_truncated_correct=False):
    if not math.isfinite(alpha) or not 0 < alpha < 1 or not math.isfinite(tau) or tau <= 0:
        raise ValueError('Require 0 < alpha < 1 and finite tau > 0')
    out=group_rewards(correctness,lengths,question_ids,truncated,alpha=1,
                      exclude_truncated_correct=exclude_truncated_correct)
    for indices in out['groups']:
        good=[lengths[i] for i in indices if out['correctness'][i]]
        target=min(good) if good else None
        for i in indices:
            penalty=0.
            if target is not None and out['correctness'][i]:
                d=max(0,lengths[i]-target)/max(target,tau)
                penalty=alpha*d/(1+d)
            out['targets'][i]=target
            out['penalties'][i]=penalty
            out['rewards'][i]=out['correctness'][i]-penalty
    return out


class CALRReward(GroupReward):
    def __init__(self,score,output,tokenizer,alpha=.1,tau=256,exclude_truncated_correct=False):
        super().__init__(score,output,tokenizer,1,exclude_truncated_correct)
        self.alpha,self.tau=alpha,tau
        self.__name__='direct_rl_calr'

    def __call__(self,completions,answer,question,question_id,completion_ids,trainer_state,**kwargs):
        n=len(completions)
        if any(len(x)!=n for x in (answer,question,question_id,completion_ids)):
            raise ValueError('Misaligned completion metadata')
        lengths=[len(ids) for ids in completion_ids]
        truncated=[not ids or ids[-1] not in self.stop_ids for ids in completion_ids]
        raw=self.score(completions,answer,question)
        out=calr_rewards(raw,lengths,question_id,truncated,alpha=self.alpha,tau=self.tau,
                           exclude_truncated_correct=self.exclude_truncated_correct)
        for indices in out['groups']:
            if len({question[i] for i in indices})!=1 or len({answer[i] for i in indices})!=1:
                raise ValueError('Conflicting group metadata')
        for indices in out['groups']:
            pick=lambda v:[v[i] for i in indices]
            cs=pick(out['correctness'])
            append_json(self.output/'rollout_groups.jsonl',dict(step=trainer_state.global_step,
                question_id=question_id[indices[0]],reward_mode='calr',alpha=self.alpha,tau=self.tau,
                rewards=pick(out['rewards']),correctness=cs,raw_answer_correctness=pick(raw),
                target_lengths=pick(out['targets']),length_penalties=pick(out['penalties']),
                output_tokens=pick(lengths),truncated=pick(truncated),all_correct=all(cs),all_wrong=not any(cs),
                completions=pick(completions),exclude_truncated_correct=self.exclude_truncated_correct))
        return out['rewards']
