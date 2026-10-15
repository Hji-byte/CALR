"""Strict answer-only AIME scoring. No symbolic evaluation or numeric guessing."""
import re

POLICY = 'strict_aime_v1'


def integer_answer(value):
    """Accept an integer spelling, with optional display wrappers; never round."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = str(value).strip()
    for left, right in [('$', '$'), (r'\(', r'\)'), (r'\[', r'\]'),
                        (r'\mathrm{', '}'), (r'\text{', '}')]:
        if text.startswith(left) and text.endswith(right):
            text = text[len(left):-len(right)].strip()
    if not re.fullmatch(r'[0-9]{1,3}', text):
        return None
    return int(text)


def extract_aime_answer(response):
    if not isinstance(response, str):
        raise TypeError('response must be text')
    # Explicit hidden thinking must not supply the final answer.
    if '</think>' in response:
        response = response.rsplit('</think>', 1)[1]
    if '<think>' in response:
        response = response.split('<think>', 1)[0]
    candidates = []
    for m in re.finditer(r'\\boxed\s*\{|<answer>', response):
        if m.group() == '<answer>':
            end = response.find('</answer>', m.end())
            value = response[m.end():end].strip() if end >= 0 else None
            if value and value.startswith(r'\boxed{') and value.endswith('}'):
                value = value[7:-1]
            candidates.append((m.start(), value, 'answer_tag'))
        else:
            depth, value = 1, None
            for i in range(m.end(), len(response)):
                depth += (response[i] == '{') - (response[i] == '}')
                if depth == 0:
                    value = response[m.end():i]
                    break
            candidates.append((m.start(), value, 'boxed'))
    lines = list(re.finditer(r'[^\n\r]+', response))
    nonempty = [m for m in lines if m.group().strip()]
    if nonempty:
        last = nonempty[-1]
        m = re.fullmatch(r'\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.*?)\s*', last.group(), re.I)
        if m and r'\boxed' not in m[1] and '<answer>' not in m[1]:
            candidates.append((last.start(), m[1].removesuffix('.').strip(), 'final_answer_line'))
    if not candidates:
        return None, 'missing_final_answer'
    _, raw, source = max(candidates, key=lambda x: x[0])
    return integer_answer(raw), source


def score_aime(response, answer, accepted_answers=None):
    golds = [answer] if accepted_answers is None else accepted_answers
    if not isinstance(golds, (list, tuple)) or not golds:
        raise ValueError('Expected nonempty gold answer list')
    values = [integer_answer(x) for x in golds]
    if any(x is None for x in values) or integer_answer(answer) not in values:
        raise ValueError('Invalid or inconsistent AIME gold answer')
    pred, source = extract_aime_answer(response)
    return dict(generated_answer=None if pred is None else str(pred),
                score=pred is not None and pred in values,
                extraction_source=source, scoring_policy=POLICY,
                reasoning_verified=False)
