"""Versioned CoT extraction policy: never guess a final answer from a number."""
import re

POLICY = "strict_cot_v4"


def extract_final_answer(response):
    """Return (answer, provenance); use the last complete explicit answer."""
    tags = list(re.finditer(r"<answer>(.*?)</answer>", response, re.S))
    if tags:
        content = tags[-1].group(1).strip()
        answer, source = extract_final_answer(content)
        return (answer if source == "boxed" else content), "answer_tag"
    boxes = []
    for match in re.finditer(r"\\boxed\s*\{", response):
        depth, start = 1, match.end()
        for i in range(start, len(response)):
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
                if depth == 0:
                    boxes.append(response[start:i].strip())
                    break
    if boxes:
        return boxes[-1], "boxed"
    # Only accept an explicit declaration on the last nonempty line. Earlier
    # declarations followed by unfinished reasoning are not a final answer.
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        match = re.fullmatch(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*(.+)",
                             lines[-1], re.I)
        if match:
            return match.group(1).strip(), "final_answer_line"
    return None, "missing_final_answer"


def normalize_answer(answer, question=""):
    """Formatting only: preserve signs, interval endpoints and set members."""
    value = str(answer).strip().rstrip(".").strip()
    value = value.replace("$", "").replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = value.replace("^{}", "").replace(r"\!", "").replace(r"\,", "")
    # TeX command delimiters carry meaning: '\cot x' must not become '\cotx'.
    value = re.sub(r"\\(sin|cos|tan|cot|sec|csc|ln|log)\s+([a-zA-Z])\b",
                   lambda m: "\\" + m.group(1) + "{" + m.group(2) + "}", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(?:\^\{?\\circ\}?|°)$", "", value)
    # Numeric angle answers may spell out the requested unit instead of using °.
    # Restrict this to questions explicitly asking for degrees and numeric values.
    if re.search(r"\bdegrees?\b", question, re.I):
        value = re.sub(r"^([+-]?\d+(?:\.\d+)?)degrees?$", r"\1", value, flags=re.I)
    # A percent requested as the answer is reported in percentage points,
    # unlike an arbitrary factor-of-100 tolerance in numeric comparison.
    if re.search(r"(?:what|which)\s+percent|(?:find|compute|calculate).*percentage|express[^?]*as\s+(?:a\s+)?percent|what[^?]*percent(?:age)?\s+(?:increase|decrease)", question, re.I):
        value = re.sub(r"\\?%$", "", value)
    # Only strip a numeric unit suffix matching the last stated length unit.
    # Never delete letters inside algebraic expressions (e.g. a or b).
    unit_aliases = {"cm":"cm", "centimeter":"cm", "centimeters":"cm",
                    "m":"m", "meter":"m", "meters":"m",
                    "km":"km", "kilometer":"km", "kilometers":"km",
                    "mm":"mm", "millimeter":"mm", "millimeters":"mm"}
    prose_question = re.sub(r"\[asy\].*?\[/asy\]", "", question, flags=re.S | re.I)
    units = re.findall(r"\b(?:centimeters?|kilometers?|millimeters?|meters?|cm|km|mm|m)\b", prose_question, re.I)
    suffix = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(centimeters?|kilometers?|millimeters?|meters?|cm|km|mm|m)", value, re.I)
    if suffix and units and unit_aliases[suffix.group(2).lower()] == unit_aliases[units[-1].lower()]:
        value = suffix.group(1)
    if re.search(r"\bgrade\b", question, re.I):
        value = re.sub(r"^(\d+)(?:\^\{(?:st|nd|rd|th)\}|st|nd|rd|th)?grade$", r"\1", value, flags=re.I)
    # The question disambiguates comma-separated solutions from thousands.
    if re.search(r"separated\s+by\s+commas", question, re.I):
        if not (value.startswith(("{", r"\{"))):
            value = "{" + value + "}"
    return value
