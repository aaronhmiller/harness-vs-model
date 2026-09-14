"""Answer extraction and matching.

Deliberately identical across harnesses. If H1 and H2 were parsed with different
leniency, the "harness effect" would partly be a parser effect -- which is
exactly the kind of silent confound this experiment is meant to avoid.
"""
from __future__ import annotations

import re
import unicodedata

CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
ANSWER_LINE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_code(text: str) -> str:
    """Pull the last fenced Python block; fall back to the raw text."""
    blocks = CODE_FENCE.findall(text or "")
    if blocks:
        return blocks[-1].strip()
    return (text or "").strip()


def extract_answer(text: str) -> str:
    """Pull the value after the last 'ANSWER:'; fall back to the last line."""
    text = (text or "").strip()
    matches = ANSWER_LINE.findall(text)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^\w\s.\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(".")


def as_number(s: str) -> float | None:
    m = NUMBER.search((s or "").replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def matches(candidate: str, accepted: list[str], numeric: bool) -> bool:
    """True if `candidate` states one of the `accepted` answers.

    Numeric items compare values with a small tolerance so that "$0.30",
    "0.3 dollars" and "30 cents"->0.30 all resolve consistently. Non-numeric
    items require the accepted string to appear as a whole phrase, so that
    "second" does not accidentally match inside "secondary".
    """
    cand = normalize(candidate)
    if not cand:
        return False

    if numeric:
        cv = as_number(cand)
        if cv is None:
            return False
        for a in accepted:
            av = as_number(a)
            if av is not None and abs(cv - av) <= max(0.01, abs(av) * 0.005):
                return True
        return False

    for a in accepted:
        an = normalize(a)
        if not an:
            continue
        if cand == an or re.search(rf"(?<!\w){re.escape(an)}(?!\w)", cand):
            return True
    return False
