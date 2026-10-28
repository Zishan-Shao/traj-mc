"""Answer extraction and equivalence for the boxed-answer math columns.

AIME and Minerva Math are prompted with the MATH-500 few-shot block, so the
model ends its solution in ``\\boxed{...}``.  Grading follows the Qwen2.5-Math
evaluation grader, which is where the Minerva Math (OCW) packaging comes from:
the LAST boxed expression counts, LaTeX is normalised, numbers compare at a
relative tolerance of 1e-4, and anything left over is compared symbolically
through latex2sympy2 -- 85 of the 272 Minerva answers are expressions, not
numbers.  Each equivalence stage can be reported on its own so a column can be
restricted to the numeric-answer subset if the symbolic stage is not trusted.
"""

from __future__ import annotations

import math
import re
import signal

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` (brace-balanced) or ``\\boxed x``."""
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
        if start < 0:
            return None
    i = text.find("{", start)
    tail = text[start + len("\\boxed"):].lstrip()
    if i < 0 or not tail.startswith("{"):
        # "\boxed 5" -- takes the next token.
        m = re.match(r"\S+", tail)
        return m.group(0) if m else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return text[i + 1:]           # unbalanced (generation cut off)


def extract_answer(text: str) -> str | None:
    """Last boxed expression; failing that "the answer is X"; then the last number."""
    boxed = last_boxed(text)
    if boxed is not None and boxed.strip():
        return boxed.strip()
    m = re.findall(r"[Tt]he (?:final )?answer is[:\s]*\$?([^\n$]+)", text)
    if m:
        return m[-1].strip().rstrip(".")
    nums = NUMBER.findall(text)
    return nums[-1] if nums else None


# ---------------------------------------------------------------- normalise

_UNIT_GROUPS = re.compile(r"\\(?:text|mathrm|textrm|mbox|textbf|mathbf|operatorname)\s*\{([^{}]*)\}")


def normalize(s: str) -> str:
    """Qwen2.5-Math-style string normalisation, extended for OCW answers."""
    s = s.strip()
    s = s.replace("\n", " ").replace("\\\\", "\\")
    s = re.sub(r"\\(?:left|right|!|,|;|:|quad|qquad)\b", "", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
    s = s.replace("\\%", "").replace("\\$", "").replace("$", "")
    s = s.replace("\\cdot", "\\times")
    s = re.sub(r"\\approx", "", s)
    # "4.5 \times 10^{33}" and bare "10^{33}" -> e-notation.
    s = re.sub(r"(-?[\d.]+)\s*\\times\s*10\^\{?(-?\d+)\}?", r"\1e\2", s)
    s = re.sub(r"(?<![\d.\w])10\^\{?(-?\d+)\}?", r"1e\1", s)
    s = re.sub(r"(-?[\d.]+)\s*[eE]\s*\{?([-+]?\d+)\}?", r"\1e\2", s)
    # Unit words:  \mathrm{~m}, \text{ cm}  -> dropped.
    s = _UNIT_GROUPS.sub(lambda m: "" if re.fullmatch(r"[\s~a-zA-Z/^{}\d]*", m.group(1)) and not re.search(r"\d", m.group(1)) else m.group(1), s)
    s = s.replace("~", " ")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)      # 1,000 -> 1000
    s = s.strip().rstrip(".").strip()
    # A trailing unit word after a number ("1.6 cm", "45 degrees").
    s = re.sub(r"^(-?[\d.]+(?:e[-+]?\d+)?)\s+[a-zA-Z][a-zA-Z/^\d\s]*$", r"\1", s)
    return s


_FRAC = re.compile(r"^(-?)\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}$")
_SLASH = re.compile(r"^(-?[\d.]+)\s*/\s*(-?[\d.]+)$")


def to_number(s: str) -> float | None:
    s = normalize(s).replace(" ", "")
    if not s:
        return None
    if s.startswith("x=") or s.startswith("y="):
        s = s[2:]
    m = _FRAC.match(s)
    if m:
        try:
            return (-1 if m.group(1) else 1) * float(m.group(2)) / float(m.group(3))
        except (ValueError, ZeroDivisionError):
            return None
    m = _SLASH.match(s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------- equivalence

class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def symbolic_equal(a: str, b: str, seconds: int = 5) -> bool:
    """latex2sympy2 parse + simplify(a - b) == 0, bounded by a wall-clock alarm."""
    try:
        from latex2sympy2 import latex2sympy
        import sympy
    except Exception:
        return False
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        ea, eb = latex2sympy(normalize(a)), latex2sympy(normalize(b))
        if ea == eb:
            return True
        diff = sympy.simplify(ea - eb)
        return diff == 0 or (diff.is_number and abs(complex(diff)) < 1e-9)
    except Exception:
        return False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def numeric_close(pred: float, gold: float, rel_tol: float) -> bool:
    if math.isclose(pred, gold, rel_tol=rel_tol):
        return True
    # Qwen's include_percentage: the model may answer in percent or fraction.
    return any(math.isclose(pred, g, rel_tol=rel_tol) for g in (gold / 100, gold * 100))


def is_equiv(pred: str | None, gold: str, rel_tol: float = 1e-4) -> bool:
    """Numeric closeness, then normalised string equality, then sympy."""
    if pred is None:
        return False
    pn, gn = to_number(pred), to_number(gold)
    if pn is not None and gn is not None:
        return numeric_close(pn, gn, rel_tol)
    a, b = normalize(pred).replace(" ", ""), normalize(gold).replace(" ", "")
    if a == b:
        return True
    return symbolic_equal(pred, gold)


def aime_equiv(pred: str | None, gold: str) -> bool:
    """AIME answers are integers 0-999; leading zeros and '.0' do not matter."""
    if pred is None:
        return False
    pn = to_number(pred)
    return pn is not None and float(pn).is_integer() and int(pn) == int(gold)
