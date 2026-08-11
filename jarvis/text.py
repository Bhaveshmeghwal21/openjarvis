# jarvis/text.py
"""Text normalization and span finding.

Shared by unit building (what gets stored as verbatim_text) and verification (whether a
quote genuinely appears in Layer 0). Both sides MUST normalize identically, or the quote
matcher reports fabrication for text that is actually present.

PDF extraction introduces: ligatures, hyphenation at line breaks, smart quotes and dashes,
non-breaking spaces, and irregular whitespace. None of these are semantic.
"""
from __future__ import annotations

import re
import unicodedata

_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}
_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", " ": " ", " ": " ", " ": " ", " ": " ",
}

# A hyphen followed by a line break, joining a word split across lines.
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Canonical form for storage and matching. Idempotent."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    for src, dst in _LIGATURES.items():
        out = out.replace(src, dst)
    for src, dst in _PUNCT.items():
        out = out.replace(src, dst)
    out = _HYPHEN_BREAK.sub(r"\1\2", out)
    out = _WHITESPACE.sub(" ", out)
    return out.strip()


def find_span(needle: str, haystack: str) -> tuple[int, int] | None:
    """Locate `needle` in `haystack` after normalizing both.

    Returns (start, end) offsets into `normalize(haystack)`, or None when absent.
    Matching stays exact after normalization: a changed number or word is not a match.

    Boundary-anchored: a plain substring search would let "2.5% error" match inside
    "12.5% error" — a materially different, fabricated number reported as present. A
    genuine verbatim quote always starts and ends on a token boundary in the source
    text, so a match is only accepted when the character immediately before and after
    it (if any) is not alphanumeric.
    """
    n = normalize(needle)
    if not n:
        return None
    h = normalize(haystack)
    idx = h.find(n)
    while idx >= 0:
        before_ok = idx == 0 or not h[idx - 1].isalnum()
        after_ok = (idx + len(n) == len(h)) or not h[idx + len(n)].isalnum()
        if before_ok and after_ok:
            return (idx, idx + len(n))
        idx = h.find(n, idx + 1)
    return None


def approx_tokens(text: str) -> int:
    """Rough token count without a tokenizer dependency.

    English prose averages ~1.3 tokens per whitespace word for BPE tokenizers. Used only
    for chunk sizing, where being off by 10% costs nothing.
    """
    words = len(normalize(text).split())
    return round(words * 1.3)