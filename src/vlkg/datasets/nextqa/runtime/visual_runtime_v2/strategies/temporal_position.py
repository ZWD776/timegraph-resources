"""Question-IR temporal position constraints for NExT-QA visual strategies."""
from __future__ import annotations
import re
from typing import Any
POSITION_BANDS = {'video_beginning': (0.0, 1.0 / 3.0), 'video_middle': (1.0 / 3.0, 2.0 / 3.0), 'video_end': (2.0 / 3.0, 1.0)}
_POSITION_PATTERNS = (('video_beginning', re.compile('\\b(?:at|near|from|in)\\s+(?:the\\s+)?(?:start|beginning)\\b|\\binitially\\b')), ('video_middle', re.compile('\\b(?:at|near|from|in)\\s+(?:the\\s+)?middle\\b|\\bmidway\\b')), ('video_end', re.compile('\\b(?:at|near|towards?|from|in)\\s+(?:the\\s+)?end\\b|\\bfinally\\b')))

def infer_position_hint(question: str, ir_anchor: Any) -> tuple[str | None, str]:
    """Return a normalized position hint and whether it came from IR or question text."""
    normalized_ir = str(ir_anchor or '').strip().lower()
    if normalized_ir in POSITION_BANDS:
        return (normalized_ir, 'ir')
    normalized_question = ' '.join(str(question or '').strip().lower().split())
    for hint, pattern in _POSITION_PATTERNS:
        if pattern.search(normalized_question):
            return (hint, 'question_text_fallback')
    return (None, 'none')

def position_matches(hint: str | None, anchor_ratio: float | None) -> bool | None:
    """Check a coarse beginning/middle/end constraint against normalized video time."""
    if hint is None or anchor_ratio is None:
        return None
    lower, upper = POSITION_BANDS[hint]
    return lower - 1e-09 <= anchor_ratio <= upper + 1e-09
