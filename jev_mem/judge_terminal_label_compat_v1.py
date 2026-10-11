"""Offline-only terminal-line Judge compatibility; never changes a model call.

Strict outcomes remain authoritative. This parser is an explicitly separate
post-failure sensitivity/recovery view, not a replacement for the frozen Judge.
"""
from __future__ import annotations

import hashlib
import re

PROTOCOL = "plain-judge-terminal-line-unanimous-label-compat-v1"
LABELS = {"[[yes]]": True, "[[no]]": False}
DECISIONS = re.compile(r"\[\[(yes|no)\]\]")


def derive_label(text):
    """Interpret text without JSON parsing, substring yes-tests or source choice."""
    if not isinstance(text, str):
        raise TypeError("Judge compatibility requires the original text string")
    tokens = [{"label": match.group(0), "start": match.start(), "end": match.end()} for match in DECISIONS.finditer(text)]
    nonempty = [(i,line.strip()) for i,line in enumerate(text.splitlines()) if line.strip()]
    exact = text.strip() in LABELS
    final = nonempty[-1][1] if nonempty else None
    label = text.strip() if exact else final
    reason = None
    if label not in LABELS:
        reason = "last_nonempty_line_not_exact_label"
    elif any(token["label"] != label for token in tokens):
        reason = "conflicting_exact_decision_label_anywhere"
    status = "strict_valid" if exact else "compat_valid" if reason is None else "rejected"
    return {"protocol": PROTOCOL, "status": status, "label": label if reason is None else None,
            "correct": LABELS[label] if reason is None else None, "rejection_reason": reason,
            "raw": text, "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "last_nonempty_line_index": nonempty[-1][0] if nonempty else None,
            "exact_decision_tokens": tokens, "raw_modified": False}


def eligible_raw_response(response, text, *, http_status):
    """Only the observed normal-text shape may extend a failed strict result."""
    if http_status != 200:
        return False, "producer_http_status_not_200"
    candidates = response.get("candidates") if isinstance(response,dict) else None
    if not isinstance(candidates,list) or len(candidates)!=1:
        return False, "requires_single_candidate"
    candidate = candidates[0]
    if not isinstance(candidate,dict) or candidate.get("finishReason") != "STOP":
        return False, "generation_not_complete_STOP"
    content = candidate.get("content",{})
    parts = content.get("parts") if isinstance(content,dict) else None
    if (not isinstance(parts,list) or len(parts)!=1 or not isinstance(parts[0],dict)
            or set(parts[0]) not in ({"text"}, {"text","thoughtSignature"}) or parts[0]["text"] != text
            or ("thoughtSignature" in parts[0] and not isinstance(parts[0]["thoughtSignature"],str))):
        return False, "requires_single_ordinary_text_part_equal_to_original_output"
    return True, None
