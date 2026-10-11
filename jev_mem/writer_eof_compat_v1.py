"""Strict one-byte EOF root closure; never alter an API cache or fact value."""
from __future__ import annotations

import copy
import hashlib
import json

PROTOCOL = "mem0-writer-single-eof-root-close-v1"


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key forbids EOF compatibility")
        result[key] = value
    return result


def _skip(text, pos):
    while pos < len(text) and text[pos] in " \r\n\t":
        pos += 1
    return pos


def derive_response(response, *, http_status):
    """Return (response, proof). Valid originals are returned without mutation.

    For malformed originals the *only* accepted change is appending one ASCII
    outer object brace. The original memory array is decoded independently.
    This function does not infer HTTP status from STOP or from cache existence.
    """
    candidates = response.get("candidates", [])
    if not candidates or candidates[0].get("finishReason") != "STOP":
        raise ValueError("Incomplete generation is not EOF compatible")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    try:
        json.loads(text)
    except json.JSONDecodeError as error:
        failure = {"type": type(error).__name__, "message": error.msg,
                   "position": error.pos, "length": len(text)}
        if error.pos != len(text):
            raise ValueError("JSON error is not at EOF") from error
    else:
        return response, {"protocol": PROTOCOL, "action": "unchanged",
                          "original_text_sha256": text_sha(text)}
    if (http_status != 200 or len(candidates) != 1 or len(parts) != 1
            or set(parts[0]) != {"text"} or not isinstance(parts[0]["text"], str)):
        raise ValueError("EOF compatibility requires HTTP200/STOP/single plain text part")
    # Explicit string/escape-aware balance check; no incomplete inner value,
    # string, array, or second root can be repaired by this policy.
    stack, in_string, escaped = [], False, False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack.pop() != ("{" if char == "}" else "["):
                raise ValueError("Mismatched JSON delimiter")
    if in_string or escaped or stack != ["{"]:
        raise ValueError("Exactly one unclosed root object is required")
    decoder = json.JSONDecoder(object_pairs_hook=_no_duplicates)
    pos = _skip(text, 0)
    if pos == len(text) or text[pos] != "{":
        raise ValueError("Root must be an object")
    key, pos = decoder.raw_decode(text, _skip(text, pos + 1))
    pos = _skip(text, pos)
    if key != "memory" or pos == len(text) or text[pos] != ":":
        raise ValueError("Root must contain only memory")
    memory, end = decoder.raw_decode(text, _skip(text, pos + 1))
    if not isinstance(memory, list) or _skip(text, end) != len(text):
        raise ValueError("Original memory array is not complete and terminal")
    # Exact original ExtractionLLM record acceptance, without coercion.
    if any(not isinstance(m, dict) or not isinstance(m.get("text"), str)
           or not m["text"].strip() for m in memory):
        raise ValueError("Original memory record is invalid")
    derived = text + "}"
    data = decoder.decode(derived)
    if list(data) != ["memory"] or data["memory"] != memory:
        raise ValueError("Derived root does not preserve the original array")
    value = copy.deepcopy(response)
    value["candidates"][0]["content"]["parts"][0]["text"] = derived
    return value, {"protocol": PROTOCOL, "action": "append_single_root_brace",
        "original_error": failure, "original_text_sha256": text_sha(text),
        "derived_text_sha256": text_sha(derived), "appended_utf8_hex": "7d",
        "original_prefix_byte_identical": True, "records": len(memory),
        "memory_sha256": text_sha(json.dumps(memory, ensure_ascii=False, sort_keys=True)),
        "http_status": http_status, "finish_reason": "STOP"}
