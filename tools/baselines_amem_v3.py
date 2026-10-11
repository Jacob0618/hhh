"""Versioned native action contract and lossless JSON TAB representation bridge.

No provider, recovery scheduler, cache admission, or mutation of v1/v2 lives here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import types
from pathlib import Path

from tools import baselines_amem_v1 as _v1
from tools import baselines_amem_v2 as _v2
from jev_mem.sdk20_cached_api import digest

BASE_V2_SHA256 = "91a376242d1179fbbbef8f4beb887e94fdeb31de58151233da3b7679530a924f"
SCHEMA = "amem-original-adapter-v3-native-action-contract"
JSON_POLICY = "strict-json-with-lossless-string-literal-tab-escape-v1"
REPLACE_POLICY = "windows-5-32-33-same-bytes-five-attempts-50ms-v1"
UPSTREAM_REVISION = _v2.UPSTREAM_REVISION
SOURCE_SHA256 = _v2.SOURCE_SHA256
SEED_K = _v2.SEED_K
AMemError = _v1.AMemError


class AMemV3Error(AMemError):
    pass


def _write_json(path: Path, value) -> None:
    """Retry only publication of already serialized bytes, never computation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # Match v1 Path.write_text(..., newline=None) bytes on the current platform.
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    payload = text.replace("\n", os.linesep).encode("utf-8")
    expected = hashlib.sha256(payload).digest()
    temporary.write_bytes(payload)
    for attempt in range(5):
        if hashlib.sha256(temporary.read_bytes()).digest() != expected:
            raise AMemV3Error("Serialized temporary JSON changed before publication")
        try:
            temporary.replace(path)
            return
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 4:
                raise
            time.sleep(0.05)


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AMemV3Error(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise AMemV3Error(f"Non-JSON numeric constant: {value}")


def _strict_json(text):
    value = json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise AMemV3Error("Generation JSON must be an object")
    return value


def decode_generation_json(raw_record: dict, *, raw_artifact_sha256: str):
    """Pure bridge for a separately admitted, immutable provider raw artifact.

    The caller verifies the artifact bytes/hash and request/context admission.
    This function validates response eligibility and preserves every JSON value.
    It never guesses missing content or repairs action spellings.
    """
    if not isinstance(raw_artifact_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_artifact_sha256):
        raise AMemV3Error("A bound raw-artifact SHA256 is required")
    signature = raw_record.get("request", {})
    body = signature.get("body", {})
    config = body.get("generationConfig", {})
    expected_config = {"thinkingConfig": {"thinkingBudget": 0}, "maxOutputTokens": 16384,
                       "responseMimeType": "application/json"}
    if (signature.get("provider") != "gemini" or signature.get("model") != "gemini-2.5-flash"
            or signature.get("operation") != "generateContent" or config != expected_config
            or set(body) - {"contents", "systemInstruction", "generationConfig"}):
        raise AMemV3Error("JSON bridge request differs from the fixed generation contract")
    contents = body.get("contents")
    if (not isinstance(contents, list) or len(contents) != 1 or contents[0].get("role") != "user"
            or not isinstance(contents[0].get("parts"), list)
            or not contents[0]["parts"] or any(set(part) != {"text"} or not isinstance(part["text"], str)
                                             for part in contents[0]["parts"])):
        raise AMemV3Error("JSON bridge requires the original plain-text user request")
    response = raw_record.get("response", {})
    candidates = response.get("candidates")
    if (response.get("modelVersion") != "gemini-2.5-flash" or response.get("promptFeedback")
            or not isinstance(candidates, list) or len(candidates) != 1):
        raise AMemV3Error("JSON bridge requires one unblocked candidate from the fixed model")
    candidate = candidates[0]
    content = candidate.get("content", {})
    parts = content.get("parts")
    if (candidate.get("finishReason") != "STOP" or candidate.get("index") != 0
            or content.get("role") != "model" or not isinstance(parts, list) or len(parts) != 1
            or set(parts[0]) != {"text"} or not isinstance(parts[0]["text"], str)
            or any(rating.get("blocked") for rating in candidate.get("safetyRatings", []))):
        raise AMemV3Error("JSON bridge requires a complete plain-text STOP response")
    text = parts[0]["text"]
    try:
        value = _strict_json(text)
        derived, changes, initial_error = text, [], None
    except json.JSONDecodeError as exc:
        initial_error = {"message": exc.msg, "offset": exc.pos, "line": exc.lineno, "column": exc.colno}
        inside, escaped, changes, segments, byte_offset = False, False, [], [], 0
        for offset, char in enumerate(text):
            replacement = char
            if inside:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    inside = False
                elif ord(char) < 32:
                    if char != "\t":
                        raise AMemV3Error("Only a literal TAB inside a JSON string can be escaped")
                    replacement = "\\t"
                    changes.append({"character_offset": offset, "utf8_byte_offset": byte_offset,
                                    "from_codepoint": "U+0009", "from_utf8_hex": "09", "to_utf8_hex": "5c74"})
            elif char == '"':
                inside = True
            segments.append(replacement)
            byte_offset += len(char.encode("utf-8"))
        if inside or escaped or not changes:
            raise AMemV3Error("Incomplete JSON or no eligible literal TAB; no repair permitted")
        derived = "".join(segments)
        try:
            value = _strict_json(derived)
        except (json.JSONDecodeError, ValueError) as error:
            raise AMemV3Error("TAB-only escaping does not produce complete strict JSON") from error
    journal = {"policy": JSON_POLICY, "raw_artifact_sha256": raw_artifact_sha256,
               "request_hash": digest(signature), "eligibility": "fixed-model-single-plain-STOP-JSON",
               "original_text_utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
               "derived_text_utf8_sha256": hashlib.sha256(derived.encode("utf-8")).hexdigest(),
               "parsed_payload_sha256": _v1._sha(value), "changes": changes,
               "initial_strict_error": initial_error, "original_raw_unchanged": True}
    return value, journal


def _private_v2():
    path = Path(_v2.__file__).resolve()
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != BASE_V2_SHA256:
        raise AMemV3Error("Frozen v2 adapter hash mismatch")
    module = types.ModuleType("_jev_amem_v3_private_v2")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    module.SCHEMA = SCHEMA
    module._runtime.SCHEMA = SCHEMA
    return module


_private = _private_v2()
_runtime = _private._runtime
# V2's controller/export helpers resolve `_v1._write_json`; give only this
# private module a facade. Never assign to imported frozen module objects.
_private_v1_facade = types.ModuleType("_jev_amem_v3_private_v1_facade")
_private_v1_facade.__dict__.update(vars(_v1))
_private_v1_facade._write_json = _write_json
_private._v1 = _private_v1_facade
_runtime._write_json = _write_json


class _Controller(_private._Controller):
    def _derive(self, raw, schema, stage):
        parsed = copy.deepcopy(_v1._parse_response(raw))
        if stage != "amem_evolution":
            return super()._derive(raw, schema, stage)
        value, normalization = _private.normalize_evolution(
            parsed, memory_count=self.memory_count, neighbor_indices=self.neighbor_indices)
        # The original schema says array<string>, with no enum. Native dispatch
        # has only if/elif: unknown strings have no branch; following actions run.
        # V2's index normalizer retains the stronger actual-neighbor/bounds gates.
        _v1._validate_json(value, schema)
        normalization["native_action_dispatch"] = {
            "policy": "unmodified-upstream-string-array-if-elif-no-else",
            "actions_unchanged": True, "original_actions": copy.deepcopy(value["actions"]),
            "should_evolve": value["should_evolve"],
            "unknown_actions": [{"position": index, "value": action}
                                for index, action in enumerate(value["actions"])
                                if action not in {"strengthen", "update_neighbor"}],
            "action_values_modified": False}
        return value, normalization


_private._Controller = _Controller
_runtime._Controller = _Controller


def build(conversation: dict, work_dir: Path, api) -> dict:
    return _private.build(conversation, work_dir, api)


def retrieve(question: str, state: dict, api) -> dict:
    return _private.retrieve(question, state, api)


def describe():
    result = _private.describe()
    result.update(schema=SCHEMA, frozen_v2_sha256=BASE_V2_SHA256, json_representation_policy=JSON_POLICY,
                  local_publication_policy=REPLACE_POLICY,
                  action_contract="original array<string>; untouched native if/elif, unknown has no branch")
    return result


def readiness():
    return _private.readiness()
