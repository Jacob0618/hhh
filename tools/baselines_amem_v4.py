"""Private frozen A-Mem v3 science plus a narrow reversible quote-token bridge.

No evaluation, content completion, typo correction, API retry, or cache admission.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
import types

from tools import baselines_amem_v3 as _base
from jev_mem.sdk20_cached_api import digest

BASE_V3_SHA256 = '32e86376104b35f302557ff1f5e475bd5ae387288b937c4cb3bbaf25f38b6a52'
SCHEMA = 'amem-original-adapter-v4-literal-quote-representation'
JSON_POLICY = 'strict-v3-or-reversible-simple-single-quoted-value-token-v1'
_source = Path(_base.__file__).read_bytes()
if hashlib.sha256(_source).hexdigest() != BASE_V3_SHA256:
    raise RuntimeError('Frozen A-Mem v3 adapter differs')


class AMemV4Error(_base.AMemV3Error):
    pass


def _single_quote_tokens(text):
    """Change only two unescaped quote delimiters per complete simple value token."""
    chars = list(text); changes = []; literals = []
    inside_double = False; escaped = False; i = 0
    while i < len(text):
        char = text[i]
        if inside_double:
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': inside_double = False
            i += 1
            continue
        if char == '"':
            inside_double = True; i += 1
            continue
        if char != "'":
            i += 1
            continue
        previous = i - 1
        while previous >= 0 and text[previous] in ' \t\r\n': previous -= 1
        if previous < 0 or text[previous] not in '[:,':
            raise AMemV4Error('Single quote is not at a JSON value-token boundary')
        end = i + 1
        while end < len(text) and text[end] != "'":
            if text[end] in '\\"' or ord(text[end]) < 32:
                raise AMemV4Error('Single-quoted value contains an ambiguous escape, quote, or control')
            end += 1
        if end == len(text):
            raise AMemV4Error('Unclosed single-quoted value token')
        following = end + 1
        while following < len(text) and text[following] in ' \t\r\n': following += 1
        if following == len(text) or text[following] not in ',]}':
            raise AMemV4Error('Single-quoted token has no unambiguous JSON value delimiter')
        literals.append({'opening_character_offset': i, 'closing_character_offset': end,
                         'value': text[i + 1:end]})
        for offset in (i, end):
            chars[offset] = '"'
            changes.append({'character_offset': offset, 'utf8_byte_offset': len(text[:offset].encode('utf-8')),
                            'from_codepoint': 'U+0027', 'to_codepoint': 'U+0022',
                            'from_utf8_hex': '27', 'to_utf8_hex': '22'})
        i = end + 1
    if inside_double or escaped or not changes:
        raise AMemV4Error('Incomplete JSON strings or no eligible single-quoted value tokens')
    derived = ''.join(chars)
    reverse = list(derived)
    for change in changes:
        reverse[change['character_offset']] = "'"
    if ''.join(reverse) != text or len(derived) != len(text):
        raise AMemV4Error('Quote-only inverse verification failed')
    return derived, changes, literals


def decode_generation_json(raw_record: dict, *, raw_artifact_sha256: str):
    """Same interface as frozen v3; v3-success payloads/journals remain identical."""
    try:
        return _base.decode_generation_json(raw_record, raw_artifact_sha256=raw_artifact_sha256)
    except (json.JSONDecodeError, _base.AMemV3Error) as exc:
        prior_error = {'type': type(exc).__name__, 'message': str(exc)}
    if not isinstance(raw_artifact_sha256, str) or re.fullmatch('[0-9a-f]{64}', raw_artifact_sha256) is None:
        raise AMemV4Error('A bound original raw artifact SHA256 is required')
    try:
        text = raw_record['response']['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError, TypeError) as exc:
        raise AMemV4Error('Original complete plain-text candidate is required') from exc
    if not isinstance(text, str):
        raise AMemV4Error('Original text must be a string')
    derived, changes, literals = _single_quote_tokens(text)
    # The fallback permits quote delimiters only; it does not compose a second
    # TAB/escape repair or accept duplicate keys, NaN, incomplete, or nonobject JSON.
    value = _base._strict_json(derived)
    record = copy.deepcopy(raw_record)
    record['response']['candidates'][0]['content']['parts'][0]['text'] = derived
    derived_record_sha = _base._v1._sha(record)
    checked, eligibility = _base.decode_generation_json(record, raw_artifact_sha256=derived_record_sha)
    if eligibility['changes'] or checked != value:
        raise AMemV4Error('A second repair or differing parsed payload is forbidden')
    journal = {'policy': JSON_POLICY, 'raw_artifact_sha256': raw_artifact_sha256,
               'request_hash': digest(raw_record['request']), 'eligibility': eligibility['eligibility'],
               'original_text_utf8_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
               'derived_text_utf8_sha256': hashlib.sha256(derived.encode('utf-8')).hexdigest(),
               'derived_record_canonical_sha256': derived_record_sha,
               'parsed_payload_sha256': _base._v1._sha(value), 'changes': changes,
               'single_quoted_value_tokens': literals, 'initial_v3_error': prior_error,
               'only_boundary_quotes_changed': True, 'inverse_verified': True,
               'original_raw_unchanged': True, 'literal_values_unchanged': True}
    return value, journal


# Private module compilation preserves the frozen v3 scientific function bodies.
# Only export/call schema labels change; shared v1/v2/v3 module objects do not.
_private = types.ModuleType('_jev_amem_v4_private_v3')
_private.__file__ = str(Path(_base.__file__))
exec(compile(_source, str(Path(_base.__file__)), 'exec'), _private.__dict__)
_private.SCHEMA = SCHEMA
_private._private.SCHEMA = SCHEMA
_private._runtime.SCHEMA = SCHEMA
_Controller = _private._Controller
_v1 = _private._v1
_v2 = _private._v2
_runtime = _private._runtime
UPSTREAM_REVISION = _private.UPSTREAM_REVISION
SOURCE_SHA256 = _private.SOURCE_SHA256
SEED_K = _private.SEED_K


def build(conversation: dict, work_dir: Path, api) -> dict:
    return _private.build(conversation, work_dir, api)


def retrieve(question: str, state: dict, api) -> dict:
    return _private.retrieve(question, state, api)


def describe():
    value = _private.describe()
    value.update(schema=SCHEMA, frozen_v3_sha256=BASE_V3_SHA256, json_representation_policy=JSON_POLICY,
                 representation_changes='only simple single-quoted value-token delimiters; existing v3 successes unchanged')
    return value


def readiness():
    return _private.readiness()
