"""R10 one complete JSON envelope; the frozen R09 inner parser is unchanged."""
import hashlib
import re
from copy import deepcopy
from jev_mem import qualifier_repair_r09 as original
from jev_mem.qualifier_repair_r09 import (ARMS, digest, request_hash,
    serialized_signature, validate_state, stable_candidates, build_request,
    parse_scores, admit_candidates)
from jev_mem.gemini38_provider_v1 import response_text

PROTOCOL_SHA256 = 'c422886b79e17358851189839795af5dd44dc7cb3c574b9dc5bcd7917d9e872c'
POLICY = {**deepcopy(original.POLICY), 'version': 'envelope-coverage-r10-v1',
          'new_proposer_calls': 0, 'envelope_decoding_count_max': 1}
ENVELOPE = re.compile(r'```json\r?\n(?P<body>.*?)\r?\n```', re.DOTALL)


def _text_ref(text):
    raw = text.encode('utf8')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'utf8_bytes': len(raw)}


def decode_text(text):
    """Return the exact inner slice and auditable delimiter offsets, or reject."""
    if not isinstance(text, str):
        raise ValueError('Actual visible model text required')
    t = text.strip()
    if len(t.encode('utf8')) > 24576:
        raise ValueError('Whole visible envelope byte cap exceeded')
    start = len(text) - len(text.lstrip())
    meta = {'visible_text': _text_ref(text), 'outer_stripped_text': _text_ref(t),
            'outer_strip_codepoint_range': [start, start + len(t)],
            'branch': 'plain', 'decoding_count': 0, 'removed_delimiter_ranges': []}
    if not t.startswith('```'):
        meta['decoded_slice'] = _text_ref(t)
        meta['decoded_slice_codepoint_range'] = [start, start + len(t)]
        return t, meta
    match = ENVELOPE.fullmatch(t)
    if match is None:
        raise ValueError('Exact complete lowercase json fence required')
    body = match.group('body')
    if '```' in body:
        raise ValueError('Nested or multiple triple-backtick sequence forbidden')
    lo, hi = match.span('body')
    meta.update(branch='fenced', decoding_count=1,
                decoded_slice=_text_ref(body),
                decoded_slice_codepoint_range=[start + lo, start + hi],
                removed_delimiter_ranges=[[start, start + lo],
                                          [start + hi, start + len(t)]])
    return body, meta


def replay_proposal(response, state):
    """Known validation failures retain evidence but never partial proposals."""
    validate_state(state)
    visible = None
    metadata = {'branch': 'unavailable', 'decoding_count': 0,
                'visible_text': None, 'decoded_slice': None,
                'removed_delimiter_ranges': []}
    try:
        visible = response_text(response)
        metadata['visible_text'] = _text_ref(visible)
        metadata['branch'] = 'rejected_envelope' if visible.strip().startswith('```') else 'plain'
        metadata['outer_stripped_text'] = _text_ref(visible.strip())
        text, metadata = decode_text(visible)
        parsed = original.parse_proposals(text, state)
        return {'status': 'complete', 'reason': None,
                'decoding': metadata, 'parsed': parsed}
    except (ValueError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
        return {'status': 'failed', 'reason': str(exc),
                'error_type': type(exc).__name__, 'decoding': metadata,
                'parsed': None}
