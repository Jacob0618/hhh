"""Conditional role-ID representation adapter; no runtime permission by itself.

Private v2 science is preserved. Full UUIDs can be represented by native aliases
only with a complete local bijection and an admitted API authorization receipt.
Original provider values remain in native raw_response; parsed_response records
the explicitly journaled derived object. No heuristic JSON repair exists here.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import textwrap
import types
import uuid

V2_SHA256 = '4b73b47919b4027e59e503d647ddb3267af1b6fe72a3d6022d57d433b36c651b'
_path = Path(__file__).with_name('baselines_hypermem_native_v2.py')
_source = _path.read_bytes()
if hashlib.sha256(_source).hexdigest() != V2_SHA256:
    raise RuntimeError('Frozen native-v2 adapter changed')
_text = _source.decode('utf-8-sig')
for old in ('_jev_hypermem_native_v2', '_jev_hypermem_adapter_native_v2'):
    if _text.count(old) != 1:
        raise RuntimeError('Private v2 package boundary differs')
    _text = _text.replace(old, old.replace('_v2', '_v3'))
_v2 = types.ModuleType('_jev_hypermem_v3_private_v2')
_v2.__file__ = str(_path)
exec(compile(_text, str(_path), 'exec'), _v2.__dict__)
_runtime = _v2._runtime
_CompatibleGeneration = _v2._Generation
HyperMemError = _runtime.HyperMemError
KIND = 'exact_complete_local_uuid_episode_roles_to_native_aliases'


def represent_episode_ids(value, simple_to_real):
    """Pure candidate derivation. Caller must separately prove admission."""
    if not isinstance(value, dict) or not isinstance(simple_to_real, dict):
        raise HyperMemError('Role representation requires object and local mapping')
    mapping = dict(simple_to_real)
    rows = value.get('episode_roles')
    if isinstance(rows, list) and all(isinstance(x, dict) and type(x.get('episode_id')) is str for x in rows):
        if {row['episode_id'] for row in rows} == set(mapping):
            _CompatibleGeneration._validate(value, 'hypermem_episode_roles', {'simple_to_real': mapping})
            return value, None  # The complete frozen-native accepted branch is untouched.
    if not mapping:
        raise HyperMemError('UUID conversion requires a nonempty local mapping')
    if list(mapping) != [f'episode_{i+1}' for i in range(len(mapping))]:
        raise HyperMemError('Native sequential alias mapping differs')
    targets = list(mapping.values())
    if any(type(x) is not str for x in targets) or len(set(targets)) != len(targets):
        raise HyperMemError('Local episode UUID mapping is not a bijection')
    try:
        if any(str(uuid.UUID(x)) != x for x in targets):
            raise ValueError('Noncanonical UUID')
    except (ValueError, AttributeError, TypeError) as exc:
        raise HyperMemError('Local episode identity is not an exact canonical UUID') from exc
    if not isinstance(rows, list) or len(rows) != len(mapping) or any(not isinstance(x, dict) for x in rows):
        raise HyperMemError('Role rows do not completely cover the local mapping')
    ids = [row.get('episode_id') for row in rows]
    if any(type(x) is not str for x in ids) or len(set(ids)) != len(ids):
        raise HyperMemError('Role identifiers are missing, duplicate or not strings')
    if set(ids) != set(targets):
        raise HyperMemError('Role UUIDs are not the exact complete local identity set')
    inverse = {real: alias for alias, real in mapping.items()}
    derived = copy.deepcopy(value)
    changes = []
    for i, row in enumerate(derived['episode_roles']):
        old = row['episode_id']
        row['episode_id'] = inverse[old]
        changes.append({'json_pointer': f'/episode_roles/{i}/episode_id', 'original': old, 'derived': row['episode_id']})
    # Explicit inverse proof; this does not remove or coerce any other field.
    inverse_value = copy.deepcopy(derived)
    for row in inverse_value['episode_roles']:
        row['episode_id'] = mapping[row['episode_id']]
    if inverse_value != value:
        raise HyperMemError('Role representation is not exactly invertible')
    _CompatibleGeneration._validate(derived, 'hypermem_episode_roles', {'simple_to_real': mapping})
    return derived, {'schema': 'hypermem-role-id-representation-v3', 'kind': KIND,
        'original_parsed_sha256': _runtime._sha(value), 'derived_parsed_sha256': _runtime._sha(derived),
        'simple_to_real': mapping, 'mapping_sha256': _runtime._sha(mapping),
        'changes': changes, 'inverse_equal': True, 'other_values_unchanged': True}


class _Generation(_CompatibleGeneration):
    def _represent(self, value, stage, local, path, request, *, cached_native=False):
        if stage != 'hypermem_episode_roles':
            return value
        mapping = local.get('simple_to_real')
        original = value
        if cached_native:
            retained = _runtime._read_json(path)
            original = retained['raw_response']
            if isinstance(original, str):
                original = json.loads(original)
            if retained['request'] != request:
                raise HyperMemError('Cached representation request differs')
        derived, journal = represent_episode_ids(original, mapping)
        if cached_native and value != derived:
            raise HyperMemError('Cached representation does not derive from original raw and current local mapping')
        if journal is None:
            if cached_native and (self.work_dir/'representations'/Path(path).name).exists():
                raise HyperMemError('Unexpected representation journal for native alias response')
            return value
        episodes = local.get('episodes')
        if not isinstance(episodes, list) or {f'episode_{i+1}': ep.event_id for i, ep in enumerate(episodes)} != mapping:
            raise HyperMemError('Role mapping is not the actual local episode list')
        evidence = {**journal, 'native_call_path': str(path), 'native_request': request,
            'local_episodes': [ep.to_dict() for ep in episodes],
            'local_topic': local['topic'].to_dict()}
        authorize = getattr(self.api, 'authorize_episode_representation', None)
        if authorize is None:
            raise _runtime.BaselineFatal('Role representation has no admitted API authority')
        if cached_native:
            receipt = authorize(evidence, original=original, derived=derived, cached_native=True)
        else:
            receipt = authorize(evidence, original=original, derived=derived)
        if not isinstance(receipt, dict) or receipt.get('status') != 'authorized':
            raise _runtime.BaselineFatal('Role representation was not admitted')
        output = self.work_dir / 'representations' / Path(path).name
        record = {**evidence, 'authorization': receipt}
        if output.exists():
            _runtime._verify_seal(output)
            if _runtime._read_json(output) != record:
                raise HyperMemError('Representation journal differs')
        else:
            if cached_native or _runtime._seal_path(output).exists():
                raise HyperMemError('Sealed representation journal is missing')
            _runtime._write_json(output, record)
            _runtime._seal(output)
        return derived


# The two added calls are the sole changes to the original generation method.
# The prompt/body/source/exception/persistence/return code is the frozen code.
_tree = ast.parse(_v2._source.decode('utf-8-sig'))
_class = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == '_Generation')
_method = next(n for n in _class.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate')
_lines = _v2._source.decode('utf-8-sig').splitlines()
_original = textwrap.dedent('\n'.join(_lines[_method.lineno-1:_method.end_lineno])) + '\n'
_marker = 'self._validate(value, stage, frame.f_locals)'
if _original.count(_marker) != 2:
    raise RuntimeError('Frozen generation validation boundary changed')
_derived_lines = []
for _line in _original.splitlines():
    if _line.strip() == _marker:
        _derived_lines.append(_line[:len(_line)-len(_line.lstrip())] + "value = self._represent(value, stage, frame.f_locals, path, request, cached_native='cached' in locals())")
    _derived_lines.append(_line)
_derived_source = '\n'.join(_derived_lines) + '\n'
_namespace = dict(_runtime.__dict__)
exec(compile(_derived_source, '<hypermem-v3-two-representation-boundaries>', 'exec'), _namespace)
_Generation.generate = _namespace['generate']
_runtime._Generation = _Generation

COMPATIBILITY = {'schema': 'hypermem-native-v3-technical-compatibility-v1',
    'missing_wait': dict(_v2.COMPATIBILITY), 'episode_role_representation': {
        'kind': KIND, 'stage': 'hypermem_episode_roles',
        'policy': 'requires_terminal_qualification_and_admitted_api_authorization',
        'allowed_change': 'episode_id values only; complete unique local UUID-to-alias bijection',
        'raw_http_bytes_mutated': False, 'heuristic_mapping': False,
        'explicit_mixed_or_incomplete_ids_allowed': False},
    'json_repair': False, 'other_science': 'unchanged_frozen_native_v2'}
build, retrieve, describe = _runtime.build, _runtime.retrieve, _runtime.describe
UPSTREAM_REVISION, CONFIG, SCHEMA = _runtime.UPSTREAM_REVISION, _runtime.CONFIG, _runtime.SCHEMA


def implementation():
    return {'frozen_v2_sha256': V2_SHA256, 'compatibility': copy.deepcopy(COMPATIBILITY),
        'original_generation_sha256': hashlib.sha256(_original.encode()).hexdigest(),
        'derived_generation_sha256': hashlib.sha256(_derived_source.encode()).hexdigest(),
        'exact_insertions': 2, 'scientific_specification': describe()}
