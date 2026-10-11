"""Private frozen HyperMem science; restore only native missing-wait default.

The response object is never edited. Explicit null/nonboolean values remain
errors. The pinned extractor itself supplies ``get('should_wait', True)``.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import types

BASE_SHA256 = '5b4d73650f9c4d69e714468bc31f4da5accf8ce83d9e83bf2112ff51b1643df3'
_path = Path(__file__).with_name('baselines_hypermem_v1.py')
_source = _path.read_bytes()
if hashlib.sha256(_source).hexdigest() != BASE_SHA256:
    raise RuntimeError('Frozen HyperMem adapter changed')
_old_package = '_PACKAGE = "_jev_hypermem_pinned_v1"'
_new_package = '_PACKAGE = "_jev_hypermem_native_v2"'
_text = _source.decode('utf-8-sig')
if _text.count(_old_package) != 1:
    raise RuntimeError('Private HyperMem package boundary differs')
_runtime = types.ModuleType('_jev_hypermem_adapter_native_v2')
_runtime.__file__ = str(_path)
exec(compile(_text.replace(_old_package, _new_package), str(_path), 'exec'), _runtime.__dict__)
_OriginalGeneration = _runtime._Generation


class _Generation(_OriginalGeneration):
    @staticmethod
    def _validate(value, stage, local):
        if (stage == 'hypermem_boundary' and isinstance(value, dict)
                and type(value.get('should_end')) is bool and 'should_wait' not in value):
            return  # Do not insert a value: original pinned native owns its default.
        return _OriginalGeneration._validate(value, stage, local)


_runtime._Generation = _Generation
COMPATIBILITY = {
    'schema': 'hypermem-native-missing-wait-default-v2',
    'stage': 'hypermem_boundary', 'required_should_end': 'explicit_boolean',
    'eligible_should_wait': 'absent_key_only', 'native_default': True,
    'response_mutated': False, 'other_validation': 'unchanged_frozen_v1',
    'private_package_only': True,
}
build = _runtime.build
retrieve = _runtime.retrieve
describe = _runtime.describe
HyperMemError = _runtime.HyperMemError
UPSTREAM_REVISION = _runtime.UPSTREAM_REVISION
CONFIG = _runtime.CONFIG
SCHEMA = _runtime.SCHEMA


def implementation():
    return {'base_sha256': BASE_SHA256, 'compatibility': dict(COMPATIBILITY),
            'scientific_specification': describe()}
