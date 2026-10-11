"""Extend frozen refinement QA400 to full1540 by running ONLY complementary1140.

All native retrieval and plain.one bodies are reused. Original400 results, their
one recovery, model protocol and fact banks stay immutable. The only generated
function edits are three reporting/cardinality constants, guarded by AST shape.
"""
from pathlib import Path as _Path
import sys as _sys
import ast as _ast
import inspect as _inspect
from types import FunctionType as _FunctionType

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from tools import run_regroup_refinement_qa_v1 as _v1
from tools import run_regroup_refinement_qa_v2 as _v2
from tools import regroup_refinement_extension1140_sources_v1 as _extension_sources

for _name, _value in vars(_v1).items():
    if not _name.startswith('__'):
        globals()[_name] = _value

sources = _extension_sources
SCHEMA = 'regroup-refinement-plain-extension1140-v1'
POLICY = {**_v1.POLICY, 'slots_per_arm': 1140, 'full_population_per_arm': 1540,
          'inherited_per_arm': 400,
          'reference_comparison': 'fixed1540_exact_original400_final_plus_disjoint1140',
          'cohort_selection': 'canonical_category1_4_minus_fixed400_ids_no_outcome_filter'}
CODE = tuple(dict.fromkeys((*_v2.CODE,
    'tools/run_regroup_refinement_extension1140_v1.py',
    'tools/regroup_refinement_extension1140_sources_v1.py',
    'tools/freeze_sdk20_full1540_v1.py')))
CARDINALITY_FIELDS = {'prepare': 'qa_total', 'retrieve': 'expected_slots', 'qa': 'planned_per_arm'}


def _cardinality_code(original, field):
    """Allow one named dict field400→1140; no prompt/function-body rewrite."""
    tree = _ast.parse(_inspect.getsource(original))
    matches = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, _ast.Constant) and key.value == field:
                    if not isinstance(value, _ast.Constant) or type(value.value) is not int or value.value != 400:
                        raise ValueError('Legacy cardinality field changed; explicit review required')
                    matches.append(value)
    if len(matches) != 1:
        raise ValueError('Expected exactly one legacy reporting cardinality')
    matches[0].value = 1140
    namespace = {}
    exec(compile(_ast.fix_missing_locations(tree), __file__ + '::' + original.__name__, 'exec'),
         original.__globals__, namespace)
    return namespace[original.__name__].__code__


REBOUND_FUNCTIONS = tuple(name for name, value in vars(_v1).items()
    if isinstance(value, _FunctionType) and value.__module__ == _v1.__name__)
for _name in REBOUND_FUNCTIONS:
    _original = getattr(_v1, _name)
    _code = (_cardinality_code(_original, CARDINALITY_FIELDS[_name])
             if _name in CARDINALITY_FIELDS else _original.__code__)
    _bound = _FunctionType(_code, globals(), _original.__name__,
                          _original.__defaults__, _original.__closure__)
    _bound.__kwdefaults__ = _original.__kwdefaults__
    _bound.__annotations__ = dict(_original.__annotations__)
    _bound.__doc__, _bound.__module__, _bound.__qualname__ = _original.__doc__, __name__, _original.__qualname__
    globals()[_name] = _bound

_base_prepare, _base_manifest = prepare, manifest
_base_publish_json = publish_json


def _verify_master_lock():
    from tools.freeze_sdk20_full1540_v1 import verify_lock
    value = verify_lock()
    full, dates = sources.full_source.cohort()
    expected = [{'query_id': q['id'], 'conversation': int(ci), 'source_qa_index': q['source_qa_index'],
                 'question': q['question'], 'reference': q['reference'], 'category': q['category'],
                 'reference_date': dates[ci]} for ci, qs in full.items() for q in qs]
    if value['dataset']['questions'] != expected:
        raise ValueError('Extension canonical1540 payload differs from master lock')
    return sources.pointer(sources.COHORT)


def _same_banks(m, inherited):
    if (m['gate'] != inherited['gate'] or m['evaluation_protocol'] != inherited['evaluation_protocol']
            or m['official_metric_provenance'] != inherited['official_metric_provenance']):
        raise ValueError('Extension differs from original400 gate or evaluation protocol')
    for ci in range(10):
        new = verify(sources.checked_pointer(m['banks'][str(ci)][m['arm']]))
        old = verify(sources.checked_pointer(inherited['banks'][str(ci)][m['arm']]))
        clean = lambda b: {k: v for k, v in b.items() if k not in ('native_dir', 'content_sha256')}
        if clean(new) != clean(old):
            raise ValueError('Extension copied facts/native lineage differ from inherited400')


def publish_json(path, value):
    """Add bindings before the new manifest is first persisted, never rewrite old files."""
    if _Path(path).name == 'run_manifest.json' and value.get('schema') == SCHEMA:
        value = {k: v for k, v in value.items() if k != 'protocol_sha256'}
        master = _verify_master_lock()
        if 'extension_lineage' not in value:
            inherited = sources.inheritance(value['arm'], audit_raw=True)
            _same_banks(value, inherited)
            value['extension_lineage'] = {'full1540_lock': master, 'inherited400': inherited,
                'extended_slots': 1140, 'inherited_slots': 400, 'combined_slots': 1540,
                'selection_does_not_read_answers_or_scores': True}
        elif value['extension_lineage']['full1540_lock'] != master:
            raise ValueError('Master lock changed before recovery publication')
        value['protocol_sha256'] = digest(value)
    return _base_publish_json(path, value)


def prepare(run_dir, gate_path):
    _verify_master_lock()
    gate = verify(gate_path)
    arm = gate.get('arm')
    if arm not in sources.ARMS:
        raise ValueError('Only the two frozen refinement arms may extend')
    old = read(sources.OLD_RUNS[arm] / 'run_manifest.json')
    if sources.pointer(gate_path) != old['gate']:
        raise ValueError('Must reuse the exact original400 source gate')
    return _base_prepare(run_dir, gate_path)


def manifest(run_dir, *, conversation=None, recheck_sources=False):
    master = _verify_master_lock()
    m = _base_manifest(run_dir, conversation=conversation, recheck_sources=recheck_sources)
    lineage = m.get('extension_lineage', {})
    if (lineage.get('full1540_lock') != master or lineage.get('extended_slots') != 1140
            or lineage.get('inherited_slots') != 400 or lineage.get('combined_slots') != 1540
            or lineage.get('selection_does_not_read_answers_or_scores') is not True):
        raise ValueError('Immutable full1540 extension binding differs')
    inherited = sources.validate_inheritance(lineage['inherited400'], audit_raw=recheck_sources)
    _same_banks(m, inherited)
    if m['phase_kind'] == 'firstpass':
        if sum(map(len, m['query_ids'].values())) != 1140:
            raise ValueError('Extension must contain exactly1140 slots')
        new_ids = {q for ids in m['query_ids'].values() for q in ids}
        if new_ids & set(inherited['query_ids']):
            raise ValueError('Extension cannot regenerate original400 questions')
    return m


if __name__ == '__main__':
    main()
