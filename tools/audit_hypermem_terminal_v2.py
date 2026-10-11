"""Pure all-ten terminal admission; exact reuse of the completed raw audit.

The hash-locked author audit is compiled privately. Only its report-publication
and fixed historical PID probe are removed; current strict controller guard is
run instead. Every source/native/producer check remains byte-for-byte source.
"""
from __future__ import annotations
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_five_baselines_reviewed_chain_v6 as chain
from tools import audit_simplemem_terminal_v2 as common
from tools.baselines_common_v1 import save_new, now

OLD = ROOT / 'runs/five_baselines_gemini25_v1_20261008/hypermem'
NEW = OLD.with_name('hypermem_native_v2')
SCHEMA = 'hypermem-native-v2-terminal-proof-v1'
AUDIT_SOURCE = 'research/baseline_execution_2026-10-08/heartbeat_20261009T0842/author_hypermem_boundary_terminal_v2.py'
AUDIT_SHA = '11b3c86c12316e077c4ec1bb0a4171e608ed6a198809aa89c6b8a0be71a27a2d'
QUALIFICATION = 'research/baseline_execution_2026-10-08/heartbeat_20261009T0842/independent_hypermem_native_contract_v1.json'
QUALIFICATION_SHA = '40e45359857f3a0ac04daa5346b34ef6f9184554a4aae2b04f68c5b7b90d6b65'
CHAIN_SHA = 'a9634bafcd35184c17ba8a24e631f444cb32096f8b0a3427db25503d52e65beb'
require, read, sha, inventory, binding = common.require, common.read, common.file_sha, common.inventory, common.binding
digest = common.digest
_active = chain.active_controllers
active_controllers = FunctionType(_active.__code__, {**_active.__globals__,
    'LAUNCHERS': (*_active.__globals__['LAUNCHERS'], 'run_five_baselines_hypermem_native_v2')},
    'active_controllers', _active.__defaults__)
_stop = chain.assert_stopped
assert_stopped = FunctionType(_stop.__code__, {**_stop.__globals__, 'active_controllers': active_controllers},
    'assert_stopped', _stop.__defaults__)


@contextmanager
def guard_context():
    require(sha(chain.__file__) == CHAIN_SHA, 'Frozen predecessor bridge changed')
    changes = ((chain.guard, 'active_controllers', active_controllers),
               (chain.guard, 'assert_stopped', assert_stopped),
               (chain.previous.simple, '_stop', assert_stopped))
    before = [(m, k, getattr(m, k)) for m, k, _ in changes]
    try:
        for m, k, v in changes: setattr(m, k, v)
        yield
    finally:
        for m, k, v in reversed(before): setattr(m, k, v)


def usage_presence(calls):
    physical = [r for r in calls if not r.get('cache_hit')]
    tokens = {}
    for field in ('promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount', 'totalTokenCount'):
        values = [r.get('usage', {}).get(field) for r in physical]
        present = [v for v in values if v is not None]
        require(all(type(v) is int and v >= 0 for v in present), 'Invalid token telemetry')
        tokens[field] = {'reported_calls': len(present), 'unreported_calls': len(values)-len(present),
                         'reported_sum': sum(present) if present else None}
    return {'physical_attempts': len(physical), 'cache_events': len(calls)-len(physical),
            'tokens': tokens, 'cost_usd': None, 'note': 'Absent telemetry is unknown; cached tokens are not new usage.'}


def _locked_terminal_audit():
    p = ROOT / AUDIT_SOURCE
    require(sha(p) == AUDIT_SHA, 'Completed author audit source changed')
    tree = ast.parse(p.read_text(encoding='utf-8-sig'))
    block = next(n for n in tree.body if isinstance(n, ast.With))
    # The hash lock plus exact AST node count excludes only historical process
    # probe/publication. No data, code, native, or physical check is bypassed.
    removed = []
    keep = []
    for n in block.body:
        names = {x.id for x in ast.walk(n) if isinstance(x, ast.Name)}
        if (isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in ('probe', 'process_rows') for x in n.targets)
                or isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and (
                    isinstance(n.value.func, ast.Name) and n.value.func.id == 'print' or
                    isinstance(n.value.func, ast.Name) and n.value.func.id == 'require' and 'process_rows' in names)
                or isinstance(n, ast.With) and 'OUT' in names):
            removed.append(n)
        else: keep.append(n)
    require(len(removed) == 6, 'Audit publication/PID boundary changed')
    block.body = keep
    env = {'__name__': '_hypermem_terminal_private_audit', '__file__': str(p)}
    exec(compile(ast.fix_missing_locations(tree), str(p), 'exec'), env)
    return env['result']


def collect():
    assert_stopped()
    with common.offline():
        result = _locked_terminal_audit()
        require(sha(ROOT / QUALIFICATION) == QUALIFICATION_SHA, 'Independent native qualification changed')
        q = read(ROOT / QUALIFICATION)
        require(q['schema'] == 'hypermem-native-boundary-independent-diagnosis-v1' and q['status'] == 'passed'
                and q['launch_admitted'] is False and q['cleanup_authorized'] is False, 'Wrong individual qualification')
        require(q['original_manifest'] == result['original_manifest'] and len(q['cases']) == 10, 'Qualification parent/cases differ')
        require(binding(q['source_inventory']['path'])==q['source_inventory'],'Independent terminal inventory changed')
        qualified_inventory=read(q['source_inventory']['path'])
        require(qualified_inventory['file_count']==380 and qualified_inventory['before_equals_after'] is True
                and qualified_inventory['files_sha256']==result['original_inventory'],'Current source differs from independently qualified 380-file terminal snapshot')
        for a, b in zip(result['rows'], q['cases']):
            require(a['conversation'] == b['conversation'] and a['raw'] == b['raw'] and a['native_call'] == b['native_call']
                    and a['physical'] == b['physical'] and a['logical'] == b['logical'], 'Independent case evidence differs')
            require(a['intent']==b['request_intent'] and a['source_messages']==b['source_messages']
                    and a['signature']==b['signature'] and a['context']==b['context'],'Independent full request/source evidence differs')
        parent = read(OLD / 'run_manifest.json')
        require(parent['questions'] == chain.base.questions(), 'Original fixed cohort changed')
        source = result.pop('source_sha256')
        result.pop('created_utc'); result.pop('elapsed_seconds'); result.pop('physical_usage')
        result.pop('source_controller_pids_observed_absent')
        result.update(schema=SCHEMA, status='passed', launch_admitted=False,
                      original_run=str(OLD), new_run=str(NEW), questions=parent['questions'],
                      specification=parent['specification'], source_code_sha256=source,
                      qualification=binding(ROOT / QUALIFICATION), audit_source=binding(ROOT / AUDIT_SOURCE),
                      auditor=binding(Path(__file__)), original_physical_usage=usage_presence([x['physical'] for x in result['rows']]))
        result['registry'] = {r['request_hash']: {k:r[k] for k in ('conversation','request_hash','signature','context','stage','raw','intent','native_call','logical','physical')} for r in result['rows']}
        require(inventory(OLD) == result['original_inventory'], 'Original changed after qualification checks')
    assert_stopped()
    return result


def validate_proof(path):
    value = read(path)
    require(value == collect(), 'Terminal admission proof differs from complete current audit')
    return value


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['create','verify']); p.add_argument('--proof', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'create': print(json.dumps(save_new(a.proof, collect())))
    else: validate_proof(a.proof); print(json.dumps({'status':'passed','proof':binding(a.proof)}))

if __name__ == '__main__': main()
