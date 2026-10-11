"""Offline single-root archive, cleanup proof, and predecessor verification.

Only the separately reviewed PowerShell companion deletes files. This module
never constructs a provider, loads credentials, or dispatches model requests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import FunctionType
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_five_baselines_gemini25_v1 as base
from tools.baselines_common_v1 import read, sha, save_new, now

METHODS = ('simplemem', 'hypermem', 'vmem', 'prism')
RUN = ROOT / 'runs/five_baselines_gemini25_v1_20261008'
SCHEMA = 'baseline-single-root-v2-archive-v1'
PLAN_SCHEMA = 'baseline-single-root-v2-cleanup-plan-v1'
TOOL_FILES = (
    'tools/run_five_baselines_reviewed_chain_v2.py',
    'tools/baseline_lifecycle_v2.py', 'tools/cleanup_reviewed_chain_v2.ps1',
    'tools/archive_amem_continuation_v4.py', 'tools/cleanup_amem_continuation_v4.ps1',
    'tools/baseline_controller_guard_v2.py',
    'tools/run_five_baselines_amem_native_v4.py', 'tools/baselines_amem_v4.py',
    'tools/run_five_baselines_amem_native_v3_admission_v2.py',
)



def folder(method):
    if method not in METHODS:
        raise ValueError('Only the four declared later methods are allowed')
    return RUN / method


def plan_path(method):
    return folder(method) / 'cleanup_plan_chain_v2.json'


def output_dir(method):
    return folder(method) / 'results/chain_lifecycle'


def safe_path(path):
    p = Path(path).absolute()
    root = ROOT.absolute()
    if not p.is_relative_to(root) or '..' in p.parts:
        raise ValueError('Path escaped exact workspace')
    for candidate in (root, *[root.joinpath(*p.relative_to(root).parts[:i])
                              for i in range(1, len(p.relative_to(root).parts) + 1)]):
        if candidate.is_symlink() or (candidate.exists() and
                getattr(candidate.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Reparse point or symbolic link is forbidden')
    if p.resolve() != p:
        raise ValueError('Path alias is forbidden')
    return p


def active_controllers(processes, own_pid):
    from tools.baseline_controller_guard_v2 import active_controllers as current
    return current(processes, own_pid)


def stopped():
    from tools.baseline_controller_guard_v2 import assert_stopped
    assert_stopped()


def binding(path):
    p = safe_path(path)
    return {'path': str(p), 'sha256': sha(p)}


def verify_binding(record):
    p = safe_path(record['path'])
    if not p.is_file() or sha(p) != record['sha256']:
        raise ValueError('Bound artifact differs: ' + str(p))
    return p


def inventory(path):
    p = safe_path(path)
    if not p.is_dir():
        raise ValueError('Required directory missing')
    result = {}
    for item in sorted(p.rglob('*')):
        safe_path(item)
        if item.is_file():
            result[str(item)] = {'sha256': sha(item), 'size': item.stat().st_size}
    return result


def code_hashes():
    return {p: sha(ROOT / p) for p in TOOL_FILES}


def evidence_closure(initial):
    """Only explicit hash-bearing references; never follow arbitrary prompt paths."""
    queue = list(initial); found = {}
    pairs = (('path', 'sha256'), ('review_path', 'review_sha256'),
             ('raw_path', 'raw_sha256'), ('source_path', 'source_sha256'))
    def visit(value):
        if isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, dict):
            for pk, hk in pairs:
                if isinstance(value.get(pk), str) and isinstance(value.get(hk), str):
                    queue.append((value[pk], value[hk]))
            for key in ('code_sha256', 'upstream_sha256', 'evidence_sha256', 'files_sha256'):
                for p, h in value.get(key, {}).items(): queue.append((p, h))
            for item in value.values(): visit(item)
    while queue:
        p, expected = queue.pop(); p = safe_path(ROOT / p)
        if str(p) in found:
            if found[str(p)] != expected: raise ValueError('Conflicting explicit evidence SHA')
            continue
        if sha(p) != expected: raise ValueError('Explicit evidence changed: ' + str(p))
        found[str(p)] = expected
        if p.suffix == '.json': visit(read(p))
    return found


def verify_tool_review(path):
    value = read(safe_path(path))
    if value.get('status') != 'passed' or any(value.get('code_sha256', {}).get(p) != h
                                             for p, h in code_hashes().items()):
        raise ValueError('Independent lifecycle code review must bind every exact helper')


def verify_result(method, review_path, *, live_inventory=True):
    root = folder(method)
    m = base.manifest(method)
    if m.get('method') != method or len(m['questions']) != 1540:
        raise ValueError('Wrong fixed method/cohort manifest')
    summary = read(root / 'results/summary.json')
    value = read(safe_path(review_path))
    if (value.get('status') != 'passed' or value.get('summary_sha256') != sha(root / 'results/summary.json')
            or value.get('method') not in (None, method)):
        raise ValueError('Independent exact result review required')
    if (summary.get('schema') != 'five-baseline-result-v1' or summary.get('method') != method
            or summary.get('audit_status') != 'no_invalid_artifacts'
            or any(summary.get('effective', {}).get(k) != 1540 for k in ('slots', 'reader_valid', 'judge_valid'))):
        raise ValueError('All 1540 effective Reader/Judge slots must be valid')
    state = read(root / 'status.json')
    if (state.get('stage') != 'completed_awaiting_result_review'
            or state.get('summary_sha256') != sha(root / 'results/summary.json')):
        raise ValueError('Method is not in its exact terminal state')
    expected = value.get('reviewed_run_inventory')
    if (safe_path(review_path).is_relative_to(root) or not isinstance(expected, dict) or not expected
            or any(not Path(p).is_absolute() or str(Path(p).resolve()) != p or not Path(p).is_relative_to(root)
                   or not isinstance(h, str) or re.fullmatch(r'[0-9a-f]{64}', h) is None
                   for p, h in expected.items())):
        raise ValueError('Independent result review must bind a nonempty complete absolute run inventory and live outside the run')
    if live_inventory and {p: rec['sha256'] for p, rec in inventory(root).items()} != expected:
        raise ValueError('Current run differs from independently reviewed full inventory')
    return m


def clone(function, **changes):
    result = FunctionType(function.__code__, {**function.__globals__, **changes}, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class Resolver:
    """SHA-checked retained files plus content-addressed generation evidence."""
    def __init__(self, target, manifest_path):
        self.manifest = read(safe_path(manifest_path))
        self.zip = zipfile.ZipFile(safe_path(target))
        try:
            if self.manifest.get('schema') != SCHEMA or self.zip.testzip() is not None:
                raise ValueError('Archive schema/CRC invalid')
            if self.zip.read('archive_manifest.json') != Path(manifest_path).read_bytes():
                raise ValueError('Internal and external archive manifests differ')
            self.members = self.manifest['members']
            self.keep = self.manifest['keep_sha256']
            root = folder(self.manifest['method'])
            expected = {'archive_manifest.json'}
            for p, rec in self.members.items():
                path = safe_path(p)
                in_work = path.is_relative_to(root / 'work')
                if (not in_work and self.manifest['evidence_sha256'].get(p) != rec['sha256']) or rec['member'] != 'blobs/' + rec['sha256']:
                    raise ValueError('Foreign archive logical path/content address')
                if in_work and self.manifest['work_inventory'].get(p) != {k: rec[k] for k in ('sha256', 'size')}:
                    raise ValueError('Archive record differs from sealed work inventory')
                expected.add(rec['member'])
            if set(self.zip.namelist()) != expected or len(self.zip.namelist()) != len(expected):
                raise ValueError('Unexpected or duplicate ZIP entries')
            for member in expected - {'archive_manifest.json'}:
                if hashlib.sha256(self.zip.read(member)).hexdigest() != member.split('/')[1]:
                    raise ValueError('Archive member SHA differs')
            for p, h in self.keep.items():
                if safe_path(p).is_relative_to(root / 'work'):
                    raise ValueError('Retained file cannot be in disposable work')
                verify_binding({'path': p, 'sha256': h})
        except BaseException:
            self.zip.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.zip.close()

    def bytes(self, path):
        p = str(safe_path(path))
        if p in self.members:
            rec = self.members[p]
            raw = self.zip.read(rec['member'])
            if len(raw) != rec['size'] or hashlib.sha256(raw).hexdigest() != rec['sha256']:
                raise ValueError('Archived raw evidence differs')
            return raw
        if p in self.keep:
            raw = Path(p).read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.keep[p]:
                raise ValueError('Retained result differs')
            return raw
        raise ValueError('Logical path not retained: ' + p)

    def read(self, path):
        return json.loads(self.bytes(path).decode('utf-8-sig'))

    def sha(self, path):
        return hashlib.sha256(self.bytes(path)).hexdigest()


def aggregate(values):
    f1 = [r['official_f1'] for r in values if r['official_f1'] is not None]
    labels = [r['judge_correct'] for r in values if r['judge_correct'] is not None]
    custom = [r['custom_metrics'].get('token_f1', r['custom_metrics'].get('f1'))
              for r in values if r['custom_metrics'] is not None]
    return {'slots': 1540, 'reader_valid': len(f1), 'judge_valid': len(labels),
            'custom_f1_valid_percent': 100 * sum(custom) / len(custom) if custom else None,
            'official_f1_valid_percent': 100 * sum(f1) / len(f1) if f1 else None,
            'official_f1_fixed_missing_zero_percent': 100 * sum(f1) / 1540,
            'judge_valid_percent': 100 * sum(labels) / len(labels) if labels else None,
            'judge_fixed_missing_zero_percent': 100 * sum(labels) / 1540,
            'missing_reader': 1540 - len(f1), 'missing_judge': 1540 - len(labels)}


def replay_all(resolver):
    method = resolver.manifest['method']; root = folder(method)
    class ArchivedRawReplay(base.RawReplay):
        generate = clone(base.RawReplay.generate, read=resolver.read, sha=resolver.sha)
    known = clone(base.known_recoverable, RawReplay=ArchivedRawReplay)
    replay = clone(base.replay_qa_leaf, read=resolver.read, RawReplay=ArchivedRawReplay, known_recoverable=known)
    manifest = resolver.read(root / 'run_manifest.json')
    questions = manifest['questions']
    if len(questions) != 1540 or len({q['query_id'] for q in questions}) != 1540 or questions != base.questions():
        raise ValueError('Archived fixed cohort differs')
    official, _ = base.verified_official_metric()
    summary = resolver.read(root / 'results/summary.json')
    checked = 0
    for phase, filename in (('firstpass', 'firstpass_per_query.json'), ('effective', 'per_query.json')):
        rows = resolver.read(root / 'results' / filename)
        if [r['query_id'] for r in rows] != [q['query_id'] for q in questions]:
            raise ValueError('Archived row identities/order differ')
        for q, row in zip(questions, rows):
            leaf = root / 'results/firstpass' / q['query_id']
            recovery = root / 'results/recovery' / q['query_id']
            if phase == 'effective' and str(recovery / 'reader.json') in resolver.keep:
                leaf = recovery
            reader, judge = replay(root, method, q, leaf)
            answer = reader['value']['prediction']['answer'] if reader['status'] == 'complete' else None
            label = base.evaluator.parse_judge_label(judge['value']['raw']) if judge['status'] == 'complete' else None
            expected = {**q, 'answer': answer,
                        'official_f1': official(answer, q['reference'], q['category']) if answer is not None else None,
                        'custom_metrics': base.evaluator.lexical_metrics(answer, q['reference']) if answer is not None else None,
                        'judge_correct': label['correct'] if label else None,
                        'reader_status': reader['status'], 'judge_status': judge['status'], 'phase': phase}
            if row != expected:
                raise ValueError('Archived full QA row differs from exact raw replay')
            checked += 1
        if summary[phase] != aggregate(rows):
            raise ValueError('Summary differs from independently replayed rows')
    return {'status': 'passed', 'firstpass_and_effective_leaves': checked, 'model_calls': 0}


def retain_work(path, root):
    parts = path.relative_to(root / 'work').parts
    if not parts or not parts[0].startswith('api_'):
        return False
    if path.parent.name == 'api_cache':
        return read(path)['request']['operation'] == 'generateContent'
    return True


def archive(method, result_review, tool_review):
    stopped(); m = verify_result(method, result_review); verify_tool_review(tool_review)
    root = folder(method); out = output_dir(method)
    if out.exists() or plan_path(method).exists():
        raise FileExistsError('Lifecycle already started; no implicit replacement')
    audit = base.audit_requests(root)
    if audit.get('status') != 'passed' or audit != read(root / 'results/raw_request_audit.json'):
        raise ValueError('Live complete raw audit differs from terminal seal')
    all_files = inventory(root); work = inventory(root / 'work')
    keep = {p: rec['sha256'] for p, rec in all_files.items() if not Path(p).is_relative_to(root / 'work')}
    members = {p: {**rec, 'member': 'blobs/' + rec['sha256']}
               for p, rec in work.items() if retain_work(Path(p), root)}
    evidence = evidence_closure([(str(root / 'run_manifest.json'), sha(root / 'run_manifest.json')),
                                 (str(result_review), sha(result_review)), (str(tool_review), sha(tool_review))])
    dataset = ROOT / 'data/locomo10.json'
    if m.get('dataset_sha256') != sha(dataset): raise ValueError('Source dataset changed')
    evidence[str(dataset)] = sha(dataset)
    locked = ROOT / 'configs/sdk20_full1540_locked_v1.json'
    evidence[str(locked)] = sha(locked)
    for p, h in evidence.items():
        # Explicit vector references are sealed, not silently retained as renamed caches.
        if Path(p).is_relative_to(root / 'work') and not retain_work(Path(p), root): continue
        members[p] = {'sha256': h, 'size': Path(p).stat().st_size, 'member': 'blobs/' + h}
    manifest = {'schema': SCHEMA, 'method': method, 'created_utc': now(),
                'work_inventory': work, 'members': members, 'keep_sha256': keep,
                'evidence_sha256': evidence,
                'disposal': 'Unretained work is sealed by SHA/size before deletion; embedding vectors and native indexes are not permanent exports.'}
    out.mkdir(parents=True, exist_ok=False)
    mp = out / 'archive_manifest.json'; target = out / 'model_evidence.zip'
    save_new(mp, manifest)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('archive_manifest.json', mp.read_bytes()); written = set()
        for p, rec in members.items():
            raw = Path(p).read_bytes()
            if hashlib.sha256(raw).hexdigest() != rec['sha256']:
                raise ValueError('Raw evidence changed while archiving')
            if rec['member'] not in written:
                z.writestr(rec['member'], raw); written.add(rec['member'])
    for p, rec in all_files.items():
        if sha(p) != rec['sha256']:
            raise ValueError('Source changed during archive')
    for p, h in evidence.items():
        if sha(p) != h: raise ValueError('Source evidence changed during archive')
    with Resolver(target, mp) as resolver:
        replay = replay_all(resolver)
    save_new(out / 'archive_replay.json', replay)
    keep.update({p: rec['sha256'] for p, rec in inventory(out).items()})
    plan = {'schema': PLAN_SCHEMA, 'method': method, 'created_utc': now(), 'workspace': str(ROOT),
            'allowed_delete_roots': [str(root / 'work')], 'archive_crc_verified': True,
            'archive': binding(target), 'archive_manifest': binding(mp), 'archive_replay': binding(out / 'archive_replay.json'),
            'result_review': binding(result_review), 'tool_review': binding(tool_review),
            'keep_sha256': keep, 'code_sha256': code_hashes(), 'deletion_executed': False}
    save_new(plan_path(method), plan)
    return {'status': 'archive_verified', 'plan': binding(plan_path(method)), 'model_calls': 0}


def verify_archive(method, *, require_work=True):
    stopped(); root = folder(method); plan = read(safe_path(plan_path(method)))
    if (plan.get('schema') != PLAN_SCHEMA or plan.get('method') != method or plan.get('workspace') != str(ROOT)
            or plan.get('allowed_delete_roots') != [str(root / 'work')] or plan.get('archive_crc_verified') is not True
            or plan.get('code_sha256') != code_hashes()):
        raise ValueError('Exact cleanup scope or source differs')
    result_review_path = verify_binding(plan['result_review'])
    verify_result(method, result_review_path, live_inventory=False)
    reviewed_inventory = read(result_review_path)['reviewed_run_inventory']
    verify_tool_review(verify_binding(plan['tool_review']))
    for p, h in plan['keep_sha256'].items():
        verify_binding({'path': p, 'sha256': h})
    target = verify_binding(plan['archive']); mp = verify_binding(plan['archive_manifest'])
    rp = verify_binding(plan['archive_replay'])
    if target != output_dir(method) / 'model_evidence.zip' or mp != output_dir(method) / 'archive_manifest.json' or rp != output_dir(method) / 'archive_replay.json':
        raise ValueError('Archive path differs from declared method')
    with Resolver(target, mp) as resolver:
        if resolver.manifest['method'] != method:
            raise ValueError('Archive method differs')
        original_inventory = {p: h for p, h in resolver.manifest['keep_sha256'].items()
                              if Path(p).is_relative_to(root)}
        original_inventory.update({p: rec['sha256'] for p, rec in resolver.manifest['work_inventory'].items()})
        if original_inventory != reviewed_inventory:
            raise ValueError('Archived run differs from independently reviewed full inventory')
        if require_work and inventory(root / 'work') != resolver.manifest['work_inventory']:
            raise ValueError('Work changed before deletion')
        if replay_all(resolver) != read(rp):
            raise ValueError('Archived replay seal differs')
    return plan


def cleanup_review(method, path, plan):
    review_path = safe_path(path); value = read(review_path)
    if (value.get('status') != 'passed' or value.get('method') != method
            or value.get('cleanup_plan_sha256') != sha(plan_path(method))
            or value.get('archive_sha256') != plan['archive']['sha256']):
        raise ValueError('Independent exact archive/cleanup approval required')
    return binding(review_path)


def removal_evidence(method, plan, review):
    root = folder(method); out = output_dir(method)
    safe_path(root / 'work')
    if (root / 'work').exists():
        raise ValueError('Work remains or reappeared')
    intent_path = out / 'cleanup_execution_intent.json'; receipt_path = out / 'removal_receipt.json'
    for path, schema, target_key in ((intent_path, 'baseline-single-root-v2-cleanup-intent-v1', 'target'),
                                     (receipt_path, 'baseline-single-root-v2-removal-receipt-v1', 'removed_path')):
        value = read(safe_path(path))
        if (value.get('schema') != schema or value.get('method') != method
                or value.get(target_key) != str(root / 'work')
                or value.get('cleanup_plan_sha256') != sha(plan_path(method))
                or value.get('cleanup_review_sha256') != review['sha256']):
            raise ValueError('Exclusive cleanup intent/removal receipt differs')
    return binding(intent_path), binding(receipt_path)


def complete(method, review_path):
    plan = verify_archive(method, require_work=False)
    review = cleanup_review(method, review_path, plan)
    intent, removal = removal_evidence(method, plan, review)
    root = folder(method)
    receipt = {'schema': 'baseline-single-root-v2-cleanup-receipt-v1', 'method': method, 'completed_utc': now(),
               'cleanup_plan': binding(plan_path(method)), 'cleanup_review': review,
               'archive': plan['archive'], 'archive_manifest': plan['archive_manifest'],
               'intent': intent, 'removal_receipt': removal, 'retained_evidence_verified': True}
    save_new(root / 'results/cleanup_receipt.json', receipt)
    completion = {'schema': 'baseline-single-root-v2-predecessor-completion-v1', 'status': 'passed', 'method': method,
                  'summary_sha256': sha(root / 'results/summary.json'), 'result_review': plan['result_review'],
                  'cleanup_receipt': binding(root / 'results/cleanup_receipt.json')}
    save_new(root / 'results/predecessor_completion.json', completion)
    return {'status': 'complete', 'model_calls': 0}


def verify_completion(method):
    plan = verify_archive(method, require_work=False)
    root = folder(method); receipt_path = root / 'results/cleanup_receipt.json'; receipt = read(safe_path(receipt_path))
    review_path = verify_binding(receipt['cleanup_review'])
    review = cleanup_review(method, review_path, plan)
    intent, removal = removal_evidence(method, plan, review)
    required = {'schema': 'baseline-single-root-v2-cleanup-receipt-v1', 'method': method,
                'cleanup_plan': binding(plan_path(method)), 'cleanup_review': review,
                'archive': plan['archive'], 'archive_manifest': plan['archive_manifest'],
                'intent': intent, 'removal_receipt': removal, 'retained_evidence_verified': True}
    if any(receipt.get(k) != v for k, v in required.items()):
        raise ValueError('Aggregate cleanup receipt differs')
    expected = {'schema': 'baseline-single-root-v2-predecessor-completion-v1', 'status': 'passed', 'method': method,
                'summary_sha256': sha(root / 'results/summary.json'), 'result_review': plan['result_review'],
                'cleanup_receipt': binding(receipt_path)}
    if read(safe_path(root / 'results/predecessor_completion.json')) != expected:
        raise ValueError('Predecessor completion differs')
    return {'status': 'passed', 'method': method, 'archive_sha256': plan['archive']['sha256'], 'model_calls': 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('archive', 'verify', 'complete', 'verify-completion'))
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--result-review', type=Path); p.add_argument('--tool-review', type=Path); p.add_argument('--cleanup-review', type=Path)
    args = p.parse_args()
    if args.command == 'archive':
        result = archive(args.method, args.result_review, args.tool_review)
    elif args.command == 'verify':
        verify_archive(args.method); result = {'status': 'passed', 'model_calls': 0}
    elif args.command == 'complete':
        result = complete(args.method, args.cleanup_review)
    else:
        result = verify_completion(args.method)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
