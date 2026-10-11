"""Single reviewed continuation of Gemini 2.5 after its retained R03 402 stop.

Only ``run`` dispatches models. ``check`` and ``summary`` activate the same exact
admission for repeatable offline lineage/raw replay, never a credential probe.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import gemini25_billing_admission_v1 as admission
from tools import prepare_gemini25_ablation_v1 as source
from tools import score_gemini25_ablation_v1 as score
from tools import run_gemini25_ablation_qa_v1 as qa
from tools import summarize_gemini25_ablation_v1 as report
from tools import run_gemini25_ablation_pipeline_v1 as pipeline
from tools.prepare_evidence_selection_r03 import Reader, save_new
from tools.run_sdk20_modules import now
from tools.run_pilot400_experiments_v2 import publish_json
from tools.check_experiment_health import shared_read

SCHEMA = 'gemini25-ablation-billing-continuation-v1'


def controller_probe():
    """Read current Windows process identities; never trust historical PIDs."""
    if os.name != 'nt':
        raise RuntimeError('This reviewed continuation requires its Windows host')
    script = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
              "Select-Object @{n='pid';e={$_.ProcessId}},@{n='parent_pid';e={$_.ParentProcessId}},"
              "@{n='creation_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}},"
              "@{n='command_line';e={$_.CommandLine}} | ConvertTo-Json -Compress")
    out = subprocess.run(['powershell.exe', '-NoProfile', '-Command', script],
                         capture_output=True, text=True, timeout=30, check=True,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    rows = json.loads(out.stdout) if out.stdout.strip() else []
    return [rows] if isinstance(rows, dict) else rows


def assert_single_controller():
    rows = controller_probe()
    bypid = {r['pid']: r for r in rows}
    if os.getpid() not in bypid or not bypid[os.getpid()].get('creation_utc'):
        raise ValueError('Current controller identity unavailable')
    entries = {
        'run_gemini25_ablation_writer_v1.py', 'run_gemini25_ablation_pipeline_v1.py',
        'resume_gemini25_ablation_after_billing_v1.py', 'score_gemini25_ablation_v1.py',
        'run_gemini25_ablation_qa_v1.py', 'prepare_gemini25_ablation_v1.py',
    }
    def entry(row):
        if not row.get('command_line'):
            raise ValueError('Python process command line unavailable')
        tokens = [t.strip('"') for t in shlex.split(row['command_line'], posix=False)]
        modules = {'tools.' + name[:-3]: name for name in entries}
        for i, token in enumerate(tokens):
            name = Path(token).name.lower()
            if name in entries:
                return name
            if i and tokens[i - 1] == '-m' and token.lower() in modules:
                return modules[token.lower()]
        return None
    own = bypid[os.getpid()]
    allowed = {os.getpid()}
    # Windows venv python.exe may be the current interpreter's redirector
    # parent. Exclude only that same-entry ancestry, not a sibling controller.
    cur = own
    while cur.get('parent_pid') in bypid:
        parent = bypid[cur['parent_pid']]
        if entry(own) != 'resume_gemini25_ablation_after_billing_v1.py' or entry(parent) != entry(own):
            break
        allowed.add(parent['pid']); cur = parent
    active = [r for r in rows if r['pid'] not in allowed and entry(r)]
    if active:
        raise ValueError('Another current Gemini25 experiment controller exists: '
                         + ','.join(str(r['pid']) for r in active))
    return {'pid': own['pid'], 'parent_pid': own['parent_pid'],
            'creation_utc': own['creation_utc'], 'duplicate_controllers': 0,
            'checked_utc': now()}


def unused_namespaces():
    paths = [admission.LIVE, admission.RUN / 'phases/r03/recovery_v1',
             admission.RUN / 'selected_inputs.json', admission.RUN / 'selections',
             admission.QA, admission.QA.with_name(admission.QA.name + '_recovery_v1')]
    paths.extend(admission.RUN / 'phases' / p for p in source.PHASES[1:])
    for p in paths:
        if p.exists():
            raise FileExistsError('One reviewed continuation only; namespace already exists: ' + str(p))


def preflight(guard, *, for_launch=False):
    print(json.dumps({'event': 'offline_validation_start', 'stage': 'retained_r03_full_audit', 'utc': now()}), flush=True)
    if for_launch:
        unused_namespaces()
        process_before = assert_single_controller()
    state = json.loads(shared_read(admission.BASE / 'live_pipeline_v1/status.json'))
    if (state.get('status') != 'needs_review' or state.get('stage') != 'r03_firstpass'
            or state.get('completed_stages') != ['source_prepare', 'retrieve', 'r03_prepare', 'r03_firstpass']):
        raise ValueError('Original stopped pipeline boundary differs')
    source.validate_phase(admission.RUN, 'r03')
    rows = guard.phase_results(Reader(), admission.RUN, 'r03', allow_recovery=False)
    if len(rows) != 1540 or Counter(r['status'] for r in rows.values()) != {'failed': 1540}:
        raise ValueError('Retained whole failed cohort differs')
    guard.bind_review(Reader()); guard.bind_firstpass(Reader())
    if for_launch:
        unused_namespaces()
        process_after = assert_single_controller()
    result = {'status': 'passed', 'requests': len(rows), 'firstpass_counts': {'failed': 1540},
            'all_original_producer_consumer_raw_checks': True,
            'scientific_code_sha256': guard.review['scientific_code_sha256'],
            'review': guard.review_ref, 'model_calls': 0, 'credential_reads': 0}
    if for_launch:
        result['controller_checks'] = [process_before, process_after]
    print(json.dumps({'event': 'offline_validation_end', 'stage': 'retained_r03_full_audit',
                      'requests': len(rows), 'utc': now()}), flush=True)
    return result


def qa_resource_gate(run):
    """An additional resource boundary; the original raw/stage audit still runs."""
    run = Path(run)
    physical = 0
    for p in sorted(run.glob('conv_*/api/api_calls.jsonl')):
        data = shared_read(p)
        if data and not data.endswith(b'\n'):
            raise ValueError('Partial QA ledger blocks continuation')
        for line in data.splitlines():
            row = json.loads(line)
            if row.get('status') in (401, 402, 403):
                raise ValueError('QA resource/auth failure requires a new review')
            physical += not row.get('cache_hit', False)
    return {'physical_attempts': physical, 'model_calls': 0}


def offline_summary(guard):
    """Requires the existing continuation intent and original QA recovery seal."""
    intent = Reader().read(admission.LIVE / 'execution_intent.json')
    if (intent.get('schema') != SCHEMA or intent.get('review') != guard.review_ref
            or intent.get('technical_code_sha256') != guard.review['technical_code_sha256']):
        raise ValueError('Continuation execution lineage differs')
    recovery = admission.QA.with_name(admission.QA.name + '_recovery_v1')
    qa_resource_gate(admission.QA)
    qa_resource_gate(recovery)
    # qa.seal originally published this; standalone replay must not rewrite it.
    seal = Reader().read(recovery / 'qa_raw_audit_seal.json')
    if seal.get('audit_status') != 'no_invalid_artifacts':
        raise ValueError('Original QA recovery raw seal missing or failed')
    with report.original.historical_guard.historical.offline():
        value = report.summarize(admission.QA, recovery)
    if value['audit_status'] != 'no_invalid_artifacts':
        raise ValueError('Final original raw audit failed')
    if any(v['unknown_intent_count'] or v['ended_success_missing_raw_count']
           for phase in value['phase_intent_inventory'].values() for v in phase.values()):
        raise ValueError('Unknown/orphan request intents block summary')
    seal_expected = {
        'schema': qa.SCHEMA + '-raw-audit-seal',
        'run_manifest': admission.pointer(recovery / 'run_manifest.json'),
        'run_outcome': admission.pointer(recovery / 'run_outcome.json'),
        'source_manifest': value['source_manifest'],
        'phase_sources': value['phase_sources'],
        'input_sha256': value['input_sha256'],
        'summary_code_sha256': value['summary_code_sha256'],
        'audit_status': value['audit_status'], 'raw_replayed': True,
        'official_f1_recomputed': True, 'custom_f1_recomputed': True,
        'model_calls': 0,
    }
    if any(seal.get(k) != v for k, v in seal_expected.items()):
        raise ValueError('Original QA raw seal does not bind the replayed artifacts')
    value['precomputation_usage'] = pipeline.precomputation_usage(pipeline.WRITER, admission.RUN)
    value['technical_continuation'] = {
        'review': guard.review_ref,
        'execution_intent': admission.pointer(admission.LIVE / 'execution_intent.json'),
        'technical_code_sha256': guard.review['technical_code_sha256'],
        'retained_firstpass_resource_exception': admission.POLICY,
    }
    return value


def check(review_path):
    guard = admission.BillingAdmission(review_path)
    with guard.activated(), report.original.historical_guard.historical.offline():
        result = preflight(guard)
        for phase in source.PHASES:
            dest = admission.RUN / 'phases' / phase
            if dest.exists():
                if not (dest / 'manifest.json').exists():
                    raise ValueError('Partial source phase has no terminal manifest')
                source.validate_phase(admission.RUN, phase)
                if (dest / 'firstpass').exists() and not (dest / 'firstpass/outcome.json').exists():
                    raise ValueError('Partial firstpass producer has no terminal outcome')
                if (dest / 'recovery_v1').exists() and not (dest / 'firstpass').exists():
                    raise ValueError('Recovery exists without its original firstpass')
                if (dest / 'firstpass').exists() and phase != 'r03':
                    guard.phase_results(Reader(), admission.RUN, phase)
        if (admission.RUN / 'phases/r03/recovery_v1').exists():
            rows = guard.phase_results(Reader(), admission.RUN, 'r03')
            result['r03_effective'] = admission.require_progress(rows)
        if (admission.RUN / 'selected_inputs.json').exists():
            value = source.validate_source(admission.RUN / 'selected_inputs.json')
            result['sealed_source_jobs'] = len(value['jobs'])
        return result


def run(review_path):
    guard = admission.BillingAdmission(review_path)
    # The fixed fresh LIVE directory is the atomic, one-shot controller claim.
    # Failed or interrupted namespaces never become an automatic restart path.
    with guard.activated():
        checked = preflight(guard, for_launch=True)
        live = admission.LIVE
        live.mkdir(parents=True, exist_ok=False)
        save_new(live / 'execution_intent.json', {
            'schema': SCHEMA, 'pid': os.getpid(), 'created_utc': now(),
            'review': guard.review_ref, 'preflight': checked,
            'source_run': str(admission.RUN), 'qa_run': str(admission.QA),
            'scientific_code_sha256': guard.review['scientific_code_sha256'],
            'technical_code_sha256': guard.review['technical_code_sha256'],
            'policy': admission.POLICY,
        })
        state = {'schema': SCHEMA, 'pid': os.getpid(), 'status': 'running',
                 'started_utc': now(), 'completed_stages': []}

        def stage(name, fn):
            state.update(stage=name, updated_utc=now())
            publish_json(live / 'status.json', state)
            print(json.dumps({'stage': name, 'event': 'start', 'utc': now()}), flush=True)
            value = fn()
            ref = save_new(live / (name + '.json'), value if isinstance(value, dict) else {'result': value})
            state['completed_stages'].append({'stage': name, 'result': ref})
            state.update(updated_utc=now()); publish_json(live / 'status.json', state)
            return value

        try:
            stage('r03_recovery', lambda: score.run_phase(admission.RUN, 'r03', recovery=True))
            stage('r03_raw_audit', lambda: admission.require_progress(
                guard.phase_results(Reader(), admission.RUN, 'r03')))
            for phase in source.PHASES[1:]:
                stage(phase + '_prepare', lambda p=phase: source.prepare_phase(admission.RUN, p))
                stage(phase + '_firstpass', lambda p=phase: score.run_phase(admission.RUN, p))
                rows = guard.phase_results(Reader(), admission.RUN, phase, allow_recovery=False)
                if any(r['status'] == 'failed' for r in rows.values()):
                    stage(phase + '_recovery', lambda p=phase: score.run_phase(admission.RUN, p, recovery=True))
                stage(phase + '_raw_audit', lambda p=phase: admission.require_progress(
                    guard.phase_results(Reader(), admission.RUN, p), require_success=False))
            stage('source_seal', lambda: source.seal(admission.RUN))
            stage('qa_prepare', lambda: qa.prepare(admission.QA, admission.RUN / 'selected_inputs.json'))
            stage('qa_firstpass', lambda: qa.run(admission.QA))
            qa_resource_gate(admission.QA)
            stage('qa_recovery_prepare', lambda: qa.prepare_recovery(admission.QA))
            recovery = admission.QA.with_name(admission.QA.name + '_recovery_v1')
            stage('qa_recovery', lambda: qa.run(recovery))
            qa_resource_gate(recovery)
            stage('qa_raw_audit', lambda: qa.seal(recovery))
            stage('final_summary_v1', lambda: offline_summary(guard))
            state.update(status='completed_awaiting_research_discussion', finished_utc=now())
        except Exception as exc:
            state.update(status='needs_review', error_type=type(exc).__name__,
                         failed_stage=state.get('stage'), finished_utc=now())
            publish_json(live / 'status.json', state)
            save_new(live / 'outcome.json', state)
            raise
        state.update(updated_utc=now()); publish_json(live / 'status.json', state)
        save_new(live / 'outcome.json', state)
        return {'status': state['status'], 'summary': str(live / 'final_summary_v1.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('check', 'summary', 'run'))
    p.add_argument('--review', required=True, type=Path)
    p.add_argument('--output', type=Path, help='Fresh report path for offline commands only')
    args = p.parse_args()
    if args.command == 'run':
        if args.output:
            p.error('run writes only its fixed one-shot live namespace')
        value = run(args.review)
    else:
        if not args.output:
            p.error('Offline commands require a fresh --output report')
        if args.output.exists():
            p.error('Offline report must not overwrite an existing artifact')
        if args.command == 'check':
            value = check(args.review)
        else:
            guard = admission.BillingAdmission(args.review)
            with guard.activated():
                value = offline_summary(guard)
        value = {'report': save_new(args.output, value), 'model_calls': 0, 'credential_reads': 0}
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
