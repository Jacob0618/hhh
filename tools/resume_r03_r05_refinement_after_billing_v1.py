"""Single reviewed continuation after the preserved rank billing boundary stop."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import prepare_r03_r05_refinement_v1 as prep
from tools import score_r03_r05_refinement_v1 as score
from tools.score_evidence_exchange_r01 import exclusive_intent, status_write, publish, now, sha
BASE = ROOT/'research/r03_r05_refinement_2026-10-06'
RUN = ROOT/'runs/r03_r05_refinement_v1_20261006'
QA = ROOT/'runs/r03_r05_refinement_qa_v1_20261006'
STOP = BASE/'heartbeat_20261006T0057/billing_boundary_stop_v1.json'
POST = BASE/'heartbeat_20261006T0057/post_stop_check_v1.json'
FINAL = BASE/'finalreview_v1.json'
OUTCOME_SHA = '9512771cff34ea8174da40e5673d38a04b89f43175a1629f3391ab7f4d97b600'


def bindings():
    """Review pins all original firstpass bytes, including response and intent files."""
    producer = RUN/'phases/rank/firstpass'
    files = [Path(__file__), ROOT/'tools/test_resume_r03_r05_refinement_after_billing_v1.py',
             BASE/'billing_resume_v1/resource_confirmation_v1.json', FINAL, STOP, POST, BASE/'live_v1/status.json',
             RUN/'phases/rank/manifest.json']
    files += sorted(p for p in producer.rglob('*') if p.is_file())
    return {str(p.resolve()): sha(p) for p in files}


def remaining_gate(rows):
    counts = Counter(r['status'] for r in rows.values())
    if set(counts)-{'complete', 'failed', 'not_dispatched'}:
        raise ValueError('Unknown rank outcome blocks dependent phases')
    # Require an actual parsed success: no all-fallback QA after resource failure.
    if not counts['complete']:
        raise ValueError('No successful rank recovery; QA blocked')
    return {'requests':len(rows), 'effective_counts':dict(counts),
            'unknown':0, 'all_raw_and_producer_consumer_verified':True, 'model_calls':0}


def admission(review_path, audit_dir):
    audit_dir = Path(audit_dir).resolve()
    if audit_dir.exists() or not audit_dir.is_relative_to(BASE.resolve()) or audit_dir == BASE.resolve():
        raise ValueError('Fresh continuation audit namespace required')
    for p in (RUN/'phases/rank/recovery_v1', RUN/'selected_inputs.json', QA,
              QA.with_name(QA.name+'_recovery_v1')):
        if p.exists(): raise FileExistsError('Unused continuation namespace required: '+str(p))
    if sha(FINAL) != '82152c39ea56f0196879bc91e4abbc0c1fb6874826f3bb30ed4523d6f45d8e7c':
        raise ValueError('Original finalreview artifact differs')
    final = prep.Reader().read(FINAL)
    expected = {**prep.codes(), 'tools/run_r03_r05_refinement_pipeline_v1.py':sha(ROOT/'tools/run_r03_r05_refinement_pipeline_v1.py')}
    if (final.get('status') != 'ready' or final.get('code_sha256') != expected
        or final.get('protocol_sha256') != sha(prep.SPEC) or final.get('models_dispatched_before_review') != 0):
        raise ValueError('Original scientific freeze mismatch')
    stop, post = (json.loads(p.read_text(encoding='utf-8')) for p in (STOP, POST))
    if (sha(RUN/'phases/rank/firstpass/outcome.json') != OUTCOME_SHA
        or stop.get('firstpass_outcome_sha256') != OUTCOME_SHA
        or stop.get('terminal_requests') != 6160 or stop.get('counts') != {'failed':6160}
        or stop.get('recovery_intent_absent_at_stop') is not True
        or stop.get('original_source_status_modified') is not False
        or post.get('firstpass_status') != 'terminal' or post.get('terminal_requests') != 6160
        or post.get('counts') != {'failed':6160} or post.get('local_errors') != []
        or any(post.get(k) is not False for k in ('rank_recovery_dir_exists','selected_inputs_exists','qa_run_exists'))):
        raise ValueError('Billing boundary stop evidence mismatch')
    reviewed = json.loads(Path(review_path).read_text(encoding='utf-8'))
    pinned = bindings()
    independent = reviewed.get('independent_review')
    if not isinstance(independent,dict) or set(independent) != {'path','sha256'}:
        raise ValueError('Independent review artifact required')
    independent_path = Path(independent['path']).resolve()
    if independent_path == Path(review_path).resolve() or sha(independent_path) != independent['sha256']:
        raise ValueError('Independent review artifact hash mismatch')
    if (reviewed.get('schema') != 'r03-r05-billing-continuation-review-v1'
        or reviewed.get('status') != 'ready' or reviewed.get('bindings') != pinned
        or reviewed.get('models_dispatched_before_review') != 0):
        raise ValueError('Independent exact-hash continuation review required')
    # Original verifier audits every producer/consumer, intent and physical ledger.
    print(json.dumps({'event':'offline_validation','stage':'rank_firstpass_full_raw_audit','utc':now()}),flush=True)
    prep.validate_phase(RUN, 'rank')
    rows = prep.phase_results(prep.Reader(), RUN, 'rank', allow_recovery=False)
    if len(rows) != 6160 or Counter(r['status'] for r in rows.values()) != {'failed':6160}:
        raise ValueError('Original known-failure cohort differs')
    if bindings() != pinned: raise ValueError('Admission evidence changed during audit')
    return pinned


def run(review_path, audit_dir):
    from tools import run_r03_r05_refinement_qa_v1 as qa
    from tools import summarize_r03_r05_refinement_v1 as summary
    pinned = admission(review_path, audit_dir)
    audit_dir = Path(audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=False)
    exclusive_intent(audit_dir/'intent.json', {'schema':'r03-r05-billing-continuation-v1',
        'pid':os.getpid(), 'created_utc':now(), 'review':{'path':str(Path(review_path).resolve()),
        'sha256':sha(review_path)}, 'bindings':pinned})
    state = {'schema':'r03-r05-billing-continuation-v1','pid':os.getpid(),
             'created_utc':now(),'status':'running','stages':[]}
    def step(name, fn):
        state.update(stage=name, updated_utc=now()); status_write(audit_dir/'status.json',state)
        print(json.dumps({'event':'start','stage':name,'utc':now()}),flush=True)
        value = fn()
        ref = prep.save_new(audit_dir/(name+'.json'), value)
        state['stages'].append({'stage':name,'result':ref,'finished_utc':now()})
        status_write(audit_dir/'status.json',state)
        return value
    try:
        recovered = step('rank_recover',lambda:score.run_phase(RUN,'rank',recovery=True))
        if recovered['status'] != 'terminal': raise ValueError('Rank recovery needs review')
        rows = prep.phase_results(prep.Reader(),RUN,'rank')
        step('rank_terminal_audit',lambda:remaining_gate(rows))
        ledger = RUN/'phases/rank/recovery_v1/api_calls.jsonl'
        if ledger.exists():
            events = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
            if any(e.get('status') == 402 and 'billing_error' in json.dumps(e) for e in events):
                raise ValueError('Recovery billing_error blocks QA')
        step('source_seal',lambda:prep.seal(RUN))
        step('qa_prepare',lambda:qa.prepare(QA,RUN/'selected_inputs.json'))
        step('qa_firstpass',lambda:qa.run(QA))
        qm, qo = qa.terminal(QA)
        recovery = None
        if qa.recovery_jobs(qm,qo):
            step('qa_recovery_prepare',lambda:qa.prepare_recovery(QA))
            recovery = QA.with_name(QA.name+'_recovery_v1')
            step('qa_recovery',lambda:qa.run(recovery)); qa.terminal(recovery)
        def finish():
            with summary.offline(): value = summary.summarize(QA,recovery,bootstrap_samples=2000)
            ref = prep.save_new(audit_dir/'final_summary_v1.json',value)
            if value['audit_status'] != 'no_invalid_artifacts': raise ValueError('Final raw audit failed')
            return {'summary':ref,'audit_status':value['audit_status'],'model_calls':0}
        step('summary',finish)
        state.update(status='completed_awaiting_research_discussion',finished_utc=now())
    except Exception as exc:
        state.update(status='needs_review',error_type=type(exc).__name__,failed_stage=state.get('stage'),updated_utc=now())
        status_write(audit_dir/'status.json',state); publish(audit_dir/'outcome.json',state)
        raise
    state.update(updated_utc=now()); status_write(audit_dir/'status.json',state)
    publish(audit_dir/'outcome.json',state)
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',required=True,type=Path)
    p.add_argument('--audit-dir',required=True,type=Path)
    a = p.parse_args()
    print(json.dumps(run(a.review,a.audit_dir),ensure_ascii=False))
