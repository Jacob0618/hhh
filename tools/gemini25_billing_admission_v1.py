"""Exact, reviewed firstpass-402 admission; no scientific selector changes.

Only an activated context changes a module attribute, and it is always restored.
The same context is required for later standalone source/QA/raw replay. All
scientific code hashes and the original dispatch-code hash remain unchanged;
the separate review and execution lineage disclose this technical adapter.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import threading

from tools import score_gemini25_ablation_v1 as score
from tools.prepare_evidence_selection_r03 import Reader, file_sha, save_new
from tools.check_experiment_health import shared_read
from tools.run_sdk20_modules import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/gemini25_ablation_2026-10-07'
HEARTBEAT = BASE / 'heartbeat_20261007T1814'
RESEARCH = HEARTBEAT / 'billing_resume_v1'
RUN = ROOT / 'runs/gemini25_ablation_source_v1_20261007'
QA = ROOT / 'runs/gemini25_ablation_qa_v1_20261007'
FIRSTPASS = RUN / 'phases/r03/firstpass'
READINESS = BASE / 'implementation_readiness_v1.json'
FULL_SOURCE = BASE / 'billing_resume_v1/independent_full_source_admission_v1.json'
STOP = HEARTBEAT / 'billing_stop_boundary_v1.json'
RESOURCE = RESEARCH / 'resource_confirmation_v1.json'
LIVE = RESEARCH / 'live_v1'
SCHEMA = 'gemini25-ablation-billing-continuation-review-v1'
TECHNICAL_CODE = (
    'tools/gemini25_billing_admission_v1.py',
    'tools/resume_gemini25_ablation_after_billing_v1.py',
    'tests/test_gemini25_billing_admission_v1.py',
    'tests/test_gemini25_billing_continuation_independent_v1.py',
)
POLICY = {
    'run_dir': str(RUN), 'phase': 'r03', 'producer': str(FIRSTPASS),
    'exception_status': 402, 'known_failed_requests': 1540,
    'unique_failed_producers': 1529, 'transport_attempts_before_402': 6,
    'failed_exact_hash_consumers': 11, 'recovery_phases_max': 1,
    'transport_retries': 2, 'recovery_resource_auth_failure': 'block',
    'scoped_r03_recovery_zero_parsed_success': 'block',
    'other_known_nonresource_failure_fallback': 'unchanged',
    'writer_retrieval_firstpass_restart': 'forbidden',
}
_ORIGINAL_PHASE_RESULTS = score.phase_results
_ACTIVATION_LOCK = threading.Lock()


def pointer(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def inventory(folder):
    """Bind every existing firstpass byte, including ledger error-body records."""
    folder = Path(folder).resolve()
    result = {}
    for p in sorted(folder.rglob('*')):
        if p.is_symlink() or not p.resolve().is_relative_to(folder):
            raise ValueError('Firstpass inventory cannot contain redirected paths')
        if p.is_file():
            result[p.relative_to(folder).as_posix()] = file_sha(p)
    if not result:
        raise ValueError('Missing retained firstpass inventory')
    return result


def sealed(path):
    return Reader().read(Path(path))


def scientific_codes():
    readiness = sealed(READINESS)
    codes = readiness['code_sha256']
    if len(codes) != 73 or codes != {p: file_sha(ROOT / p) for p in codes}:
        raise ValueError('Original 73-file scientific freeze differs')
    return codes


def review_requirements(failure_review, independent_review):
    """Return evidence to bind; deliberately does not declare any review ready."""
    failure_review, independent_review = map(lambda p: Path(p).resolve(),
                                           (failure_review, independent_review))
    files = [READINESS, FULL_SOURCE, STOP, RESOURCE, failure_review, independent_review,
             HEARTBEAT / 'snapshot_01.json', HEARTBEAT / 'snapshot_02.json',
             BASE / 'live_pipeline_v1/status.json',
             BASE / 'live_pipeline_v1/execution_intent.json',
             RUN / 'run_manifest.json', RUN / 'phases/r03/manifest.json',
             ROOT / 'runs/gemini25_ablation_writer_v1_20261007/all_bank_review.json']
    return {
        'schema': SCHEMA, 'policy': POLICY,
        'models_dispatched_before_review': 0,
        'failure_review': pointer(failure_review),
        'independent_review': pointer(independent_review),
        'resource_confirmation': pointer(RESOURCE),
        'bindings': {str(p.resolve()): file_sha(p) for p in files},
        'scientific_code_sha256': scientific_codes(),
        'technical_code_sha256': {p: file_sha(ROOT / p) for p in TECHNICAL_CODE},
        'firstpass_file_sha256': inventory(FIRSTPASS),
    }


def require_progress(rows, *, require_success=True):
    counts = Counter(r['status'] for r in rows.values())
    if set(counts) - {'complete', 'failed', 'not_dispatched'}:
        raise ValueError('Unknown outcome blocks dependent phases')
    if require_success and not counts['complete']:
        raise ValueError('Nonempty phase has zero parsed successes; downstream blocked')
    return {'terminal': len(rows), 'counts': dict(counts), 'model_calls': 0}


class BillingAdmission:
    def __init__(self, review_path):
        self.review_path = Path(review_path).resolve()
        self.review_ref = pointer(self.review_path)
        self.review = sealed(self.review_path)
        expected = review_requirements(self.review['failure_review']['path'],
                                       self.review['independent_review']['path'])
        if (self.review.get('status') != 'ready'
                or any(self.review.get(k) != v for k, v in expected.items())):
            raise ValueError('Exact independent continuation review required')
        if self.review_path in (Path(self.review['failure_review']['path']).resolve(),
                                Path(self.review['independent_review']['path']).resolve()):
            raise ValueError('Continuation cannot review itself')
        independent = Reader().read(self.review['independent_review']['path'], sealed=False)
        if (independent.get('schema') != 'gemini25-ablation-billing-independent-review-v1'
                or independent.get('status') != 'review_passed'
                or independent.get('model_calls') != 0
                or any(independent.get('code_sha256', {}).get(p) != expected['technical_code_sha256'][p]
                       for p in TECHNICAL_CODE[:2])):
            raise ValueError('Independent technical review is not passed for this code')
        failure = Reader().read(self.review['failure_review']['path'], sealed=False)
        if (failure.get('status') != 'review_passed_original_firstpass_terminal_resource_failure'
                or Path(failure.get('run_dir', '')).resolve() != RUN
                or failure.get('phase') != 'r03'
                or Path(failure.get('firstpass_dir', '')).resolve() != FIRSTPASS
                or failure.get('firstpass_file_sha256') != expected['firstpass_file_sha256']):
            raise ValueError('Independent firstpass failure review differs')
        confirmation = sealed(RESOURCE)
        if (confirmation.get('schema') != 'gemini25-billing-resource-confirmation-v1'
                or confirmation.get('firstpass_run') != str(FIRSTPASS)
                or confirmation.get('no_probe') is not True
                or confirmation.get('model_calls') != 0):
            raise ValueError('User resource confirmation binding differs')
        self.firstpass_inventory = expected['firstpass_file_sha256']

    def bind_review(self, reader):
        if pointer(self.review_path) != self.review_ref:
            raise ValueError('Continuation review changed')
        reader.raw_pointer(self.review_path, ref=self.review_ref)
        for path, sha in self.review['bindings'].items():
            ref = {'path': path, 'sha256': sha}
            if pointer(path) != ref:
                raise ValueError('Continuation evidence changed')
            reader.raw_pointer(path, ref=ref)
        for path, sha in self.review['technical_code_sha256'].items():
            ref = {'path': str((ROOT / path).resolve()), 'sha256': sha}
            if pointer(ref['path']) != ref:
                raise ValueError('Technical continuation code changed')
            reader.raw_pointer(ref['path'], ref=ref)

    def bind_firstpass(self, reader):
        if inventory(FIRSTPASS) != self.firstpass_inventory:
            raise ValueError('Reviewed firstpass bytes or file population changed')
        for rel, sha in self.firstpass_inventory.items():
            path = (FIRSTPASS / rel).resolve()
            reader.raw_pointer(path, ref={'path': str(path), 'sha256': sha})

    def permits_402(self, run, phase, producer):
        return (Path(run).resolve() == RUN and phase == 'r03'
                and Path(producer).resolve() == FIRSTPASS)

    def phase_results(self, reader, run, phase, *, allow_recovery=True):
        # Keep the original terminal proof checks. Only the final 402 check has
        # a reviewed, byte-bound exception, confined to this exact firstpass.
        run = Path(run).resolve()
        self.bind_review(reader)
        scoped = run == RUN and phase == 'r03'
        if scoped:
            self.bind_firstpass(reader)
        pm = reader.read(run / 'phases' / phase / 'manifest.json'); refs = pm['requests']

        def terminal(producer, requests):
            outcome = reader.read(producer / 'outcome.json')
            intent = reader.read(producer / 'execution_intent.json', sealed=False)
            if (outcome['schema'] != score.SCHEMA or outcome['status'] != 'terminal' or outcome.get('local_errors')
                or outcome['expected_requests'] != len(requests) or outcome['terminal_requests'] != len(requests)
                or intent['requests'] != requests or intent['transport_retries'] != 2
                or intent['dispatch_code_sha256'] != file_sha(score.__file__)
                or intent['manifest_sha256'] != file_sha(run / 'phases' / phase / 'manifest.json')):
                raise ValueError('Whole phase is not audited terminal')
            rows = {}
            for ref in requests:
                req = reader.bound(ref); rid = req['request_id']
                result_ref = reader.pointer(producer / 'responses' / (rid + '.json'))
                row = score.verify_result(reader, result_ref, ref, producer, 2)
                rows[rid] = {**row, 'result_ref': result_ref}
            if {Path(r['result_ref']['path']).resolve() for r in rows.values()} != {p.resolve() for p in (producer / 'responses').glob('*.json')}:
                raise ValueError('Missing/extra terminal responses')
            eligible = {reader.bound(ref)['request_hash'] for ref in requests if reader.bound(ref)['dispatch_eligible']}
            if {p.stem for p in (producer / 'request_intents').glob('*.json')} != eligible:
                raise ValueError('Unmanifested/missing API intent')
            if {p.stem for p in (producer / 'api_cache').glob('*.json')} != {r['request_hash'] for r in rows.values() if r.get('raw')}:
                raise ValueError('Unmanifested/missing raw response')
            allowed = {(reader.bound(ref)['request_hash'], score.shared.digest(score.context(reader.bound(ref))))
                       for ref in requests if reader.bound(ref)['dispatch_eligible']}
            for p in (producer / 'request_intents').glob('*.json'):
                iv = reader.read(p, sealed=False)
                if (p.stem, score.shared.digest(iv.get('context'))) not in allowed or score.shared.request_hash(iv['signature']) != p.stem:
                    raise ValueError('Foreign API producer')
            lp = producer / 'api_calls.jsonl'
            if lp.exists():
                data = shared_read(lp); reader.raw_pointer(lp)
                if data and not data.endswith(b'\n'):
                    raise ValueError('Incomplete ledger')
                for line in data.splitlines():
                    e = json.loads(line)
                    if ((e.get('request_hash'), score.shared.digest(e.get('context'))) not in allowed
                        or (e.get('stage'), e.get('provider'), e.get('model'), e.get('operation')) != (score.STAGE, 'jev', 'jev-1.13.0', 'systemone')):
                        raise ValueError('Foreign ledger request/context')
                    if e.get('status') in (401, 402, 403):
                        if not (e['status'] == 402 and self.permits_402(run, phase, producer)):
                            raise ValueError('Resource/auth failure blocks downstream; no all-fallback evaluation')
            return rows

        result = terminal(run / 'phases' / phase / 'firstpass', refs)
        expected_failed = POLICY['known_failed_requests']
        if scoped and (len(result) != expected_failed or Counter(r['status'] for r in result.values()) != {'failed': expected_failed}):
            raise ValueError('Reviewed firstpass failed cohort differs')
        retry = run / 'phases' / phase / 'recovery_v1'
        failures = [ref for ref in refs if result[reader.bound(ref)['request_id']]['status'] == 'failed']
        if allow_recovery and failures and not retry.exists():
            raise ValueError('Uniform recovery is still required')
        # Even allow_recovery=False cannot hide an already-existing partial or
        # resource-failed recovery in a later offline/re-entry check.
        if retry.exists():
            if not failures:
                raise ValueError('Unnecessary recovery namespace')
            recovered = terminal(retry, failures)
            require_progress(recovered, require_success=scoped)
            if allow_recovery:
                for rid, row in recovered.items():
                    result[rid] = {**row, 'firstpass_result_ref': result[rid]['result_ref']}
        if allow_recovery:
            require_progress(result, require_success=scoped)
        if scoped:
            self.bind_firstpass(reader)
        return result

    @contextmanager
    def activated(self):
        if not _ACTIVATION_LOCK.acquire(blocking=False):
            raise RuntimeError('A billing admission context is already active')
        installed = self.phase_results
        try:
            if score.phase_results is not _ORIGINAL_PHASE_RESULTS:
                raise RuntimeError('Scorer is already modified; refusing nested admission')
            self.bind_review(Reader())
            score.phase_results = installed
            try:
                yield self
            finally:
                score.phase_results = _ORIGINAL_PHASE_RESULTS
        finally:
            _ACTIVATION_LOCK.release()
