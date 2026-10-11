"""Offline first-pass plus recovery overlay accounting; never edits source runs."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.data import load_data, queries
from jev_mem.evaluation import lexical_metrics
from jev_mem.locomo_official_metrics import official_locomo_f1
from tools.summarize_experiments import usage_cost, PRICES_USD_PER_MILLION


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def accept_overlay(row, q, *, failed_keys, seen, seal_sha):
    """Reject inappropriate denominators and score/schema drift before merging."""
    digest = hashlib.sha256(json.dumps({k: v for k, v in row.items() if k != 'row_sha256'},
                            sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    if row.get('row_sha256') != digest:
        raise ValueError('Recovery row hash mismatch')
    key = (row['architecture'], row['conversation'], row['method'], row['query_id'])
    if (row.get('status') not in {'complete', 'blocked', 'failed', 'unknown', 'unbound'}
            or key not in failed_keys or key in seen or row['seal_sha256'] != seal_sha):
        raise ValueError('Overlay is duplicated, not originally failed, or bound to another seal')
    seen.add(key)
    if row['status'] != 'complete':
        return None
    if (row['query_id'] != q['id'] or row['question'] != q['question'] or row['reference'] != q['reference']
            or row['category'] != q['category'] or row.get('reader_protocol') != 'brief_answer_local_evidence_ids_v2'):
        raise ValueError('Overlay question/reader binding mismatch')
    selected = row['selected_ids']
    if (not isinstance(selected, list) or any(not isinstance(x, str) or not x for x in selected)
            or len(selected) != 20 or len(set(selected)) != 20 or row.get('selected_count') != 20):
        raise ValueError('Recovery must retain exactly20 unique facts')
    answer = row['prediction']['answer']
    judgment = row['judgment']
    completeness = judgment.get('completeness')
    if (not isinstance(answer, str) or type(judgment.get('correct')) is not bool
            or isinstance(completeness, bool) or not isinstance(completeness, (int, float))
            or not math.isfinite(completeness) or not 0 <= completeness <= 1
            or not isinstance(judgment.get('reason'), str)
            or row.get('reader_model') != 'gemini-2.5-flash' or row.get('judge_model') != 'gemini-3.5-flash-lite'
            or not row.get('request_events')):
        raise ValueError('Recovery answer/judge format invalid')
    scores = lexical_metrics(answer, q['reference'])
    if any(isinstance(row[name], bool) or not isinstance(row[name], (int, float))
           or not math.isfinite(row[name]) or abs(scores[name] - row[name]) > 1e-12
           for name in ('token_f1', 'exact_match')):
        raise ValueError('Custom metric differs from reconstruction')
    return {'query_id': q['id'], 'conversation': row['conversation'], 'category': q['category'],
            'strict_correct': judgment['correct'], 'custom_token_f1': scores['token_f1'],
            'custom_exact_match': scores['exact_match'],
            'official_locomo_f1': official_locomo_f1(answer, q['reference'], q['category'])}


def group(rows, requested=1540):
    metrics = {}
    for name in ('strict_correct', 'custom_token_f1', 'custom_exact_match', 'official_locomo_f1'):
        value = sum(float(r[name]) for r in rows)
        metrics[name] = {'completed_only': value / len(rows) if rows else None,
                         'missing_zero_sensitivity': value / requested,
                         'fully_observed': value / requested if len(rows) == requested else None}
    return {'requested': requested, 'completed': len(rows), 'missing': requested - len(rows),
            'technical_failure_rate': (requested - len(rows)) / requested, 'metrics': metrics}


def validate_frozen_replay(store, row, failure, recovery_dir):
    """Recompute selectors and evaluator from bound original/overlay raw responses."""
    from jev_mem import recovery_v1 as rec, selection, evaluation
    from jev_mem.providers import ModelAPI
    q, retrieval, reference_date, bindings = store.inputs(failure)
    if row.get('input_bindings') != bindings or row.get('policy') != rec.POLICY:
        raise ValueError('Overlay sealed input/policy binding mismatch')
    events = {(e['stage'], e['request_hash']): e for e in row['request_events']}
    if len(events) != len(row['request_events']):
        raise ValueError('Duplicate recovery request event')
    visited = set()

    class BoundResponses:
        generate = ModelAPI.generate
        score_many = ModelAPI.score_many

        def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', **_):
            sig = {'provider': provider, 'model': model, 'body': body, 'operation': operation}
            h = rec.request_hash(sig); key = (stage, h)
            e = events.get(key)
            if e is None or e.get('status') != 'valid':
                raise ValueError('Frozen replay lacks a valid matching response event')
            if e.get('source') == 'sealed_cache':
                valid, sources = store.cached(sig, stage)
                if valid is None or e.get('chosen_source') != valid[1] or e.get('sources') != sources:
                    raise ValueError('Original cache source binding differs')
                response = valid[0]
            elif e.get('source') in {'overlay_response', 'new_http'}:
                raw = read(recovery_dir / 'attempts' / h / 'response.json')
                if raw.get('request') != sig or raw.get('seal_sha256') != store.seal_sha256:
                    raise ValueError('Recovery raw response is not bound to reconstructed payload')
                response = raw['response']
            else:
                raise ValueError('Unknown response source')
            if rec.digest(response) != e.get('response_sha256'):
                raise ValueError('Response hash differs from recorded event')
            visited.add(key)
            return copy.deepcopy(response)

    api = BoundResponses()
    if row['method'] == 'sdk20':
        selected = retrieval['sdk20']; trace = {'method': 'sdk20'}
    else:
        result = selection.select(q['question'], retrieval['pool60'], api, method=row['method'], top_k=20,
                                  seed_count=8, round_sizes=(6, 6), shortlist_size=32,
                                  conditional_threshold=.5, mmr_weight=.7, max_workers=1)
        selected, trace = result.selected, result.trace
    if row['selected_ids'] != [f['id'] for f in selected] or row['selection_trace'] != trace:
        raise ValueError('Recovered selection differs from frozen replay')
    expected = evaluation.answer_and_judge(api, q['question'], q['reference'], selected, reference_date=reference_date)
    if any(row.get(k) != v for k, v in expected.items()) or visited != set(events):
        raise ValueError('Recovered evaluator/result/request set differs from raw-response replay')


def actual_recovery_usage(directory):
    """Reconcile the one-attempt ledger and transport evidence without resending.

    A durable started marker consumes the retry allowance even if the process
    crashed before reaching the wire. Such an unlogged intent is not called a
    confirmed paid HTTP request, and its possible usage is never filled with zero.
    """
    import math
    from jev_mem.recovery_v1 import POLICY, request_hash

    directory = Path(directory)
    path = directory / 'transport/api_calls.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    physical_by_hash, cached_by_hash = {}, {}

    def hash_ok(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

    for row in rows:
        if not isinstance(row, dict) or not hash_ok(row.get('request_hash')):
            raise ValueError('Transport log lacks a valid request hash')
        h = row['request_hash']
        if 'cache_hit' in row and type(row['cache_hit']) is not bool:
            raise ValueError('cache_hit must be boolean when present')
        if row.get('cache_hit'):
            cached_by_hash.setdefault(h, []).append(row)
        else:
            if h in physical_by_hash:
                raise ValueError('More than one new physical HTTP recorded for the same hash')
            physical_by_hash[h] = row

    attempt_dirs = {p.name: p for p in (directory / 'attempts').iterdir() if p.is_dir()} if (directory / 'attempts').exists() else {}
    cache_dir = directory / 'transport/api_cache'
    transport_caches = {p.stem: p for p in cache_dir.glob('*.json')} if cache_dir.exists() else {}
    evidence_hashes = set(physical_by_hash) | set(cached_by_hash) | set(transport_caches)
    if not evidence_hashes <= set(attempt_dirs):
        raise ValueError('Transport evidence exists without an attempt ledger')
    if any(not hash_ok(h) for h in attempt_dirs):
        raise ValueError('Attempt directory name must be a request SHA256')
    binding_path = directory / 'binding.json'
    binding = read(binding_path) if binding_path.exists() else None
    if attempt_dirs or rows or transport_caches:
        if (not isinstance(binding, dict) or not hash_ok(binding.get('seal_sha256'))
                or binding.get('policy') != POLICY):
            raise ValueError('Nonempty recovery accounting requires its bound seal and policy')

    def checked_usage(value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError('Provider usage must be an object')
        for key in ('input_tokens', 'output_tokens', 'promptTokenCount', 'candidatesTokenCount', 'thoughtsTokenCount'):
            if key in value and (type(value[key]) not in (int, float)
                                 or not math.isfinite(value[key]) or value[key] < 0):
                raise ValueError('Provider usage contains invalid token counts')
        return value

    def response_usage(response):
        if not isinstance(response, dict):
            raise ValueError('Persisted raw response must be an object')
        return checked_usage(response.get('usageMetadata', response.get('usage', {})))

    def check_signature_fields(row, plan, *, context=None):
        signature = plan['request']
        for field in ('provider', 'model', 'operation'):
            if row.get(field) != signature[field]:
                raise ValueError('Transport ' + field + ' differs from planned request')
        if row.get('stage') != plan['stage']:
            raise ValueError('Transport stage differs from planned request')
        if context is not None:
            actual = row.get('context')
            if not isinstance(actual, dict) or any(actual.get(k) != v or type(actual.get(k)) != type(v)
                                                  for k, v in context.items()):
                raise ValueError('Transport consumer differs from the owner of the started attempt')

    counters = Counter()
    costs, entries = [], []
    for h, folder in sorted(attempt_dirs.items()):
        plan_path = folder / 'planned.json'
        if not plan_path.exists():
            raise ValueError('Attempt ledger is missing planned.json')
        plan = read(plan_path)
        if (not isinstance(plan, dict) or plan.get('schema') != 'jev-mem-recovery-request-v1'
                or plan.get('request_hash') != h or plan.get('seal_sha256') != binding['seal_sha256']
                or plan.get('policy') != POLICY or type(plan.get('max_new_http_attempts')) is not int
                or plan['max_new_http_attempts'] != 1):
            raise ValueError('Planned request binding or one-attempt policy differs')
        signature = plan.get('request')
        if (not isinstance(signature, dict) or set(signature) != {'provider', 'model', 'operation', 'body'}
                or request_hash(signature) != h):
            raise ValueError('Planned signature does not reproduce the directory request hash')
        stage_contract = {'jev_score': ('jev', 'jev-1.13.0', 'systemone'),
                          'qa_reader': ('gemini', 'gemini-2.5-flash', 'generateContent'),
                          'strict_judge': ('gemini', 'gemini-3.5-flash-lite', 'generateContent')}
        if tuple(signature.get(k) for k in ('provider', 'model', 'operation')) != stage_contract.get(plan.get('stage')):
            raise ValueError('Attempt stage/model is outside frozen recovery QA')
        counters['planned'] += 1
        started_path, response_path, outcome_path = (folder / name for name in
                                                       ('attempt_started.json', 'response.json', 'outcome.json'))
        started = read(started_path) if started_path.exists() else None
        raw_response = read(response_path) if response_path.exists() else None
        outcome = read(outcome_path) if outcome_path.exists() else None
        transport_cache = read(transport_caches[h]) if h in transport_caches else None
        physical = physical_by_hash.get(h)
        cached = cached_by_hash.get(h, [])
        if started is None:
            if any(value is not None for value in (raw_response, outcome, transport_cache, physical)) or cached:
                raise ValueError('Response, outcome or transport evidence without a started attempt')
            entries.append({'request_hash': h, 'stage': plan['stage'], 'status': 'planned_not_started',
                            'attempt_allowance_consumed': False, 'usage_unknown': False,
                            'known_cost_component_usd': 0.0, 'cost_total_usd': 0.0})
            continue
        if (not isinstance(started, dict) or started.get('request_hash') != h
                or started.get('seal_sha256') != binding['seal_sha256']):
            raise ValueError('Started marker binding differs')
        owner = (started.get('proof') or {}).get('consumer')
        if (not isinstance(owner, dict) or set(owner) != {'architecture', 'conversation', 'query_id', 'method'}
                or owner['architecture'] not in ('mem0', 'independent')
                or type(owner['conversation']) is not int or not 0 <= owner['conversation'] <= 9
                or not isinstance(owner['query_id'], str) or not isinstance(owner['method'], str)):
            raise ValueError('Started marker requires a valid recovery consumer identity')
        counters['started'] += 1
        persisted = None
        for record, is_overlay in ((raw_response, True), (transport_cache, False)):
            if record is None:
                continue
            if (not isinstance(record, dict) or record.get('request') != signature
                    or request_hash(record['request']) != h or record.get('stage') != plan['stage']):
                raise ValueError('Persisted response/cache request differs from the ledger')
            if is_overlay and (record.get('request_hash') != h or record.get('seal_sha256') != binding['seal_sha256']):
                raise ValueError('Overlay response hash or seal differs')
            if 'request_hash' in record and record['request_hash'] != h:
                raise ValueError('Persisted response has a mismatched explicit request hash')
            response_usage(record.get('response'))
            if persisted is not None and json.dumps(persisted, sort_keys=True, ensure_ascii=False) != json.dumps(record['response'], sort_keys=True, ensure_ascii=False):
                raise ValueError('Overlay response differs from transport cache')
            persisted = record['response']
        if raw_response is not None:
            counters['responses'] += 1
        if transport_cache is not None:
            counters['transport_caches'] += 1
        outcome_status = outcome.get('status') if isinstance(outcome, dict) else None
        if outcome is not None:
            if (not isinstance(outcome, dict) or outcome.get('request_hash') != h
                    or outcome_status not in ('complete', 'failed', 'unknown')):
                raise ValueError('Outcome identity or status is invalid')
            if 'seal_sha256' in outcome and outcome['seal_sha256'] != binding['seal_sha256']:
                raise ValueError('Outcome seal differs')
            if outcome_status == 'complete' and raw_response is None:
                raise ValueError('Complete outcome lacks its persisted response')
            known_http_failure = (physical is not None and type(physical.get('status')) is int
                                  and 400 <= physical['status'] <= 599)
            if outcome_status == 'failed' and raw_response is None and not known_http_failure:
                raise ValueError('Failed outcome lacks either a known HTTP failure or a schema-validated response')
            if outcome_status == 'failed' and raw_response is not None:
                counters['schema_validation_failed'] += 1
            counters['outcome_' + outcome_status] += 1
        for cache_row in cached:
            check_signature_fields(cache_row, plan)
            if persisted is None or checked_usage(cache_row.get('usage')) != response_usage(persisted):
                raise ValueError('Cache log has no matching persisted response/usage')
        if physical is not None:
            check_signature_fields(physical, plan, context=owner)
            if type(physical.get('attempt')) is not int or physical['attempt'] != 1:
                raise ValueError('Recovery physical log must be first and only HTTP attempt')
            status = physical.get('status')
            if type(status) is int and 100 <= status <= 599:
                if status >= 400:
                    counters['http_failures'] += 1
                    network_status = 'http_failure'
                    if persisted is not None:
                        raise ValueError('HTTP failure unexpectedly has a persisted success response')
                elif 200 <= status < 400:
                    counters['http_successes'] += 1
                    counters['http200'] += status == 200
                    network_status = 'http_response_success'
                else:
                    counters['http_other'] += 1
                    network_status = 'http_other_status'
            elif status == 'transport_error':
                counters['transport_errors'] += 1
                network_status = 'transport_error'
                if persisted is not None:
                    raise ValueError('A sole transport-error attempt cannot also have a persisted response')
            else:
                raise ValueError('Transport log has an unknown status encoding')
            usage = checked_usage(physical.get('usage'))
            if persisted is not None and usage != response_usage(persisted):
                raise ValueError('Physical log usage differs from persisted response usage')
            billing_row = physical
            billing_source = 'physical_log'
        else:
            counters['started_unlogged'] += 1
            if persisted is None:
                counters['started_unlogged_no_response'] += 1
                network_status = 'started_unlogged_no_response'
                usage = {}
                billing_source = 'unknown_started_attempt'
            else:
                counters['response_backed_unlogged'] += 1
                network_status = 'response_backed_unlogged'
                usage = response_usage(persisted)
                billing_source = 'persisted_response_without_log'
            billing_row = {**{k: signature[k] for k in ('provider', 'model', 'operation')}, 'usage': usage}
        cost = usage_cost(billing_row)
        if network_status in ('started_unlogged_no_response', 'transport_error'):
            cost = {**cost, 'known_component_usd': 0.0, 'unknown': True, 'cost_usd': None}
        token_fields = ('input_tokens', 'output_tokens') if signature['provider'] == 'jev' else ('promptTokenCount', 'candidatesTokenCount')
        token_usage_unknown = (not all(field in usage for field in token_fields)
                               or network_status in ('started_unlogged_no_response', 'transport_error'))
        counters['unknown_usage'] += token_usage_unknown
        if cost['unknown'] and physical is None:
            counters['unlogged_unknown_cost'] += 1
        costs.append(cost)
        entries.append({'request_hash': h, 'stage': plan['stage'], 'status': network_status,
                        'outcome_status': outcome_status, 'attempt_allowance_consumed': True,
                        'physical_log_present': physical is not None, 'response_present': persisted is not None,
                        'billing_source': billing_source, 'usage_unknown': token_usage_unknown,
                        'cost_unknown': bool(cost['unknown']),
                        'recorded_usage': usage if usage else None,
                        'known_cost_component_usd': cost['known_component_usd'],
                        'cost_total_usd': None if cost['unknown'] else cost['cost_usd']})

    physical = list(physical_by_hash.values())
    known_cost = sum(c['known_component_usd'] for c in costs)
    unknown = sum(bool(c['unknown']) for c in costs)
    return {'logged_physical_attempts': len(physical),
            'http200': counters['http200'], 'http_success_responses': counters['http_successes'],
            'http_failure_responses': counters['http_failures'], 'transport_error_attempts': counters['transport_errors'],
            'http_other_status_attempts': counters['http_other'],
            'http_failures_or_transport': counters['http_failures'] + counters['transport_errors'] + counters['http_other'],
            'cache_hits_in_transport_log': sum(map(len, cached_by_hash.values())),
            'planned_requests': counters['planned'], 'attempt_allowance_slots_consumed': counters['started'],
            'planned_not_started': counters['planned'] - counters['started'],
            'persisted_overlay_responses': counters['responses'], 'persisted_transport_caches': counters['transport_caches'],
            'started_but_unlogged_attempts': counters['started_unlogged'],
            'started_but_unlogged_without_response': counters['started_unlogged_no_response'],
            'response_backed_unlogged_attempts': counters['response_backed_unlogged'],
            'unlogged_attempts_with_unknown_cost': counters['unlogged_unknown_cost'],
            'confirmed_physical_attempts_lower_bound': len(physical) + counters['response_backed_unlogged'],
            'possible_physical_attempts_upper_bound': counters['started'],
            'schema_validation_failed_attempts': counters['schema_validation_failed'],
            'outcome_failed_attempts': counters['outcome_failed'],
            'outcome_unknown_attempts': counters['outcome_unknown'],
            'actual_known_cost_usd': known_cost, 'actual_unknown_cost_calls': unknown,
            'attempts_with_unknown_token_usage': counters['unknown_usage'],
            'actual_cost_complete': unknown == 0,
            'actual_total_cost_usd': known_cost if unknown == 0 else None,
            'unknown_cost_usd': None if unknown else 0.0,
            'attempt_audit': entries, 'prices_usd_per_million': PRICES_USD_PER_MILLION,
            'note': 'Configured usage-price estimate, not an invoice. Started markers consume the sole retry allowance but do not prove the request reached the wire. Unlogged started intents without response retain unknown usage/cost; bound persisted responses may recover known usage. HTTP status and response-format outcome are distinct. Cache rows do not add usage or fees.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seal', type=Path, required=True)
    p.add_argument('--recovery-dir', type=Path, required=True)
    p.add_argument('--firstpass-dir', type=Path, default=ROOT / 'research/heartbeat_checks/20260927T2241')
    p.add_argument('--firstpass-binding', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    seal = read(args.seal); seal_sha = file_sha(args.seal)
    from jev_mem.recovery_v1 import SealStore
    store = SealStore(args.seal)
    failures_path = args.seal.parent / seal['failures_path']
    if file_sha(failures_path) != seal['failures_sha256']:
        raise ValueError('Sealed failure list changed')
    failures = read(failures_path)
    failed_keys = {(r['architecture'], r['conversation'], r['method'], r['query_id']) for r in failures}
    dataset = load_data(); qs = {q['id']: q for ci, item in enumerate(dataset) for q in queries(item, ci)}
    bound = read(args.firstpass_binding or args.seal.parent / 'firstpass_reports.json')
    if bound.get('seal_sha256') != seal_sha or bound.get('schema') != 'jev-mem-firstpass-report-binding-v1':
        raise ValueError('First-pass report binding differs from seal')
    first = {}
    for run in seal['runs']:
        arch = run['architecture']; entry = bound['reports'][arch]; path = ROOT / entry['path']
        if file_sha(path) != entry['sha256']:
            raise ValueError('First-pass scientific report bytes changed')
        report = read(path)
        if (report['issues'] or set(report['per_query_metrics']) != set(run['methods'])
                or report['requested_questions'] != 1540 or report['architecture'] != arch):
            raise ValueError('First-pass report protocol/validation differs')
        for method, rr in report['per_query_metrics'].items():
            expected = set(qs) - {f['query_id'] for f in failures if f['architecture'] == arch and f['method'] == method}
            ids = [r['query_id'] for r in rr]
            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError('First-pass success IDs differ from seal complement')
        if sum(len(rr) for rr in report['per_query_metrics'].values()) != run['successful_evaluations']:
            raise ValueError('First-pass scientific report count differs from seal')
        first[arch] = report
    combined = {arch: {m: {r['query_id']: r for r in rows} for m, rows in report['per_query_metrics'].items()}
                for arch, report in first.items()}
    seen, statuses, added, row_hashes = set(), Counter(), Counter(), {}
    for path in sorted((args.recovery_dir / 'execute/consumers').glob('*/conv_*/*/*.json')):
        row = read(path)
        if row.get('schema') != 'jev-mem-technical-recovery-result-v1':
            raise ValueError('Wrong overlay schema')
        key = (row['architecture'], row['conversation'], row['method'], row['query_id'])
        failure = next((f for f in failures if (f['architecture'], f['conversation'], f['method'], f['query_id']) == key), None)
        if failure is None or row.get('original_failure') != failure:
            raise ValueError('Overlay original failure provenance differs')
        if row['status'] == 'complete':
            validate_frozen_replay(store, row, failure, args.recovery_dir)
        metric = accept_overlay(row, qs[row['query_id']], failed_keys=failed_keys, seen=seen, seal_sha=seal_sha)
        statuses[row['status']] += 1
        row_hashes[path.relative_to(args.recovery_dir).as_posix()] = file_sha(path)
        if metric:
            target = combined[row['architecture']][row['method']]
            if row['query_id'] in target:
                raise ValueError('Cannot overwrite first-pass success')
            target[row['query_id']] = metric
            added[(row['architecture'], row['method'])] += 1
    report = {'schema': 'jev-mem-technical-recovery-summary-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'seal_sha256': seal_sha, 'firstpass_is_unchanged': True, 'expected_failure_consumers': len(failed_keys),
              'recovery_rows': len(seen), 'missing_recovery_rows': len(failed_keys - seen),
              'recovery_statuses': dict(statuses), 'runs': {}, 'recovery_usage': actual_recovery_usage(args.recovery_dir),
              'overlay_file_sha256': row_hashes, 'model_calls_made': 0,
              'limitations': 'This summary validates cohort, reader, budget, lexical scores and artifact bindings. It does not replace raw-response or source-faithfulness review. All results include development conversations0/1.'}
    for arch, methods in combined.items():
        report['runs'][arch] = {'firstpass_issues': first[arch]['issues'], 'methods': {
            m: {'firstpass': group(first[arch]['per_query_metrics'][m]),
                'after_recovery': group(list(rows.values())), 'recovered': added[(arch, m)]}
            for m, rows in methods.items()}, 'per_query_metrics_after_recovery': {m: list(rows.values()) for m, rows in methods.items()}}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'summary.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    lines = ['# First pass and bounded technical recovery', '',
             'Failed/missing rows remain in the1540-per-method denominator. Zero-fill is a sensitivity analysis; fully observed scores are null if anything is missing.', '',
             '| Architecture/method | First pass | Recovered | Final completed | Official F1 (completed / missing=0) | Judge (completed / missing=0) |',
             '|---|---:|---:|---:|---:|---:|']
    for arch, run in report['runs'].items():
        for m, row in run['methods'].items():
            final = row['after_recovery']; f1 = final['metrics']['official_locomo_f1']; judge = final['metrics']['strict_correct']
            lines.append(f"| {arch}/{m} | {row['firstpass']['completed']}/1540 | {row['recovered']} | {final['completed']}/1540 | {f1['completed_only']:.4f}/{f1['missing_zero_sensitivity']:.4f} | {judge['completed_only']:.4f}/{judge['missing_zero_sensitivity']:.4f} |")
    with (args.output_dir / 'summary.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print(json.dumps({'output': str(args.output_dir), 'expected_failures': len(failed_keys), 'recovery_rows': len(seen), 'statuses': dict(statuses)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
