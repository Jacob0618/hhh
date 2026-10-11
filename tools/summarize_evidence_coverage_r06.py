"""Offline R06 raw QA audit; complete fixed-pool facet source is validated separately.

The new snapshot and context views avoid rereading shared bytes/scanning all
contexts per stage. Every bound byte is hashed again at the end before output.
They do not change parsing, metrics, recovery admission or request inheritance.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_evidence_exchange_r01 as prior
from tools import run_evidence_coverage_qa_r06 as runner
from tools.run_sdk20_modules import save_new
from tools.run_sdk20_judge_retest_v2 import read

SCHEMA = 'evidence-coverage-r06-plain-qa-summary-v1'
ARMS = runner.ARMS
CODE = tuple(dict.fromkeys((*prior.CODE, *runner.CODE, 'tools/summarize_evidence_coverage_r06.py')))
original = prior.original
_cached_snapshot_json = prior._cached_snapshot_json
_ordinary_audit_stage = prior._ordinary_audit_stage


class Snapshot(original.Snapshot):
    """Owned immutable-byte view; final unchanged() still rereads every binding.

    This caches first-read bytes, never accepts new on-disk contents silently.
    Deliberate mutation between reads is detected by final unchanged(), and by
    the inherited summary's second full input SHA pass before publication.
    """
    def __init__(self):
        super().__init__()
        self._bytes = {}
        self._context_indexes = {}

    def raw(self, path):
        key = Path(path).resolve()
        if key not in self._bytes:
            self._bytes[key] = super().raw(key)
        return self._bytes[key]


class ContextIndexed(Mapping):
    """Restrict values() scans to consumer context; producer items/get stay full.

    The frozen audit stage already filters values() by equality with this
    context. Physical/cache producer checks use items()/get() and must keep
    other consumer/producer contexts of the same exact request available.
    """
    def __init__(self, snapshot, indexed, context):
        self.indexed = indexed
        token = id(indexed)
        cache = getattr(snapshot, '_context_indexes', None)
        if cache is None:
            cache = {}
            snapshot._context_indexes = cache
        entry = cache.get(token)
        if entry is None or entry[0] is not indexed:
            groups = defaultdict(list)
            for rows in indexed.values():
                for row in rows:
                    groups[original.old.fingerprint(row.get('context', {}))].append(row)
            entry = (indexed, groups)
            cache[token] = entry
        self.rows = entry[1].get(original.old.fingerprint(context), [])

    def __getitem__(self, key):
        return self.indexed[key]

    def __iter__(self):
        return iter(self.indexed)

    def __len__(self):
        return len(self.indexed)

    def values(self):
        return (self.rows,)

    def items(self):
        return self.indexed.items()

    def get(self, key, default=None):
        return self.indexed.get(key, default)


_helpers = {**vars(prior), 'runner': runner, 'SCHEMA': SCHEMA, 'ARMS': ARMS, 'CODE': CODE}
_stage = runner.clone(prior.audit_stage, _helpers)


def audit_stage(snapshot, path, expected, api_dir, context, indexed, callback, *, no_dispatch=False):
    return _stage(snapshot, path, expected, api_dir, context,
                  ContextIndexed(snapshot, indexed, context), callback, no_dispatch=no_dispatch)


_helpers['audit_stage'] = audit_stage
_scope = {**prior._scope, 'runner': runner, 'ARMS': ARMS, 'SCHEMA': SCHEMA, 'CODE': CODE,
    'Snapshot': Snapshot,
    'ledger': original._clone(original.old.ledger, {'runner': runner, 'evaluator': original.evaluator}),
    'audit_stage': audit_stage}
for name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
    _scope[name] = original._clone(getattr(original, name), _scope)
for name in ('audit_phase', 'record_rows', 'aggregate', 'readiness'):
    globals()[name] = _scope[name]


def stopping_result(rows):
    result = prior.stopping_result(rows)
    result.pop('independent_component_count', None)
    result.update(component_family='original_bank_retrieval_selection_shared_with_r03',
        independent_of_source_append=True,
        candidate_component_family_count=1,
        successful_jev_components_counted_here=int(result['strict_gt_3pp_fixed_denominator']['jev']),
        root_stop_count_requires_actual_three_reviewer_result_consensus=True)
    return result


_helpers.update(audit_phase=audit_phase, record_rows=record_rows, aggregate=aggregate,
                readiness=readiness, stopping_result=stopping_result)
_summarize = runner.clone(prior.summarize, _helpers)


def summarize(firstpass, recovery=None, *, bootstrap_samples=2000):
    result = _summarize(firstpass, recovery, bootstrap_samples=bootstrap_samples)
    manifest = read(Path(firstpass) / 'run_manifest.json')
    source = runner.source_validate(manifest['source_manifest']['path'])
    protocol = source.get('coverage_protocol')
    if not isinstance(protocol, dict) or set(protocol) != {'path', 'sha256'}:
        raise original.InvalidArtifact('Explicit final coverage protocol binding required')
    result.update(coverage_protocol=protocol,
        selection_counts=source.get('selection_counts'),
        evidence_contract='Complete query-derived facets and all per-facet candidate scores;20 original records from exact pinned common pool restored to pool order. No sourceappend or record rewrite.',
        prior_raw_proof_lineage=runner.proof_for(manifest).get('reuse_revalidation'),
        summary_snapshot_policy='Cache immutable bytes and consumer-context row indexes during replay; original final all-file byte SHA verification remains mandatory.',
        limitations=[
            'Query-only facets are syntactically checked, not certified faithful decompositions; originalQ facet0 remains.',
            'Raw Noul pYES and column-normalized BM25 are different uncalibrated utilities. Max-coverage is not a probability or semantic support certificate.',
            'Marginal-gain saturation and zero-fill do not certify list diversity, missing bridges, completeness or Reader use of present evidence.',
            'Any shared facet failure returns both methods originalN0; any JEV column failure returns JEV originalN0, with source lineage preserved.',
            'No added source or retrieval pool capacity. Facts absent from every candidate cannot be restored by this module.',
            'Fixed20 and unchanged per-record text are not equal Reader tokens across memberships. Bytes are not measured tokens.',
            'Facet generator costs are shared cold overhead; JEV pays additional per-facet matrices. Historical inherited usage is separate from new physical requests.',
            'Inherited exact requests, including validNO, are correlated consumers and not independent repetitions or free cold-start costs.',
            'Judge labels are model observations, not independently established human correctness.',
            'Direction is informed by repeated prior QA diagnoses on seen1540; not a heldout estimate. Per-query gold/answers/labels are forbidden method input.',
            'This is the R03 retrieval family and is independent from R05 sourceappend; cannot count both retrieval rounds as separate components.',
            'Numerical success and next-round admission require actual threshold and three-reviewer result discussion.'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firstpass', type=Path, required=True)
    parser.add_argument('--recovery', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--readiness', action='store_true')
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1: raise ValueError('Positive bootstrap count required')
    if not args.readiness and args.output is None: parser.error('--output required')
    with original.historical_guard.historical.offline():
        result = (readiness(args.firstpass, args.recovery) if args.readiness else
                  summarize(args.firstpass, args.recovery, bootstrap_samples=args.bootstrap_samples))
    if args.output: save_new(args.output, result)
    import json
    print(json.dumps(result if args.readiness else {'output': str(args.output),
        'audit_status': result['audit_status'], 'model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__': main()
