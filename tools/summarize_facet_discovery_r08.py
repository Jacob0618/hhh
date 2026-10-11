"""Offline R08 raw QA audit; complete fixed-pool facet source is validated separately.

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
from tools import run_facet_discovery_qa_r08 as runner
from tools.run_sdk20_modules import save_new
from tools.run_sdk20_judge_retest_v2 import read

SCHEMA = 'facet-discovery-r08-plain-qa-summary-v1'
ARMS = runner.ARMS
CODE = tuple(dict.fromkeys((*prior.CODE, *runner.CODE, 'tools/summarize_facet_discovery_r08.py')))
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
        independent_of_source_append='candidate_pending_actual_source_review',
        candidate_component_family_count=1,
        numerical_threshold_met=bool(result['strict_gt_3pp_fixed_denominator']['jev']),
        successful_jev_components_counted_here=0,
        component_qualification='pending_actual_three_reviewer_result_consensus',
        root_stop_count_requires_actual_three_reviewer_result_consensus=True)
    return result


_helpers.update(audit_phase=audit_phase, record_rows=record_rows, aggregate=aggregate,
                readiness=readiness, stopping_result=stopping_result)
_summarize = runner.clone(prior.summarize, _helpers)


def summarize(firstpass, recovery=None, *, bootstrap_samples=2000):
    result = _summarize(firstpass, recovery, bootstrap_samples=bootstrap_samples)
    manifest = read(Path(firstpass) / 'run_manifest.json')
    source = runner.source_validate(manifest['source_manifest']['path'])
    protocol = source.get('discovery_protocol')
    if not isinstance(protocol, dict) or set(protocol) != {'path', 'sha256'}:
        raise original.InvalidArtifact('Explicit final discovery protocol binding required')
    result.update(discovery_protocol=protocol,
        selection_counts=source.get('selection_counts'),
        evidence_contract='Frozen query-only facets discover original-bank candidates outside native80; shared80 with protected originalR01first60; fulloriginalQ R03 scoring and samepoolBM25 select20 original records restored to pool order. No sourceappend or record rewrite.',
        prior_raw_proof_lineage=runner.proof_for(manifest).get('reuse_revalidation'),
        summary_snapshot_policy='Cache immutable bytes and consumer-context row indexes during replay; original final all-file byte SHA verification remains mandatory.',
        limitations=[
            'Sealed query-only facets are discovery queries, not facts or certified faithful decompositions. No new facet generation.',
            'Shared80 preserves originalR01first60 and replaces at most20 tail records using stable roundrobin dense rankings outside originalnative80. Final20 need not preserve baseline membership.',
            'OriginalR03 wholeQ predicate, BM25 and stable pool-order output are reused. ID provenance is external; HTTP record literal remains text/date only.',
            'Dense pool-out candidates may be broad or redundant, displacing useful lexical tail. The original bank cannot reveal relations omitted at writing.',
            'Raw Noul probabilities are uncalibrated ranking utilities, not semantic truth or whole-answer sufficiency.',
            'm1 or zero outsiders preserves exact oldR01pool; known discovery failure instead returns both arms exactN0. JEV known score failure returns JEV N0; unknown blocks seal.',
            'Fixed20 does not match Reader tokens; the discovery scan examines the whole conversation bank despite fixed80 selection capacity.',
            'OldR06 facet generation and exact wholeR03 score reuse are historical producers; current physical embedding/JEV misses and QA are incremental, unknown usage not zero.',
            'Exact same requests, including validNO, are correlated cached consumers rather than independent repeats or free cold start.',
            'Narrow uniform postterminal technical recovery excludes malformedHTTP200, nonretryableHTTP and validNO; missing outcomes remain in fixed1540.',
            'Judge labels are observations requiring source case review; no human accuracy, heldout or causal prior-round comparison claim.',
            'Retrieval family remains one component distinct from R05 sourceappend; root success count requires threshold plus actual three-reviewer closure.'])
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
