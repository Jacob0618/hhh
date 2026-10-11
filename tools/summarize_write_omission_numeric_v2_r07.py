"""Offline numeric-v2 compatible R07 raw QA audit; complete persistent write-repair source is validated separately.

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
from tools import run_write_omission_numeric_v2_qa_r07 as runner
from tools.run_sdk20_modules import save_new
from tools.run_sdk20_judge_retest_v2 import read

SCHEMA = 'write-omission-r07-numeric-plain-qa-summary-v1'
ARMS = runner.ARMS
CODE = tuple(dict.fromkeys((*prior.CODE, *runner.CODE, 'tools/summarize_write_omission_r07.py', 'tools/summarize_write_omission_numeric_r07.py', 'tools/summarize_write_omission_numeric_v2_r07.py')))
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
    result.update(component_family='persistent_query_independent_write_omission_repair',
        independent_component_status='not_established_by_this_summary',
        operationally_distinct_from_querytime_append=True,
        shared_source_information_bottleneck=True,
        candidate_component_family_count=1,
        numeric_jev_threshold_met=bool(result['strict_gt_3pp_fixed_denominator']['jev']),
        successful_jev_components_counted_here=0,
        component_success_status='pending_actual_three_reviewer_result_consensus',
        root_stop_count_requires_actual_three_reviewer_result_consensus=True)
    return result


_helpers.update(audit_phase=audit_phase, record_rows=record_rows, aggregate=aggregate,
                readiness=readiness, stopping_result=stopping_result)
_summarize = runner.clone(prior.summarize, _helpers)


def summarize(firstpass,recovery=None,*,bootstrap_samples=2000):
    result=_summarize(firstpass,recovery,bootstrap_samples=bootstrap_samples)
    manifest=read(Path(firstpass)/'run_manifest.json')
    source=runner.source_validate(manifest['source_manifest']['path'])
    protocol=source.get('write_omission_protocol')
    if not isinstance(protocol,dict) or set(protocol)!={'path','sha256'}:
        raise original.InvalidArtifact('Explicit final write-omission protocol binding required')
    result.update(write_omission_protocol=protocol,
        numeric_compatibility_protocol=source['numeric_compatibility_protocol'],
        numeric_io_erratum=source['numeric_io_erratum'],
        numeric_v1_prelaunch_check_failure_retained=True,
        effective_bank_gate=source['effective_bank_gate'],
        original_native_firstpass={'complete':12,'failed':4,'not_run':4},
        native_technical_continuation='Four complete strict-failed cheap banks independently re-audited/exported; four never-started JEV banks first attempted in fresh copies. Original failed records preserved.',
        selection_counts=source.get('selection_counts'),
        evidence_contract='Original1637 facts preserved, shared query-independent bounded proposals; cheap all format-valid versus JEV support/novelty admission; native SDK20 on augmented isolated banks, unchanged Reader/Judge.',
        prior_raw_proof_lineage=runner.proof_for(manifest).get('reuse_revalidation'),
        summary_snapshot_policy='Immutable first-read bytes/context index; final all-input SHA validation remains.',
        limitations=[
            'Original native strict20 gate did not pass. Effective banks use an explicit exact-trajectory numeric compatibility audit; no general epsilon and no scientific/QA changes.',
            'Source quotes certify bytes/locations, not whole-relation entailment. Bounded original history is not exhaustive.',
            'Baseline-prefix representational novelty and source faithfulness are separate; raw .8/.5 scores are uncalibrated heuristics.',
            'Shared proposer capped at two/session can miss valuable information or create supported but unhelpful records.',
            'JEV acceptance may reduce quantity as well as error; same maximum records does not mean equal storage or token budget.',
            'Original records preserved does not guarantee they remain selected after native retrieval; added records can displace necessary evidence.',
            'Source omitted image search metadata, kept descriptive captions; this is not identical old Writer input modality.',
            'Proposer/JEV, new native memory/entity embedding/index, retrieval and QA costs require separate actual ledger accounting; inherited writer costs and unknown usage are not zero.',
            'Common exact requests/old YES and NO reuse are correlated, not independent repetitions.',
            'Model Judge labels are not human truth; current metric configuration remains frozen despite known ambiguity.',
            'All1540 already seen in adaptive development; no blind-test/generalization claim.',
            'Persistent new records differ operationally from R05 query-conditioned raw append but share source-information augmentation. Threshold alone does not establish a second distinct family.',
            'Success-component counting requires actual three-reviewer source/result discussion.'])
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
