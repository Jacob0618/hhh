"""Offline R09 snapshot and zero-update retrieval continuation raw QA audit; complete persistent write-repair source is validated separately.

The new snapshot and context views avoid rereading shared bytes/scanning all
contexts per stage. Every bound byte is hashed again at the end before output.
They do not change parsing, metrics, recovery admission or request inheritance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_evidence_exchange_r01 as prior
from tools import run_qualifier_repair_retrieval_qa_r09 as runner
from tools.run_sdk20_modules import save_new
from tools.run_sdk20_judge_retest_v2 import read
from tools.check_experiment_health import shared_read

SCHEMA = 'qualifier-repair-r09-retrieval-plain-qa-summary-v1'
ARMS = runner.ARMS
CODE = tuple(dict.fromkeys((*prior.CODE, *runner.CODE, 'tools/summarize_qualifier_repair_r09.py', 'tools/summarize_qualifier_repair_snapshot_r09.py', 'tools/summarize_qualifier_repair_retrieval_r09.py')))
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
    result.update(component_family='persistent_query_independent_qualifier_repair',
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


def source_metadata(ref):
    """Read metadata from the exact source already fully audited twice.

    Hash and parse one immutable byte snapshot. This does not replace either
    full source replay in the inherited summarizer.
    """
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise original.InvalidArtifact('Exact audited source reference required')
    data = shared_read(Path(ref['path']))
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise original.InvalidArtifact('Audited source changed before metadata read')
    value = json.loads(data.decode('utf-8-sig'))
    if value.get('schema') != runner.SOURCE_SCHEMA:
        raise original.InvalidArtifact('Audited source metadata schema differs')
    return value


def summarize(firstpass,recovery=None,*,bootstrap_samples=2000):
    result=_summarize(firstpass,recovery,bootstrap_samples=bootstrap_samples)
    manifest=read(Path(firstpass)/'run_manifest.json')
    if manifest['source_manifest'] != result['source_manifest']:
        raise original.InvalidArtifact('Summary source binding changed')
    source=source_metadata(result['source_manifest'])
    runner.technical_bindings(source)
    protocol=source.get('qualifier_repair_protocol')
    if not isinstance(protocol,dict) or set(protocol)!={'path','sha256'}:
        raise original.InvalidArtifact('Explicit final qualifier-repair protocol binding required')
    result.update(qualifier_repair_protocol=protocol,
        snapshot_compatibility_protocol=source['snapshot_compatibility_protocol'],
        effective_bank_gate=source['effective_bank_gate'], technical_manifest=source['technical_manifest'],
        retrieval_compatibility_protocol=source['retrieval_compatibility_protocol'],
        retrieval_manifest=source['retrieval_manifest'], retrieval_gate=source['retrieval_gate'],
        original_retrieval_failure_preserved=True,
        original_native_failure_preserved=True,
        scientific_protocol_unchanged_by_snapshot_continuation=True,
        selection_counts=source.get('selection_counts'),
        evidence_contract='Original1637 IDs and observation dates preserved; <=2 anchored current-birth replacements/session, cheap common valid proposals versus JEV support and preservation admission. Native SDK UPDATE and SDK20 on isolated banks, unchanged Reader/Judge.',
        prior_raw_proof_lineage=runner.proof_for(manifest).get('reuse_revalidation'),
        summary_snapshot_policy='Immutable first-read bytes/context index; final all-input SHA validation remains.',
        limitations=[
            'Four zero-UPDATE native clones lacked the lock file required for a second closed copy. Original failed retrieval is retained; only exactly equivalent original N0 locked sources substitute for those four physical copies. All20 retrievals are new cache-only runs with prior1900 logical selections checked exactly.',
            'Original native firstpass failed a sealed-envelope versus body comparison; old outcomes retained. Reviewed cheap metadata exports and first-attempt fresh JEV stores pass a separate explicit gate, not the original strict outcome.',
            'The snapshot compatibility only authenticates and separates the content seal; all native numeric/history/source checks remain exact. Deliberate proposer JSON-fence rejection is unchanged.',
            'Exact quote/anchor validation certifies location, not whole replacement entailment or event identity.',
            'JEV support AND preservation thresholds are uncalibrated heuristics, not correctness probabilities or no-loss guarantees.',
            'Current plus original last-ten-turn historical source is bounded; absent support is not evidence an old proposition was false.',
            'Shared proposer is capped at two replacements/session; empty/invalid/noop/rejected are distinct, not proof of no opportunities.',
            'Active replacement can lose supported clauses; frozen original banks and lineage are preserved but not extra Reader input.',
            'Fact count/IDs/date fixed does not imply unchanged retrieval: new vectors, entity cleanup and SDK UPDATE may change selected unrevised facts.',
            'Cheap and JEV may change different amounts of text/index state;20 records are not equal tokens.',
            'Source excludes image-search query/URLs and retains descriptive captions, not independently verified visual truth.',
            'All proposer/JEV/update/entity embedding/index/retrieval/QA costs need physical ledger accounting; inheritance and unknown usage are not zero.',
            'Exact full-request inheritance including old YES and NO introduces correlated results, not independent repetitions.',
            'Reader/Judge configuration and gold remain frozen despite known equivalent-answer Judge sensitivity.',
            'All1540 have been seen during adaptive development; no blind-test or generalization claim.',
            'Persistent UPDATE differs operationally from R05 query-time append and R07 ADD but shares the source-information bottleneck.',
            'A numeric threshold alone does not qualify another component family; actual three-reviewer source/result discussion is required.'])
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
