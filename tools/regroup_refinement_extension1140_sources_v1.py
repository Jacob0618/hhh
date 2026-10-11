"""Query-blind complementary1140 admission for the two frozen refinement arms.

The original400 and its recovery are only hash-bound and read. No native client,
credentials or API call is created by this source adapter.
"""
from __future__ import annotations
from pathlib import Path
from tools import regroup_refinement_qa_sources_v2 as legacy
from tools import mem0_regroup_plain_qa_sources_v1 as full_source

ROOT, CONFIG = legacy.ROOT, legacy.CONFIG
ARMS = ('source_local_context', 'jev_refined')
COHORT = ROOT / 'configs/sdk20_full1540_locked_v1.json'
PILOT_COHORT = legacy.COHORT
GATE_SCHEMA, CHECKS = legacy.GATE_SCHEMA, legacy.CHECKS
pointer, checked_pointer = legacy.pointer, legacy.checked_pointer
copy_bank, inventory, protocol_config = legacy.copy_bank, legacy.inventory, legacy.protocol_config
admit_gate, verify, read, digest = legacy.admit_gate, legacy.verify, legacy.read, legacy.digest
SOURCE_HASH = legacy.SOURCE_HASH
OLD_RUNS = {arm: ROOT / ('runs/refinement_qa_' + arm + '_v2') for arm in ARMS}
OLD_SUMMARIES = {
    'source_local_context': ROOT / 'research/heartbeat_checks/20260930T1253/qa_context_final_summary_v1.json',
    'jev_refined': ROOT / 'research/heartbeat_checks/20260930T1253/qa_jev_final_summary_v1.json',
}


def split_cohort(full_rows, old_rows):
    full = [q for rows in full_rows.values() for q in rows]
    old = [q for rows in old_rows.values() for q in rows]
    full_ids, old_ids = [q['id'] for q in full], [q['id'] for q in old]
    if (len(full_ids) != 1540 or len(set(full_ids)) != 1540 or
            len(old_ids) != 400 or len(set(old_ids)) != 400 or not set(old_ids) <= set(full_ids)):
        raise ValueError('Exactly the canonical1540 and original400 unique IDs are required')
    excluded = set(old_ids)
    by_id = {q['id']: q for q in full}
    for q in old:
        if any(q[k] != by_id[q['id']][k] for k in ('question', 'reference', 'category', 'source_qa_index')):
            raise ValueError('Inherited400 question/reference identity differs from canonical1540')
    rows = {ci: [q for q in qs if q['id'] not in excluded] for ci, qs in full_rows.items()}
    if sum(map(len, rows.values())) != 1140 or any(not qs for qs in rows.values()):
        raise ValueError('Complement must contain1140 questions across all ten conversations')
    return rows


def cohort():
    full, dates = full_source.cohort()
    old, old_dates = legacy.cohort()
    if dates != old_dates:
        raise ValueError('Inherited400 reference dates differ from canonical1540')
    return split_cohort(full, old), dates


def _summary_checks(summary, arm, first, recovery):
    if (summary.get('schema') != 'regroup-refinement-single-arm400-qa-summary-v1'
            or summary.get('arm') != arm or summary.get('planned_per_arm') != 400
            or summary.get('audit_status') != 'no_invalid_artifacts'
            or summary.get('audit_errors') or summary.get('recovery_supplied') is not True):
        raise ValueError('A closed, audited original400 final summary is required')
    phases = summary.get('phase_sources', [])
    if (len(phases) != 2 or phases[0]['manifest'] != pointer(first / 'run_manifest.json')
            or phases[0]['outcome'] != pointer(first / 'run_outcome.json')
            or phases[1]['manifest'] != pointer(recovery / 'run_manifest.json')
            or phases[1]['outcome'] != pointer(recovery / 'run_outcome.json')):
        raise ValueError('Inherited400 summary phase lineage differs')


def inheritance(arm, *, audit_raw=True):
    if arm not in ARMS:
        raise ValueError('Only the two completed refinement arms can be extended')
    from tools import run_regroup_refinement_qa_v2 as old_driver
    first = OLD_RUNS[arm]
    recovery = first.with_name(first.name + '_recovery_v1')
    # No sealing/mutation: both closed source/outcome files must already exist.
    for path in (first/'source_jobs.json', first/'run_outcome.json', recovery/'run_outcome.json'):
        if not path.is_file():
            raise ValueError('Original400 must already be closed before extension')
    m, out = old_driver.terminal(first)
    rm, rout = old_driver.terminal(recovery)
    summary = verify(OLD_SUMMARIES[arm])
    _summary_checks(summary, arm, first, recovery)
    old_rows, dates = legacy.cohort()
    originals = {q['id']: (int(ci), q) for ci, qs in old_rows.items() for q in qs}
    jobs = m['jobs']
    if (m['arm'] != arm or len(jobs) != 400 or len({j['query_id'] for j in jobs}) != 400
            or {j['query_id'] for j in jobs} != set(originals)
            or out['total'] != 400 or rm['parent']['run_dir'] != str(first.resolve())):
        raise ValueError('Original400 population differs from frozen sampled identities')
    for job in jobs:
        ci, q = originals[job['query_id']]
        if (job['conversation'] != ci or job['reference_date'] != dates[str(ci)]
                or any(job[k] != q[k] for k in ('question', 'reference', 'category'))):
            raise ValueError('Inherited400 payload differs from frozen source question')
        old_driver.check_job(job, arm)
    refs = {name: pointer(path) for name, path in {
        'manifest': first/'run_manifest.json', 'outcome': first/'run_outcome.json',
        'source_jobs': first/'source_jobs.json', 'recovery_manifest': recovery/'run_manifest.json',
        'recovery_outcome': recovery/'run_outcome.json', 'final_summary': OLD_SUMMARIES[arm],
        'sample_manifest': PILOT_COHORT}.items()}
    value = {'schema': 'refinement-inherited400-lineage-v1', 'arm': arm,
             'firstpass_run': str(first.resolve()), 'recovery_run': str(recovery.resolve()),
             'references': refs, 'query_ids': [q['id'] for qs in old_rows.values() for q in qs],
             'job_inputs_sha256': digest(jobs), 'gate': m['gate'],
             'banks': m['banks'], 'evaluation_protocol': m['evaluation_protocol'],
             'official_metric_provenance': m['official_metric_provenance'],
             'old400_final_reused_not_regenerated': True}
    validate_inheritance(value, audit_raw=audit_raw)
    return value


def validate_inheritance(value, *, audit_raw=False):
    arm = value.get('arm')
    if arm not in ARMS or value.get('schema') != 'refinement-inherited400-lineage-v1':
        raise ValueError('Invalid inherited400 lineage')
    first = OLD_RUNS[arm].resolve()
    recovery = first.with_name(first.name + '_recovery_v1')
    expected_paths = {
        'manifest': first/'run_manifest.json', 'outcome': first/'run_outcome.json',
        'source_jobs': first/'source_jobs.json', 'recovery_manifest': recovery/'run_manifest.json',
        'recovery_outcome': recovery/'run_outcome.json', 'final_summary': OLD_SUMMARIES[arm],
        'sample_manifest': PILOT_COHORT}
    if set(value['references']) != set(expected_paths):
        raise ValueError('Inherited400 reference population differs')
    for key, path in expected_paths.items():
        if checked_pointer(value['references'][key]).resolve() != path.resolve():
            raise ValueError('Inherited400 source path changed')
    if value['firstpass_run'] != str(first) or value['recovery_run'] != str(recovery):
        raise ValueError('Inherited400 canonical namespace changed')
    rows, _ = legacy.cohort()
    if value['query_ids'] != [q['id'] for qs in rows.values() for q in qs]:
        raise ValueError('Inherited400 order or population changed')
    m = read(first/'run_manifest.json')
    jobs = verify(first/'source_jobs.json')['jobs']
    if (value['job_inputs_sha256'] != digest(jobs) or value['gate'] != m['gate']
            or value['banks'] != m['banks'] or value['evaluation_protocol'] != m['evaluation_protocol']
            or value['official_metric_provenance'] != m['official_metric_provenance']
            or value.get('old400_final_reused_not_regenerated') is not True):
        raise ValueError('Inherited400 model, facts, input or gate lineage changed')
    summary = verify(value['references']['final_summary']['path'])
    _summary_checks(summary, arm, first, recovery)
    if audit_raw:
        from tools.run_sdk20_modules import sha
        for path, expected in summary['input_sha256'].items():
            if sha(path) != expected:
                raise ValueError('An audited original400 input/raw artifact changed: ' + path)
    return value

