"""Explicit complete-facts admission; native stores and QA behavior stay frozen.

A per-arm ten-bank gate must reference the same terminal uniform twenty-bank
review. Only strict_complete and a specifically proved metadata-only export are
eligible. This module never publishes a gate/export or opens a native client.
"""
from __future__ import annotations

import copy
from pathlib import Path

from tools import regroup_refinement_qa_sources_v1 as legacy
from tools import mem0_regroup_plain_qa_sources_v2 as common

ROOT, SOURCE_HASH = legacy.ROOT, legacy.SOURCE_HASH
ARMS = ('source_local_context', 'jev_refined')
GATE_SCHEMA = 'regroup-refinement-qa-gate-v2'
REVIEW_SCHEMA = 'regroup-refinement-uniform20-review-v2'
REVIEW_STATUS = 'passed_complete_facts_with_disclosed_native_defects'
CHECKS = ('twenty_terminal_banks_reviewed', 'twenty_complete_fact_banks',
          'source_dates_and_generation_chain_verified', 'current_sdk20_index_dependencies_reviewed',
          'native_index_limitations_disclosed', 'content_quality_findings_disclosed')
COHORT, CONFIG = legacy.COHORT, legacy.CONFIG
pointer, checked_pointer = legacy.pointer, legacy.checked_pointer
copy_bank, inventory, protocol_config, cohort = legacy.copy_bank, legacy.inventory, legacy.protocol_config, legacy.cohort
verify, read, digest = legacy.verify, legacy.read, legacy.digest


def export_module():
    # Late import permits offline schema tests without importing an unfinished
    # reviewer. Production always resolves the frozen module included in CODE.
    from tools import review_regroup_refinement_banks_v2
    return review_regroup_refinement_banks_v2


def reviewed_population(review_path):
    review = export_module().check_review(review_path)
    expected = {(ci, arm) for ci in range(10) for arm in ARMS}
    rows = review.get('banks', [])
    if (review.get('schema') != REVIEW_SCHEMA or review.get('status') != REVIEW_STATUS
            or len(rows) != 20 or {(r.get('conversation'), r.get('arm')) for r in rows} != expected):
        raise ValueError('Exactly the current twenty reviewed complete fact banks are required')
    result = {}
    for row in rows:
        kind = row.get('source_kind')
        required_state = {'strict_complete': 'complete', 'reviewed_export': 'failed'}.get(kind)
        if (required_state is None or row.get('original_write_status') != required_state
                or type(row.get('groups')) is not int or row['groups'] <= 0
                or row['groups'] != row.get('expected_groups')
                or type(row.get('facts')) is not int or row['facts'] < 20
                or not isinstance(row.get('native_index_limitations'), list)
                or (kind == 'reviewed_export' and (not row['native_index_limitations'] or not row.get('proof')))):
            raise ValueError('Partial, unknown or unreviewed failure is not an admissible source')
        result[(row['conversation'], row['arm'])] = row
    return review, result


def assemble_rows(review_path, arm, export_bindings):
    """Construct per-arm source rows, without writing a gate or copying a store.

    export_bindings maps exactly the exceptional conversation IDs in this arm to
    sealed reviewed_export pointers. Semantic approval remains an explicit gate.
    """
    if arm not in ARMS:
        raise ValueError('Only the two current refinement arms are permitted')
    _, population = reviewed_population(review_path)
    expected_exports = {ci for ci in range(10) if population[ci, arm]['source_kind'] == 'reviewed_export'}
    if set(export_bindings) != expected_exports:
        raise ValueError('Every exceptional source needs exactly one explicit export binding')
    review_ref = pointer(review_path)
    rows = []
    for ci in range(10):
        original = population[ci, arm]
        export_ref = export_bindings.get(ci)
        bank_ref = original.get('bank')
        if export_ref is not None:
            loaded = export_module().check_export_bank(export_ref)
            if (loaded.get('uniform_row') != original
                    or loaded['export'].get('uniform_twenty_bank_review') != review_ref):
                raise ValueError('Export belongs to a different complete-bank review')
            bank_ref = loaded['review']['bank']
        rows.append({'conversation': ci, 'arm': arm, 'source_kind': original['source_kind'],
                     'review_row_sha256': digest(original), 'bank': bank_ref, 'reviewed_export': export_ref,
                     'audits': copy.deepcopy(original['audits']),
                     'native_index_limitations': copy.deepcopy(original['native_index_limitations'])})
    return rows


def admit_gate(path):
    gate = verify(path)
    if (gate.get('schema') != GATE_SCHEMA or gate.get('status') != 'passed' or gate.get('arm') not in ARMS
            or any(gate.get('checks', {}).get(k) is not True for k in CHECKS)
            or not isinstance(gate.get('limitations'), list) or not gate['limitations']):
        raise ValueError('Explicit uniform20 structural and source-quality review is required')
    quality = gate.get('content_quality_reviews')
    if not isinstance(quality, list) or not quality:
        raise ValueError('Source semantic findings must be separately bound')
    for ref in quality:
        checked_pointer(ref)
    uniform_ref = gate['uniform_twenty_bank_review']
    review_path = checked_pointer(uniform_ref)
    _, population = reviewed_population(review_path)
    arm = gate['arm']
    rows = gate.get('banks', [])
    if len(rows) != 10 or {(r.get('conversation'), r.get('arm')) for r in rows} != {(ci, arm) for ci in range(10)}:
        raise ValueError('Exactly ten distinct current-arm sources required')
    admitted, resources, parents = {}, [], {}
    for row in rows:
        ci = row['conversation']
        original = population[ci, arm]
        if (row.get('review_row_sha256') != digest(original)
                or row.get('source_kind') != original['source_kind']
                or row.get('audits') != original['audits']
                or row.get('native_index_limitations') != original['native_index_limitations']):
            raise ValueError('Gate row differs from the uniform reviewed source')
        for kind in ('source_date', 'index'):
            refs = row['audits'].get(kind)
            if not isinstance(refs, list) or not refs:
                raise ValueError('Per-bank source/date and index evidence required')
            for ref in refs:
                checked_pointer(ref)
        manifest_path = checked_pointer(original['source_manifest'])
        if manifest_path.name != 'run_manifest.json':
            raise ValueError('Original writer manifest required')
        origin = manifest_path.parent.resolve()
        if origin not in parents:
            parents[origin] = legacy.writer.manifest(origin)
        parent = parents[origin]
        if parent['arms'] != [arm]:
            raise ValueError('Writer arm differs from the reviewed source')
        folder = origin / f'conv_{ci}' / arm
        for field, expected in (('write_status', folder / 'write_status.json'),
                                ('outcome', folder / 'write_outcome.json'),
                                ('group_inputs', folder / 'group_inputs.json')):
            if checked_pointer(original[field]).resolve() != expected:
                raise ValueError('Uniform review points to another original bank lineage')
        status = read(original['write_status']['path'])
        outcome = verify(original['outcome']['path'])
        if (status.get('status') != original['original_write_status'] or not status.get('finished_utc')
                or outcome.get('status') != status['status'] or outcome.get('write_status') != original['write_status']
                or any(v.get('protocol_sha256') != parent['protocol_sha256'] for v in (status, outcome))
                or any((v.get('conversation'), v.get('arm')) != (ci, arm) for v in (status, outcome))):
            raise ValueError('Original terminal status or outcome changed')
        bank_path = checked_pointer(row['bank'])
        bank = verify(bank_path)
        native = Path(original['native_dir']).resolve()
        if native != folder / 'native':
            raise ValueError('Only the original unmodified native index may be copied')
        if row['source_kind'] == 'strict_complete':
            if (row.get('reviewed_export') is not None or row['bank'] != original.get('bank')
                    or bank_path.resolve() != folder / 'bank.json'
                    or status.get('bank') != row['bank'] or outcome.get('bank') != row['bank']):
                raise ValueError('Strict source may not silently change into an export')
        else:
            if original.get('bank') is not None or not row.get('reviewed_export'):
                raise ValueError('Original failed source requires an explicit reviewed export')
            loaded = export_module().check_export_bank(row['reviewed_export'])
            if (loaded.get('source_kind') != 'reviewed_export' or loaded.get('uniform_row') != original
                    or loaded['export'].get('uniform_twenty_bank_review') != uniform_ref
                    or loaded['review'].get('bank') != row['bank'] or loaded['bank'] != bank
                    or Path(loaded['native']).resolve() != native):
                raise ValueError('Metadata export does not match this exact failed source')
        if (bank.get('complete') is not True or bank.get('protocol_sha256') != parent['protocol_sha256']
                or (bank.get('conversation'), bank.get('arm')) != (ci, arm)):
            raise ValueError('Complete fact metadata bank identity changed')
        common._common_chain({**original, 'bank': row['bank']}, bank, status, native, parent)
        provenance = {**copy.deepcopy(row), 'native_inventory': copy.deepcopy(original['native_inventory']),
                      'original_review': copy.deepcopy(original), 'uniform_twenty_bank_review': uniform_ref,
                      'source_manifest': original['source_manifest']}
        admitted[ci, arm] = {'review': provenance, 'bank': bank, 'native': native}
        resources.append(parent['refinement']['resources'])
    if any(r != resources[0] for r in resources):
        raise ValueError('Compared native stores use different frozen local resources')
    legacy.old.validate_resources(resources[0]['snapshot'], resources[0]['files_sha256'])
    return gate, admitted, resources[0]
