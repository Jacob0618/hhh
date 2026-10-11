"""Immutable full1540 comparison contract. Offline; no model or credential access.

The user-selected anchor is the completed plain-text SDK20 baseline, not the
historical JSON experiment. New drivers must verify this contract before work.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_sdk20_modules import digest, sha, save_new, verify, now
from tools import run_sdk20_plain_qa_v1 as plain
from tools import run_regroup_refinement_qa_v2 as refinement
from tools import mem0_regroup_plain_qa_sources_v1 as source
from tools.summarize_pilot400 import verified_official_metric
from jev_mem.data import SOURCE_HASH

LOCK = ROOT / 'configs/sdk20_full1540_locked_v1.json'
BASELINE = ROOT / 'research/reports/sdk20_plain_qa_2026-09-30/final_summary.json'
SCHEMA = 'user-frozen-native-sdk20-plain-full1540-v1'
FINAL_VIEW = 'final_with_uniform_technical_recovery'
BASE_F1 = 54.6095082528163
BASE_JUDGE = 66.03896103896105


def pointer(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p)}


def full_cohort():
    rows, dates = source.cohort()
    return [dict(conversation=ci, query_id=q['id'], source_qa_index=q['source_qa_index'],
                 question=q['question'], reference=q['reference'], category=q['category'],
                 reference_date=dates[str(ci)])
            for ci in range(10) for q in rows[str(ci)]]


def contract():
    config = source.protocol_config()
    summary = verify(BASELINE)
    if summary['audit_status'] != 'no_invalid_artifacts' or summary['audit_errors']:
        raise ValueError('Baseline original raw-response audit is not complete')
    slots = [x for x in summary['slots_by_view'][FINAL_VIEW] if x['method'] == 'baseline_sdk20']
    questions = full_cohort()
    expected = {q['query_id']: q for q in questions}
    if len(slots) != 1540 or len({s['query_id'] for s in slots}) != 1540 or set(expected) != {s['query_id'] for s in slots}:
        raise ValueError('Canonical baseline no longer covers exactly1540 questions')
    for row in slots:
        q = expected[row['query_id']]
        if (row['conversation'], row['category']) != (q['conversation'], q['category']):
            raise ValueError('Baseline question identity mismatch')
        if any(row['scores'].get(k) is None for k in ('official_locomo_f1', 'judge_correct')):
            raise ValueError('The user-selected baseline must have1540 actual scores')
    f1 = 100 * sum(r['scores']['official_locomo_f1'] for r in slots) / 1540
    judge = 100 * sum(r['scores']['judge_correct'] for r in slots) / 1540
    if abs(f1-BASE_F1) > 1e-10 or abs(judge-BASE_JUDGE) > 1e-10:
        raise ValueError('User-selected54.610 /66.039 baseline changed')
    original = plain.read(ROOT / 'runs/sdk20_plain_qa_v1/run_manifest.json')
    for path, expected_hash in original['code_sha256'].items():
        if sha(ROOT / path) != expected_hash:
            raise ValueError('Original baseline implementation changed: '+path)
    # Inherit the already-run refinement SDK/runtime bindings; do not silently
    # declare a newly installed SDK equivalent to the old baseline.
    old = plain.read(ROOT / 'runs/refinement_qa_jev_refined_v2/run_manifest.json')
    for path, expected_hash in {**old['sdk_sha256'], **old['runtime_sha256']}.items():
        if sha(ROOT / path) != expected_hash:
            raise ValueError('Frozen native SDK/runtime changed: '+path)
    if importlib.metadata.version('mem0ai') != old['sdk_version']:
        raise ValueError('Frozen Mem0 SDK version changed')
    _, metric = verified_official_metric()
    if metric != original['official_metric_provenance']:
        raise ValueError('Baseline official LoCoMo scorer changed')
    code = tuple(dict.fromkeys((*plain.CODE, *refinement.CODE,
        'jev_mem/data.py', 'jev_mem/sdk20_type_adapter.py',
        'tools/freeze_sdk20_full1540_v1.py')))
    return {
        'schema': SCHEMA,
        'user_instruction': '2026-10-02: baseline54.610F1/66.039Judge, all1540, freeze configuration',
        'baseline': {'method': 'baseline_sdk20', 'summary': pointer(BASELINE),
                     'view': FINAL_VIEW, 'official_f1_percent': BASE_F1,
                     'judge_percent': BASE_JUDGE, 'reader_valid':1540, 'judge_valid':1540},
        'dataset': {'source':pointer(ROOT/'data/locomo10.json'), 'source_sha256':SOURCE_HASH,
                    'categories':[1,2,3,4], 'conversations':10, 'planned_per_arm':1540,
                    'questions':questions, 'question_identity_sha256':digest(questions),
                    'reference_date':'last original session observation date; no module-specific replacement'},
        'plain_text_configuration':pointer(source.CONFIG),
        'models':{**config['models'], 'writer':'gemini-2.5-flash',
                  'embedding':'gemini-embedding-001', 'embedding_dimensions':768, 'jev':'jev-1.13.0'},
        'generation':config['generation'], 'prompts':config['prompts'],
        'retrieval':{'base':'native Memory.search', 'top_k':20, 'rerank':False,
                     'semantic_threshold':0.1, 'native_internal_limit':80,
                     'input_order':'native SDK20 order, except declared fact-type soft-ranking module',
                     'reader_fact_fields':['id','text','observation_date'],
                     'equal_fact_count_is_not_equal_token_budget':True},
        'evaluation':{'protocol':config['evaluation_protocol'], 'main_view':'final_after_one_uniform_technical_recovery',
                      'final_priority':plain.POLICY['final_priority'],
                      'judge_compat_protocol':plain.POLICY['judge_compat_protocol'],
                      'reader_text':'unchanged nonempty original text; no JSON answer coercion',
                      'official_metric_provenance':metric,
                      'denominator':1540, 'missing':'explicit null plus valid_count/1540; never drop or fabricate scores',
                      'paired':'same question IDs, report actual common-valid count',
                      'firstpass':'retain separate original compatible and strict results'},
        'execution':{'conversation_workers':8, 'qa_threads_per_conversation':4,
                     'transport_retries':2, 'technical_recovery':plain.POLICY['recovery'],
                     'successful_reader':plain.POLICY['successful_reader'],
                     'valid_yes_and_no':plain.POLICY['valid_yes_and_no'],
                     'unknown':plain.POLICY['unknown']},
        'extension':{'arms':['source_local_context','jev_refined'], 'new_queries_per_arm':1140,
                     'inherited_queries_per_arm':400, 'inherited_cohort':pointer(ROOT/'configs/pilot400_manifest_v1.json'),
                     'selection':'full1540 minus fixed400 IDs, independent of answers/scores/success',
                     'reuse':'same arm, bank, selected facts, prompt/model/request and raw lineage only; no rewriting old records',
                     'no_new_writer_or_segmentation':True},
        'module_policy':'Only the declared module may change; shared Reader/Judge/retrieval budget/scoring/cohort never change silently',
        'historical_json_config':'configs/canonical_sdk20_protocol_v1.json remains historical; not the current evaluation anchor',
        'code_sha256':{p:sha(ROOT/p) for p in code},
        'sdk_version':old['sdk_version'], 'sdk_sha256':old['sdk_sha256'], 'runtime_sha256':old['runtime_sha256'],
    }


def freeze():
    if LOCK.exists():
        raise FileExistsError('Immutable lock already exists; use check, never overwrite')
    save_new(LOCK, {'created_utc':now(), **contract()})
    return verify_lock()


def verify_lock():
    value = verify(LOCK)
    comparable = {k:v for k,v in value.items() if k not in ('created_utc','content_sha256')}
    if comparable != contract():
        raise ValueError('Locked full1540 protocol drift; do not run with changed settings')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','check'])
    args=parser.parse_args()
    value=freeze() if args.action=='freeze' else verify_lock()
    print(json.dumps({'status':'verified','path':str(LOCK),'sha256':sha(LOCK),
                      'questions':1540,'baseline_f1':value['baseline']['official_f1_percent'],
                      'baseline_judge':value['baseline']['judge_percent'],'model_calls':0},ensure_ascii=False))


if __name__=='__main__':main()
