"""Separately reviewed four-work-root archive for the one-query result overlay.

The frozen archive engine retains logical paths and content-addressed bytes.
Only final artifact locations and the result reconstruction contract change.
No original controller/status or previously passed lifecycle source is edited.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import types

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from tools import archive_amem_continuation_v4 as old
from tools import seal_amem_retrieval_continuation_v1 as union
from tools import run_five_baselines_amem_retrieval_v1 as entry
from tools.baselines_common_v1 import read, sha

ROOT=old.ROOT
ROOTS=old.ROOTS
FINAL=union.FINAL
WORK_OWNER=union.WORK_OWNER
DEFAULT_PLAN=FINAL/'cleanup_plan_retrieval_v1.json'
FROZEN='tools/archive_amem_continuation_v4.py'
FROZEN_SHA='1fe96446b7712dc9b99236acaf04911516d2a2388360affe33aead337e342cc1'
TOOL_FILES=(FROZEN,'tools/archive_amem_retrieval_continuation_v1.py','tools/cleanup_amem_retrieval_continuation_v1.ps1',
            'tools/baseline_controller_guard_v2.py','tools/baseline_controller_guard_v3.py',
            'tools/seal_amem_retrieval_continuation_v1.py','tools/run_five_baselines_amem_retrieval_v1.py',
            'tools/audit_amem_retrieval_admission_v1.py','tools/baselines_amem_retrieval_v1.py',
            'tools/run_five_baselines_amem_native_v4.py','tools/baselines_amem_v4.py')


def _engine():
    data=(ROOT/FROZEN).read_bytes()
    if hashlib.sha256(data).hexdigest()!=FROZEN_SHA:raise ValueError('Frozen four-root archive source changed')
    source=data.decode('utf-8')
    replacements={
        "ROOTS['v4']/'results/summary.json'":"FINAL/'results/summary.json'",
        "ROOTS['v4']/'results/cleanup_receipt.json'":"FINAL/'results/cleanup_receipt.json'",
        "ROOTS['v4']/'results/predecessor_completion.json'":"FINAL/'results/predecessor_completion.json'",
        "ROOTS['v4']/'cleanup_plan_native_v4.json'":"FINAL/'cleanup_plan_retrieval_v1.json'",
        "return ROOTS['v4']/'results/continuation_archive'":"return FINAL/'results/continuation_archive'",
        "new=ROOTS['v4'];receipt_path=":"new=FINAL;receipt_path=",
        '    verify_native_seal()':'    verify_native_seal(result_review)',
        'amem-four-root-':'amem-four-root-retrieval-',
        'amem-native-v4-predecessor-completion-v1':'amem-retrieval-v1-predecessor-completion-v1',
        'from tools.baseline_controller_guard_v2 import assert_stopped':'from tools.baseline_controller_guard_v3 import assert_stopped',
    }
    for before,after in replacements.items():
        if before not in source:raise ValueError('Frozen lifecycle relocation boundary changed: '+before)
        source=source.replace(before,after)
    module=types.ModuleType('_jev_four_root_retrieval_archive_v1');module.__file__=str(ROOT/FROZEN)
    exec(compile(source,str(ROOT/FROZEN),'exec'),module.__dict__)
    module.FINAL=FINAL;module.TOOL_FILES=TOOL_FILES
    return module


engine=_engine()
_base_closure=engine.closure
SCHEMA=engine.SCHEMA
PLAN_SCHEMA=engine.PLAN_SCHEMA
Resolver=engine.Resolver
safe_path=engine.safe_path
file_inventory=engine.file_inventory
binding=engine.binding
verify_binding=engine.verify_binding
output_dir=engine.output_dir


def verify_result_review(path, *, live_inventory=True):
    review=read(safe_path(path));summary_path=FINAL/'results/summary.json';summary=read(summary_path)
    if (review.get('status')!='passed' or review.get('summary_sha256')!=sha(summary_path)
            or summary.get('schema')!=union.SCHEMA or summary.get('self_root')!=str(FINAL)
            or summary.get('parent_root')!=str(ROOTS['v4']) or summary.get('work_owner')!=str(WORK_OWNER)
            or summary.get('method')!='amem' or summary.get('audit_status')!='no_invalid_artifacts'
            or any(summary.get('effective',{}).get(k)!=1540 for k in ('slots','reader_valid','judge_valid'))):
        raise ValueError('Independent complete single-query-union result review required')
    state=read(FINAL/'status.json')
    if state.get('stage')!='complete_awaiting_result_review' or state.get('summary_sha256')!=sha(summary_path):
        raise ValueError('Exact sidecar completed status and summary required')
    expected=review.get('reviewed_run_inventory')
    if (Path(path).resolve().is_relative_to(ROOTS['v4']) or not isinstance(expected,dict) or not expected
            or any(not Path(p).is_absolute() or str(Path(p).resolve())!=p or not Path(p).is_relative_to(ROOTS['v4'])
                   or not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for p,h in expected.items())):
        raise ValueError('Result review must bind complete parent v4 plus sidecar inventory outside the run')
    if live_inventory and file_inventory(ROOTS['v4'])!=expected:
        raise ValueError('Entire reviewed original-plus-sidecar inventory changed')
    if live_inventory:
        verify_seed_review(review,view=union.LiveView())
    return review


def verify_seed_review(result_review, *, view):
    """Numeric search is independently verified before vectors are removed.

    Post-cleanup this verifies the retained proof and exact state/text bindings;
    it intentionally does not claim to recompute the discarded numeric search.
    """
    record=result_review.get('retrieval_seed_review')
    if not isinstance(record,dict) or Path(record.get('path','')).is_relative_to(ROOTS['v4']):
        raise ValueError('External independent actual-query seed review required')
    value=union.bound(view,record)
    m=view.read(FINAL/'run_manifest.json');proof=union.bound(view,m['proof'])
    expected={'schema':'amem-query-retrieval-seed-independent-review-v1','status':'passed',
              'query_id':union.QUERY_ID,'model_calls':0,'credential_reads':0,
              'query_embedding_calls':1,'corpus_embedding_calls':0,
              'query_embedding_recomputed':True,'source_inventory_before_after_equal':True,'full_result_equal':True}
    if any(value.get(k)!=v or (type(v) is bool and type(value.get(k)) is not bool) for k,v in expected.items()):
        raise ValueError('Actual numeric query verification is incomplete')
    if any(type(value.get(k)) is not int for k in ('model_calls','credential_reads','query_embedding_calls','corpus_embedding_calls')):
        raise ValueError('Numeric verification counts must be exact integers')
    if value.get('bank_state')!=proof['bank']['state'] or value.get('embeddings')!=proof['bank']['embeddings']:
        raise ValueError('Independent numeric verification used a different original bank')
    state=union.bound(view,value['bank_state'])
    key=entry.amem._v1._sha({'question':proof['missing_question']['question'],'memory_sha256':state['memory_sha256']})
    capture_path=WORK_OWNER/'results/c2/retrieval'/(union.QUERY_ID+'.json')
    native_path=WORK_OWNER/'native/c2/retrieval'/(key+'.json')
    if value.get('retrieval')!={'path':str(capture_path),'sha256':view.sha(capture_path)} or value.get('native_result')!={'path':str(native_path),'sha256':view.sha(native_path)}:
        raise ValueError('Independent numeric verification is not bound to the exact sidecar result')
    result=union.bound(view,value['native_result']);capture=union.bound(view,value['retrieval'])
    if (capture.get('status')!='complete' or capture.get('value')!=result
            or value.get('seed_indices')!=result['seed_indices'] or len(result['seed_indices'])!=10
            or any(type(i) is not int for i in result['seed_indices'])):
        raise ValueError('Independently verified ordered seeds or full native capture differs')
    return record


def closure(initial):
    return _base_closure([*initial,(str(FINAL/'run_manifest.json'),sha(FINAL/'run_manifest.json'))])


def _archived_contract(view):
    if {k:Path(v) for k,v in view.manifest['roots'].items()}!=ROOTS:
        raise ValueError('Exactly the original four work owners required')
    manifest=view.read(FINAL/'run_manifest.json');proof=union.bound(view,manifest['proof'])
    parent,questions,routes=union.contract(manifest,proof,view=view)
    # Preserve all original roots, including their failures; compare inventory
    # metadata even for the explicitly disposable numeric encoder bytes.
    records={r['path']:r['sha256'] for r in view.manifest['records']}
    excluded=[str(FINAL),str(WORK_OWNER)]
    if proof.get('source_inventory_exclusions')!=excluded:
        raise ValueError('Only exact declared sidecar subtrees may be excluded from parent inventory')
    original={p:h for p,h in records.items() if Path(p).is_relative_to(ROOTS['v4'])
              and not any(Path(p).is_relative_to(Path(x)) for x in excluded)}
    if original!=proof['source_inventory']:
        raise ValueError('Archived pre-sidecar original parent inventory changed')
    parent_proof=union.bound(view,parent['technical_continuation']['proof'])
    if set(parent_proof['history_inventories'])!={'v1','v2','v3'}:
        raise ValueError('All three original ancestors must remain bound')
    for name,inventory in parent_proof['history_inventories'].items():
        expected={str((ROOTS[name]/p).resolve()):h for p,h in inventory.items()}
        actual={p:h for p,h in records.items() if Path(p).is_relative_to(ROOTS[name])}
        if actual!=expected:raise ValueError('Archived historical ancestor differs: '+name)
    for p,h in manifest['upstream_sha256'].items():
        if view.sha(ROOT/p)!=h:raise ValueError('Archived native source differs')
    review=union.bound(view,manifest['independent_review'])
    if (review.get('status')!='passed' or review.get('scope')!='single_query_core_implementation_and_actual_admission'
            or review.get('proof')!=manifest['proof'] or review.get('code_sha256')!=manifest['code_sha256']
            or review.get('archive_cleanup_downstream_authorized') is not False):
        raise ValueError('Archived original core-only admission review differs')
    for p,h in review['evidence_sha256'].items():
        if view.sha(p)!=h:raise ValueError('Archived independent core evidence changed')
    return manifest,proof


def replay_all(view):
    manifest,proof=_archived_contract(view)
    review=union.bound(view,view.manifest['result_review'])
    seed_review=verify_seed_review(review,view=view)
    summary=view.read(FINAL/'results/summary.json')
    state=view.read(FINAL/'status.json')
    if state.get('stage')!='complete_awaiting_result_review' or state.get('summary_sha256')!=view.sha(FINAL/'results/summary.json'):
        raise ValueError('Archived sidecar final status differs')
    # New union core has been frozen before its only model dispatch. Its pure
    # reconstruction replays original successful QA plus the admitted sidecar
    # method raw/parser/native context and first/recovered QA exactly.
    values=union.reconstruct(manifest,proof,view=view,created_utc=summary['created_utc'])
    for name,value in values.items():
        if view.read(FINAL/'results'/name)!=value:raise ValueError('Archived union seal output differs: '+name)
    if (summary.get('audit_status')!='no_invalid_artifacts' or summary['effective']['reader_valid']!=1540
            or summary['effective']['judge_valid']!=1540):raise ValueError('Incomplete effective QA cannot complete cleanup')
    return {'status':'passed','model_calls':0,'vault_reads':0,'original_success_qa_replays':1539,
            'original_missing_firstpass_retained':union.QUERY_ID,'effective_slots':1540,
            'summary_sha256':view.sha(FINAL/'results/summary.json'),
            'raw_request_audit_sha256':view.sha(FINAL/'results/raw_request_audit.json'),
            'method_request_audit_sha256':view.sha(FINAL/'results/method_request_audit.json'),
            'retrieval_seed_review':seed_review,
            'pre_sidecar_parent_inventory_files':len(proof['source_inventory']),
            'method':'Frozen union core and exact entry method audit over retained logical paths; prior inventories preserved; numeric query recomputation verified before cleanup, formatter/evidence replayed after cleanup'}


def verify_native_seal(result_review):
    manifest,proof=entry.manifest(return_proof=True)
    view=engine.LiveResolver(manifest,read(FINAL/'results/summary.json'))
    view.manifest['result_review']=binding(result_review)
    # The old LiveResolver is accepted as a live physical view by the union
    # only when it is this explicit subclass; otherwise virtual paths provide
    # identical immutable logical reads from its complete file inventory.
    view.bytes=lambda p:Path(p).read_bytes()
    return replay_all(view)


engine.verify_result_review=verify_result_review
engine.closure=closure
engine.verify_native_seal=verify_native_seal
engine.replay_all=replay_all
archive=engine.archive
verify_cleanup=engine.verify_cleanup
complete=engine.complete
verify_completion=engine.verify_completion
main=engine.main


if __name__=='__main__':main()
