"""Strict R11 primary table from actual exact-request QA producers; no model calls."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.run_sdk20_judge_retest_v2 import verify,bound,pointer,sha
from tools.run_sdk20_modules import save_new,digest
from tools import run_source_set_qa_r11 as runner
from tools import summarize_source_set_r11 as strict_summary


def project(summary_path,output):
    actual=verify(summary_path)
    if actual.get('audit_status')!='no_invalid_artifacts' or actual.get('rounding_sensitivity') is not True:
        raise ValueError('Complete raw-audited rounding summary required')
    for filename,hashed in actual['input_sha256'].items():
        if not Path(filename).is_file() or sha(filename)!=hashed:raise ValueError('Audited producer input changed')
    effective=bound(actual['source_manifest']);original=bound(effective['strict_source'])
    oldjobs={j['job_id']:j for j in original['jobs']};newjobs={j['job_id']:j for j in effective['jobs']}
    if len(oldjobs)!=4620 or set(oldjobs)!=set(newjobs):raise ValueError('Full same-cohort manifests required')
    out=deepcopy(actual);views={};proofs={}
    for view,rows in actual['slots_by_view'].items():
        indexed={r['job_id']:r for r in rows};derived=[];proofs[view]={}
        for jid,j in oldjobs.items():
            candidate=jid
            if runner.request_signature(j,'reader')!=runner.request_signature(newjobs[jid],'reader'):
                candidate='baseline/'+j['query_id']
                if j['method']!='p':raise ValueError('Only P rounding input can differ')
            srcjob=newjobs[candidate];row=indexed[candidate]
            if any(j[k]!=srcjob[k] for k in ('conversation','query_id','question','reference','reference_date','category')):
                raise ValueError('Projection query identity changed')
            rsig=runner.request_signature(j,'reader')
            if rsig!=runner.request_signature(srcjob,'reader'): raise ValueError('Cannot inherit a different full Reader request')
            answer=row.get('reader_answer')
            jsig=runner.request_signature(j,'judge',answer or '')
            if jsig!=runner.request_signature(srcjob,'judge',answer or ''):raise ValueError('Judge request identity differs')
            d=deepcopy(row)
            d.update(job_id=jid,method=j['method'],selection=j['selection'],selected_ids=j['selected_ids'],
                strict_exact_projection={'producer_job_id':candidate,'producer_summary':pointer(summary_path),
                    'producer_reader_artifact':row.get('reader_artifact'),'producer_judge_artifact':row.get('judge_artifact'),
                    'reader_signature_sha256':digest(rsig),'judge_signature_sha256':digest(jsig),'new_model_calls':0})
            derived.append(d);proofs[view][jid]=d['strict_exact_projection']
        views[view]=derived
    out.update(schema='source-set-r11-strict-exact-qa-projection-v1',source_manifest=effective['strict_source'],
        rounding_sensitivity=False,strict_primary=True,actual_execution_summary=pointer(summary_path),
        projection_explanation='Each strict input matches either the actual effective same-arm request or actual unchanged N0 request. Scores remain attached to real producer artifacts; this is not a second model run.',
        projection_code_sha256={str(Path(__file__).resolve()):sha(__file__)},
        execution_accounting='All inherited QA physical usage is the actual shared effective/compatible execution, not a separate strict cold run. Projection adds no model calls.',
        projection_proofs=proofs,slots_by_view=views,selector_manifest=original['selector_manifest'],
        selection_counts=original['selection_counts'],
        source_append_contract_audit=strict_summary.augmentation_audit(original,bound(original['source_set_protocol'])),
        selector_lineage={j['job_id']:{'lineage':j.get('lineage'),'selection':j['selection']} for j in original['jobs']},
        views={k:strict_summary.aggregate(rows,bootstrap_samples=2000) for k,rows in views.items()},
        stopping_by_view={k:strict_summary.stopping_result(rows) for k,rows in views.items()})
    out.pop('content_sha256',None)
    out['input_sha256'].update({str(Path(summary_path).resolve()):sha(summary_path),effective['strict_source']['path']:effective['strict_source']['sha256']})
    save_new(output,out)
    return {'output':str(output),'projected_slots':4620,'model_calls':0}
