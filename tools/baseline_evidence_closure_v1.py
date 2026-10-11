"""Strict current-experiment closure with fixed prior numeric reference leaves.

No path is guessed from prose. Unknown scalar maps and missing evidence fail
closed. Four previously accepted numeric documents are retained whole under
their exact path, SHA, and schema; their former experiment is not re-executed.
"""
from pathlib import Path
import re

ANNOTATION_REPORT='research/baseline_execution_2026-10-08/heartbeat_20261008T2147/root_candidate_collect_v1.json'
ANNOTATION_REPORT_SHA='c449c19e767f8291338bd119485d6c313e347b41115f7d876382c5cbe336b951'
ANNOTATION_SOURCE='tools/audit_amem_retrieval_admission_v1.py'
ANNOTATION_SOURCE_SHA='948d149c48a027f5df609b1848c7bf7c57afcd4f339fbb797fafe8be29819e69'
HISTORICAL_CONFIG='configs/sdk20_full1540_locked_v1.json'
HISTORICAL_CONFIG_SHA='3a795ff19e32847f02a583b93f094b60a9cbaf247f51bb9d36f4cf12993dc75b'
HISTORICAL_TARGET='research/reports/sdk20_plain_qa_2026-09-30/final_summary.json'
HISTORICAL_TARGET_SHA='75f67aefe7cadfb2c514ff24232c1f4598fbe73e64c9f1c3e0a92926f3c491c1'
HISTORICAL_DECISION='research/baseline_execution_2026-10-08/heartbeat_20261008T2147/root_historical_annotation_classification_v1.json'
HISTORICAL_DECISION_SHA='42df1658b20101e7606d6fd813a80052a5734086cab0056bc14c24c6f9a797dc'
HISTORICAL_REVIEW='research/baseline_execution_2026-10-08/downstream_dispatch_retrieval_closure_v1/historical_annotation_independent_v1.json'
HISTORICAL_REVIEW_SHA='df92198cc546b0592a264518a04a8c7b8468007f5e3d319ad2ade477efae528a'
PRIOR_DECISION='research/baseline_execution_2026-10-08/heartbeat_20261008T2147/root_prior_numeric_reference_scope_v1.json'
PRIOR_DECISION_SHA='106682dbe72ec32ac00c1d6b298f7b9e0f31e6453747f2502db438e2b2264ad2'
PRIOR_REVIEW='research/baseline_execution_2026-10-08/downstream_dispatch_retrieval_closure_v1/prior_reference_numeric_boundary_independent_v1.json'
PRIOR_REVIEW_SHA='6e37fa575185371f6c0a24dd245937d8da90dd5aa0b44a0c6aff1b6b14e0824c'
PRIOR_DOCUMENTS={
    'research/baseline_execution_2026-10-08/heartbeat_20261008T2046/reference_mem0_readiness_addendum_v2.json':(
        '58a6913ae26b576368a23d24d07509e21f6ceaa211e952d0f7bfae0693d349db',
        'mem0-gemini25-reference-readiness-addendum-v2'),
    'research/gemini25_ablation_2026-10-07/heartbeat_20261007T1814/billing_resume_v1/live_v1/final_summary_v1.json':(
        'a73ac20650ac3f135ae07404ca4042a1ccb42fe1574ca9109c5edd63fb2d15d3',
        'gemini25-ablation-plain-qa-summary-v1'),
    'research/gemini25_ablation_2026-10-07/result_review_v1/canonical_comparison_v1.json':(
        'a8929d04989181866ee5af2bf3715992174130922e869f22423a8710b34fdc1a',
        'gemini25-ablation-canonical-comparison-v1'),
    'research/gemini25_ablation_2026-10-07/result_review_v1/independent_aggregation_v1.json':(
        '4390d9a836046a53baae990dd1473aa0fcbb0354f45b6f3f7a0a2a390c371389',
        'gemini25-ablation-independent-aggregation-v1'),
}
MAP_KEYS=('code_sha256','upstream_sha256','evidence_sha256','files_sha256')
PAIRS=(('path','sha256'),('review_path','review_sha256'),('raw_path','raw_sha256'),
       ('source_path','source_sha256'),('native_call_path','native_call_sha256'))


def _digest(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _historical_annotation(value,document,document_sha,pointer,root):
    return (document==root/HISTORICAL_CONFIG and document_sha==HISTORICAL_CONFIG_SHA
            and pointer=='$.baseline.summary' and isinstance(value.get('path'),str)
            and Path(value['path'])==root/HISTORICAL_TARGET and value.get('sha256')==HISTORICAL_TARGET_SHA)


def collect(initial, *, root, safe_path, read, sha, annotations=None):
    """Return canonical path->SHA; only read files via supplied frozen helpers."""
    root=Path(root).resolve();queue=list(initial);records={};safe_cache={}
    prior={root/name:value for name,value in PRIOR_DOCUMENTS.items()}
    def enqueue(path,value):
        if not isinstance(path,str) or not _digest(value):
            raise ValueError('Explicit evidence references require a path and lowercase SHA256')
        queue.append((path,value))
    def visit(value,document,document_sha,pointer='$'):
        if isinstance(value,list):
            for i,item in enumerate(value):visit(item,document,document_sha,f'{pointer}[{i}]')
        elif isinstance(value,dict):
            for pk,hk in PAIRS:
                if pk not in value or hk not in value:continue
                if pk=='path' and hk=='sha256' and _historical_annotation(value,document,document_sha,pointer,root):
                    if (root/HISTORICAL_TARGET).exists():
                        raise ValueError('Classified unavailable historical target reappeared; review required')
                    # Both immutable decisions and their deletion/source evidence
                    # are retained even when the caller does not request a journal.
                    enqueue(HISTORICAL_DECISION,HISTORICAL_DECISION_SHA)
                    enqueue(HISTORICAL_REVIEW,HISTORICAL_REVIEW_SHA)
                    if annotations is not None:
                        annotations.append({'document':str(document),'document_sha256':document_sha,
                            'json_pointer':pointer,'historical_target_path':value['path'],
                            'historical_target_sha256':value['sha256'],
                            'treatment':'known unavailable nonoperative historical annotation; deleted September scores are NOT replayable',
                            'classification':{'path':str(root/HISTORICAL_DECISION),'sha256':HISTORICAL_DECISION_SHA},
                            'independent_review':{'path':str(root/HISTORICAL_REVIEW),'sha256':HISTORICAL_REVIEW_SHA}})
                else:enqueue(value[pk],value[hk])
            for key in MAP_KEYS:
                if key not in value:continue
                hashes=value[key]
                if isinstance(hashes,dict):
                    for path,h in hashes.items():enqueue(path,h)
                    continue
                if (document==root/ANNOTATION_REPORT and document_sha==ANNOTATION_REPORT_SHA
                        and pointer=='$' and key=='code_sha256' and hashes==ANNOTATION_SOURCE_SHA
                        and value.get('scope')=='root_current_candidate_full_readonly_collect_not_production_prepare'):
                    enqueue(ANNOTATION_SOURCE,ANNOTATION_SOURCE_SHA)
                    if annotations is not None:
                        annotations.append({'document':str(document),'document_sha256':document_sha,
                            'json_pointer':'$.code_sha256','source_path':str(root/ANNOTATION_SOURCE),
                            'source_sha256':ANNOTATION_SOURCE_SHA,'treatment':'explicit source binding; scalar annotation retained unchanged'})
                else:
                    raise ValueError(f'Unknown non-map evidence field: {document} {pointer}.{key}')
            for key,item in value.items():visit(item,document,document_sha,pointer+'.'+str(key))
    while queue:
        name,expected=queue.pop()
        if not isinstance(name,str) or not _digest(expected):
            raise ValueError('Initial evidence references require a path and lowercase SHA256')
        # The key is the caller's original string, not a Path-normalized alias.
        # Every first spelling receives the unchanged full safety validation.
        if name not in safe_cache:safe_cache[name]=safe_path(root/name)
        p=safe_cache[name];canonical=str(p)
        if p in prior and expected!=prior[p][0]:
            raise ValueError('Prior numeric reference expected SHA differs: '+canonical)
        if canonical in records:
            if records[canonical]!=expected:raise ValueError('Conflicting bound evidence SHA')
            continue
        if sha(p)!=expected:raise ValueError('Closure SHA differs: '+canonical)
        records[canonical]=expected
        if p.suffix.lower()=='.json':
            value=read(p)
            if p in prior:
                if not isinstance(value,dict) or value.get('schema')!=prior[p][1]:
                    raise ValueError('Prior numeric reference schema differs: '+canonical)
                # Retain the whole fixed accepted bundle and both exact scope
                # authorities even if no optional journal was requested.
                for relative,(h,_) in PRIOR_DOCUMENTS.items():enqueue(relative,h)
                enqueue(PRIOR_DECISION,PRIOR_DECISION_SHA)
                enqueue(PRIOR_REVIEW,PRIOR_REVIEW_SHA)
                if annotations is not None:
                    annotations.append({'document':canonical,'document_sha256':expected,
                        'json_pointer':'$','schema':value['schema'],
                        'treatment':'fixed accepted prior numeric reference leaf; complete original bytes retained; no prior experiment raw/code replay claimed',
                        'classification':{'path':str(root/PRIOR_DECISION),'sha256':PRIOR_DECISION_SHA},
                        'independent_numeric_review':{'path':str(root/PRIOR_REVIEW),'sha256':PRIOR_REVIEW_SHA}})
            else:visit(value,p,expected)
    return records
