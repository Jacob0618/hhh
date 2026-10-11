"""Build then freeze explicit source-bound inputs for symmetric index recovery QA."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_mem0_sparse1540 as base
from tools import recover_mem0_sparse1540_index_v3 as recovery
from tools.run_pilot400_experiments_v2 import read
from jev_mem.pilot400 import fingerprint

def code_hashes():
    files=list(base.CODE_FILES)+['tools/recover_mem0_sparse1540_index_v2.py',
        'tools/recover_mem0_sparse1540_index_v3.py','tools/freeze_mem0_sparse1540_inputs_v2.py',
        'tools/run_mem0_sparse1540_recovered_qa_v2.py','tools/audit_mem0_sparse1540_inputs_v2.py',
        'tools/audit_mem0_sparse1540_retrieval.py']
    return {s:base.sha(ROOT/s) for s in files}

def draft():
    source=base.verify_manifest();recovery.validate_coverage()
    rm=read(recovery.OUT/'recovery_manifest.json');rows=[]
    for ci in range(10):
        for arm in base.ARMS:
            if ci==3:
                path=recovery.folder(arm);result=read(path/'repair_result.json')
                if read(path/'retrieve_status.json')['status']!='complete':raise ValueError('Repaired retrieval incomplete')
                bankpath=Path(result['bank']['path']);bank=read(bankpath)
                if base.sha(bankpath)!=result['bank']['sha256']:raise ValueError('Repaired bank changed')
                index_recovery=recovery.binding(path/'repair_result.json')
                retrieval_dir=path/'retrieval';native=path/'native'
            else:
                bank,bh=base.load_bank(ci,arm,source)
                bankpath=base.conv_path(ci)/('session_bank.json' if arm=='session_rank' else 'regroup_bank.json')
                retrieval_dir=base.conv_path(ci)/'retrieval'/arm
                if read(base.conv_path(ci)/'retrieve_status.json')['status']!='complete':raise ValueError('Native retrieval incomplete')
                index_recovery=None
                native=(base.BASE/f'conv_{ci}') if arm=='session_rank' else (base.conv_path(ci)/'native')
            file_hashes={}
            for q in base.questions(ci):
                p=retrieval_dir/(q['id']+'.json')
                base.validate_retrieval(read(p),q,arm,bank,base.sha(bankpath));file_hashes[q['id']]=base.sha(p)
            if {p.stem for p in retrieval_dir.glob('*.json')}!=set(file_hashes):raise ValueError('Extra/missing retrieval')
            rows.append({'conversation':ci,'arm':arm,'bank_path':str(bankpath),'bank_sha256':base.sha(bankpath),
                'retrieval_dir':str(retrieval_dir),'retrieval_files':file_hashes,'index_recovery':index_recovery,
                'native_ledger_path':str(native/'api_calls.jsonl'),'native_cache_dir':str(native/'api_cache')})
    out={'policy':recovery.POLICY,'schema':'mem0-sparse1540-explicit-inputs-v2','created_utc':base.now(),
        'source_manifest':recovery.binding(base.OUT/'manifest.json'),
        'recovery_policy':recovery.binding(recovery.POLICY_PATH),'recovery_manifest':recovery.binding(recovery.OUT/'recovery_manifest.json'),
        'coverage':recovery.binding(recovery.COVERAGE),'rows':rows,'models':source['models'],
        'code_sha256':code_hashes(),
        'question_ids':[q['id'] for q in base.questions()],'provider_retries':2,'conversation_workers':8,
        'query_workers':4,'final_unique_facts':20,'model_calls_to_build_manifest':0}
    base.save_once(recovery.OUT/'input_draft.json',out)
    return {'draft':str(recovery.OUT/'input_draft.json'),'rows':20,'questions_per_arm':1540}

def freeze(audit_path):
    d=read(recovery.OUT/'input_draft.json');audit=read(audit_path)
    expected={'policy':recovery.POLICY,'admitted':True,
        'source_manifest_sha256':d['source_manifest']['sha256'],'recovery_policy_sha256':d['recovery_policy']['sha256'],
        'rows_sha256':fingerprint(d['rows']),'question_ids_sha256':fingerprint(d['question_ids']),'model_calls':0}
    if any(audit.get(k)!=v for k,v in expected.items()):raise ValueError('Independent input audit is not admitted or is unbound')
    if d['code_sha256']!=code_hashes():raise ValueError('Reviewed code changed after input draft')
    d['input_audit']=recovery.binding(audit_path);d['frozen_utc']=base.now()
    for row in d['rows']:
        if base.sha(row['bank_path'])!=row['bank_sha256']:raise ValueError('Bank changed after independent audit')
        for q,h in row['retrieval_files'].items():
            if base.sha(Path(row['retrieval_dir'])/(q+'.json'))!=h:raise ValueError('Retrieval changed after independent audit')
    base.save_once(recovery.OUT/'input_manifest.json',d)
    return {'manifest':str(recovery.OUT/'input_manifest.json'),'sha256':base.sha(recovery.OUT/'input_manifest.json')}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('draft','freeze'));p.add_argument('--audit',type=Path)
    args=p.parse_args()
    if args.stage=='freeze' and args.audit is None:p.error('freeze requires --audit')
    import json
    print(json.dumps(draft() if args.stage=='draft' else freeze(args.audit),indent=2))
