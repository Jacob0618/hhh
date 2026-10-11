"""Offline fresh-bank identity, one-capture selection and full-source admission tests."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from test_hybrid_native_capture_v1 import Memory, labels
from jev_mem.routing import TYPE_KEYS
from tools import prepare_gemini38_hybrid_v1 as h


def bank(memory=None, arm='session'):
    memory=memory or Memory()
    facts=[{**f,'user_id':'u'}for f in memory.facts]
    return {'schema':h.writer.SCHEMA if arm=='session' else h.writer.grouped.SCHEMA,
            'conversation':0,'arm':arm,'protocol_sha256':'p','complete':True,'facts':facts,
            'source_dataset_sha256':h.SOURCE_HASH,'user_id':'u'}


class FreshHybridTests(unittest.TestCase):
    def test_fact_tag_failure_is_terminal_retained_and_not_implicitly_replayed(self):
        calls=[]
        class FailedAPI:
            def __init__(self):self.events=[]
            def scoped(self,**ctx):return FailedAPI()
            def score_many(self,*args,**kwargs):
                calls.append('offline_failed_attempt');self.events.append({'mock':'retained'})
                raise RuntimeError('offline provider failure')
        with tempfile.TemporaryDirectory()as d:
            run=Path(d);folder=run/'retrieval/conv_0';folder.mkdir(parents=True)
            session=next(h.session_envelopes(h.load_data()[0]))['session_id']
            b=bank();b['facts']=b['facts'][:20]
            for f in b['facts']:f['input_scope']={'session_id':session}
            bp=run/'bank.json';h.save_new(bp,b);ref=h.pointer(bp)
            m={'protocol_sha256':'p','banks':[{'conversation':0,'arm':'session','bank':ref}]}
            with patch.object(h,'manifest',return_value=(m,{})):
                result=h.tag_worker(run,0,api_factory=lambda p,ci,m:FailedAPI())
                self.assertEqual(result['status'],'failed');self.assertEqual(result['failed'],20)
                self.assertEqual(len(calls),20)
                state=h.read(folder/'tag_status.json');self.assertTrue(state['terminal']);self.assertEqual(state['failed'],20)
                failed=h.verify(folder/'tags/id0.json');self.assertEqual(failed['status'],'failed')
                self.assertEqual(failed['api_events'],[{'mock':'retained'}]);self.assertTrue((folder/'tag_intents/id0.json').exists())
                with self.assertRaises(RuntimeError):h.tag_worker(run,0,api_factory=lambda p,ci,m:FailedAPI())
                self.assertEqual(len(calls),20)

    def test_all_forty_bank_gate_and_invalid_report_fail_closed(self):
        with tempfile.TemporaryDirectory()as d:
            run=Path(d);manifest_file=run/'run_manifest.json';manifest_file.write_text('{}')
            inputs=[];review_rows=[];audit_rows=[];semantic_rows=[]
            for ci in range(10):
                for arm in h.BANK_ARMS:
                    folder=run/f'conv_{ci}'/arm;folder.mkdir(parents=True)
                    ip=folder/'input.json';h.save_new(ip,{'envelopes':[{}]});inp=h.pointer(ip)
                    inputs.append({'conversation':ci,'arm':arm,'input':inp})
                    jp=folder/'journal.json';h.save_new(jp,{'model':'new3.8'});journal=h.pointer(jp)
                    b=bank(arm=arm);b.update(conversation=ci,group_inputs=inp,journal_files=[journal],native_dir=str(folder/'native'))
                    bp=folder/'bank.json';h.save_new(bp,b);bref=h.pointer(bp)
                    status={'conversation':ci,'arm':arm,'protocol_sha256':'p','status':'complete','bank':bref,
                            'completed_groups':1,'expected_groups':1,'facts':len(b['facts'])}
                    sp=folder/'write_status.json';sp.write_text(__import__('json').dumps(status));sref=h.pointer(sp)
                    op=folder/'write_outcome.json';h.save_new(op,{**status,'write_status':sref})
                    row={'conversation':ci,'arm':arm,'bank':bref,'write_status':sref,'outcome':h.pointer(op),
                         'native_dir':str(folder/'native'),'native_inventory':{},'exceptions':[],
                         'checks':{k:True for k in h.CHECKS}}
                    review_rows.append(row)
                    audit_rows.append({'conversation':ci,'arm':arm,'bank':bref,'native_inventory':{},'native_dir':str(folder/'native'),
                                       'strict_passed':True,'source_generation_persistence_passed':True,'orphan_fact_ids':[],
                                       'journals':[{'journal':journal}],'semantic_review_targets':[]})
                    semantic_rows.append({'conversation':ci,'arm':arm,'bank':bref,'notes':['source review'],
                                          'blocking_implementation_error':False,'reviewed_journals':[journal]})
            audit={'schema':'gemini38-terminal-source-index-audit-v1','strict_passed':True,'exceptions':[],
                   'protocol_sha256':'p','writer_manifest':h.pointer(manifest_file),'conversations':list(range(10)),
                   'banks':audit_rows,'source_file_hashes':{}}
            ap=run/'reviews/structural.json';h.save_new(ap,audit);aref=h.pointer(ap)
            sem={'schema':'gemini38-writer-source-semantic-review-v1','status':'reviewed_with_limitations',
                 'qa_gold_used':False,'protocol_sha256':'p','structural_audit':aref,'banks':semantic_rows}
            semp=run/'reviews/semantic.json';h.save_new(semp,sem)
            for row in review_rows:row.update(source_date_audit=aref,index_audit=aref)
            gate={'schema':h.GATE_SCHEMA,'status':'passed','protocol_sha256':'p','exceptions':[],
                  'writer_manifest':h.pointer(manifest_file),'semantic_review':h.pointer(semp),'banks':review_rows,
                  'source_date_audits':[aref],'index_audits':[aref]}
            gp=run/'all_bank_review.json';h.save_new(gp,gate)
            parent={'protocol_sha256':'p','inputs':inputs}
            with patch.object(h.writer,'manifest',return_value=parent),patch.object(h,'verify_new_config',return_value={}),patch.object(h,'inventory',return_value={}):
                self.assertEqual(len(h.admit_gate(run)[2]),40)
                # Preserve valid SHA/content seals while replacing the report:
                # caller-written gate booleans cannot certify another protocol.
                bad=deepcopy(audit);bad['protocol_sha256']='old'
                badp=run/'reviews/bad.json';h.save_new(badp,bad);badref=h.pointer(badp)
                changed=deepcopy(gate);changed['source_date_audits']=[badref]
                changed['banks'][0]['source_date_audit']=badref
                gatep=run/'bad_gate.json';h.save_new(gatep,changed)
                with self.assertRaises(ValueError):h.admit_gate(run,gatep)
                bad=deepcopy(sem);bad['banks'].pop()
                badp=run/'reviews/bad_semantic.json';h.save_new(badp,bad)
                changed=deepcopy(gate);changed['semantic_review']=h.pointer(badp)
                gatep=run/'bad_semantic_gate.json';h.save_new(gatep,changed)
                with self.assertRaises(ValueError):h.admit_gate(run,gatep)

    def test_new_native_one_capture_produces_n0_and_all_h_without_mutation(self):
        m=Memory();b=bank(m);before=deepcopy(b)
        result=h.capture_current(m,'test query',b,{'sha256':'a'*64},labels(b['facts']),
                                 {k:.5 for k in TYPE_KEYS},_scorer_module=m.module)
        self.assertEqual(set(result['selections']),{'baseline',*h.H_ARMS})
        self.assertEqual(result['selections']['baseline']['selected_ids'],[f'id{i}'for i in range(20)])
        self.assertEqual(m.vector_store.calls,[('dense',80,{'user_id':'u'}),('keyword',80,{'user_id':'u'})])
        self.assertEqual(b,before)
        for value in result['selections'].values():
            self.assertEqual(len(value['selected_ids']),20)
        self.assertIn('not independent historical replay',result['native_replay_scope'])

    def test_current_fact_missing_tags_and_wrong_store_text_rejected(self):
        m=Memory();b=bank(m);tags=labels(b['facts']);tags.pop('id80')
        with self.assertRaises(ValueError):
            h.capture_current(m,'test query',b,{'sha256':'a'*64},tags,{k:.5 for k in TYPE_KEYS},_scorer_module=m.module)
        m=Memory();b=bank(m);m.vector_store.dense[0].payload['data']='old bank text'
        with self.assertRaises(ValueError):
            h.capture_current(m,'test query',b,{'sha256':'a'*64},_scorer_module=m.module)

    def test_rehashed_native_rank_tampering_fails_local_scorer_replay(self):
        m=Memory();b=bank(m)
        cap=h.capture_current(m,'test query',b,{'sha256':'a'*64},_scorer_module=m.module)['capture']
        cap['eligible'][0]['score']+=.03
        cap['capture_sha256']=h.digest({k:v for k,v in cap.items()if k!='capture_sha256'})
        with self.assertRaises(ValueError):h.validate_native_capture(cap,h.fact_projection(b),'u',_scorer_module=m.module)

    def test_session_and_group_schema_explicit_and_old_protocol_rejected(self):
        for arm in h.BANK_ARMS:
            b=bank(arm=arm);self.assertTrue(h._fact_bank(b,0,arm,'p'))
            b['protocol_sha256']='historical'
            with self.assertRaises(ValueError):h._fact_bank(b,0,arm,'p')
        b=bank();b['schema']=h.writer.grouped.SCHEMA
        with self.assertRaises(ValueError):h._fact_bank(b,0,'session','p')

    def test_full_source_10780_exact_jobs_and_tampered_gold_rejected(self):
        # All source admission invariants remain active; only expensive verified
        # store replay is replaced by a deterministic fixture in this test.
        cohort=h.cohort()[0]
        jobs={}
        for ci in range(10):
            for arm in h.BANK_ARMS:
                methods=('baseline',*h.H_ARMS)if arm=='session'else(arm,)
                jobs[(ci,arm)]=[{'job_id':method+'/'+q['id'],'method':method,'reference':q['reference']}
                                  for q in cohort[str(ci)]for method in methods]
        expected=[j for ci in range(10)for arm in h.BANK_ARMS for j in jobs[(ci,arm)]]
        self.assertEqual(len(expected),10780)
        with tempfile.TemporaryDirectory()as d:
            run=Path(d);(run/'retrieval').mkdir();mp=run/'retrieval/run_manifest.json';mp.write_text('{}')
            for ci in range(10):
                h.save_new(run/'retrieval'/f'conv_{ci}'/'retrieval_outcome.json',{'conversation':ci,'status':'complete',
                           'protocol_sha256':'p','queries':4*len(cohort[str(ci)])})
            metric=h.verified_official_metric()[1]
            m={'protocol_sha256':'p','code_sha256':{},'official_metric_provenance':metric,'new_execution_config':h.pointer(h.NEW_CONFIG)}
            source={'schema':h.SOURCE_SCHEMA,'status':'sealed','arms':list(h.ARMS),'source_dataset_sha256':h.SOURCE_HASH,
                    'questions_per_arm':1540,'parent_run':str(run),'retrieval_manifest':h.pointer(mp),
                    'protocol_sha256':'p','code_sha256':{},'official_metric_provenance':metric,'jobs':expected,
                    'new_execution_config':h.pointer(h.NEW_CONFIG)}
            path=run/'source.json';h.save_new(path,source)
            with patch.object(h,'manifest',return_value=(m,{})),patch.object(h,'_selection_jobs',side_effect=lambda r,m,ci,a:jobs[(ci,a)]):
                self.assertEqual(len(h.validate_source(path)['jobs']),10780)
                changed=deepcopy(source);changed['jobs'][0]['reference']='changed gold'
                altered=run/'altered.json';h.save_new(altered,changed)
                with self.assertRaises(ValueError):h.validate_source(altered)
                missing=run/'retrieval/conv_9/retrieval_outcome.json';missing.unlink()
                with self.assertRaises(FileNotFoundError):h.validate_source(path)


if __name__=='__main__':unittest.main()
