"""Offline rejection and frozen-body tests for the new C uniform20 policy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import review_boundary_variants_banks_v2 as r
from tools import boundary_variants_qa_sources_v2 as s
from tools import run_boundary_variants_qa_v2 as q
from tools import run_boundary_variants_qa_v1 as q1

class ReviewTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        for name in ('socket.create_connection','requests.Session.request','jev_mem.providers.load_credentials'):
            p=patch(name,side_effect=AssertionError('No network/credentials'));p.start();self.addCleanup(p.stop)

    def save(self,path,value):r.old.save(path,value);return path
    def raw(self,path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
    def fixtures(self):
        parent=self.root/'parent';self.save(parent/'manifest.json',{'parent':True});self.save(parent/'global.json',{'global':True})
        runs={a:parent/a/'writer' for a in r.ARMS};ms={};targets=[]
        for arm,run in runs.items():
            m={'protocol_sha256':'protocol-'+arm,'arms':[arm],'boundary_variant':{'arm':arm,
                'parent':r.old.pointer(parent/'manifest.json'),'global':r.old.pointer(parent/'global.json')}}
            self.raw(run/'run_manifest.json',m);ms[str(run)]=m
            for ci in range(10):
                folder=run/f'conv_{ci}'/arm;bad=(ci,arm)==(3,r.ARMS[0]);state={'conversation':ci,'arm':arm,
                    'protocol_sha256':m['protocol_sha256'],'status':'failed' if bad else 'complete',
                    'expected_groups':2,'completed_groups':2,'facts':20,'finished_utc':'2026-10-03T01:00:00+00:00'}
                if bad:state['error']={'type':'ValueError','message':r.ERROR}
                else:state['bank']=r.old.pointer(self.save(folder/'bank.json',{'complete':True}))
                self.raw(folder/'write_status.json',state)
                targets.append({'conversation':ci,'arm':arm,'folder':folder,'run':run,'state':state,'manifest':m,'outcome_value':state})
        controller=parent/'parallel_controller_v1/write_expansion';cm={'schema':r.parallel.VERSION,
            'policy':{'stage':'write','arms':list(r.ARMS),'conversations':list(range(10))},
            'parent':r.boundary.pointer(parent/'manifest.json'),
            'code_sha256':{p:r.old.sha(r.ROOT/p) for p in r.parallel.FILES}}
        mp=r.old.pointer(self.save(controller/'manifest.json',cm));outcomes={}
        for ci in range(10):
            terms=[]
            for arm in r.ARMS:
                t=next(t for t in targets if (t['conversation'],t['arm'])==(ci,arm));job=controller/'conversations'/f'conv_{ci}'/arm
                status=t['state']['status'];term={'schema':r.parallel.VERSION,'controller_manifest':mp,'stage':'write',
                    'conversation':ci,'arm':arm,'status':status,'finished_utc':'2026-10-03T01:00:01+00:00',
                    'outputs':{} if status=='failed' else {'write_status':r.old.pointer(t['folder']/'write_status.json'),'bank':r.old.pointer(t['folder']/'bank.json')}}
                tp=r.old.pointer(self.save(job/'terminal.json',term));terms.append(term)
                self.save(job/'intent.json',{'controller_manifest':mp,'nonce':'nonce','stage':'write','conversation':ci,'arm':arm})
                self.raw(job/'worker_state.json',{'status':status,'terminal':tp})
            outcomes[str(ci)]={'conversation':ci,'finished_utc':'2026-10-03T01:00:01+00:00','arms':terms,
                'status':'failed' if ci==3 else 'complete'}
        state={'status':'failed','finished_utc':'2026-10-03T01:00:02+00:00','controller_manifest':mp,'stage':'write',
            'requested_conversations':list(range(10)),'arms':list(r.ARMS),'outcomes':outcomes,'pid':42,
            'parent_process_identity':{'pid':42,'creation_ticks':1},'nonce':'nonce'}
        self.raw(controller/'parent_state.json',state)
        return runs,ms,targets,controller,state

    def test_exact_closed20_failed_parent_not_relabelled(self):
        runs,ms,ts,c,st=self.fixtures()
        with patch.object(r.cstrict.writer,'manifest',side_effect=lambda p:ms[str(p)]),patch.object(r.parallel,'process_identity',side_effect=RuntimeError):
            got=r.terminal(runs)
        self.assertEqual(len(got),20);self.assertEqual(sum(t['state']['status']=='failed' for t in got),1)
        self.assertEqual(json.loads((c/'parent_state.json').read_text())['status'],'failed')

    def test_partial_or_active_whole_batch_refused_before_sqlite(self):
        runs,ms,ts,c,st=self.fixtures();t=ts[-1]
        for changes in ({'status':'running'},{'completed_groups':1},{'error':{'type':'ValueError','message':'other'},'status':'failed'}):
            state={**t['state'],**changes};self.raw(t['folder']/'write_status.json',state)
            with patch.object(r.cstrict.writer,'manifest',side_effect=lambda p:ms[str(p)]),patch.object(r.old,'points',side_effect=AssertionError('SQLite too early')):
                with self.assertRaises(ValueError):r.terminal(runs)

    def test_controller_running_interrupted_unknown_or_missing_arm_rejected(self):
        runs,ms,ts,c,st=self.fixtures()
        for state in [{**st,'status':'running'},{**st,'status':'interrupted'},{**st,'outcomes':{}},
                      {**st,'outcomes':{**st['outcomes'],'3':{**st['outcomes']['3'],'arms':st['outcomes']['3']['arms'][:1]}}}]:
            self.raw(c/'parent_state.json',state)
            with patch.object(r.parallel,'process_identity',side_effect=RuntimeError):
                with self.assertRaises(ValueError):r.closed_controllers(runs,ts)

    def test_sealed_worker_source_drift_or_live_parent_rejected(self):
        runs,ms,ts,c,st=self.fixtures()
        with patch.object(r.parallel,'process_identity',return_value=st['parent_process_identity']):
            with self.assertRaisesRegex(ValueError,'live'):r.closed_controllers(runs,ts)
        job=c/'conversations/conv_3'/r.ARMS[0];worker=json.loads((job/'worker_state.json').read_text());worker['status']='complete';self.raw(job/'worker_state.json',worker)
        with patch.object(r.parallel,'process_identity',side_effect=RuntimeError):
            with self.assertRaisesRegex(ValueError,'Per-arm'):r.closed_controllers(runs,ts)

    def test_unknown_failure_proof_rejected_without_evidence_or_sqlite(self):
        runs,ms,ts,c,st=self.fixtures();proof=self.save(self.root/'foreign.json',{'schema':'context-c3-entity-failure-audit-v1'})
        with patch.object(r,'terminal',return_value=ts),patch.object(r.old,'points',side_effect=AssertionError('No SQLite')):
            with self.assertRaisesRegex(ValueError,'pinned'):r._audit_body(runs,{(3,r.ARMS[0]):proof},self.root/'review.json')
        self.assertFalse((self.root/'review.evidence').exists())

    def test_population_exact20_and_original_failed_mapping(self):
        rows=[{'conversation':ci,'arm':a,'source_kind':'strict_complete','original_write_status':'complete',
            'groups':2,'expected_groups':2,'facts':20,'native_index_limitations':['scope'],'proof':None} for ci in range(10) for a in r.ARMS]
        with patch.object(s.review,'check_review',return_value={'banks':rows}):self.assertEqual(len(s.population('fixture')[1]),20)
        for changed in [rows[:-1],[*rows[:-1],rows[0]],copy.deepcopy(rows)]:
            if len(changed)==20 and len({(x['conversation'],x['arm']) for x in changed})==20:
                changed[0].update(source_kind='reviewed_export',original_write_status='complete',proof={'path':'proof'})
            with patch.object(s.review,'check_review',return_value={'banks':changed}):
                with self.assertRaises(ValueError):s.population('fixture')

    def test_export_aggregate_must_cover_all20(self):
        rows=[{'conversation':ci,'arm':a} for ci in range(10) for a in r.ARMS]
        value={'schema':r.EXPORT_SCHEMA,'status':'reviewed_metadata_exports_complete','source_review':{'path':'review'},
            'source_mutations':0,'native_bytes_copied':0,'model_calls':0,'exports':rows}
        with patch.object(s,'verify',return_value=value):self.assertEqual(len(s.export_population('fixture',value['source_review'])),20)
        for edits in ({'exports':rows[:-1]},{'native_bytes_copied':1},{'source_review':{}},{'model_calls':1}):
            with patch.object(s,'verify',return_value={**value,**edits}):
                with self.assertRaises(ValueError):s.export_population('fixture',value['source_review'])

    def test_frozen_qa_bodies_and_full1540_scope(self):
        self.assertIs(q.plain.one,q1.plain.one);self.assertIs(q.sources,s);self.assertEqual(q.POLICY['slots_per_arm'],1540)
        self.assertEqual(q.POLICY['selection'],q1.POLICY['selection']);self.assertIs(q._manifest_core.__code__,q1._manifest_core.__code__)
        for name in ('qa','retrieve_worker','seal_firstpass','prepare_recovery','_qa_core','_prepare_core'):
            self.assertIs(getattr(q,name).__code__,getattr(q1,name).__code__)
        self.assertIs(s.cohort,s.legacy.cohort);self.assertIs(s.copy_bank,s.legacy.copy_bank)

    def test_v1_gate_not_silently_widened(self):
        with patch.object(s,'verify',return_value={'schema':s.legacy.GATE_SCHEMA,'status':'passed','source_arm':r.ARMS[0],'arm':s.ARMS[0]}):
            with self.assertRaisesRegex(ValueError,'v2'):s.admit_gate('fixture')
        with patch.object(s.legacy,'verify',return_value={'schema':s.GATE_SCHEMA,'status':'passed'}):
            with self.assertRaises(ValueError):s.legacy.admit_gate('fixture')

    def test_two_current_pins_do_not_transfer_to_another_arm_or_protocol(self):
        self.assertEqual(set(r.SUPPORTED_FAILURES),set(r.SUPPORTED_ORPHANS))
        self.assertEqual(len(r.SUPPORTED_FAILURES),2)
        for key,h in r.SUPPORTED_FAILURES.items():
            other=r.ARMS[1] if key[1]==r.ARMS[0] else r.ARMS[0]
            target={'conversation':key[0],'arm':other,'manifest':{'protocol_sha256':key[2]},'state':{}}
            with patch.object(r.old,'pointer',return_value={'path':'moved','sha256':h}),patch.object(r,'verify') as v:
                with self.assertRaisesRegex(ValueError,'pinned'):r._failure_binding('moved',target)
                v.assert_not_called()

    def test_both_pinned_proof_inspection_shapes_and_incomplete_refusal(self):
        for arm in r.ARMS:
            p=r.verify(r.ROOT/f'research/planned_execution_2026-10-03/checks/20261003T0658/{"c1" if arm==r.ARMS[0] else "c2"}_conv3_failure_review.json')
            self.assertTrue(r._proof_inspection(p)['writer_prompt_reconstructed_and_raw_verified'])
            bad=copy.deepcopy(p);field='source_date_raw_inspection' if 'source_date_raw_inspection' in p else 'strict_body_inspection'
            bad[field]['source_generation_persistence_passed']=False
            with self.assertRaisesRegex(ValueError,'strict-body'):r._proof_inspection(bad)

if __name__=='__main__':unittest.main()
