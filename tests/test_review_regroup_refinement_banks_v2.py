"""Offline namespace, terminal, proof-binding and metadata export regressions."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from tools import review_regroup_refinement_banks_v2 as m


class ReviewRefinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); m.old.HASHES.clear()
        for p in (patch('socket.socket.connect', side_effect=AssertionError('Network prohibited')),
                  patch('jev_mem.providers.load_credentials', side_effect=AssertionError('Credentials prohibited'))):
            p.start(); self.addCleanup(p.stop)

    def save(self, p, x, sealed=True):
        p.parent.mkdir(parents=True, exist_ok=True)
        if sealed: return m.old.save(p, x)
        p.write_text(json.dumps(x), encoding='utf-8'); return m.pointer(p)

    def replace(self, p, x, sealed=True):
        m.old.HASHES.pop(str(p.resolve()), None)
        if sealed: x = {**x, 'content_sha256': m.digest(x)}
        p.write_text(json.dumps(x), encoding='utf-8')

    def population_fixture(self):
        runs = {a: self.root/a for a in m.ARMS}
        for arm, r in runs.items():
            base = {'arms':[arm], 'refinement':{'arm':arm}, 'inputs':[]}
            manifest = {**base, 'protocol_sha256':m.digest(base)}
            self.save(r/'run_manifest.json', manifest, False)
            for ci in range(10):
                d = r/f'conv_{ci}'/arm
                state = {'conversation':ci,'arm':arm,'status':'complete','finished_utc':'2026-09-30T00:00:00Z',
                         'protocol_sha256':manifest['protocol_sha256'],'completed_groups':1,'expected_groups':1,'facts':1}
                ref = self.save(d/'write_status.json',state,False)
                self.save(d/'write_outcome.json',dict(state,write_status=ref))
        return runs

    def mutate_state(self, runs, ci, arm, **changes):
        d=runs[arm]/f'conv_{ci}'/arm;p=d/'write_status.json';s=json.loads(p.read_text());s.update(changes)
        self.replace(p,s,False);self.replace(d/'write_outcome.json',dict(s,write_status=m.pointer(p)))

    def test_terminal_exact20_preflight_no_sqlite(self):
        runs=self.population_fixture()
        with patch.object(m.old,'points',side_effect=AssertionError('No SQLite preflight')):
            self.assertEqual(len(m.terminal(runs)),20)
            self.mutate_state(runs,2,m.ARMS[0],status='running')
            with self.assertRaisesRegex(ValueError,'terminal'):m.terminal(runs)

    def test_running_without_outcome_reports_terminal_gate(self):
        runs=self.population_fixture();self.mutate_state(runs,2,m.ARMS[0],status='running')
        (runs[m.ARMS[0]]/'conv_2'/m.ARMS[0]/'write_outcome.json').unlink()
        with patch.object(m.old,'points',side_effect=AssertionError('No SQLite preflight')):
            with self.assertRaisesRegex(ValueError,'All20 must be terminal'):m.terminal(runs)

    def test_partial_and_unknown_failures_block(self):
        for changes in ({'completed_groups':0},{'status':'failed','error':{'message':'other'}},
                        {'status':'failed','error':{'message':m.ERROR},'completed_groups':0}):
            with self.subTest(changes=changes):
                runs=self.population_fixture() if not (self.root/m.ARMS[0]).exists() else {a:self.root/a for a in m.ARMS}
                self.mutate_state(runs,3,m.ARMS[0],**dict({'status':'complete','error':None,'completed_groups':1},**changes))
                with self.assertRaises(ValueError):m.terminal(runs)

    def test_wrong_protocol_and_original_outcome_binding_block(self):
        runs=self.population_fixture();self.mutate_state(runs,0,m.ARMS[1],protocol_sha256='different')
        with self.assertRaisesRegex(ValueError,'identity'):m.terminal(runs)

    def test_population_duplicates_missing_extra_block(self):
        rows=[{'conversation':c,'arm':a} for c in range(10) for a in m.ARMS]
        m._population(rows)
        for altered in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):m._population(altered)

    def test_audit_running_or_partial_creates_no_output_or_evidence(self):
        runs=self.population_fixture();self.mutate_state(runs,8,m.ARMS[0],status='running')
        p=self.root/'review.json'
        with patch.object(m.strict,'audit',side_effect=AssertionError('No bank audit')):
            with self.assertRaises(ValueError):m.audit(runs,{},p)
        self.assertFalse(p.exists());self.assertFalse(p.with_suffix('.evidence').exists())

    def test_known_error_without_explicit_proof_blocks(self):
        runs=self.population_fixture();self.mutate_state(runs,3,m.ARMS[0],status='failed',error={'message':m.ERROR})
        with patch.object(m.strict,'audit',side_effect=AssertionError('No strict work before proof')):
            with self.assertRaisesRegex(ValueError,'proof'):m.audit(runs,{},self.root/'review.json')

    def test_unpinned_old_or_other_proof_rejected_before_sqlite(self):
        runs=self.population_fixture();self.mutate_state(runs,3,m.ARMS[0],status='failed',error={'message':m.ERROR})
        target=next(t for t in m.terminal(runs) if (t['conversation'],t['arm'])==(3,m.ARMS[0]))
        p=self.root/'old.json';self.save(p,{'schema':'old-report','cause_evidence':{'verified':True}})
        with patch.object(m.old,'points',side_effect=AssertionError('No native reads')):
            with self.assertRaisesRegex(ValueError,'pinned proof'):m._failure_binding(p,target)

    def test_source_pointer_drift_blocks(self):
        p=self.root/'source.json';ref=self.save(p,{'a':1});self.replace(p,{'a':2})
        with self.assertRaisesRegex(ValueError,'drift'):m._checked(ref)

    def failure_binding_fixture(self):
        runs=self.population_fixture();self.mutate_state(runs,3,m.ARMS[0],status='failed',error={'message':m.ERROR})
        target=next(t for t in m.terminal(runs) if (t['conversation'],t['arm'])==(3,m.ARMS[0]))
        f=target['folder'];p=self.root/'proof.json'
        self.save(p,{'schema':'context-c3-entity-failure-audit-v1',
            'status':'real_native_same_canonical_stale_payload_omission_confirmed',
            'conversation':3,'arm':m.ARMS[0],'protocol_sha256':target['manifest']['protocol_sha256'],
            'manifest':m.pointer(target['run']/'run_manifest.json'),
            'write_status':m.pointer(f/'write_status.json'),'write_outcome':m.pointer(f/'write_outcome.json'),
            'counts':{'groups':1,'facts':1},'qa_gate_passed':False,'recovery_or_export_performed':False,
            'checks':dict.fromkeys(['all36_original_group_source_date_effective_prompt_raw_add_chains',
                'facts_dense768_sparse_and_bm25_exact','existing_entity_dense_and_links_valid'],True),
            'source_file_hashes':{},'source_folder_inventory':m._folder_inventory(f)})
        key=(3,m.ARMS[0],target['manifest']['protocol_sha256'])
        return p,target,{key:m.pointer(p)['sha256']}

    def test_failure_binding_rejects_added_or_missing_source_files(self):
        p,target,pins=self.failure_binding_fixture();f=target['folder']
        with patch.object(m,'SUPPORTED_FAILURES',pins):
            m._failure_binding(p,target)
            # Native lock files are intentionally outside the immutable inventory.
            (f/'.lock').write_text('lock')
            m._failure_binding(p,target)
            extra=f/'unreviewed_raw.json';extra.write_text('{}')
            with self.assertRaisesRegex(ValueError,'folder inventory drift'):m._failure_binding(p,target)
            extra.unlink();(f/'write_outcome.json').unlink()
            with self.assertRaises((ValueError,FileNotFoundError)):m._failure_binding(p,target)

    def test_folder_inventory_rejects_symlinks_and_junctions(self):
        folder=self.root/'bank';folder.mkdir();p=folder/'extra.json';p.write_text('{}')
        with patch.object(Path,'is_symlink',lambda self:self.name=='extra.json'):
            with self.assertRaisesRegex(ValueError,'symlink or junction'):m._folder_inventory(folder)
        original_lstat = Path.lstat
        def fake_lstat(path):
            return SimpleNamespace(st_file_attributes=m.stat.FILE_ATTRIBUTE_REPARSE_POINT) if path==p else original_lstat(path)
        with patch.object(Path,'lstat',fake_lstat),patch.object(Path,'is_symlink',return_value=False):
            with self.assertRaisesRegex(ValueError,'symlink or junction'):m._folder_inventory(folder)

    def test_row_requires_both_nonempty_audit_types(self):
        runs=self.population_fixture();target=m.terminal(runs)[0];f=target['folder'];s=target['state']
        row={'conversation':target['conversation'],'arm':target['arm'],'original_write_status':'complete',
             'write_status':m.pointer(f/'write_status.json'),'outcome':m.pointer(f/'write_outcome.json'),
             'source_manifest':m.pointer(target['run']/'run_manifest.json'),
             'group_inputs':self.save(f/'group_inputs.json',{}),'groups':1,'expected_groups':1,'facts':1,
             'native_dir':str(f/'native'),'native_inventory':{}}
        for audits in ({},{'source_date':[]},{'source_date':[],'index':[]},
                       {'source_date':[row['outcome']],'index':[]},
                       {'source_date':[row['outcome']],'index':[row['outcome']],'other':[row['outcome']]}):
            with self.subTest(audits=audits),patch.object(m.old,'native_inventory',return_value={}):
                with self.assertRaisesRegex(ValueError,'[Aa]udit'):m._validate_row(dict(row,audits=audits),target)

    def test_proof_uses_same_code_with_private_bindings_and_no_old_global_changes(self):
        original, old_folder, old_ids = m.old.overwrite_proof, m.old.folder, m.old.ORPHANS
        originals = dict(original.__globals__)
        bound = m._isolated_overwrite_proof(self.root/'native_owner', 'new_arm', 'new_orphan')
        self.assertIs(bound.__code__, original.__code__)
        self.assertIsNot(bound.__globals__, original.__globals__)
        self.assertEqual(bound.__globals__['folder'](3,'new_arm'),self.root/'native_owner')
        self.assertEqual(bound.__globals__['ORPHANS'], {'new_arm':'new_orphan'})
        self.assertIs(m.old.folder,old_folder);self.assertIs(m.old.ORPHANS,old_ids)
        self.assertIs(m.old.overwrite_proof,original)
        self.assertEqual(set(original.__globals__),set(originals))
        self.assertTrue(all(original.__globals__[k] is v for k,v in originals.items()))

    def test_new_destination_required_before_review(self):
        with patch.object(m,'ROOT',self.root),patch.object(m,'check_review',side_effect=AssertionError('No review')):
            for dest in (self.root/'outside',self.root/'runs'/'existing'):
                if dest.name=='existing':dest.mkdir(parents=True)
                with self.assertRaises(ValueError):m.export(self.root/'review.json',dest)

    def test_sealed_partial_export_rejected_early(self):
        p=self.root/'reviewed_export.json';self.save(p,{'schema':m.EXPORT_SCHEMA,'source_kind':'reviewed_export',
            'original_write_status':'failed','reviewed_export_status':'partial','native_index_policy':'original_unrepaired'})
        with patch.object(m,'check_review',side_effect=AssertionError('No source use')):
            with self.assertRaisesRegex(ValueError,'Wrong reviewed export'):m.check_export_bank(p)

    def export_fixture(self):
        ci,arm=3,m.ARMS[0];src=self.root/'writer'/f'conv_{ci}'/arm;n=src/'native';n.mkdir(parents=True)
        fact={'id':'original-uuid','text':'Original fact','event':'ADD','input_scope':{'group_id':'g'}}
        vector=[1.0]+[0.0]*767
        journal={'status':'success','sequence':0,'group_id':'g','fact_records':[fact]}
        jp=n/'group_audit'/'owner'/'g.json';jref=self.save(jp,journal,False)
        mf=self.save(self.root/'writer'/'run_manifest.json',{'protocol_sha256':'p','dataset_sha256':'source'})
        inp=self.save(src/'group_inputs.json',{'envelopes':[{'group_id':'g'}]})
        sref=self.save(src/'write_status.json',{'status':'failed','error':{'message':m.ERROR}},False)
        oref=self.save(src/'write_outcome.json',{'status':'failed','write_status':sref})
        row={'conversation':ci,'arm':arm,'source_kind':'reviewed_export','original_write_status':'failed',
             'bank':None,'write_status':sref,'outcome':oref,'source_manifest':mf,'group_inputs':inp,
             'native_dir':str(n),'native_inventory':{'facts.sqlite':'same'},'groups':1,'expected_groups':1,'facts':1,
             'orphan_fact_ids':['original-uuid'],'facts_projection_sha256':m.digest([dict(fact,embedding=vector)]),
             'native_index_limitations':['original unrepaired orphan']}
        rp=self.root/'uniform.json';self.save(rp,{'fixture':True});review={'banks':[row]}
        return rp,review,row,n,vector,fact

    def test_metadata_export_positive_keeps_failed_and_uuid_and_source_bytes(self):
        rp,review,row,n,vector,fact=self.export_fixture();dest=self.root/'runs'/'new';dest.parent.mkdir()
        before={str(p):p.read_bytes() for p in (self.root/'writer').rglob('*') if p.is_file()}
        with patch.object(m,'ROOT',self.root),patch.object(m,'check_review',return_value=review),\
             patch.object(m.old,'points',return_value={'original-uuid':SimpleNamespace(vector={'':vector})}),\
             patch.object(m.old,'native_inventory',return_value=row['native_inventory']):
            ref=m.export(rp,dest);self.assertEqual(m.verify(ref['path'])['status'],'reviewed_metadata_exports_complete')
            loaded=m.check_export_bank(dest/'conv_3'/m.ARMS[0])
            self.assertEqual(loaded['uniform_row'],row)
            self.assertEqual(loaded['bank']['facts'],[dict(fact,embedding=vector)])
            self.assertEqual(loaded['export']['original_write_status'],'failed')
        self.assertFalse((n.parent/'bank.json').exists())
        self.assertEqual(before,{str(p):p.read_bytes() for p in (self.root/'writer').rglob('*') if p.is_file()})

    def test_export_projection_tamper_and_native_drift_rejected(self):
        rp,review,row,n,vector,fact=self.export_fixture();dest=self.root/'runs'/'new';dest.parent.mkdir()
        with patch.object(m,'ROOT',self.root),patch.object(m,'check_review',return_value=review),\
             patch.object(m.old,'points',return_value={'original-uuid':SimpleNamespace(vector={'':vector})}),\
             patch.object(m.old,'native_inventory',return_value=row['native_inventory']):
            m.export(rp,dest)
            row['facts_projection_sha256']='wrong'
            with self.assertRaisesRegex(ValueError,'row differs'):m.check_export_bank(dest/'conv_3'/m.ARMS[0])
            row['facts_projection_sha256']=m.digest([dict(fact,embedding=vector)])
            with patch.object(m.old,'native_inventory',return_value={'facts.sqlite':'changed'}):
                with self.assertRaisesRegex(ValueError,'inventory differs'):m.check_export_bank(dest/'conv_3'/m.ARMS[0])

    def test_check_review_validates20_before_sources(self):
        p=self.root/'r.json';self.save(p,{'schema':m.SCHEMA,'status':m.STATUS,'native_index_policy':'original_unrepaired',
            'review_tool':m.pointer(Path(m.__file__)),'checks':dict.fromkeys(['all20_terminal','all20_complete_fact_groups',
            'uniform_source_date_raw_history_required_vectors_reviewed','original_native_unchanged','no_unknown_failure_admitted'],True),
            'banks':[{'conversation':0,'arm':m.ARMS[0]}]})
        with patch.object(m,'terminal',side_effect=AssertionError('No terminal read')):
            with self.assertRaisesRegex(ValueError,'20'):m.check_review(p)


if __name__=='__main__':unittest.main()
