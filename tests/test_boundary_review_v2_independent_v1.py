"""Independent fail-closed source checks; fixtures only, no export/native/API."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from tools import review_boundary_variants_banks_v2 as r
from tools import boundary_variants_qa_sources_v2 as sources
from tools import run_boundary_variants_qa_v2 as qa
from tools import run_boundary_variants_qa_v1 as qa1


class BoundaryReviewCounterexamples(unittest.TestCase):
    def test_review_save_adapter_does_not_mutate_the_original_module(self):
        self.assertIsNot(r.old,r.base.old)
        self.assertIs(r.base.old.save,r._original_save)
        self.assertIs(r.old.save,r._save)
        self.assertIsNot(r.old.save,r.base.old.save)

    def targets(self):
        output=[]
        for ci in range(10):
            for arm in r.ARMS:
                folder=Path('offline')/arm/f'conv_{ci}'/arm
                output.append({'conversation':ci,'arm':arm,'folder':folder,'run':folder.parents[1],
                    'manifest':{'protocol_sha256':'test-protocol'},
                    'state':{'status':'complete','facts':24,'completed_groups':3,'expected_groups':3}})
        return output

    def test_exact_twenty_population_not_nineteen_or_duplicate(self):
        rows=self.targets();r._population(rows)
        for bad in (rows[:-1],rows[:-1]+[rows[0]],rows+[rows[0]]):
            with self.assertRaisesRegex(ValueError,'current20'):r._population(bad)

    def test_only_current_proof_identity_is_pinned_not_old_c3(self):
        target=self.targets()[6]  # c3/c1
        target['state'].update(status='failed')
        target['manifest']['protocol_sha256']='607a8ed53abe1a31f5a2a1dbc8b462f76ca983db36d51553f100a199e07f33f5'
        with patch.object(r.old,'pointer',return_value={'path':'historical','sha256':'eda4bc6c807d40ef2a4ce09a3074b1ce51e14d006b9626db43b0a729d130c87c'}),\
             patch.object(r,'verify') as verify,patch.object(r.old,'native_inventory') as inventory:
            with self.assertRaisesRegex(ValueError,'Current independently pinned'):r._failure_binding('old-proof',target)
            verify.assert_not_called();inventory.assert_not_called()
        self.assertNotIn((3,'source_local_context',target['manifest']['protocol_sha256']),r.SUPPORTED_FAILURES)

    def test_unknown_failure_class_cannot_be_metadata_exception(self):
        for status,typ,message in [('running','ValueError',r.ERROR),('failed','RuntimeError',r.ERROR),
                                   ('failed','ValueError','parse failure'),('complete','ValueError',r.ERROR)]:
            with self.assertRaisesRegex(ValueError,'Unknown failure'):
                r._failed_state({'status':status,'error':{'type':typ,'message':message}})

    def test_same_error_in_other_arm_still_needs_its_own_current_identity_proof(self):
        target=self.targets()[7]  # c3/c2, same ValueError does not transfer c1 proof
        target['state'].update(status='failed',error={'type':'ValueError','message':r.ERROR})
        target['manifest']['protocol_sha256']=next(iter(r.SUPPORTED_FAILURES))[2]
        c1_sha=next(iter(r.SUPPORTED_FAILURES.values()))
        r._failed_state(target['state'])  # error class is eligible for review only
        with patch.object(r.old,'pointer',return_value={'path':'c1-proof','sha256':c1_sha}),\
             patch.object(r,'verify') as verify,patch.object(r.old,'native_inventory') as inventory:
            with self.assertRaisesRegex(ValueError,'Current independently pinned'):
                r._failure_binding('c1-proof',target)
            verify.assert_not_called();inventory.assert_not_called()

    def test_current_c2_key_refuses_c1_sha_wrong_protocol_and_unpinned_hash(self):
        target=self.targets()[7];target['state'].update(status='failed',error={'type':'ValueError','message':r.ERROR})
        c2_protocol='f27eb8e8f8c079125ed264787f53916da206886d8928929ed231ebdcc16095d8'
        c1_sha=r.SUPPORTED_FAILURES[next(k for k in r.SUPPORTED_FAILURES if k[1]=='c1_multiscale')]
        for protocol,sha in [(c2_protocol,c1_sha),('wrong-protocol','1ceaeb59bc2fcd89b2d1445d7da24653c97b7b7bdc4203a72e467828599ccbe8'),
                            (c2_protocol,'unreviewed-proof-sha')]:
            target['manifest']['protocol_sha256']=protocol
            with patch.object(r.old,'pointer',return_value={'path':'candidate-proof','sha256':sha}),\
                 patch.object(r,'verify') as verify,patch.object(r.old,'native_inventory') as native:
                with self.assertRaisesRegex(ValueError,'Current independently pinned'):
                    r._failure_binding('candidate-proof',target)
                verify.assert_not_called();native.assert_not_called()

    def test_all_statuses_terminal_and_complete_prefix_before_controller_or_sqlite(self):
        runs={arm:Path('offline')/arm/'writer' for arm in r.ARMS}
        manifest={'arms':[r.ARMS[0]],'boundary_variant':{'arm':r.ARMS[0],
            'parent':{'path':'p','sha256':'s'},'global':{'path':'g','sha256':'s'}},'protocol_sha256':'test'}
        def source_manifest(path):
            arm=Path(path).parts[-2];value=copy.deepcopy(manifest);value['arms']=[arm];value['boundary_variant']['arm']=arm;return value
        def state(path):
            p=Path(path);ci=int(p.parts[-3].split('_')[1]);arm=p.parts[-2]
            return {'conversation':ci,'arm':arm,'protocol_sha256':'test','status':'complete',
                'finished_utc':'time','completed_groups':3,'expected_groups':3,'facts':24}
        for defect in ('running','partial','missing_finished'):
            def damaged(path):
                value=state(path)
                if (value['conversation'],value['arm'])==(9,r.ARMS[1]):
                    if defect=='running':value['status']='running'
                    elif defect=='partial':value['completed_groups']=2
                    else:value['finished_utc']=None
                return value
            with patch.object(r.cstrict.writer,'manifest',side_effect=source_manifest),\
                 patch.object(r.boundary,'pointer',side_effect=lambda p:{'path':str(p),'sha256':'s'}),\
                 patch.object(r.old,'read',side_effect=damaged),patch.object(r,'closed_controllers') as closed,\
                 patch.object(r.old,'points') as points,patch.object(r.old,'rows') as sqlite:
                with self.assertRaises(ValueError):r.terminal(runs)
                closed.assert_not_called();points.assert_not_called();sqlite.assert_not_called()

    def test_failed_controller_is_terminal_but_never_pretends_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'parallel_controller_v1/write_fixture';folder.mkdir(parents=True)
            runs={a:root/a/'writer' for a in r.ARMS};targets=self.targets()
            ref=lambda p:{'path':str(Path(p)),'sha256':'fixture'}
            cm={'schema':r.parallel.VERSION,'parent':ref(root/'manifest.json'),
                'code_sha256':{p:'fixture-code' for p in r.parallel.FILES},
                'policy':{'stage':'write','arms':list(r.ARMS),'conversations':list(range(10))}}
            mp=ref(folder/'manifest.json');docs={str(folder/'manifest.json'):cm};outcomes={}
            targets[6]['state']['status']='failed'
            targets[6]['state']['error']={'type':'ValueError','message':r.ERROR}
            for ci in range(10):
                terms=[]
                for arm in r.ARMS:
                    t=next(t for t in targets if (t['conversation'],t['arm'])==(ci,arm))
                    t['state']['finished_utc']='1';job=folder/'conversations'/f'conv_{ci}'/arm
                    term={'schema':r.parallel.VERSION,'controller_manifest':mp,'stage':'write',
                        'conversation':ci,'arm':arm,'status':t['state']['status'],'finished_utc':'2',
                        'outputs':{'write_status':ref(t['folder']/'write_status.json'),
                            'bank':ref(t['folder']/'bank.json')} if t['state']['status']=='complete' else {}}
                    docs[str(job/'terminal.json')]=term;terms.append(term)
                    docs[str(job/'intent.json')]={'controller_manifest':mp,'nonce':'unique',
                        'stage':'write','conversation':ci,'arm':arm}
                    docs[str(job/'worker_state.json')]={'status':term['status'],'terminal':ref(job/'terminal.json')}
                outcomes[str(ci)]={'conversation':ci,'finished_utc':'2',
                    'status':'failed' if ci==3 else 'complete','arms':terms}
            state={'status':'failed','finished_utc':'3','controller_manifest':mp,'stage':'write',
                'requested_conversations':list(range(10)),'arms':list(r.ARMS),'outcomes':outcomes,
                'pid':999,'parent_process_identity':{'pid':999,'creation_ticks':1},'nonce':'unique'}
            docs[str(folder/'parent_state.json')]=state
            with patch.object(r,'verify',side_effect=lambda p:copy.deepcopy(docs[str(p)])),\
                 patch.object(r.old,'read',side_effect=lambda p:copy.deepcopy(docs[str(p)])),\
                 patch.object(r.old,'pointer',side_effect=ref),patch.object(r.boundary,'pointer',side_effect=ref),\
                 patch.object(r.old,'sha',return_value='fixture-code'),patch.object(r.parallel,'process_identity',return_value=None):
                closed=r.closed_controllers(runs,targets)
                self.assertEqual(len(closed['controllers'][0]['workers']),20)
                state['status']='complete'
                with self.assertRaisesRegex(ValueError,'aggregate status'):r.closed_controllers(runs,targets)
                state['status']='running'
                with self.assertRaisesRegex(ValueError,'closed terminal'):r.closed_controllers(runs,targets)
                state['status']='failed';outcomes['3']['status']='complete'
                with self.assertRaisesRegex(ValueError,'conversation status'):r.closed_controllers(runs,targets)

    def test_required_index_or_source_checks_cannot_be_waived(self):
        proof_path=r.ROOT/'research/planned_execution_2026-10-03/checks/20261003T0658/c1_conv3_failure_review.json'
        proof=json.loads(proof_path.read_text(encoding='utf-8'))  # metadata only; no native read
        key=next(iter(r.SUPPORTED_FAILURES));target={'conversation':key[0],'arm':key[1],
            'run':Path(proof['source_manifest']['path']).parent,'folder':Path(proof['write_status']['path']).parent,
            'manifest':{'protocol_sha256':key[2]},'state':{'expected_groups':33,'facts':243}}
        refs={str(proof_path):{'path':str(proof_path),'sha256':r.SUPPORTED_FAILURES[key]},
              proof['source_manifest']['path']:proof['source_manifest'],proof['write_status']['path']:proof['write_status']}
        for check in ('facts_dense_sparse_history','existing_entity_dense_links_valid','group_source_date_raw_persistence'):
            bad=copy.deepcopy(proof);bad['checks'][check]=False
            with patch.object(r.old,'pointer',side_effect=lambda p:refs[str(p)]),\
                 patch.object(r,'verify',return_value=bad),patch.object(r.old,'native_inventory') as native:
                with self.assertRaisesRegex(ValueError,'required-index proof'):r._failure_binding(proof_path,target)
                native.assert_not_called()

    def test_export_text_date_or_embedding_changes_refused_without_native_or_export(self):
        for field,bad_value in [('text','altered'),('observation_date','2099-01-01'),('embedding',[.5]*768)]:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder=root/'metadata/conv_3/c1_multiscale';folder.mkdir(parents=True)
                native=root/'empty-native-fixture';native.mkdir()
                fact={'id':'original-id','text':'original text','observation_date':'2023-01-01','embedding':[.25]*768}
                journal=r.old.save(root/'journal.json',{'sequence':0,'status':'success',
                    'fact_records':[{k:v for k,v in fact.items() if k!='embedding'}]})
                source=r.old.save(root/'source.json',{'protocol_sha256':'original-protocol','dataset_sha256':'frozen-data'})
                state=r.old.save(root/'status.json',{'status':'failed'})
                row={'conversation':3,'arm':'c1_multiscale','source_kind':'reviewed_export',
                    'original_write_status':'failed','facts':1,'groups':1,'orphan_fact_ids':['original-id'],
                    'facts_projection_sha256':r.digest([fact]),'native_dir':str(native),'native_inventory':{},
                    'write_status':state,'outcome':state,'source_manifest':source,'group_inputs':{},'bank':None}
                review=r.old.save(root/'review.json',{'banks':[row]})
                changed=copy.deepcopy(fact);changed[field]=bad_value
                bp=r.old.save(folder/'bank.json',{'schema':r.BANK_SCHEMA,'complete':True,'conversation':3,'arm':'c1_multiscale',
                    'protocol_sha256':'original-protocol','source_dataset_sha256':'frozen-data','group_inputs':{},
                    'source_review':review,'original_write_status':'failed','source_kind':'reviewed_export',
                    'native_index_policy':'original_unrepaired','native_dir':str(native),'source_inventory':{},
                    'journal_files':[journal],'facts':[changed]})
                r.old.save(folder/'reviewed_export.json',{'schema':r.EXPORT_SCHEMA,'conversation':3,'arm':'c1_multiscale',
                    'source_kind':'reviewed_export','original_write_status':'failed','reviewed_export_status':'complete',
                    'source_review':review,'uniform_twenty_bank_review':review,'review_row_sha256':r.digest(row),
                    'native_index_policy':'original_unrepaired','bank':bp,'export_tool':r.old.pointer(r.__file__),
                    'original_write_status_ref':state,'original_outcome_ref':state,'source_manifest':source,
                    'native_dir':str(native),'native_inventory':{},'orphan_fact_ids':['original-id']})
                with patch.object(r,'check_review',return_value={'banks':[row]}),patch.object(r.old,'points') as points,\
                     patch.object(r.old,'rows') as sqlite,patch.object(r,'export') as production_export:
                    with self.assertRaisesRegex(ValueError,'text/date/vector projection'):r.check_export_bank(folder)
                    points.assert_not_called();sqlite.assert_not_called();production_export.assert_not_called()

    def test_source_population_rejects_partial_failed_erasure_and_missing_proof(self):
        rows=[{'conversation':t['conversation'],'arm':t['arm'],'groups':3,'expected_groups':3,'facts':24,
            'source_kind':'strict_complete','original_write_status':'complete','proof':None,
            'native_index_limitations':['unchanged native']} for t in self.targets()]
        rows[6].update(source_kind='reviewed_export',original_write_status='failed',proof={'path':'current-proof'})
        with patch.object(r,'check_review',return_value={'banks':rows}):
            _,population=sources.population('fixture-review')
            self.assertEqual(len(population),20)
        for change in ({'groups':2},{'facts':19},{'proof':None},{'original_write_status':'complete'},
                       {'source_kind':'strict_complete'},{'native_index_limitations':[]}):
            bad=copy.deepcopy(rows);bad[6].update(change)
            with patch.object(r,'check_review',return_value={'banks':bad}):
                with self.assertRaisesRegex(ValueError,'Partial/unknown/unproved'):sources.population('fixture-review')

    def test_export_outcome_requires_all20_and_zero_mutation(self):
        exports=[{'conversation':t['conversation'],'arm':t['arm']} for t in self.targets()]
        ref={'path':'current20-review','sha256':'sealed'}
        value={'schema':r.EXPORT_SCHEMA,'status':'reviewed_metadata_exports_complete','source_review':ref,
            'source_mutations':0,'native_bytes_copied':0,'model_calls':0,'exports':exports}
        with patch.object(sources,'verify',return_value=value):self.assertEqual(len(sources.export_population('fixture',ref)),20)
        for bad in ({**value,'exports':exports[:-1]}, {**value,'exports':exports[:-1]+[exports[0]]},
                    {**value,'native_bytes_copied':1},{**value,'source_mutations':1},{**value,'model_calls':1}):
            with patch.object(sources,'verify',return_value=bad):
                with self.assertRaises(ValueError):sources.export_population('fixture',ref)

    def test_original_sdk20_reader_cohort_and_cache_only_retrieval_objects_reused(self):
        self.assertIs(qa.plain.one,qa1.plain.one)
        self.assertIs(qa.native_qa.retrieve_arm,qa1.native_qa.retrieve_arm)
        self.assertIs(qa.retrieve_worker.__code__,qa1.retrieve_worker.__code__)
        self.assertIs(sources.protocol_config,sources.legacy.protocol_config)
        self.assertIs(sources.cohort,sources.legacy.cohort)
        self.assertEqual(qa.POLICY['slots_per_arm'],1540)
        self.assertEqual(qa.POLICY['cache_policy'],qa1.POLICY['cache_policy'])
        self.assertNotEqual(qa.SCHEMA,qa1.SCHEMA)


if __name__=='__main__':unittest.main()
