"""Independent review of lifecycle additions, not of this reviewer's v3 core.

Only in-memory records and isolated temporary archives are used. No real
controller process, production artifact, model, credential, or deletion runs.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_hypermem_native_v3 as a
from tools import run_five_baselines_reviewed_chain_v8 as chain


def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def sha(data):return hashlib.sha256(data).hexdigest()


class View:
    def __init__(self):self.data={}
    def put(self,p,v):self.data[str(p)]=encoded(v);return {'path':str(p),'sha256':self.sha(p)}
    def bytes(self,p):return self.data[str(p)]
    def read(self,p):return json.loads(self.bytes(p))
    def sha(self,p):return sha(self.bytes(p))


class ResultGate(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=a.ROOT/'research');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.old=self.root/'old';self.parent=self.root/'parent';self.new=self.root/'new'
        self.s=ExitStack();self.addCleanup(self.s.close)
        for name,path in [('OLD',self.old),('PARENT',self.parent),('NEW',self.new)]:self.s.enter_context(patch.object(a,name,path))
        self.view=View();self.review=self.root/'result_review.json'
        qs=[{'query_id':f'q{i}'} for i in range(1540)]
        original=self.view.put(self.old/'run_manifest.json',{'questions':qs})
        parent=self.view.put(self.parent/'run_manifest.json',{'questions':qs,'parent':original})
        self.cohort={'questions':qs,'original_inventory':{original['path']:original['sha256']},'current_inventory':{parent['path']:parent['sha256']}}
        self.s.enter_context(patch.object(a,'_cohort',return_value=self.cohort))
        proof=self.view.put(self.root/'proof.json',{'questions':qs})
        review=self.view.put(self.root/'core_review.json',{'schema':a._core().REVIEW_SCHEMA,'status':'passed',
            'proof_sha256':proof['sha256'],'code_sha256':{},'upstream_sha256':{},'specification':{},'evidence_sha256':{}})
        self.view.put(self.new/'run_manifest.json',{'schema':'hypermem-native-v3-continuation-manifest-v1','new_run':str(self.new),
            'original_manifest':original,'parent_manifest':parent,'terminal_proof':proof,'review':review,
            'questions':qs,'code_sha256':{},'upstream_sha256':{},'specification':{}})
        for root in (self.old,self.parent):self.view.put(root/'status.json',{'stage':'needs_review'})
        counts={'slots':1540,'reader_valid':1539,'judge_valid':1538,'missing_reader':1,'missing_judge':2}
        self.summary={'schema':'five-baseline-result-v1','method':'hypermem','audit_status':'no_invalid_artifacts',
            'firstpass':deepcopy(counts),'effective':deepcopy(counts)}
        self.update()

    def update(self):
        summary=self.view.put(self.new/'results/summary.json',self.summary)
        self.view.put(self.new/'status.json',{'stage':'completed_awaiting_result_review','summary_sha256':summary['sha256']})
        self.view.put(self.review,{'schema':a.RESULT_SCHEMA,'status':'passed','method':'hypermem','summary_sha256':summary['sha256'],
            'reviewed_run_inventories':{'original':self.cohort['original_inventory'],'parent':self.cohort['current_inventory'],
                'continuation':{summary['path']:summary['sha256']}}})

    def test_complete_fixed_slots_accept_honest_missing_after_permitted_recovery(self):
        result=a.result_review(self.review,self.view)
        self.assertEqual(result[2]['effective']['judge_valid'],1538)
        self.assertEqual(result[2]['effective']['missing_judge'],2)

    def test_boolean_or_float_missing_counts_are_not_integer_telemetry(self):
        original=deepcopy(self.summary)
        for phase in ('firstpass','effective'):
            for field,valid in (('missing_reader','reader_valid'),('missing_judge','judge_valid')):
                for bad in (False,0.0):
                    with self.subTest(phase=phase,field=field,bad=bad):
                        self.summary=deepcopy(original);self.summary[phase][valid]=1540;self.summary[phase][field]=bad;self.update()
                        with self.assertRaises(ValueError):a.result_review(self.review,self.view)

    def test_changed_parent_inventory_and_overwritten_original_failure_rejected(self):
        review=self.view.read(self.review);review['reviewed_run_inventories']['parent']={str(self.parent/'unknown.json'):'a'*64}
        self.view.put(self.review,review)
        with self.assertRaises(ValueError):a.result_review(self.review,self.view)
        self.update();self.view.put(self.old/'status.json',{'stage':'completed_awaiting_result_review'})
        with self.assertRaises(ValueError):a.result_review(self.review,self.view)


class ArchiveBoundary(unittest.TestCase):
    def test_original_parent_new_raw_and_journals_cannot_be_disposed(self):
        for owner in a.roots().values():
            for name in ('api_method/api_cache/key.json','api_method/request_intents/key.json','prefix_v3/c0/requests.jsonl',
                'replacement_v3/known_invalid_episode_once.json','native/c0/representations/role.json','native/c0/cache/unique.txt'):
                self.assertIsNone(a.disposition(owner/'work'/name))
            with self.assertRaises(ValueError):a.disposition(owner/'work/native/c0/calls/response.json.tmp')
        with self.assertRaises(ValueError):a.disposition(a.ROOT/'research/nonproduction/cache/hypergraph_bm25_index_conv_0.pkl')

    def test_crc_valid_unique_evidence_disposal_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td:
            root=Path(td);path=a.PARENT/'work/api_method/api_cache/same_key.json';data=b'original raw'
            m={'schema':a.SCHEMA,'records':[{'path':str(path),'sha256':sha(data),'size':len(data),
                'retained':False,'disposal_reason':'runnable_hypermem_bm25_vector_index'}]}
            mp=root/'manifest.json';mp.write_bytes(encoded(m));zp=root/'archive.zip'
            with zipfile.ZipFile(zp,'x') as z:z.writestr('archive_manifest.json',mp.read_bytes())
            with self.assertRaises(ValueError):a.Resolver(zp,mp)

    def test_safe_live_certificate_operations_reject_foreign_before_io(self):
        a._sync();foreign=Path(a.ROOT.anchor)/'not_workspace'/'untrusted.json'
        for name in ('read','sha','bytes'):
            with self.subTest(name=name),patch.object(Path,'open',side_effect=AssertionError('Unexpected foreign read')),\
                    patch.object(Path,'read_bytes',side_effect=AssertionError('Unexpected foreign read')):
                with self.assertRaises(ValueError):getattr(a._engine.seal.LiveView(),name)(foreign)


class DispatcherBoundary(unittest.TestCase):
    def test_current_hyper_completion_and_optional_vmem_are_both_required(self):
        for method in ('vmem','prism'):
            owners=['amem','simplemem','hypermem']+(['vmem'] if method=='prism' else [])
            refs={o:{'completion':a.ROOT/'research/synthetic'/o/'completion.json'} for o in owners}
            bind=lambda p:{'path':str(p),'sha256':sha(str(p).encode())}
            accepted={o:{k:bind(p) for k,p in refs[o].items()} for o in ('amem','simplemem')};calls=[]
            with patch.object(chain,'completion_paths',return_value=refs),patch.object(chain,'binding',side_effect=bind),\
                patch.object(chain,'assert_stopped'),patch.object(chain,'attested_predecessors',return_value=accepted),\
                patch.object(chain.hyper,'verify_completion',side_effect=lambda:(calls.append('hypermem') or {'status':'passed'})),\
                patch.object(chain.previous.previous.frozen.lifecycle,'verify_completion',side_effect=lambda m:(calls.append(m) or {'status':'passed'})):
                self.assertEqual(set(chain.strong_predecessors(method)),set(owners))
            self.assertEqual(calls,['hypermem']+(['vmem'] if method=='prism' else []))

    def test_hyper_failure_has_no_legacy_or_earlier_result_fallback(self):
        refs={o:{'completion':a.ROOT/'research/synthetic'/o/'completion.json'} for o in ('amem','simplemem','hypermem')}
        bind=lambda p:{'path':str(p),'sha256':'a'*64};accepted={o:{k:bind(p) for k,p in refs[o].items()} for o in ('amem','simplemem')}
        with patch.object(chain,'completion_paths',return_value=refs),patch.object(chain,'binding',side_effect=bind),\
            patch.object(chain,'assert_stopped'),patch.object(chain,'attested_predecessors',return_value=accepted),\
            patch.object(chain.hyper,'verify_completion',return_value={'status':'needs_review'}),\
            patch.object(chain.previous.previous.frozen.lifecycle,'verify_completion') as old:
            with self.assertRaises(ValueError):chain.strong_predecessors('vmem')
            old.assert_not_called()


class HistoricalByteVersions(unittest.TestCase):
    def test_two_later_heartbeats_rebind_only_twenty_eight_logs_and_twenty_statuses(self):
        total=0
        with a._read_context():
            for filename,status_count in (('heartbeat0949_native01.json',11),('heartbeat0949_native02.json',9)):
                path=(a.ROOT/a.OBSERVER_REPORT).with_name(filename);original=a.read(path);before=a.sha(path)
                logs=next(r for r in a.observer_registries() if Path(r['report']['path'])==path)
                statuses=a.status_registry(path)
                self.assertFalse(any(Path(r['report']['path'])==path for r in a.started_registries()))
                for fn,registry,extra,count in (
                    (a.historical_observer_projection,logs,'historical_prefix_capsule_registry',14),
                    (a.historical_status_projection,statuses,'historical_status_capsule_registry',status_count)):
                    with self.subTest(document=filename,kind=fn.__name__):
                        notes=[];actual=fn(original,path,notes);expected=deepcopy(original)
                        self.assertEqual(len(registry['records']),count);self.assertEqual(len(notes),count)
                        for row in registry['records']:
                            group,key=row['field_path'];expected[group][key]['path']=row['capsule']['path']
                            self.assertEqual(a.sha(row['capsule']['path']),row['observed_sha256'])
                        expected[extra]=registry['registry_binding'];self.assertEqual(actual,expected)
                        with self.assertRaises(ValueError):fn({**original,'unqualified_extra':False},path)
                        total+=count
                for p,h in statuses.get('unchanged_terminal_statuses',{}).items():
                    self.assertEqual(original['files'][p]['path'],p)
                    self.assertEqual(a.sha(p),h)
                self.assertEqual(a.sha(path),before)
        self.assertEqual(total,48)

    def test_all_sixty_four_registered_versions_preserve_every_other_field(self):
        total=0
        with a._read_context():
            for relative in (a.OBSERVER_REPORT,a.EMPTY_SEAL_OBSERVER):
                path=a.ROOT/relative;original=a.read(path);before=a.sha(path)
                select=lambda rows:next(r for r in rows if Path(r['report']['path'])==path)
                expected_counts=(14,9,11) if path.name=='native_snapshot04.json' else (14,5,11)
                variants=[(a.historical_observer_projection,select(a.observer_registries()),'historical_prefix_capsule_registry'),
                    (a.historical_started_projection,select(a.started_registries()),'historical_started_capsule_registry'),
                    (a.historical_status_projection,a.status_registry(path),'historical_status_capsule_registry')]
                for (fn,registry,extra),count in zip(variants,expected_counts):
                    with self.subTest(document=path.name,kind=fn.__name__):
                        notes=[];actual=fn(original,path,notes);expected=deepcopy(original)
                        self.assertEqual(len(registry['records']),count);self.assertEqual(len(notes),count)
                        for row in registry['records']:
                            group,key=row['field_path'];expected[group][key]['path']=row['capsule']['path']
                            self.assertEqual(a.sha(row['capsule']['path']),row['observed_sha256'])
                        expected[extra]=registry['registry_binding'];self.assertEqual(actual,expected)
                        total+=count
                self.assertEqual(a.sha(path),before)
        self.assertEqual(total,64)

    def test_one_empty_observation_is_retained_without_changing_terminal_seal(self):
        observer=a.ROOT/a.EMPTY_SEAL_OBSERVER;lineage_path=a.ROOT/a.EMPTY_SEAL_LINEAGE
        lineage=a.read(lineage_path);source=a.ROOT/a.EMPTY_SEAL_SOURCE
        before={str(p):a.sha(p) for p in (observer,lineage_path,source,Path(lineage['terminal_native']['path']))}
        self.assertIs(lineage['historical_seal_valid'],False)
        self.assertEqual(source.read_bytes().decode('ascii'),lineage['terminal_native']['sha256'])
        with a._read_context():
            for document in (observer,lineage_path):
                original=a.read(document);notes=[]
                actual=a.historical_empty_seal_projection(original,document,notes);expected=deepcopy(original)
                if document==observer:
                    expected['files'][str(source)]['path']=lineage['capsule']['path']
                    expected['historical_empty_read_capsule_lineage']={'path':str(lineage_path),'sha256':a.EMPTY_SEAL_LINEAGE_SHA}
                else:expected['observed_descriptor']['path']=lineage['capsule']['path']
                self.assertEqual(actual,expected);self.assertEqual(len(notes),1)
                self.assertIs(notes[0]['historical_seal_valid'],False)
                self.assertIs(notes[0]['terminal_seal_verified'],True)
        self.assertEqual({p:a.sha(p) for p in before},before)

    def test_empty_capsule_rule_rejects_any_other_document_field_or_terminal_hash(self):
        observer=a.ROOT/a.EMPTY_SEAL_OBSERVER;original=a.read(observer);source=a.ROOT/a.EMPTY_SEAL_SOURCE
        with a._read_context():
            with self.assertRaises(ValueError):a.historical_empty_seal_projection(original,observer.with_name('native_snapshot05.json'))
            for field,bad in [('bytes',1),('sha256','a'*64),('path',str(source.with_name('other.sha256')))]:
                changed=deepcopy(original);changed['files'][str(source)][field]=bad
                with self.subTest(field=field),self.assertRaises(ValueError):a.historical_empty_seal_projection(changed,observer)
            actual_sha=a.sha
            with patch.object(a,'sha',side_effect=lambda p:'0'*64 if Path(p)==source else actual_sha(p)):
                with self.assertRaises(ValueError):a.historical_empty_seal_projection(original,observer)

    def test_exact_failed_source_versions_preserve_report_and_all_other_fields(self):
        path=a.ROOT/a.FAILED_PREFIX_REPORT;original=a.read(path);before=a.sha(path);notes=[]
        actual=a.historical_failed_source_projection(original,path,notes)
        self.assertEqual(a.sha(path),before)
        self.assertEqual({k:v for k,v in actual.items() if k not in ('code_sha256','historical_source_snapshot_lineage')},
            {k:v for k,v in original.items() if k!='code_sha256'})
        self.assertEqual(sorted(actual['code_sha256'].values()),sorted(original['code_sha256'].values()))
        self.assertEqual(len(actual['code_sha256']),4);self.assertEqual(len(notes),1)
        for p,h in actual['code_sha256'].items():self.assertEqual(a.sha(p),h)
        with self.assertRaises(ValueError):a.historical_failed_source_projection(original,path.with_name('other.json'))
        with self.assertRaises(ValueError):a.historical_failed_source_projection({**original,'status':'passed'},path)

    def test_three_exact_observer_projections_only_rebind_certified_paths(self):
        path=a.ROOT/a.OBSERVER_REPORT;original=a.read(path);original_sha=a.sha(path)
        with a._read_context():
            select=lambda rows:next(r for r in rows if Path(r['report']['path'])==path)
            variants=[(a.historical_observer_projection,select(a.observer_registries()),'historical_prefix_capsule_registry',14),
                (a.historical_started_projection,select(a.started_registries()),'historical_started_capsule_registry',5),
                (a.historical_status_projection,a.status_registry(),'historical_status_capsule_registry',11)]
            for fn,registry,extra,count in variants:
                with self.subTest(kind=fn.__name__):
                    notes=[];actual=fn(original,path,notes);expected=deepcopy(original)
                    self.assertEqual(len(registry['records']),count);self.assertEqual(len(notes),count)
                    for row in registry['records']:
                        group,key=row['field_path'];expected[group][key]['path']=row['capsule']['path']
                        self.assertEqual(a.sha(row['capsule']['path']),row['observed_sha256'])
                    expected[extra]=registry['registry_binding'];self.assertEqual(actual,expected)
                    with self.assertRaises(ValueError):fn(original,path.with_name('unqualified_other_snapshot.json'))
                    with self.assertRaises(ValueError):fn({**original,'unqualified_extra_field':True},path)
        self.assertEqual(a.sha(path),original_sha)

    def test_capsule_changed_bytes_fail_before_any_historical_equivalence_claim(self):
        path=a.ROOT/a.OBSERVER_REPORT;value=a.read(path);original_sha=a.sha
        with a._read_context():
            select=lambda rows:next(r for r in rows if Path(r['report']['path'])==path)
            for fn,registry in [(a.historical_observer_projection,select(a.observer_registries())),
                (a.historical_started_projection,select(a.started_registries())),(a.historical_status_projection,a.status_registry())]:
                capsule=Path(registry['records'][0]['capsule']['path'])
                with self.subTest(kind=fn.__name__),patch.object(a,'sha',side_effect=lambda p,c=capsule:'0'*64 if Path(p)==c else original_sha(p)):
                    with self.assertRaises(ValueError):fn(value,path)


if __name__=='__main__':unittest.main(verbosity=2)
