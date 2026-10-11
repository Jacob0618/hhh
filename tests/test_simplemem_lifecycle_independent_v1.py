"""Independent adversarial lifecycle tests; temporary files only, no controller."""
from contextlib import ExitStack
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import archive_simplemem_native_v2 as a


class RetentionAndResolver(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=a.ROOT/'research')
        self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to((a.ROOT/'research').resolve()))
        self.addCleanup(self.temp.cleanup)
        self.old=self.root/'old';self.new=self.root/'new'
        self.context=ExitStack();self.addCleanup(self.context.close)
        self.context.enter_context(patch.object(a,'OLD',self.old))
        self.context.enter_context(patch.object(a,'NEW',self.new))

    def package(self,records,blobs):
        mp=self.root/'manifest.json';zp=self.root/'evidence.zip'
        mp.write_text(json.dumps({'schema':a.SCHEMA,'records':records}),encoding='utf8')
        with zipfile.ZipFile(zp,'x') as z:
            z.writestr('archive_manifest.json',mp.read_bytes())
            for name,data in blobs.items():z.writestr(name,data)
        return zp,mp

    def test_unique_text_is_never_dropped_by_directory_or_temporary_suffix(self):
        for name in ('state.json','cache/audit.json','__pycache__/unique_notes.json','__pycache__/unique_log.txt'):
            p=self.new/'work/native/c0/simplemem'/name
            try:reason=a.disposition(p)
            except ValueError:continue
            self.assertIsNone(reason,str(p))
        with self.assertRaises(ValueError):a.disposition(self.new/'work/native/c0/simplemem/response.json.tmp')

    def test_crc_valid_false_unique_note_is_rejected(self):
        p=self.old/'work/native/c0/simplemem/memory.json';data=b'{"unique":true}'
        record={'path':str(p),'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                'retained':False,'disposal_reason':'runnable_lancedb_vector_and_fts_index'}
        zp,mp=self.package([record],{})
        with self.assertRaisesRegex(ValueError,'retention'):a.Resolver(zp,mp)

    def test_retained_requires_bool_not_numeric_flag(self):
        p=self.old/'work/native/c0/simplemem/lancedb/file.lance'
        record={'path':str(p),'size':3,'sha256':hashlib.sha256(b'abc').hexdigest(),
                'retained':0,'disposal_reason':'runnable_lancedb_vector_and_fts_index'}
        zp,mp=self.package([record],{})
        with self.assertRaisesRegex(ValueError,'Noncanonical'):a.Resolver(zp,mp)

    def test_same_request_name_distinct_raw_and_no_live_fallback(self):
        records=[];blobs={};paths=[]
        for root,data in ((self.old,b'{"owner":"first"}'),(self.new,b'{"owner":"second"}')):
            p=root/'work/api_method/api_cache/samekey.json';paths.append(p)
            h=hashlib.sha256(data).hexdigest();member='blobs/'+h
            records.append({'path':str(p),'size':len(data),'sha256':h,'retained':True,'member':member});blobs[member]=data
        db=self.new/'work/native/c0/simplemem/lancedb/file.lance';db.parent.mkdir(parents=True);db.write_bytes(b'database')
        records.append({'path':str(db),'size':8,'sha256':hashlib.sha256(b'database').hexdigest(),'retained':False,'disposal_reason':'runnable_lancedb_vector_and_fts_index'})
        unknown=self.root/'unknown.json';unknown.write_text('{"available_live":true}',encoding='utf8')
        zp,mp=self.package(records,blobs)
        with a.Resolver(zp,mp) as view:
            self.assertNotEqual(view.bytes(paths[0]),view.bytes(paths[1]))
            for p in (unknown,db):
                with self.assertRaisesRegex(ValueError,'unavailable'):view.bytes(p)
                with self.assertRaisesRegex(ValueError,'unavailable'):view.sha(p)
            self.assertFalse(view._simplemem_path_type(db).exists())
            self.assertNotIn(view._simplemem_path_type(db),list(view._simplemem_path_type(self.new/'work').rglob('*')))

    def test_reappeared_work_blocks_completion_even_with_valid_receipts(self):
        out=self.root/'output';out.mkdir();plan=out/'plan.json';plan.write_text('{}',encoding='utf8')
        ph=hashlib.sha256(plan.read_bytes()).hexdigest();rh='a'*64
        values={'allowed_delete_roots':[str(self.old/'work'),str(self.new/'work')]}
        (out/'cleanup_execution_intent.json').write_text(json.dumps({'schema':'simplemem-native-v2-cleanup-intent-v1','targets':values['allowed_delete_roots'],'cleanup_plan_sha256':ph,'cleanup_review_sha256':rh}),encoding='utf8')
        for key,root in (('original',self.old),('continuation',self.new)):
            (out/('removal_'+key+'.json')).write_text(json.dumps({'schema':'simplemem-native-v2-removal-receipt-v1','removed_path':str(root/'work'),'cleanup_plan_sha256':ph,'cleanup_review_sha256':rh}),encoding='utf8')
        (self.new/'work').mkdir(parents=True)
        with patch.object(a,'OUTPUT',out),patch.object(a,'DEFAULT_PLAN',plan):
            with self.assertRaisesRegex(ValueError,'remains or reappeared'):a.removal_evidence(values,{'sha256':rh})


class ExactNegativeEvidenceBoundary(unittest.TestCase):
    """Virtual graph hashes isolate policy predicates; no full evidence scan."""
    def graph(self,*,change_negative=None,alternate_path=False,initial_sha=None,missing_positive_dependency=False):
        root=a.ROOT;negative=root/a.NEGATIVE;positive=root/a.POSITIVE;decision=root/a.NEGATIVE_DECISION
        source='tools/simplemem_prefix_api_v2.py';real_sha=a.sha
        good={'schema':a.core.REVIEW_SCHEMA,'fixture_only':True,'production_launch_authorized':False,
            'purpose':'Synthetic review input for isolated research prepare test; not the final independent review or permission to launch.',
            'code_sha256':{source:'84a6f321445715939d7d1c36207b5ace72b8f80f9ccedbad35e0ab2c8d7a7d3e'}}
        if missing_positive_dependency:
            good['evidence_sha256']={str(root/'research/does-not-exist-independent-fixture-evidence.txt'):'d'*64}
        bad=deepcopy(good);bad['code_sha256'][source]='0'*64
        if change_negative:change_negative(bad)
        other=root/'research/other-unclassified-negative-fixture.json'
        documents={negative:bad,positive:good,decision:{'scope':'virtual exact decision'},other:bad}
        trusted={negative:a.NEGATIVE_SHA,positive:a.POSITIVE_SHA,decision:a.NEGATIVE_DECISION_SHA,other:a.NEGATIVE_SHA}
        def fake_read(path):
            p=Path(path)
            if p in documents:return deepcopy(documents[p])
            raise FileNotFoundError(str(p))
        def fake_sha(path):return trusted[Path(path)] if Path(path) in trusted else real_sha(path)
        with patch.object(a,'read',side_effect=fake_read),patch.object(a,'sha',side_effect=fake_sha):
            journal=[];result=a.evidence_closure([(str(other if alternate_path else negative),initial_sha or a.NEGATIVE_SHA)],annotations=journal)
        return result,journal

    def test_only_exact_document_can_skip_its_deliberately_invalid_map(self):
        result,journal=self.graph()
        self.assertEqual(result[str(a.ROOT/a.NEGATIVE)],a.NEGATIVE_SHA)
        self.assertEqual(result[str(a.ROOT/a.POSITIVE)],a.POSITIVE_SHA)
        self.assertEqual(result[str(a.ROOT/'tools/simplemem_prefix_api_v2.py')],'84a6f321445715939d7d1c36207b5ace72b8f80f9ccedbad35e0ab2c8d7a7d3e')
        self.assertEqual(len(journal),1)
        self.assertIn('nonoperative admission test leaf',journal[0]['treatment'])
        with self.assertRaises(ValueError):self.graph(alternate_path=True)
        with self.assertRaises(ValueError):self.graph(initial_sha='a'*64)

    def test_exact_schema_flags_and_sole_changed_field_are_required(self):
        for key,value in [('schema','other'),('fixture_only',1),('fixture_only',False),('production_launch_authorized',0),('production_launch_authorized',True),('purpose','different')]:
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                self.graph(change_negative=lambda bad:bad.update({key:value}))
        with self.assertRaises(ValueError):self.graph(change_negative=lambda bad:bad.update({'extra_unclassified_field':True}))

    def test_positive_document_still_recurses_and_missing_dependency_is_rejected(self):
        with self.assertRaises((ValueError,FileNotFoundError)):
            self.graph(missing_positive_dependency=True)

    def test_locked_actual_negative_bytes_and_only_field_difference(self):
        good=a.read(a.ROOT/a.POSITIVE);bad=a.read(a.ROOT/a.NEGATIVE)
        self.assertEqual(a.sha(a.ROOT/a.POSITIVE),a.POSITIVE_SHA)
        self.assertEqual(a.sha(a.ROOT/a.NEGATIVE),a.NEGATIVE_SHA)
        self.assertEqual(a.sha(a.ROOT/a.NEGATIVE_DECISION),a.NEGATIVE_DECISION_SHA)
        good_map=good['code_sha256'];bad_map=bad['code_sha256']
        self.assertEqual([p for p in good_map if good_map[p]!=bad_map[p]],['tools/simplemem_prefix_api_v2.py'])
        fixed=deepcopy(bad);fixed['code_sha256']=good_map
        self.assertEqual(fixed,good)


class PinnedRuntimeAndLaunch(unittest.TestCase):
    def test_external_lancedb_exception_is_one_source_path_and_content(self):
        source=a.ROOT/'runtime/baseline_env_v1/Lib/site-packages/lancedb/fts.py'
        expected='960450edf67d4870d7ee714373ef78fa51c7ab7e68d41e66c9b895c1cce5f0f7'
        self.assertEqual(a.sha(source),expected)
        self.assertIsNone(a.disposition(source))
        with patch.object(a,'sha',return_value='f'*64):
            with self.assertRaises(ValueError):a.disposition(source)
        for p in (source.with_name('another.py'),source.with_name('vectors.lance'),
                  a.ROOT/'research/example/lancedb/fts.py',
                  a.ROOT/'research/example/lancedb/data/fragment.lance'):
            with self.subTest(path=str(p)),self.assertRaises(ValueError):a.disposition(p)

    def test_ps_checks_literal_and_final_chains_and_code_before_exclusive_intent(self):
        ps=(a.ROOT/'tools/cleanup_simplemem_native_v2.ps1').read_text('utf8')
        function=ps.split('function AssertUnderWorkspace',1)[1].split('function WriteExclusive',1)[0]
        self.assertLess(function.index('AssertLiteralChain $literal'),function.index('Resolve-Path'))
        self.assertLess(function.index('Resolve-Path'),function.index('AssertLiteralChain $resolved'))
        call=ps.index('WriteExclusive $intentPath')
        for required in ('foreach ($prop in $plan.keep_sha256.', 'foreach ($prop in $plan.code_sha256.',
                         "throw 'Reparse inside work'",'& $python $helper verify'):
            self.assertLess(ps.index(required),call)
        self.assertLess(call,ps.index('Remove-Item -LiteralPath $targets[$i] -Recurse -Force'))
        self.assertEqual(ps.count('Remove-Item -LiteralPath'),1)


if __name__=='__main__':unittest.main()
