"""Audit-only preprocessing regression; no store/search/network/production writes."""
import copy,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from tools import audit_qualifier_repair_query_trim_r09 as m
from jev_mem.local_bm25_context_v2 import local_bm25_context,forbid_network

class QueryTrimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=m.erratum();cls.completed=m.completed_run(spec)
        cls.original=m.sealed_read(cls.completed/'run_manifest.json')
        jobs=m.parent.old.old_jobs(m.bound(cls.original['protocol']['frozen_source']))
        cls.job=next(j for j in jobs if j['query_id']=='c1_q22')
        pack=m.bound(cls.original['embedding_cache_pack']['manifest'])
        cls.entries=[e for e in pack['entries'] if e['conversation']==1]
        cls.selections={arm:m.sealed_read(m.parent.retrieval_folder(cls.completed,1,arm)/'selections/c1_q22.json')
            for arm in ('cheap','jev')}
    def context(self):
        r=self.original['resources'];return local_bm25_context(r['snapshot'],r['files_sha256'])
    def test_actual_two_arm_events_correct_raw_question_preserved(self):
        before=copy.deepcopy(self.job)
        self.assertTrue(self.job['question'].endswith(' '))
        with self.context(),forbid_network():
            for arm,s in self.selections.items():
                with self.assertRaises(RuntimeError):m.parent.validate_cache_events(s,self.job,self.entries,arm)
                self.assertEqual(m.validate_cache_events(s,self.job,self.entries,arm),2)
                self.assertEqual(s['cache_events'][0]['request_hash'],
                    '656c2f227a5f3d83c2fba0b76e6849f90ad38badd206469cba0ca9f5be01bf84')
        self.assertEqual(self.job,before)
    def test_leading_and_trailing_whitespace_uses_exact_sdk_without_mutation(self):
        base=self.job['question'].strip();j=copy.deepcopy(self.job);j['question']=' \t\n'+base+'\r\n '
        before=copy.deepcopy(j)
        with self.context(),forbid_network():self.assertEqual(m.validate_cache_events(self.selections['jev'],j,self.entries,'jev'),2)
        self.assertEqual(j,before)
    def test_empty_or_nonstring_uses_frozen_sdk_rejection(self):
        for q in ('',' \t\r\n',None,7,[]):
            j={**self.job,'question':q}
            with forbid_network(),self.assertRaises(ValueError):m.validate_cache_events(self.selections['jev'],j,self.entries,'jev')
    def test_wrong_identity_context_raw_hash_and_extra_event_rejected(self):
        with self.context(),forbid_network():
            for mode in ('qid','arm','context','hash','target','extra'):
                s=copy.deepcopy(self.selections['jev'])
                if mode=='qid':s['query_id']='foreign'
                elif mode=='arm':
                    s['method']='cheap'
                    for e in s['cache_events']:e['context']['method']='cheap'
                elif mode=='context':s['cache_events'][0]['context']['phase']='other'
                elif mode=='hash':s['cache_events'][0]['request_hash']='0'*64
                elif mode=='target':s['cache_events'][0]['target']['sha256']='0'*64
                else:s['cache_events'].append(copy.deepcopy(s['cache_events'][0]))
                with self.assertRaises(ValueError):m.validate_cache_events(s,self.job,self.entries,'jev')
    def test_private_gate_pointer_alias_exact_and_original_globals_unmodified(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)/'completed';run.mkdir();out=Path(td)/'derived_gate.json'
            ref=m.save_new(out,{'status':'passed'})
            original_pointer=m.parent.pointer
            scope=m._scope(run,ref)
            self.assertEqual(scope.pointer(run/'retrieval_gate.json'),ref)
            self.assertEqual(scope.pointer(out),m.pointer(out))
            self.assertIs(m.parent.pointer,original_pointer)
            out.write_bytes(out.read_bytes()+b' ')
            with self.assertRaises(ValueError):scope.pointer(run/'retrieval_gate.json')
    def test_source_assembler_keeps_original_question_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            directory=Path(td);gate=m.save_new(directory/'retrieval_gate.json',{'status':'passed'})
            original_value={'jobs':[copy.deepcopy(self.job)],'retrieval_gate':gate,
                'retrieval_manifest':m.pointer(self.completed/'run_manifest.json')}
            private=SimpleNamespace(_source_value=lambda run,original:copy.deepcopy(original_value))
            with patch.object(m,'erratum',return_value={'outputs':{'directory':str(directory)}}),\
                patch.object(m,'_scope',return_value=private),\
                patch.object(m,'pointer',side_effect=lambda p:m.parent.pointer(p) if Path(p).name!='manifest.json'
                    else {'path':str(Path(p).resolve()),'sha256':'bound-test-manifest'}):
                result=m.source_value({'code_sha256':{'new':'h'}},self.original,self.completed)
            self.assertEqual(result['jobs'],original_value['jobs'])
            self.assertTrue(result['jobs'][0]['question'].endswith(' '))
            self.assertEqual(result['retrieval_manifest'],original_value['retrieval_manifest'])
    def test_foreign_source_location_and_erratum_refs_fail_closed(self):
        spec=m.erratum()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'selected_inputs.json'
            m.save_new(path,{'query_event_audit_erratum':m.pointer(m.ERRATUM)})
            with self.assertRaises(ValueError):m.validate_source(path)
        with patch.object(m,'ERRATUM_SHA','0'*64),self.assertRaises(ValueError):m.erratum()

if __name__=='__main__':unittest.main()

