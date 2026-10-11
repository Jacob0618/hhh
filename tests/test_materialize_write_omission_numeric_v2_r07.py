"""Actual prepared 286-reference regression plus minimal v2 isolation/spawn seams."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
from tools import materialize_write_omission_numeric_r07 as v1
from tools import materialize_write_omission_numeric_v2_r07 as v2


class IOTests(unittest.TestCase):
    def test_actual_prepared286refs_including18_multiline_ledgers_all_bytebound(self):
        erratum=v2.erratum()
        m=v1.bound(erratum['prepared_v1_native_manifest'],sealed=True)
        self.assertEqual(len(m['input_refs']),286)
        lines=[r for r in m['input_refs'] if r['path'].endswith('.jsonl')]
        self.assertEqual(len(lines),18)
        v2.validate_input_bytes(m['input_refs'])
        # Reproduce the original wrong generic loader on an actual multiline ledger.
        row=next(r for r in lines if Path(r['path']).name=='api_calls.jsonl'
                 and len(Path(r['path']).read_text().splitlines())>1)
        with self.assertRaises(json.JSONDecodeError):v1.bound(row)
        # Dedicated semantic reader is still the unchanged original ledger validator.
        ledger=v1.old.embedding_ledger(v1.api_dir(v1.spec()['original_run'],4),4)
        self.assertGreater(len(ledger),1)
    def test_temporary_jsonl_and_binary_single_byte_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for filename,data in [('ledger.jsonl',b'{"x":1}\n{"x":2}\n'),('blob.bin',b'\xff\x00\x01')]:
                path=Path(tmp)/filename;path.write_bytes(data);ref=v1.pointer(path)
                v2.validate_input_bytes([ref]);path.write_bytes(data+b' ')
                with self.assertRaises(ValueError):v2.validate_input_bytes([ref])
    def test_exact_canonical_path_and_pointer_shape_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'record';path.write_text('x');ref=v1.pointer(path)
            for malformed in ({**ref,'extra':1},{'path':ref['path'],'sha256':'wrong'}):
                with self.assertRaises(ValueError):v2.validate_input_bytes([malformed])


class IsolationTests(unittest.TestCase):
    def test_no_frozen_global_leak_and_all_v2_workers_are_top_level(self):
        before=dict(vars(v1));ctx=v2._context()
        self.assertEqual(vars(v1),before)
        self.assertIs(ctx.prepare.__globals__['manifest'],v2.manifest)
        self.assertIs(ctx.materialize.__globals__['materialize_first_attempt'],v2.materialize_first_attempt)
        self.assertIs(ctx.retrieve.__globals__['retrieve_conversation'],v2.retrieve_conversation)
        self.assertEqual(ctx.spec()['namespace'],v2.erratum()['namespace'])
        self.assertNotEqual(ctx.spec()['namespace'],v1.spec()['namespace'])
        for f in (v2.materialize_first_attempt,v2.retrieve_conversation):
            self.assertIs(pickle.loads(pickle.dumps(f)),f)
    def test_actual_spawn_v2_worker_enters_original_bank_lookup_offline(self):
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
            future=pool.submit(v2.retrieve_conversation,'unused',0,{'admission':{}},{'banks':{}},[],[])
            with self.assertRaises(KeyError):future.result(timeout=30)
    def test_new_manifest_publication_adds_erratum_before_single_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run_manifest.json';v2._publish(path,{'code_sha256':{},'numeric_compatibility_protocol':v1.pointer(v1.TECH_SPEC)})
            result=v1.sealed_read(path)
            self.assertEqual(result['numeric_io_erratum'],v1.pointer(v2.ERRATUM))
            self.assertEqual(result['numeric_compatibility_protocol'],v1.pointer(v1.TECH_SPEC))
            with self.assertRaises(FileExistsError):v2._publish(path,{})
    def test_derived_source_bound_to_erratum(self):
        ctx=v2._context()
        with patch.object(v2,'_context',return_value=type('Context',(),{
            '_unversioned_seal_value':staticmethod(lambda r:{'jobs':[]})})()):
            source=v2._seal_value('unused')
        self.assertEqual(source['numeric_io_erratum'],v1.pointer(v2.ERRATUM))
        self.assertIs(ctx.validate_source.__globals__['_seal_value'],v2._seal_value)


if __name__=='__main__':unittest.main()
