"""Synthetic observer tests only; never calls observe() against a live run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'research/baseline_execution_2026-10-08/heartbeat_20261008T1945/production_v4_boundary_check_v1.py'
spec=importlib.util.spec_from_file_location('v4_boundary_observer_test_module',PATH)
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


class BoundaryObserverTests(unittest.TestCase):
    def test_missing_is_pending_and_hot_unterminated_tail_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'ledger.jsonl';snapshot=w.Snapshot()
            with self.assertRaises(w.Pending):snapshot.json(p)
            p.write_bytes(b'{"request_hash":"a"}\n{"request_hash":"partial"')
            snapshot=w.Snapshot()
            self.assertEqual(snapshot.lines(p),[{'request_hash':'a'}])
            self.assertEqual(snapshot.trailing_partial_logs,[str(p)])

    def test_malformed_completed_log_row_is_integrity_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'ledger.jsonl';p.write_bytes(b'broken\n')
            with self.assertRaises(w.EvidenceError):w.Snapshot().lines(p)

    def test_native_request_signature_matches_frozen_contract_without_imports(self):
        req={'prompt':'test','system':'sys','max_output':16384,'json_output':True}
        sig=w.signature(req)
        self.assertEqual(sig['body']['generationConfig'],{'maxOutputTokens':16384,'thinkingConfig':{'thinkingBudget':0},'responseMimeType':'application/json'})
        self.assertEqual(sig['body']['contents'][0]['parts'],[{'text':'test'}])
        with self.assertRaises(w.EvidenceError):w.signature({**req,'max_output':8192})

    def test_quote_journal_replays_only_exact_original_bytes(self):
        text='{"tags": [\'value\']}';derived='{"tags": ["value"]}'
        signature={'provider':'gemini','model':'gemini-2.5-flash','operation':'generateContent','body':{}}
        raw={'request':signature,'response':{'modelVersion':'gemini-2.5-flash','candidates':[{'finishReason':'STOP','content':{'role':'model','parts':[{'text':text}]}}]}}
        changes=[{'character_offset':i,'utf8_byte_offset':i,'from_utf8_hex':'27','to_utf8_hex':'22'} for i,c in enumerate(text) if c=="'"]
        journal={'raw_artifact_sha256':'a'*64,'request_hash':w.request_sha(signature),'original_text_utf8_sha256':w.byte_sha(text.encode()),
            'derived_text_utf8_sha256':w.byte_sha(derived.encode()),'parsed_payload_sha256':w.payload_sha({'tags':['value']}),'changes':changes}
        self.assertEqual(w.replay_journal(raw,journal,'a'*64),{'tags':['value']})
        changes[0]['to_utf8_hex']='78'
        with self.assertRaises(w.EvidenceError):w.replay_journal(raw,journal,'a'*64)

    def test_duplicate_keys_and_failed_native_rejected_started_pending(self):
        with self.assertRaises(w.EvidenceError):w.strict_json('{"x":1,"x":2}')
        with self.assertRaises(w.EvidenceError):w._success({'status':'failed'},'fixture')
        with self.assertRaises(w.Pending):w._success({'status':'started'},'fixture')


if __name__=='__main__':unittest.main()
