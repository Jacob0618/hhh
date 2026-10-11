from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import audit_mem0_regroup_eof_helpers_v1 as audit


ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def pointer(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def bound(p):
    if sha(p['path']) != p['sha256']:
        raise ValueError('Pointer mismatch')
    return read(p['path'])


def verify(path):
    value = read(path)
    if value['content_sha256'] != digest({k:v for k,v in value.items() if k != 'content_sha256'}):
        raise ValueError('Seal mismatch')
    return value


def write(path, value, sealed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if sealed:
        value = copy.deepcopy(value)
        value['content_sha256'] = digest(value)
    path.write_text(json.dumps(value), encoding='utf8')


def response(text):
    return {'candidates': [{'finishReason':'STOP', 'content': {'parts':[{'text':text}]}}]}


class IndependentAuditTests(unittest.TestCase):
    def setUp(self):
        self.network = patch('socket.socket.connect', side_effect=AssertionError('No network'))
        self.network.start(); self.addCleanup(self.network.stop)

    def test_valid_and_eof_without_response_mutation(self):
        for text, action in [('{"memory":[]}', 'unchanged'),
                (' {"memory":[{"text":"embedded \\"brace}\\"", "attributed_to":"user"}]\n', 'append_single_root_brace')]:
            raw = response(text); before = copy.deepcopy(raw)
            parsed, proof = audit.independent_parse(raw, 200)
            self.assertEqual(raw, before)
            self.assertEqual(proof['action'], action)
            self.assertIsInstance(parsed['memory'], list)

    def test_real_original_response_one_byte_proof(self):
        proof = read(ROOT/'research/mem0_regroup_conv8_eof_root_close_offline_proof_2026-09-29.json')
        raw = bound(proof['original_raw']); original = copy.deepcopy(raw)
        parsed, actual = audit.independent_parse(raw['response'], 200)
        self.assertEqual(parsed['memory'], proof['memory_records_unmodified'])
        self.assertEqual(actual['derived_text_sha256'], proof['derivation']['derived_response_text_sha256'])
        self.assertEqual(actual['appended_utf8_hex'], '7d')
        self.assertEqual(raw, original)

    def test_malformed_nonroot_and_invalid_records_refused(self):
        for text in ['{"memory":[{"text":"x"}', '{"memory":[{"text":"x}',
                     '{"memory":[{"text":"x","text":"y"}]', '{"other":1,"memory":[]',
                     '{"memory":[],"other":1', '{"memory":[{"text":12}]',
                     '{"memory":[null]', '{"memory":[{"text":""}]', '{"memory":[],']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.independent_parse(response(text), 200)

    def test_incomplete_or_non200_generation_refused(self):
        raw = response('{"memory":[]')
        with self.assertRaises(ValueError): audit.independent_parse(raw, 500)
        raw['candidates'][0]['finishReason'] = 'MAX_TOKENS'
        with self.assertRaises(ValueError): audit.independent_parse(raw, 200)
        raw = response('{"memory":[]'); raw['candidates'] *= 2
        with self.assertRaises(ValueError): audit.independent_parse(raw, 200)

    def test_terminal_refuses_before_any_outcome_or_database_read(self):
        for status, count in [('in_progress', 28), ('failed', 29), ('complete', 28)]:
            reads = []
            def reader(path):
                reads.append(Path(path).name)
                return {'status':status, 'completed_groups':count, 'expected_groups':29}
            def forbidden(*args): raise AssertionError('Preflight touched additional artifact')
            with self.assertRaises(ValueError):
                audit.terminal_preflight(ROOT/'unused', reader, forbidden, forbidden, digest)
            self.assertEqual(reads, ['write_status.json'])

    def fixture(self, root):
        folder = root/'new/conv_8/dated_merged'; original = root/'old'
        gid = 'global_event_0000'
        signature = {'provider':'gemini', 'model':'gemini-2.5-flash', 'operation':'generateContent', 'body':{}}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        raw = {'request':signature, 'response':response('{"memory":[]}'), 'stage':'mem0_extract'}
        rawpath = folder/'native/api_cache'/(key+'.json'); write(rawpath, raw)
        oldraw = original/'conv_8/dated_merged/native/api_cache'/(key+'.json'); write(oldraw, raw)
        event = {'request_hash':key, 'stage':'mem0_extract', 'cache_hit':False, 'status':200,
                 'provider':'gemini', 'model':'gemini-2.5-flash', 'operation':'generateContent', 'usage':{},
                 'context':{'conversation':8, 'method':'dated_merged', 'group_id':gid}}
        ledgerpath = original/'conv_8/dated_merged/native/api_calls.jsonl'
        ledgerpath.write_text(json.dumps(event)+'\n', encoding='utf8'); blob=ledgerpath.read_bytes()
        evidence = {'path':str(ledgerpath.resolve()), 'prefix_bytes':len(blob), 'prefix_sha256':hashlib.sha256(blob).hexdigest(),
                    'physical_event_line':1, 'physical_event_sha256':digest(event), 'physical_event':event}
        pin = {'group_id':gid, 'sequence':0, 'raw':pointer(oldraw), 'signature':signature}
        binding = {'source_run':str(original), 'pins':[pin]}
        parsed, proof = audit.independent_parse(raw['response'],200)
        journal = {'group_id':gid, 'sequence':0, 'llm_calls':[{'response':json.dumps(parsed), 'parsed_memory':[]}]}
        record = {'schema':audit.SCHEMA, 'group_id':gid, 'request_hash':key, 'original_raw':pointer(rawpath),
                  'derived_response_persisted_to_cache':False, 'source_pin':pin['raw'],
                  'compat_source_sha256':sha(ROOT/'jev_mem/writer_eof_compat_v1.py'),
                  'physical_ledger':evidence, 'parse_proof':proof}
        rp=folder/'native/eof_parse_lineage'/(gid+'.json'); write(rp,record,True)
        current = dict(event, cache_hit=True)
        args=(folder,journal,rawpath,raw,binding,read,bound,verify,sha,digest,[current])
        return args, rp, record

    def test_lineage_matches_real_physical_evidence_and_rejects_pin_change(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = self.fixture(Path(directory))
            parsed, _ = audit.check_lineage(*args)
            self.assertEqual(parsed, {'memory':[]})
            args[4]['pins'][0]['signature'] = {'changed':True}
            with self.assertRaises(ValueError): audit.check_lineage(*args)

    def test_lineage_rejects_redispatch_and_tampered_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rp, record = self.fixture(Path(directory))
            args[-1][0]['cache_hit']=False
            with self.assertRaises(ValueError): audit.check_lineage(*args)
            args[-1][0]['cache_hit']=True
            record['physical_ledger']['prefix_sha256']='0'*64; write(rp,record,True)
            with self.assertRaises(ValueError): audit.check_lineage(*args)

    def test_lineage_rejects_false_parse_proof_and_changed_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            args, rp, record = self.fixture(Path(directory))
            args[1]['llm_calls'][0]['parsed_memory']=[{'text':'added'}]
            with self.assertRaises(ValueError): audit.check_lineage(*args)
            args[1]['llm_calls'][0]['parsed_memory']=[]
            record['parse_proof']['original_text_sha256']='0'*64; write(rp,record,True)
            with self.assertRaises(ValueError): audit.check_lineage(*args)


if __name__ == '__main__': unittest.main()
