import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('snapshot_phase61', ROOT/'research/gemini38_retest_2026-10-03/snapshot_retrieval_phase_61.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def save(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row), encoding='utf8')


class SnapshotTests(unittest.TestCase):
    def test_fact_counts_failures_pending_and_final_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); r = run/'retrieval'; c = r/'conv_0'; bank = run/'bank.json'
            save(bank, {'facts': [{'id':'a'}, {'id':'b'}, {'id':'c'}]})
            save(r/'run_manifest.json', {'schema':'fixture','protocol_sha256':'p','questions':2,
                'bank_arms':['session','source_local_context','boundary_multiscale','boundary_neighbor'],
                'banks':[{'conversation':0,'arm':'session','bank':{'path':str(bank)}}]})
            for fid, status in [('a','complete'), ('b','failed')]:
                save(c/'tags'/f'{fid}.json', {'fact_id':fid,'conversation':0,'protocol_sha256':'p','status':status,'error_type':'Example' if status=='failed' else None})
                save(c/'tag_intents'/f'{fid}.json', {'pid':123})
            save(c/'tag_intents/c.json', {'pid':123})
            save(c/'retrieval_status.json', {'completed':1,'total':8,'terminal':False})
            save(c/'retrieval_outcome.json', {'status':'complete','queries':8,'protocol_sha256':'p'})
            save(c/'session/selections/c0_q1.json', {'reference':'DO_NOT_EMIT_GOLD'})
            out = s.phase_snapshot(run); first = out['conversations'][0]
            self.assertEqual(out['totals']['tag_statuses'], {'complete':1,'failed':1})
            self.assertEqual(first['tag_intents_without_record_ids'], ['c'])
            self.assertEqual(first['tags']['missing_record_ids'], ['c'])
            self.assertEqual(first['states']['retrieval_status.json']['terminal'], False)
            self.assertEqual(first['states']['retrieval_outcome.json']['status'], 'complete')
            self.assertEqual(out['totals']['selection_files'], 1)
            self.assertNotIn('DO_NOT_EMIT_GOLD', json.dumps(out))
            self.assertEqual(out['read_errors'], [])

    def test_complete_line_api_counts_and_error_visibility(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d); api = run/'retrieval/conv_0/tag_api'; api.mkdir(parents=True)
            rows = [{'status':429,'attempt':1,'request_hash':'same','context':{'conversation':0}},
                    {'status':200,'attempt':2,'request_hash':'same','context':{'conversation':0}},
                    {'status':200,'cache_hit':True,'context':{'conversation':0}}]
            (api/'api_calls.jsonl').write_bytes((''.join(json.dumps(x)+'\n' for x in rows)+'{"unfinished"').encode())
            bad = run/'retrieval/conv_0/tags/bad.json'; bad.parent.mkdir(); bad.write_bytes(b'{')
            out = s.phase_snapshot(run)
            self.assertEqual(out['totals']['api_counts'], {'physical':2,'http_429':1,'http_200':1,'cache':1})
            self.assertGreater(out['ledgers'][0]['bytes'], out['ledgers'][0]['complete_bytes'])
            self.assertEqual(len(out['read_errors']), 1)

    def test_delta_growth_regression_and_missing(self):
        before = {'ledgers':[{'path':'a','bytes':100,'complete_bytes':90,'counts':{'physical':2}},
                             {'path':'missing','bytes':20,'complete_bytes':20,'counts':{}}],
                  'totals':{'tag_statuses':{'complete':1},'selection_files':3}}
        now = {'ledgers':[{'path':'a','bytes':150,'complete_bytes':140,'counts':{'physical':3}},
                          {'path':'new','bytes':8,'complete_bytes':8,'counts':{'cache':1}}],
               'totals':{'tag_statuses':{'complete':2,'failed':1},'selection_files':5}}
        delta = s.ledger_deltas(now, before)
        self.assertEqual(delta['ledgers'][0]['counts_added']['physical'],1)
        self.assertEqual(delta['selection_files_added'],2)
        self.assertEqual(delta['previous_ledgers_now_missing'], ['missing'])
        now['ledgers'][0].update(bytes=80,counts={'physical':1})
        self.assertTrue(s.ledger_deltas(now,before)['ledgers'][0]['shrank_or_counts_regressed'])


if __name__ == '__main__': unittest.main()
