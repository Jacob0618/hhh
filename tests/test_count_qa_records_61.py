import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('count_qa61',ROOT/'research/gemini38_retest_2026-10-03/count_qa_records_61.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class Counts(unittest.TestCase):
    def write(self,root,stage,name,status):
        p=Path(root)/'conv_0'/stage/'baseline'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'stage':stage,'conversation':0,'method':'baseline','query_id':Path(name).stem,'status':status}))

    def test_intent_and_similar_names_excluded(self):
        with tempfile.TemporaryDirectory()as d:
            for name in ('c0_q1.intent.json','c0_q1.json.tmp.json','c0_q1_extra.json'):
                self.write(d,'reader',name,'complete')
            self.write(d,'reader','c0_q1.json','complete')
            out=c.snapshot(d)
            self.assertEqual(out['stages']['reader']['matched_record_files'],1)
            self.assertEqual(out['stages']['reader']['ignored_other_json_files'],3)
            self.assertEqual(out['stages']['reader']['slots_without_readable_record'],10779)

    def test_statuses_and_fixed_denominator_not_success_by_file_count(self):
        with tempfile.TemporaryDirectory()as d:
            for i,status in enumerate(('complete','failed','not_run')):self.write(d,'judge',f'c0_q{i}.json',status)
            out=c.snapshot(d)
            self.assertEqual(out['stages']['judge']['by_status'],{'complete':1,'failed':1,'not_run':1})
            self.assertEqual(out['stages']['judge']['expected_slots'],10780)
            self.assertEqual(out['stages']['judge']['by_conversation']['conv_0']['failed'],1)
            self.assertEqual(out['stages']['judge']['by_method']['baseline']['complete'],1)
            self.assertIsNone(out['health_verdict']);self.assertFalse(out['transactional'])

    def test_delta_and_denominator_guard(self):
        with tempfile.TemporaryDirectory()as d:
            before=c.snapshot(d);self.write(d,'qa','c0_q0.json','failed');after=c.snapshot(d)
            self.assertEqual(c.delta(after,before)['qa']['status_counts_delta'],{'failed':1})
            after['stages']['qa']['expected_slots']=1
            with self.assertRaises(ValueError):c.delta(after,before)


if __name__=='__main__':unittest.main()
