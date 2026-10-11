import json
import tempfile
import unittest
from pathlib import Path
from tools.snapshot_write_omission_r07 import snapshot_omission

class NativeProgressTests(unittest.TestCase):
    def test_nested_native_additions_and_terminal_state_are_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);bank=run/'materialization/conv_3/jev'
            (bank/'additions').mkdir(parents=True)
            (bank/'additions/0000.json').write_text(json.dumps({'status':'added','candidate':{'candidate_id':'c3_s001_m00'}}),encoding='utf-8')
            (bank/'materialization_outcome.json').write_text(json.dumps({'status':'complete','pid':123}),encoding='utf-8')
            result=snapshot_omission(run)
            self.assertEqual(result['read_errors'],[])
            progress=result['native_banks'][str(bank.relative_to(run))]
            self.assertEqual(progress['additions'],1)
            self.assertEqual(progress['statuses'],{'added':1})
            self.assertEqual(progress['last_candidate_id'],'c3_s001_m00')
            self.assertEqual(result['native_stage_files'][str((bank/'materialization_outcome.json').relative_to(run))]['state']['status'],'complete')

if __name__=='__main__':unittest.main()
