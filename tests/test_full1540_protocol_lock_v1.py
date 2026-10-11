"""Offline checks for the user's immutable full1540 anchor."""
import copy
import unittest
from unittest.mock import patch
from tools import freeze_sdk20_full1540_v1 as lock


class Full1540LockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value=lock.verify_lock()

    def test_real_anchor_and_cohort(self):
        v=self.value
        self.assertEqual(v['baseline']['reader_valid'],1540)
        self.assertEqual(v['baseline']['judge_valid'],1540)
        self.assertAlmostEqual(v['baseline']['official_f1_percent'],54.6095082528163)
        self.assertAlmostEqual(v['baseline']['judge_percent'],100*1017/1540)
        q=v['dataset']['questions']
        self.assertEqual(len(q),1540)
        self.assertEqual(len({r['query_id'] for r in q}),1540)
        self.assertEqual({r['category'] for r in q},{1,2,3,4})
        self.assertEqual(v['generation']['json_output'],False)
        self.assertEqual(v['retrieval']['top_k'],20)

    def test_shared_setting_drift_rejected(self):
        original={k:v for k,v in self.value.items() if k not in ('created_utc','content_sha256')}
        for section,key,new in [('dataset','planned_per_arm',400),('generation','temperature',0.2),
                                ('models','reader','another-model'),('retrieval','top_k',60)]:
            changed=copy.deepcopy(self.value);changed[section][key]=new
            with self.subTest(section=section,key=key),patch.object(lock,'verify',return_value=changed),patch.object(lock,'contract',return_value=original):
                with self.assertRaisesRegex(ValueError,'protocol drift'):lock.verify_lock()

    def test_freeze_never_overwrites(self):
        with self.assertRaises(FileExistsError):lock.freeze()


if __name__=='__main__':unittest.main()
