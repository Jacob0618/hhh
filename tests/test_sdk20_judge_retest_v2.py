"""Reuse the v1 orchestration tests against the versioned numeric-reference guard."""
import copy
import hashlib
import unittest
from unittest.mock import patch

from tests import test_sdk20_judge_retest_v1 as base
from tools import run_sdk20_judge_retest_v1 as v1
from tools import run_sdk20_judge_retest_v2 as v2


class DriverV2Tests(base.DriverTests):
    def setUp(self):
        p=patch.object(base,'d',v2);p.start();self.addCleanup(p.stop)
        super().setUp()

    def test_integer_gold_kept_as_integer_and_original_prompt_unchanged(self):
        for n in (2022,2,3):
            j=base.job();j['reference']=n
            before=copy.deepcopy(j)
            with self.assertRaisesRegex(ValueError,'reference strings'):v1.check_jobs([j])
            v2.check_jobs([j]);self.assertEqual(j,before);self.assertIs(type(j['reference']),int)
            self.assertEqual(v2.signature(j),v1.signature(j))
            run,m=self.phase([j],name='int_'+str(n))
            _,transport=self.execute(run,['[[yes]]'])
            self.assertEqual(transport.calls,[v1.signature(j)['body']])
            self.assertEqual(self.artifact(run,j,'strict')['status'],'complete')

    def test_reference_guard_remains_narrow_and_rejects_bool_and_containers(self):
        for ref in (True,False,None,2.0,{'answer':2},[2]):
            j=base.job();j['reference']=ref
            with self.assertRaises(ValueError):v2.check_jobs([j])

    def test_frozen_v1_is_unchanged(self):
        expected='5d11105b9150620509abaca3d4963c21da24f1c4a94cbb877ab873fd248e3fc4'
        self.assertEqual(hashlib.sha256(v1.Path(v1.__file__).read_bytes()).hexdigest(),expected)


if __name__=='__main__':unittest.main()
