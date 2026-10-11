import unittest
from jev_mem.evaluation import answer_and_judge


class API:
    def __init__(self, prediction=None, judgment=None):
        self.responses = iter([prediction or {"answer":"yes","used_fact_ids":["f1"]},
                               judgment if judgment is not None else {"correct":True,"completeness":1,"reason":"match"}])
    def generate(self,*args,**kwargs):
        return next(self.responses)


class EvaluationTests(unittest.TestCase):
    def test_budget_and_duplicate_rejection_before_api(self):
        for facts in ([{"id":"f","text":"x"}]*2,[{"id":str(i),"text":"x"} for i in range(21)]):
            with self.assertRaises(ValueError):
                answer_and_judge(None,"q","yes",facts,reference_date="2023-01-01")

    def test_unknown_citation_and_invalid_judge_rejected(self):
        cases=[API(prediction={"answer":"yes","used_fact_ids":["not_visible"]}),
               API(prediction={"answer":"yes","used_fact_ids":[{}]}),
               API(judgment={"correct":"true","completeness":1,"reason":"x"}),
               API(judgment={"correct":True,"completeness":float("nan"),"reason":"x"})]
        for api in cases:
            with self.assertRaises(ValueError):
                answer_and_judge(api,"q","yes",[{"id":"f","text":"yes"}],reference_date="2023-01-01")

    def test_success_does_not_assert_source_recall(self):
        result=answer_and_judge(API(),"q","yes",[{"id":"f","text":"yes"}],reference_date="2023-01-01")
        self.assertEqual(result["token_f1"],1)
        self.assertEqual(result["prediction"]["used_fact_ids"],["f"])
        self.assertIsNone(result["source_evidence_metrics"])


if __name__ == "__main__":
    unittest.main()
