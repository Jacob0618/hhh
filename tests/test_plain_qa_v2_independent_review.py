"""Independent process binding and complete-denominator checks; zero model calls."""
import multiprocessing
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

from tools import run_mem0_regroup_plain_qa_v2 as runner
from tools import mem0_regroup_plain_qa_sources_v2 as sources
from tools import summarize_mem0_regroup_plain_qa_v1 as summary


def _mock_base_worker(run_dir, ci):
    if runner.frozen.sources is not sources or runner.frozen.worker is not runner.worker:
        raise AssertionError("Windows child did not install v2 source binding")
    if not set(runner.EXTRA_CODE) <= set(runner.frozen.CODE):
        raise AssertionError("Child manifest code set omitted v2 validators")
    return {"conversation": ci, "source_module": runner.frozen.sources.__name__,
            "worker_module": runner.worker.__module__, "model_calls": 0}


def _initialize_offline_spawn():
    # Patches are installed inside the newly spawned process, not inherited
    # from the parent. The actual top-level v2 worker is still submitted.
    runner.manifest = lambda *args, **kwargs: {}
    runner._BASE_WORKER = _mock_base_worker
    import requests
    from jev_mem import providers
    def forbidden(*args, **kwargs):
        raise AssertionError("No API or credentials in independent spawn probe")
    requests.Session.request = forbidden
    providers.load_credentials = forbidden


class IndependentPlainQAReview(unittest.TestCase):
    def test_spawned_process_installs_named_v2_worker_binding(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_offline_spawn) as pool:
            result = pool.submit(runner.worker, "unused_mock_namespace", 8).result(timeout=45)
        self.assertEqual(result["source_module"], sources.__name__)
        self.assertEqual(result["worker_module"], runner.__name__)
        self.assertEqual(result["model_calls"], 0)

    def test_all_3080_missing_slots_keep_separate_full_denominators(self):
        cohort, _ = sources.cohort()
        slots = [{"query_id": q["id"], "conversation": ci, "category": q["category"], "method": arm,
                  **{stage + "_status": "missing" for stage in ("selection", "reader", "judge", "qa")},
                  "scores": {metric: None for metric in summary.METRICS}, "errors": []}
                 for ci in range(10) for arm in summary.ARMS for q in cohort[str(ci)]]
        self.assertEqual(len(slots), 3080)
        result = summary.aggregate(slots, bootstrap_samples=2)
        for arm in summary.ARMS:
            value = result["arms"][arm]
            self.assertEqual(value["planned_denominator"], 1540)
            self.assertEqual(value["statuses"]["reader"], {"missing": 1540})
            for metric in summary.METRICS:
                scored = value["metrics"][metric]
                self.assertEqual(scored["valid_denominator"], 0)
                self.assertIsNone(scored["complete_only_mean"])
                self.assertIsNone(scored["full_cohort_observed_mean"])
                self.assertTrue(scored["zero_imputed_sensitivity"]["is_imputation_not_observed_score"])
        for metric in summary.METRICS:
            self.assertFalse(result["paired_merged_minus_unmerged"][metric]["bootstrap"]["eligible"])


if __name__ == "__main__":
    unittest.main()
