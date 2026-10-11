"""Offline refinement QA wiring: no credentials or real networking permitted."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import run_regroup_refinement_qa_v1 as d
from tools import regroup_refinement_qa_sources_v1 as s
from tests.test_sdk20_judge_retest_v1 import Transport, KEYS
from tests.test_sdk20_plain_qa_v1 import job as old_job
from jev_mem.plain_text_api_v1 import PlainTextEvaluationAPI
from jev_mem.evaluation_plain_text_v1 import reader_input


def job(arm="jev_refined", qid="c0_q0", ci=0):
    return old_job(qid, arm, ci)


class RefinementQATests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="refinement_qa_offline_")
        self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.run = self.root / "main"; self.run.mkdir()
        for target in ("requests.Session.post", "requests.Session.request", "socket.create_connection", "jev_mem.providers.load_credentials"):
            p = patch(target, side_effect=AssertionError("No real credentials/network")); p.start(); self.addCleanup(p.stop)

    def m(self, jobs, phase="firstpass"):
        return {"schema": d.SCHEMA, "protocol_sha256": "mock-protocol-" + phase, "arm": "jev_refined",
                "phase_kind": phase, "phase_id": "refinement_mock_" + phase, "jobs": jobs,
                "source_manifest": {"path": str(self.root / "mock_source.json"), "sha256": "mock"}}

    def factory(self, path, ci, m):
        return PlainTextEvaluationAPI(path, keys=KEYS, cache_roots=(), retries=2,
            context={"architecture": d.plain.SCHEMA, "conversation": ci, "phase": "qa", "phase_id": m["phase_id"]})

    def execute(self, jobs, outputs, *, run=None, m=None):
        run = run or self.run; m = m or self.m(jobs)
        transport = Transport(outputs)
        with patch.object(d, "execution_manifest", return_value=m), patch("requests.Session", return_value=transport):
            out = d.qa(run, api_factory=self.factory)
        return out, transport, m

    def row(self, j, stage, run=None):
        return d.verify(d.plain.stage_files(run or self.run, j)[stage])

    def test_actual_fixed_cohort_is_400_with_all_conversations(self):
        rows, dates = s.cohort()
        self.assertEqual(sum(map(len, rows.values())), 400)
        self.assertEqual(set(rows), {str(i) for i in range(10)})
        self.assertTrue(all(rows.values())); self.assertEqual(len(dates), 10)
        self.assertEqual(s.copy_bank, d.native_qa.sources.copy_bank)

    def test_native_selection_and_reader_source_fields_are_exact(self):
        j = job(); q = {"id": j["query_id"], "question": j["question"], "reference": 2022, "category": 2}
        bank = {"facts": [dict(f, metadata="not in Reader") for f in j["selected_facts"]]}
        m = {"protocol_sha256": "x", "reference_dates": {"0": j["reference_date"]},
             "banks": {"0": {"jev_refined": {"path": "x", "sha256": "y"}}}}
        p = self.run / "conv_0/selection/jev_refined/c0_q0.json"
        selected = {**d.native_qa.identity(0, "jev_refined", q, m), "bank_sha256": "bh", "status": "complete",
                    "selected_ids": list(reversed(j["selected_ids"])), "capture": {"base20_ids": list(reversed(j["selected_ids"]))}}
        d.save_new(p, selected)
        with patch.object(d.native_qa, "get_bank", return_value=(bank, "bh")):
            frozen = d.job_from_selection(self.run, 0, "jev_refined", q, m)
        self.assertEqual(frozen["reference"], 2022)
        self.assertEqual(frozen["selected_ids"], list(reversed(j["selected_ids"])))
        self.assertEqual(frozen["selected_facts"][0], j["selected_facts"][-1])
        self.assertTrue(all(set(f) == {"id", "text", "observation_date"} for f in frozen["selected_facts"]))

    def test_no_json_and_reader_is_persisted_before_judge(self):
        j = job(); j["reference"] = 2022; m = self.m([j]); transport = Transport(["2022", "[[yes]]"])
        original = transport.post
        def post(url, **kwargs):
            if "3.5-flash-lite" in url:
                self.assertEqual(self.row(j, "reader")["result"]["metrics"]["official_locomo_f1"], 1)
            return original(url, **kwargs)
        transport.post = post
        with patch.object(d, "execution_manifest", return_value=m), patch("requests.Session", return_value=transport):
            out = d.qa(self.run, api_factory=self.factory)
        payload = json.loads(transport.calls[0]["contents"][0]["parts"][0]["text"])
        self.assertEqual(payload, reader_input(j["question"], j["selected_facts"], reference_date=j["reference_date"]))
        self.assertNotIn("reference", payload)
        for call in transport.calls:
            self.assertNotIn("responseMimeType", call["generationConfig"])
            self.assertNotIn("responseSchema", call["generationConfig"])
        self.assertEqual(out["counts"]["qa"], {"complete": 1})
        self.assertEqual(out["planned_per_arm"], 400)

    def test_format_failure_keeps_reader_f1_and_same_compat_rule(self):
        j = job(); out, _, _ = self.execute([j], ["Paris", "Explanation\n[[yes]]"])
        self.assertEqual(self.row(j, "reader")["result"]["metrics"]["official_locomo_f1"], 1)
        self.assertEqual(self.row(j, "judge")["status"], "failed")
        self.assertEqual(self.row(j, "compat")["status"], "compat_valid")
        self.assertEqual(self.row(j, "qa")["resolved_judge"]["source"], "firstpass_eligible_compat")
        self.assertEqual(out["counts"]["judge"], {"failed": 1})

    def test_all_strict_failures_recover_including_compat_valid(self):
        j = job(); out, _, m = self.execute([j], ["Paris", "Explanation\n[[yes]]"])
        candidates = d.recovery_jobs(m, out)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["reader_source"], out["results"][0]["reader"])
        retry = self.root / "recovery"; retry.mkdir()
        rm = self.m(candidates, "technical_recovery")
        _, transport, _ = self.execute(candidates, ["[[no]]"], run=retry, m=rm)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(self.row(j, "reader", retry)["result"], self.row(j, "reader")["result"])
        self.assertEqual(self.row(j, "qa", retry)["resolved_judge"], {"source": "retry_strict", "correct": False, "label": "[[no]]"})

    def test_valid_no_is_not_retried(self):
        j = job(); out, _, m = self.execute([j], ["Paris", "[[no]]"])
        self.assertEqual(d.recovery_jobs(m, out), [])

    def test_prior_stage_intent_does_not_dispatch(self):
        j = job(); m = self.m([j])
        d.native_qa.durable_intent(d.plain.stage_files(self.run, j)["reader"].with_suffix(".intent.json"), {})
        out, transport, _ = self.execute([j], [], m=m)
        self.assertEqual(transport.calls, [])
        self.assertEqual(self.row(j, "reader")["error"]["type"], "UnresolvedStageIntent")
        with self.assertRaises(ValueError):
            d.recovery_jobs(m, out)

    def test_unknown_or_local_recovery_outcomes_rejected(self):
        j = job(); out, _, m = self.execute([j], ["Paris", "[[yes]]"])
        missing = copy.deepcopy(out); missing["results"][0]["judge"] = None
        with self.assertRaisesRegex(ValueError, "Unknown"):
            d.recovery_jobs(m, missing)
        local = copy.deepcopy(out); local["outcomes"][0]["local_errors"] = [{"error": "local"}]
        with self.assertRaisesRegex(ValueError, "Local"):
            d.recovery_jobs(m, local)
        with self.assertRaisesRegex(ValueError, "Only one"):
            d.recovery_jobs({**m, "phase_kind": "technical_recovery"}, out)

    def test_check_job_rejects_wrong_arm_order_duplicates_and_extra_metadata(self):
        variants = []
        j = job(); j["method"] = "dated_merged"; variants.append(j)
        j = job(); j["selected_ids"].reverse(); variants.append(j)
        j = job(); j["selected_ids"][1] = j["selected_ids"][0]; variants.append(j)
        j = job(); j["selected_facts"][0]["gold"] = "no"; variants.append(j)
        for j in variants:
            with self.assertRaises(ValueError): d.check_job(j, "jev_refined")

    def test_failed_selection_never_calls_reader(self):
        j = job(); q = {"id": j["query_id"], "question": j["question"], "reference": j["reference"], "category": j["category"]}
        m = {"protocol_sha256": "x", "reference_dates": {"0": "2023-01-01"}}
        p = self.run / "conv_0/selection/jev_refined/c0_q0.json"
        d.save_new(p, {**d.native_qa.identity(0, "jev_refined", q, m), "bank_sha256": "bh", "status": "failed"})
        with patch.object(d.native_qa, "get_bank", return_value=({"facts": []}, "bh")):
            with self.assertRaisesRegex(ValueError, "no partial400"):
                d.job_from_selection(self.run, 0, "jev_refined", q, m)

    def test_gate_rejects_partial_arm_population_before_store_access(self):
        for value in ({"schema": s.GATE_SCHEMA, "status": "failed", "arm": "jev_refined"},
                      {"schema": s.GATE_SCHEMA, "status": "passed", "arm": "jev_refined",
                       "checks": {k: True for k in s.CHECKS}, "limitations": ["quality unknown"],
                       "source_manifest": {"path": str(self.root / "run_manifest.json"), "sha256": "fake"}, "banks": []}):
            path = self.root / ("gate" + str(len(list(self.root.glob('gate*')))) + ".json")
            d.save_new(path, value)
            with self.assertRaises((ValueError, FileNotFoundError)):
                s.admit_gate(path)

    def test_prepare_is_model_free_single_arm_400_and_embedding_only(self):
        (self.root / "runs").mkdir()
        gate_path = self.root / "gate.json"; d.save_new(gate_path, {"test": "bound gate"})
        gate = {"arm": "jev_refined", "limitations": ["mock source quality limitation"]}
        admitted = {(ci, "jev_refined"): {"mock": ci} for ci in range(10)}
        rows = {str(ci): [{"id": f"c{ci}_q{k}"} for k in range(40)] for ci in range(10)}
        dates = {str(ci): "2023-01-01" for ci in range(10)}
        def copied(run, ci, arm, bank):
            self.assertEqual(arm, "jev_refined"); self.assertEqual(bank["mock"], ci)
            p = run / f"conv_{ci}" / arm / "bank.json"; d.save_new(p, {"ci": ci})
            return s.pointer(p)
        binding = {"manifest": {"path": "embedding_only", "sha256": "exact"},
                   "cache_roots": {str(ci): [f"embedding_pack/{ci}"] for ci in range(10)}}
        with patch.object(d, "ROOT", self.root), patch.object(d, "CODE", ()), patch.object(d, "RUNTIME", ()), \
             patch.object(s, "admit_gate", return_value=(gate, admitted, {"mock": "resources"})), \
             patch.object(s, "protocol_config", return_value={"evaluation_protocol": "frozen"}), \
             patch.object(s, "cohort", return_value=(rows, dates)), patch.object(s, "copy_bank", side_effect=copied) as cp, \
             patch.object(d.embedding_pack, "audit_sources", return_value={"plan": "embedding only"}), \
             patch.object(d.embedding_pack, "create_pack", return_value=binding), \
             patch.object(d, "PlainTextEvaluationAPI", side_effect=AssertionError("No model clients during prepare")), \
             patch.object(d.importlib.metadata, "version", return_value="2.2.1"), \
             patch.object(d.importlib.metadata, "distribution") as dist:
            dist.return_value.locate_file.return_value = self.root / "no_sdk_files"
            result = d.prepare(self.root / "runs/qa", gate_path)
        self.assertEqual(result["qa_total"], 400); self.assertEqual(result["model_calls"], 0)
        self.assertEqual(cp.call_count, 10)
        m = d.read(self.root / "runs/qa/run_manifest.json")
        self.assertEqual(m["cache_roots"], binding["cache_roots"])
        self.assertEqual(m["phase_kind"], "firstpass")
        self.assertNotIn("jobs", m)


if __name__ == "__main__":
    unittest.main()
