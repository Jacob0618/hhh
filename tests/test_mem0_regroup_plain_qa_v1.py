"""Synthetic source/QA tests. Never open original stores or dispatch models."""
import copy
from contextlib import contextmanager, nullcontext
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import mem0_regroup_plain_qa_sources_v1 as src
from tools import run_mem0_regroup_plain_qa_v1 as run
from tools.run_sdk20_modules import save_new, verify, sha
from tools.run_pilot400_experiments_v2 import publish_json
from jev_mem.providers import ModelAPI
from jev_mem.plain_text_api_v1 import check_plain_text_request
from jev_mem.evaluation_plain_text_v1 import READER_STAGE, JUDGE_STAGE


class FakeAPI:
    generate = ModelAPI.generate

    def __init__(self, *args, shared=None, context=None, **kwargs):
        self.shared = shared if shared is not None else {"outputs": {READER_STAGE: "Paris", JUDGE_STAGE: "[[yes]]"}, "calls": []}
        self.context, self.events = context or {}, []

    def scoped(self, **context):
        return type(self)(shared=self.shared, context={**self.context, **context})

    def _redact(self, value):
        return str(value)

    def request(self, provider, model, body, *, stage, operation="generateContent"):
        check_plain_text_request(provider, model, body, stage=stage, operation=operation)
        if stage == JUDGE_STAGE and "before_judge" in self.shared:
            self.shared["before_judge"]()
        self.shared["calls"].append({"stage": stage, "body": copy.deepcopy(body), "context": self.context})
        self.events.append({"stage": stage, "mock": True})
        out = self.shared["outputs"][stage]
        if isinstance(out, Exception):
            raise out
        return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": out}]}}]}


class PlainQATests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dated_plain_qa_mock_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.q = {"id": "c0_q0", "question": "Where?", "reference": "Paris", "category": 1}
        self.facts = [{"id": "f" + str(i), "text": "Paris " + str(i), "observation_date": None} for i in range(20)]
        self.bank = {"facts": self.facts, "user_id": "mock-user"}
        self.m = {"protocol_sha256": "mock-protocol", "reference_dates": {"0": "2023-06-30"}}
        for target in ("requests.Session.request", "requests.Session.post", "socket.create_connection",
                       "jev_mem.providers.load_credentials"):
            blocked = patch(target, side_effect=AssertionError("Offline test forbids network or credentials"))
            blocked.start()
            self.addCleanup(blocked.stop)

    def selection(self, *, status="complete", q=None):
        q = q or self.q
        arm = run.ARMS[0]
        path = self.root / "conv_0/selection" / arm / (q["id"] + ".json")
        save_new(path, {**run.identity(0, arm, q, self.m), "status": status, "bank_sha256": "mock-bank",
                       "selected_ids": [f["id"] for f in self.facts] if status == "complete" else [],
                       "error": None if status == "complete" else {"type": "MockRetrievalFailure"}})
        return path

    def qa(self, api):
        return run.qa_one(self.root, 0, run.ARMS[0], self.q, self.m, self.bank, "mock-bank", api,
                          lambda answer, ref, category: .75)

    def test_reader_and_both_f1_metrics_are_persisted_before_judge(self):
        self.selection()
        rp = self.root / "conv_0/reader" / run.ARMS[0] / "c0_q0.json"
        api = FakeAPI()
        def before_judge():
            row = verify(rp)
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["result"]["metrics"], {"official_locomo_f1": .75, "custom_token_f1": 1., "custom_exact_match": 1.})
            self.assertEqual(row["reference_date"], "2023-06-30")
        api.shared["before_judge"] = before_judge
        final = self.qa(api)
        self.assertEqual(final["status"], "complete")
        self.assertTrue(final["judge_correct"])
        payload = json.loads(api.shared["calls"][0]["body"]["contents"][0]["parts"][0]["text"])
        self.assertEqual(payload["conversation_reference_date"], "2023-06-30")
        self.assertTrue(all(f["observation_date"] is None for f in payload["facts"]))

    def test_invalid_judge_retains_reader_metrics_and_cannot_become_false(self):
        self.selection()
        api = FakeAPI()
        api.shared["outputs"][JUDGE_STAGE] = "yes because Paris"
        final = self.qa(api)
        self.assertEqual((final["status"], final["reader_status"], final["judge_status"]), ("failed", "complete", "failed"))
        self.assertEqual(final["reader_metrics"]["official_locomo_f1"], .75)
        self.assertIsNone(final["judge_correct"])
        judge = verify(final["judge"]["path"])
        self.assertIsNone(judge["result"])
        self.assertEqual(judge["raw_invalid_label"], "yes because Paris")
        self.assertEqual(self.qa(api), final)
        self.assertEqual(len(api.shared["calls"]), 2)

    def test_valid_no_is_terminal_and_no_correctness_retry(self):
        self.selection()
        api = FakeAPI()
        api.shared["outputs"][JUDGE_STAGE] = "[[no]]"
        final = self.qa(api)
        self.assertEqual(final["status"], "complete")
        self.assertIs(final["judge_correct"], False)
        self.qa(api)
        self.assertEqual(len(api.shared["calls"]), 2)

    def test_reader_failure_means_no_judge_dispatch(self):
        self.selection()
        api = FakeAPI()
        api.shared["outputs"][READER_STAGE] = RuntimeError("simulated transport failure")
        final = self.qa(api)
        self.assertEqual((final["reader_status"], final["judge_status"]), ("failed", "not_run"))
        self.assertIsNone(final["reader_metrics"])
        self.assertIsNone(final["judge_correct"])
        self.assertEqual(len(api.shared["calls"]), 1)
        self.qa(api)
        self.assertEqual(len(api.shared["calls"]), 1)

    def test_selection_failure_has_two_explicit_no_dispatch_records(self):
        self.selection(status="failed")
        api = FakeAPI()
        final = self.qa(api)
        self.assertEqual((final["reader_status"], final["judge_status"]), ("not_run", "not_run"))
        self.assertEqual(final["selected_ids"], [])
        self.assertIsNone(final["reader_metrics"])
        self.assertEqual(api.shared["calls"], [])

    def test_interrupted_stage_intent_is_terminal_without_dispatch(self):
        path = self.root / "reader/q.json"
        expected = {"protocol_sha256": "p", "query_id": "q"}
        run.durable_intent(path.with_suffix(".intent.json"), expected)
        with patch("builtins.print") as unused:
            row = run.stage_once(path, expected, lambda: unused("dispatch"), FakeAPI())
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["error"]["type"], "UnresolvedStageIntent")
        unused.assert_not_called()

    def test_missing_or_four_bank_gate_rejected_before_namespace_creation(self):
        gate = self.root / "gate.json"
        save_new(gate, {"schema": "mem0-dated-regroup-development-gate-v1", "status": "passed", "banks": []})
        with patch.object(run, "ROOT", self.root), patch.object(src, "protocol_config", return_value={}), \
                patch.object(run.embedding_pack, "audit_sources") as cache_audit, \
                patch.object(run.native, "open_memory", side_effect=AssertionError("Never open any source")):
            with self.assertRaises(ValueError):
                run.prepare(self.root / "runs/new-qa", gate)
            with self.assertRaises(FileNotFoundError):
                run.prepare(self.root / "runs/new-qa", self.root / "missing.json")
            self.assertFalse((self.root / "runs/new-qa").exists())
            cache_audit.assert_not_called()

    def test_namespace_rejects_nested_source_before_reading_gate(self):
        with patch.object(run, "ROOT", self.root), patch.object(src, "protocol_config") as config:
            with self.assertRaises(ValueError):
                run.prepare(self.root / "runs/old/new-qa", self.root / "missing.json")
            config.assert_not_called()
            self.assertFalse((self.root / "runs").exists())

    def fake_native(self):
        native = self.root / "source/native"
        (native / "qdrant/collection/facts").mkdir(parents=True)
        (native / "history.sqlite").write_bytes(b"closed synthetic history")
        (native / "qdrant/collection/facts/storage.sqlite").write_bytes(b"closed synthetic store")
        (native / "qdrant/meta.json").write_text("{}")
        (native / "qdrant/.lock").write_text("unused lock")
        (native / "api_cache").mkdir()
        (native / "api_cache/not-copied.json").write_text("{}")
        (native / "api_calls.jsonl").write_text("not copied")
        return native

    def test_copy_is_byte_identical_excludes_locks_api_and_never_opens_source(self):
        native = self.fake_native()
        initial = src.inventory(native)
        source = {"native": native, "review": {"native_inventory": initial, "bank": {"path": "mock", "sha256": "mock"}},
                  "bank": {**self.bank, "fact_support": "unverified_group_input_scope"}}
        with patch.object(run.native, "open_memory", side_effect=AssertionError("No open during prepare")):
            binding = src.copy_bank(self.root / "qa", 0, run.ARMS[0], source)
        bank = verify(binding["path"])
        self.assertEqual(src.inventory(bank["native_dir"]), initial)
        self.assertEqual(src.inventory(native), initial)
        self.assertEqual(bank["facts"], self.facts)
        self.assertFalse((Path(bank["native_dir"]) / "api_calls.jsonl").exists())
        self.assertFalse((Path(bank["native_dir"]) / "qdrant/.lock").exists())
        with self.assertRaises(FileExistsError):
            src.copy_bank(self.root / "qa", 0, run.ARMS[0], source)

    def test_any_nonempty_sqlite_wal_or_journal_blocks_copy(self):
        native = self.fake_native()
        for name in ("history.sqlite-wal", "qdrant/collection/facts/storage.sqlite-wal", "qdrant/collection/facts/storage.sqlite-journal"):
            p = native / name
            p.write_bytes(b"uncheckpointed")
            with self.assertRaises(ValueError):
                src.inventory(native)
            p.unlink()

    def test_real_cohort_and_pinned_plain_config_offline(self):
        rows, dates = src.cohort()
        self.assertEqual(sum(map(len, rows.values())), 1540)
        self.assertEqual(len(dates), 10)
        config = src.protocol_config()
        self.assertFalse(config["generation"]["json_output"])

    def fake_twenty_gate(self):
        origins = [self.root / name for name in ("development", "fourth", "remaining")]
        parents = {str(p.resolve()): {"protocol_sha256": p.name, "inputs": []} for p in origins}
        audit = self.root / "synthetic-audit.json"
        publish_json(audit, {"note": "mock structural/source review only"})
        rows = []
        for ci in range(10):
            for arm in run.ARMS:
                origin = origins[1] if (ci, arm) == (1, "dated_merged") else origins[0] if ci < 2 else origins[2]
                folder = origin / f"conv_{ci}" / arm
                native = folder / "native"
                (native / "qdrant").mkdir(parents=True)
                (native / "history.sqlite").write_bytes(b"closed mock history")
                (native / "qdrant/meta.json").write_text("{}")
                inp, journal = folder / "group_inputs.json", native / "audit.json"
                save_new(inp, {"envelopes": [{}]})
                publish_json(journal, {"status": "success"})
                parents[str(origin.resolve())]["inputs"].append({"conversation": ci, "arm": arm, "input": src.pointer(inp)})
                ident = {"conversation": ci, "arm": arm, "protocol_sha256": origin.name}
                bp = folder / "bank.json"
                save_new(bp, {**ident, **self.bank, "complete": True, "source_dataset_sha256": src.SOURCE_HASH,
                              "group_inputs": src.pointer(inp), "journal_files": [src.pointer(journal)], "native_dir": str(native.resolve())})
                sp = folder / "write_status.json"
                publish_json(sp, {**ident, "status": "complete", "bank": src.pointer(bp), "completed_groups": 1,
                                  "expected_groups": 1, "facts": 20})
                op = folder / ("recovery_outcome.json" if ci < 2 else "write_outcome.json")
                save_new(op, {**ident, "status": "complete", "bank": src.pointer(bp), "write_status": src.pointer(sp)})
                rows.append({"conversation": ci, "arm": arm, "bank": src.pointer(bp), "write_status": src.pointer(sp),
                             "outcome": src.pointer(op), "native_inventory": src.inventory(native),
                             "audits": {"source_date": [src.pointer(audit)], "index": [src.pointer(audit)]}})
        resource = {"snapshot": "mock snapshot", "files_sha256": {"mock": "mock"}}
        parents[str(origins[2].resolve())]["remaining_expansion"] = {"resources": resource}
        for origin in origins:
            publish_json(origin / "run_manifest.json", parents[str(origin.resolve())])
        for row in rows:
            origin = origins[1] if (row["conversation"], row["arm"]) == (1, "dated_merged") else origins[0] if row["conversation"] < 2 else origins[2]
            row["source_manifest"] = src.pointer(origin / "run_manifest.json")
        gate = self.root / "synthetic-20-gate.json"
        save_new(gate, {"schema": src.GATE_SCHEMA, "status": "passed", "checks": {k: True for k in src.CHECKS},
                        "banks": rows, "limitations": ["Synthetic fixture, not a real reviewed bank"]})
        return origins, parents, gate

    def test_twenty_gate_accepts_exact_bindings_and_rejects_tampered_outcome(self):
        origins, parents, gate = self.fake_twenty_gate()
        with patch.object(src, "DEVELOPMENT", origins[0]), patch.object(src, "FOURTH", origins[1]), \
                patch.object(src, "REMAINING", origins[2]), patch.object(src, "source_manifests", return_value=parents), \
                patch.object(src, "validate_resources"), \
                patch.object(run.native, "open_memory", side_effect=AssertionError("No store open during admission")):
            value, banks, resource = src.admit_gate(gate)
            self.assertEqual(len(banks), 20)
            self.assertEqual(set(banks), {(c, a) for c in range(10) for a in run.ARMS})
            target = Path(value["banks"][0]["outcome"]["path"])
            target.write_text("{}")
            with self.assertRaises(ValueError):
                src.admit_gate(gate)

    def retrieval_setup(self, startup=False):
        copy_native = self.fake_native()
        bank = {**self.bank, "native_dir": str(copy_native), "source_inventory": src.inventory(copy_native)}
        queries = [self.q, {**self.q, "id": "c0_q1", "question": "Second?"}]
        calls, active = [], {}
        @contextmanager
        def capture():
            active.clear()
            yield active
        class Memory:
            vector_store = SimpleNamespace(_get_bm25_encoder=lambda: object())
            def search(self, question, **kwargs):
                calls.append((question, kwargs))
                if question == "Where?":
                    raise RuntimeError("One failed query")
                ids = [f["id"] for f in bank["facts"]]
                active.update(base20_ids=ids, eligible_ids=ids)
                return {"results": [{"id": i} for i in ids]}
        spacy = SimpleNamespace(get_nlp_full=lambda: object(), get_nlp_lemma=lambda: object())
        with patch.object(run, "get_bank", return_value=(bank, "mock-bank")), \
                patch.object(run.native, "open_memory", side_effect=RuntimeError("startup failed") if startup else None, return_value=Memory()), \
                patch.object(run.native, "close_memory"), patch.object(run, "capture_native_pool", capture), \
                patch.object(run, "Embeddings", lambda api: api), \
                patch.dict("sys.modules", {"mem0.utils": SimpleNamespace(spacy_models=spacy)}):
            run.retrieve_arm(self.root, 0, run.ARMS[0], queries, self.m, FakeAPI())
        return queries, calls

    def test_single_search_failure_does_not_drop_remaining_slots(self):
        queries, calls = self.retrieval_setup()
        states = [verify(self.root / "conv_0/selection" / run.ARMS[0] / (q["id"] + ".json"))["status"] for q in queries]
        self.assertEqual(states, ["failed", "complete"])
        for _, kw in calls:
            self.assertEqual(kw, {"top_k": 20, "filters": {"user_id": "mock-user"}, "rerank": False})
        api = FakeAPI()
        for q in queries:
            final = run.qa_one(self.root, 0, run.ARMS[0], q, self.m, self.bank, "mock-bank", api, lambda *a: 1.)
            self.assertIn(final["status"], ("failed", "complete"))
        self.assertEqual(len(list((self.root / "conv_0/qa" / run.ARMS[0]).glob("*.json"))), 2)
        self.assertEqual(len(api.shared["calls"]), 2)

    def test_startup_failure_creates_all_selection_failure_records(self):
        queries, calls = self.retrieval_setup(startup=True)
        self.assertEqual(calls, [])
        for q in queries:
            value = verify(self.root / "conv_0/selection" / run.ARMS[0] / (q["id"] + ".json"))
            self.assertEqual(value["status"], "failed")
            self.assertEqual(value["selected_ids"], [])


if __name__ == "__main__":
    unittest.main()
