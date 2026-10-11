"""Independent continuation checks: real source/raw/MiniLM, no live provider."""
import copy
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from tools import audit_amem_format_failure_v2 as admission
from tools import run_five_baselines_format_v2 as wrapper
from tools import baselines_amem_v1 as v1
from tools import baselines_amem_v2 as v2
from tools.baselines_common_v1 import BaselineAPI, BaselineFatal, read, sha


class OfflinePrefixComplete(BaseException):
    """Test boundary before the third note's first model request."""


class IndependentContinuationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.new = self.temp / "amem_format_v2"
        self.bound = wrapper.proof()
        self.stack.enter_context(patch.object(wrapper, "NEW", self.new))
        self.stack.enter_context(patch.object(wrapper, "_proof", self.bound))
        self.stack.enter_context(patch("jev_mem.providers.load_credentials", side_effect=AssertionError("no credentials")))
        self.stack.enter_context(patch("requests.sessions.Session.request", side_effect=AssertionError("no network")))
        self.stack.enter_context(patch.dict("os.environ", {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}))

    def tearDown(self):
        self.stack.close()

    def api(self, ci=0, cls=None):
        return (cls or wrapper.PrefixAPI)(self.new / "work/api_method", keys={"GEMINI_API_KEY": "independent-offline-only"},
            context={"method": "amem", "conversation": ci, "phase": "build"})

    def send(self, api, key):
        record = self.bound["records"][key]
        signature = read(record["raw_path"])["request"]
        return api.request(signature["provider"], signature["model"], signature["body"], stage=record["stage"])

    def test_recompute_exact_actual_prefix_admission_without_provider(self):
        actual = admission.audit()
        self.assertEqual({k: v for k, v in actual.items() if k != "created_utc"},
                         {k: v for k, v in self.bound.items() if k != "created_utc"})
        self.assertEqual((actual["physical_http200"], actual["known_representation_failures"],
                          actual["unknown"], actual["qa_calls"]), (40, 10, 0, 0))

    def test_missing_original_raw_is_fatal_before_super_dispatch(self):
        # An absent temporary OLD is the post-proof missing-source counterexample;
        # the frozen actual OLD namespace is never modified.
        with patch.object(wrapper, "OLD", self.temp / "missing_original"), \
                patch.object(BaselineAPI, "request", side_effect=AssertionError("must not dispatch")) as dispatch:
            api = self.api()
            with self.assertRaises(BaselineFatal):
                self.send(api, self.bound["prefixes"]["0"][0])
            dispatch.assert_not_called()
            self.assertEqual(api.prefix_position, 0)

    def test_amem_build_in_wrong_namespace_cannot_fall_back_to_new_requests(self):
        with self.assertRaises(BaselineFatal):
            wrapper.PrefixAPI(self.temp / "wrong_namespace/api_method", keys={"GEMINI_API_KEY": "offline-only"},
                context={"method": "amem", "conversation": 0, "phase": "build"})

    def test_proof_changed_after_prepare_rejected_by_manifest_binding(self):
        original_proof = wrapper.PROOF
        local_proof = self.temp / "proof.json"
        local_proof.write_bytes(original_proof.read_bytes())
        review = self.temp / "review.json"
        review.write_text(json.dumps({"prefix_proof_sha256": sha(local_proof)}), encoding="utf-8")
        prepared = {"review": {"path": str(review)}, "technical_continuation": {
            "original_manifest_sha256": wrapper.MANIFEST_SHA, "original_run": str(wrapper.OLD),
            "prefix_proof": {"path": str(local_proof), "sha256": sha(local_proof)}}}
        with patch.object(wrapper, "PROOF", local_proof), patch.object(wrapper, "_manifest", return_value=prepared):
            self.assertEqual(wrapper.manifest("amem"), prepared)
            local_proof.write_bytes(local_proof.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "proof binding"):
                wrapper.manifest("amem")

    def test_unresolved_local_intent_cannot_redispatch_after_prefix(self):
        from jev_mem.gemini25_ablation_provider_v1 import generation_body
        from jev_mem.sdk20_cached_api import digest
        api = self.api()
        for key in self.bound["prefixes"]["0"]:
            self.send(api, key)
        body = generation_body("independent unknown request", system="unmodified fake system", json_output=True, max_output=16384)
        signature = {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent", "body": body}
        key = digest(signature)
        intent = api.run_dir / "request_intents" / (key + ".json")
        intent.parent.mkdir(parents=True, exist_ok=True)
        intent.write_text(json.dumps({"request_hash": key, "signature": signature}), encoding="utf-8")
        with self.assertRaisesRegex(BaselineFatal, "unresolved logical request"):
            api.request("gemini", "gemini-2.5-flash", body, stage="amem_analysis")
        self.assertFalse((api.cache_dir / (key + ".json")).exists())
        self.assertEqual(api.prefix_position, 4)

    def test_duplicate_controller_run_does_not_change_terminal_status(self):
        self.new.mkdir(parents=True)
        marker = self.new / "execution_intent.json"
        marker.write_text('{"already_started":true}', encoding="utf-8")
        status = self.new / "status.json"
        status.write_text('{"stage":"original_terminal"}', encoding="utf-8")
        before = (marker.read_bytes(), status.read_bytes())
        with patch("sys.argv", ["wrapper", "run", "--method", "amem"]), \
                patch.object(wrapper, "install"), patch.object(wrapper.base, "paths", return_value=self.new), \
                patch.object(wrapper.base, "run", side_effect=AssertionError("no controller reentry")) as run:
            with self.assertRaises(FileExistsError):
                wrapper.main()
            run.assert_not_called()
        self.assertEqual(before, (marker.read_bytes(), status.read_bytes()))

    def test_all_full_source_prefixes_rebuild_with_actual_local_minilm(self):
        """Reconstruct from all source turns; stop before any unrecorded request."""
        data_path = wrapper.ROOT / "data/locomo10.json"
        self.assertEqual(sha(data_path), wrapper.base.DATA_SHA)
        data = read(data_path)
        initial_states = {}
        original_snapshot = v2._runtime._snapshot
        def snapshot(*args, **kwargs):
            state = original_snapshot(*args, **kwargs)
            if state["completed_notes"] == 1:
                initial_states[Path(state["work_dir"]).name] = copy.deepcopy(state)
            return state
        class OnlyAdmittedPrefix(wrapper.PrefixAPI):
            def generate(self, *args, **kwargs):
                if self.prefix_position == 4:
                    raise OfflinePrefixComplete()
                return super().generate(*args, **kwargs)
        def reconstruct(ci):
            api = self.api(ci, OnlyAdmittedPrefix)
            self.assertTrue(api.prefix_admitted, {"context": api.context, "run_dir": str(api.run_dir),
                                                  "expected_root": str(wrapper.NEW)})
            work = self.new / "work/native" / f"c{ci}"
            source = wrapper.base.conversation_only(data[ci])
            with self.assertRaises(OfflinePrefixComplete):
                v2.build(source, work, api)
            checkpoint = read(work / "checkpoint.json")
            original = read(wrapper.OLD / "work/native" / f"c{ci}/checkpoint.json")
            self.assertEqual(checkpoint["completed_notes"], 2)
            self.assertEqual(checkpoint["input_sha256"], original["input_sha256"])
            self.assertEqual(len(api.events), 4)
            self.assertTrue(all(e["source"] == "inherited_exact_request" for e in api.events))
            self.assertEqual([e["request_hash"] for e in api.events], self.bound["prefixes"][str(ci)])
            for field in ("notes", "provenance", "corpus", "evo_cnt", "memory_sha256", "embeddings_sha256"):
                self.assertEqual(initial_states[f"c{ci}"][field], original[field])
            converted = read(work / "calls/note_000002_amem_evolution.json")
            saved = read(wrapper.OLD / "work/native" / f"c{ci}/calls/note_000002_amem_evolution.json")
            self.assertEqual(converted["request"], saved["request"])
            self.assertEqual(converted["raw_response"], saved["raw_response"])
            self.assertEqual(converted["normalization"]["actual_neighbor_indices"], [0])
            self.assertEqual(converted["normalization"]["changes"], [{"json_pointer": "/suggested_connections/0",
                "from": "0", "from_type": "string", "to": 0, "to_type": "integer"}])
            derived = converted["parsed_response"]
            self.assertEqual(derived["suggested_connections"], [0])
            self.assertEqual({k: v for k, v in derived.items() if k != "suggested_connections"},
                             {k: v for k, v in saved["raw_response"].items() if k != "suggested_connections"})
            return {"conversation": ci, "completed_notes": 2, "exact_raws_reused": 4}
        with patch.object(v2._runtime, "_snapshot", side_effect=snapshot):
            with ThreadPoolExecutor(max_workers=10) as pool:
                rebuilt = list(pool.map(reconstruct, range(10)))
        self.assertEqual(sum(r["completed_notes"] for r in rebuilt), 20)
        self.assertIsNotNone(v1._MODEL)
        audit = wrapper.audit_requests(self.new)
        self.assertEqual(audit["errors"], [])
        self.assertEqual(audit["inherited_unique_prefix_requests"], 40)
        self.assertEqual(list((self.new / "work/api_method/request_intents").glob("*.json")), [])
        self.assertEqual(wrapper.proof()["original_inventory"], self.bound["original_inventory"])


if __name__ == "__main__":
    unittest.main()
