"""Offline adapter tests: real pinned V-Mem retrieval, deterministic fake vectors."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import baselines_vmem_v1 as adapter


class FakeEncoder:
    dimension = 384

    def encode(self, texts, batch_size=64):
        import numpy as np
        texts = [texts] if isinstance(texts, str) else list(texts)
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimension
                matrix[row, index] += 1
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix

    def embed(self, text):
        return self.encode([text])[0]

    def embed_batch(self, texts):
        return self.encode(texts)


class NoModelAPI:
    def generate(self, *args, **kwargs):
        raise AssertionError("V-Mem text must not call a generation API")

    def embed(self, *args, **kwargs):
        raise AssertionError("V-Mem uses local MiniLM, not a remote embedding API")


def conversation(name="Ada", turns=4):
    return {"speaker_a": name, "speaker_b": "Bob", "session_2": [
        {"dia_id": f"D2:{i}", "speaker": name, "text": f"{name} collects item{i} and visits museum{i}.",
         "blip_caption": "A red bird." if i == 0 else "", "gold_evidence": "GOLD_SENTINEL"}
        for i in range(turns)], "session_2_date_time": "2023-05-02",
        "session_1": [{"dia_id": "D1:0", "speaker": name, "text": "Welcome to the archive."}],
        "session_1_date_time": "2023-05-01", "event_summary": "LEAK_SENTINEL"}


class VMemTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.encoder_patch = patch.object(adapter, "_ENCODER", FakeEncoder())
        self.encoder_patch.start()
        adapter._RUNTIMES.clear()

    def tearDown(self):
        self.encoder_patch.stop()
        adapter._RUNTIMES.clear()
        self.tmp.cleanup()

    def test_native_loader_parity_caption_date_and_no_gold(self):
        raw = conversation()
        state = adapter.build(raw, self.directory, NoModelAPI())
        source_file = self.directory / "comparison_input.json"
        source_file.write_text(json.dumps([{"conversation": adapter.clean_conversation(raw)}]), encoding="utf-8")
        native_rounds = adapter._SOURCE.load("load_locomo").load_locomo_rounds(path=str(source_file))
        saved = json.loads((self.directory / "rounds.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, {item.round_id: item.to_dict() for item in native_rounds})
        result = adapter.retrieve("What does Ada collect?", state, NoModelAPI())
        self.assertIn("2023-05-02", result["context"])
        self.assertIn("A red bird.", result["context"])
        self.assertNotIn("SENTINEL", json.dumps(result))
        self.assertTrue(all(Path(path).is_file() for path in state["export_paths"]))
        self.assertEqual(state["round_count"], 5)
        self.assertEqual(result["trace"]["effective_top_k"], 12)

    def test_native_aggregation_budget_and_checkpoint_reload(self):
        state = adapter.build(conversation(turns=45), self.directory, NoModelAPI())
        first = adapter.retrieve("List all activities Ada mentioned", state, NoModelAPI())
        self.assertEqual(first["trace"]["effective_top_k"], 30)
        self.assertEqual(first["trace"]["effective_window"], 0)
        self.assertGreater(len(first["evidence"]), 20)
        adapter._RUNTIMES.clear()
        second = adapter.retrieve("List all activities Ada mentioned", json.loads(json.dumps(state)), NoModelAPI())
        self.assertEqual(first, second)
        self.assertEqual(adapter.build(conversation(turns=45), self.directory, NoModelAPI()), state)
        with self.assertRaises(RuntimeError):
            adapter.build(conversation(name="Alice"), self.directory, NoModelAPI())

    def test_conversations_remain_isolated_under_parallel_build_and_query(self):
        def run(name):
            state = adapter.build(conversation(name=name), self.directory / name, NoModelAPI())
            return adapter.retrieve(f"What does {name} collect?", state, NoModelAPI())["context"]
        with ThreadPoolExecutor(max_workers=2) as pool:
            left, right = list(pool.map(run, ["Ada", "Zelda"]))
        self.assertNotIn("Zelda", left)
        self.assertNotIn("Ada", right)

    def test_sample_or_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            adapter.clean_conversation({"conversation": conversation(), "qa": []})
        raw = conversation()
        raw["session_2"][0]["dia_id"] = "D1:0"
        with self.assertRaises(ValueError):
            adapter.clean_conversation(raw)

    def test_every_native_file_and_export_is_checked_on_reuse_and_retrieve(self):
        raw = conversation()
        state = adapter.build(raw, self.directory, NoModelAPI())
        self.assertEqual(set(state["files_sha256"]), {"rounds.json", "vectors.npy", "vec_meta.json",
                         "keyword_index.json", "vmem_memory.json", "vmem_memory.txt"})
        for name in state["files_sha256"]:
            path = self.directory / name
            original = path.read_bytes()
            path.write_bytes(original + b"CORRUPTED_ARTIFACT")
            try:
                with self.subTest(artifact=name):
                    with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                        adapter.build(raw, self.directory, NoModelAPI())
                    with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                        adapter.retrieve("What did Ada collect?", state, NoModelAPI())
                    adapter._RUNTIMES.clear()
                    with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                        adapter.retrieve("What did Ada collect?", state, NoModelAPI())
            finally:
                path.write_bytes(original)
        self.assertTrue(adapter.retrieve("What did Ada collect?", state, NoModelAPI())["context"])

    def test_native_round_text_mutation_and_uncommitted_inventory_are_rejected(self):
        state = adapter.build(conversation(), self.directory, NoModelAPI())
        path = self.directory / "rounds.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        next(iter(data.values()))["units"][0]["text"] = "CORRUPTED_SOURCE_TEXT"
        path.write_text(json.dumps(data), encoding="utf-8")
        adapter._RUNTIMES.clear()
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            adapter.retrieve("Where?", state, NoModelAPI())
        legacy_state = {k: v for k, v in state.items() if k != "files_sha256"}
        with self.assertRaisesRegex(RuntimeError, "inventory"):
            adapter.retrieve("Where?", legacy_state, NoModelAPI())


if __name__ == "__main__":
    unittest.main()
