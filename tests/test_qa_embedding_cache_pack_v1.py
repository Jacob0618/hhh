"""Synthetic raw-cache fixtures only; never credentials, network, or native clients."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev_mem import qa_embedding_cache_pack_v1 as pack


def raw_record(text="Q", task="RETRIEVAL_QUERY", dimension=768):
    signature = {"provider": "gemini", "model": "gemini-embedding-001", "operation": "batchEmbedContents",
                 "body": {"requests": [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": text}]},
                                        "taskType": task, "outputDimensionality": dimension}]}}
    saved = {"request": signature, "response": {"embeddings": [{"values": [1.0] + [0.0] * 767}]},
             "stage": "mem0_embedding_search"}
    return pack.request_digest(signature), (json.dumps(saved, indent=2) + "\n").encode(), signature


class EmbeddingPackTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="query_embedding_pack_mock_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.base, self.sdk = self.root / "old_session", self.root / "old_sdk"
        for name, value in (("BASE", self.base), ("SDK", self.sdk)):
            p = patch.object(pack, name, value)
            p.start()
            self.addCleanup(p.stop)
        for name in ("requests.Session.request", "requests.Session.post", "socket.create_connection",
                     "jev_mem.providers.load_credentials"):
            p = patch(name, side_effect=AssertionError("Offline cache test must never dispatch"))
            p.start()
            self.addCleanup(p.stop)

    def save(self, path, raw):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def plan(self):
        key, raw, signature = raw_record()
        anchor = pack.pointer(self.save(self.root / "anchor.json", b"{}"))
        entries, full = [], []
        for ci in range(10):
            src = self.save(self.base / f"conv_{ci}/api_cache/{key}.json", raw)
            entries.append({"conversation": ci, "request_hash": key, "signature": signature,
                            "source": pack.pointer(src), "source_rank": 0,
                            "sealed_original_relative_path": f"conv_{ci}/api_cache/{key}.json",
                            "sealed_original_sha256": pack.sha(src),
                            "query_ids": [f"fixture_c{ci}_q{i}" for i in range(154)]})
            full.extend({"conversation": ci, "query_id": f"fixture_c{ci}_q{i}", "request_hash": key,
                         "sdk_trimmed": False} for i in range(154))
        return {"schema": pack.SCHEMA + "-plan", "source_precedence": pack.PRECEDENCE,
                "entries": entries, "full_queries": full,
                "original_seal": anchor, "original_inventory": anchor, "sdk_manifest": anchor,
                "sdk_search_code": anchor, "retrieval_ledgers": [anchor],
                "counts": {"questions": 1540, "query_or_entity_requests": 10,
                           "per_conversation": {str(ci): 1 for ci in range(10)}},
                "model_calls": 0, "native_clients": 0, "policy": "synthetic fixture only"}

    def test_query768_raw_replays_original_embed_and_rejects_qa_document_dimensions(self):
        key, raw, signature = raw_record(" Q ")
        self.assertEqual(pack.validate_raw(raw, key), signature)
        for task, dimension in (("RETRIEVAL_DOCUMENT", 768), ("RETRIEVAL_QUERY", 3072)):
            badkey, badraw, _ = raw_record(task=task, dimension=dimension)
            with self.assertRaises(ValueError):
                pack.validate_raw(badraw, badkey)
        qa = {"request": {"provider": "gemini", "model": "gemini-2.5-flash", "operation": "generateContent", "body": {}},
              "response": {"candidates": []}}
        with self.assertRaises(ValueError):
            pack.validate_raw(json.dumps(qa).encode(), pack.request_digest(qa["request"]))

    def test_invalid_first_source_never_falls_through_to_successful_second(self):
        key, raw, _ = raw_record()
        first = self.save(self.base / f"conv_0/api_cache/{key}.json", raw)
        self.save(self.sdk / f"conv_0/api/api_cache/{key}.json", raw)
        bad = json.loads(raw)
        bad["response"]["embeddings"][0]["values"] = [0.0] * 768
        first.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(RuntimeError):
            pack._first_source(0, key, {f"conv_0/api_cache/{key}.json": pack.sha(first)})

    def test_missing_completed_raw_stops_without_dispatch(self):
        with self.assertRaisesRegex(ValueError, "unknown/failed"):
            pack._first_source(0, "a" * 64, {})

    def test_sealed_byte_pack_roundtrip_and_injection_rejection(self):
        plan = self.plan()
        before = {r["source"]["path"]: Path(r["source"]["path"]).read_bytes() for r in plan["entries"]}
        binding = pack.create_pack(self.root / "new_run", plan)
        checked = pack.verify_pack(binding, check_sources=True)
        self.assertEqual(checked["counts"]["questions"], 1540)
        self.assertEqual(len(checked["entries"]), 10)
        for row in checked["entries"]:
            self.assertEqual(Path(row["target"]["path"]).read_bytes(), before[row["source"]["path"]])
        for path, raw in before.items():
            self.assertEqual(Path(path).read_bytes(), raw)
        pack.verify_pack(binding, conversation=3)
        self.save(Path(binding["cache_roots"]["3"][0]) / "unexpected_qa.json", b"{}")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            pack.verify_pack(binding, conversation=3)

    def test_plan_identity_and_namespace_tampering_rejected(self):
        plan = self.plan()
        for change in ("coverage", "source"):
            broken = copy.deepcopy(plan)
            if change == "coverage":
                broken["full_queries"].pop()
            else:
                broken["entries"][0]["source"]["path"] = str(self.root / "outside.json")
            with self.assertRaises(ValueError):
                pack.create_pack(self.root / change, broken)
            self.assertFalse((self.root / change).exists())
        with self.assertRaisesRegex(ValueError, "original source"):
            pack.create_pack(self.base / "new_subfolder", plan)
        self.assertFalse((self.base / "new_subfolder").exists())


if __name__ == "__main__":
    unittest.main()
