import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_pilot400_usage_v1 import summarize_rows, partitions, parse_ledger, physical_usage
PRICES = {"jev-1.13.0": {"input": .042, "output": 0.0},
          "gemini-2.5-flash": {"input": .30, "output_including_thinking": 2.50}}


def row(stage="qa", **kwargs):
    return {"request_hash": "same", "context": {"driver_stage": stage}, "cache_hit": False,
            "status": 200, "operation": "systemone", "model": "jev-1.13.0",
            "usage": {"input_tokens": 1000, "output_tokens": 80}, "latency_ms": 5, **kwargs}


def encoded(rows):
    return b"".join(json.dumps(x).encode() + b"\n" for x in rows)


class UsageTests(unittest.TestCase):
    def test_repeated_hash_physical_attempts_and_cache(self):
        rows = [row(status="transport_error", usage={}), row(), row(cache_hit=True)]
        result = summarize_rows(rows, PRICES)
        self.assertEqual(result["physical_attempts"], 2)
        self.assertEqual(result["cache_hits"], 1)
        self.assertEqual(result["unique_physical_request_hashes"], 1)
        self.assertEqual(result["extra_physical_attempts_with_repeated_hash"], 1)
        self.assertEqual(result["unknown_cost_attempts"], 1)
        self.assertIsNone(result["total_cost_usd"])
        self.assertEqual(result["known_cost_component_usd"], .000042)
        self.assertEqual(result["recorded_tokens"]["input"], 1000)

    def test_embedding_unpriced_not_free(self):
        result = summarize_rows([row(model="gemini-embedding-001", operation="batchEmbedContents", usage={})], PRICES)
        self.assertEqual(result["unknown_cost_attempts"], 1)
        self.assertIsNone(result["total_cost_usd"])
        self.assertEqual(result["unknown_cost_reasons"]["unpriced_model"], 1)

    def test_partial_usage_preserves_known_component(self):
        result = summarize_rows([row(model="gemini-2.5-flash", operation="generateContent", usage={"promptTokenCount": 1000})], PRICES)
        self.assertEqual(result["known_cost_component_usd"], .0003)
        self.assertEqual(result["unknown_cost_attempts"], 1)
        self.assertIsNone(result["total_cost_usd"])

    def test_thinking_and_output_both_count(self):
        result = summarize_rows([row(model="gemini-2.5-flash", operation="generateContent", usage={"promptTokenCount": 1000, "candidatesTokenCount": 100, "thoughtsTokenCount": 50})], PRICES)
        self.assertEqual(result["known_cost_component_usd"], .000675)
        self.assertEqual(result["unknown_cost_attempts"], 0)

    def test_count_tokens_excluded_but_transport_still_unknown(self):
        result = summarize_rows([row(operation="countTokens", usage={}), row(operation="countTokens", usage={}, status="transport_error")], PRICES)
        self.assertEqual(result["unknown_cost_attempts"], 1)
        self.assertEqual(result["known_cost_component_usd"], 0)

    def test_byte_prefix_and_stage_context_enforced(self):
        write = encoded([row("write")])
        tag = write + encoded([row("tag")])
        final = tag + encoded([row("qa")])
        self.assertEqual({k: len(v) for k, v in partitions(write, tag, final).items()}, {"write": 1, "tag": 1, "qa": 1})
        with self.assertRaisesRegex(ValueError, "byte-exact"):
            partitions(write, encoded([row("tag")]), final)
        with self.assertRaisesRegex(ValueError, "context disagreement"):
            partitions(write, tag, tag + encoded([row("tag")]))

    def test_partial_tail_snapshot_only(self):
        good = encoded([row()])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parse_ledger(good + b'{"request_hash":')
        raw, rows = parse_ledger(good + b'{"request_hash":', snapshot=True)
        self.assertEqual(raw, good)
        self.assertEqual(len(rows), 1)

    def test_bool_usage_not_numeric(self):
        result = summarize_rows([row(usage={"input_tokens": True})], PRICES)
        self.assertEqual(result["unknown_cost_attempts"], 1)


if __name__ == "__main__":
    unittest.main()

