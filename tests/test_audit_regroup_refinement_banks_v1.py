"""Offline source/index-audit guard tests; no native Memory or model client."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools import audit_regroup_refinement_banks_v1 as audit


def source_fixture(env):
    sessions = [{"session_id": "session_1", "observation_date": "2026-09-01",
                 "source_turn_ids": ["D1:1", "D1:2", "D1:3", "D1:4"],
                 "messages": [{"role": "user", "content": f"Alice: original turn {i}."} for i in range(4)]}]
    segments = [{"segment_id": f"s{i}", "source_turn_ids": [f"D1:{i+1}"]} for i in range(4)]
    stamp = {"source_turn_id": "D1:4", "session_id": "session_1", "observation_date": "2026-09-01", "source_order_index": 3}
    envelope = {"group_id": "g3", "segment_ids": ["s3"], "source_turn_ids": ["D1:4"],
                "messages": [sessions[0]["messages"][3]], "turn_sources": [stamp],
                "source_order_indices": [3], "source_observation_dates": ["2026-09-01"],
                "observation_date": "2026-09-01"}
    messages = [env["render_source"](envelope["messages"][0], stamp)]
    journal = {**envelope, "user_id": "mock_user", "writer_messages": messages, "current_date": "2026-09-27"}
    journal["request_sha256"] = audit.index_audit.digest(journal)
    context = env["build_context_plan"](envelope, source_sessions=sessions, segments=segments)
    journal["source_local_context"] = context
    journal["history_context"] = context["history_context"]
    journal["effective_request_sha256"] = audit.index_audit.digest({
        "native_request_sha256": journal["request_sha256"], "source_local_context_sha256": context["sha256"]})
    return sessions, {"segments": segments}, envelope, journal, context


class AuditGuardTests(unittest.TestCase):
    def setUp(self):
        audit.index_audit.HASHES.clear()

    def test_running_preflight_refuses_before_opening_database(self):
        manifest = {"arms": ["source_local_context"], "refinement": {
            "schema": "mem0-regroup-refinement-writer-v1", "arm": "source_local_context", "source_local_context": True}}
        manifest["protocol_sha256"] = audit.index_audit.digest(manifest)
        with patch.object(audit.index_audit, "read", side_effect=[manifest, {"status": "running"}]), \
                patch.object(audit.index_audit, "rows", side_effect=AssertionError("SQLite forbidden")) as rows, \
                patch.object(audit.index_audit, "points", side_effect=AssertionError("SQLite forbidden")) as points:
            with self.assertRaisesRegex(ValueError, "not closed complete"):
                audit.terminal_preflight(Path("D:/jev mem/runs/mock_running"), [0])
            rows.assert_not_called()
            points.assert_not_called()

    def test_terminal_without_close_marker_is_refused(self):
        manifest = {"arms": ["jev_refined"], "refinement": {
            "schema": "mem0-regroup-refinement-writer-v1", "arm": "jev_refined", "source_local_context": False}}
        manifest["protocol_sha256"] = audit.index_audit.digest(manifest)
        state = {"status": "complete", "completed_groups": 10, "expected_groups": 10}
        with patch.object(audit.index_audit, "read", side_effect=[manifest, state]):
            with self.assertRaisesRegex(ValueError, "not closed complete"):
                audit.terminal_preflight(Path("D:/jev mem/runs/mock_not_closed"), [0])

    def test_effective_source_context_uses_actual_predecessors(self):
        env = audit.prompt_environment()
        sessions, grouping, envelope, journal, context = source_fixture(env)
        messages, history, returned = audit.effective_inputs(journal, envelope, [], grouping, sessions, True, env)
        self.assertEqual([r["source_turn_id"] for r in history], ["D1:2", "D1:3"])
        self.assertEqual(returned, context)
        self.assertIn("[SEGMENT s3]", messages[0]["content"])
        self.assertNotEqual(messages, journal["writer_messages"])

    def test_effective_context_and_hash_tampering_rejected(self):
        env = audit.prompt_environment()
        sessions, grouping, envelope, journal, context = source_fixture(env)
        changed = deepcopy(journal)
        changed["history_context"] = []
        with self.assertRaisesRegex(ValueError, "Effective source-local context differs"):
            audit.effective_inputs(changed, envelope, [], grouping, sessions, True, env)
        changed = deepcopy(journal)
        changed["effective_request_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Effective extraction fingerprint"):
            audit.effective_inputs(changed, envelope, [], grouping, sessions, True, env)

    def test_original_native_request_hash_still_required(self):
        env = audit.prompt_environment()
        sessions, grouping, envelope, journal, context = source_fixture(env)
        journal["request_sha256"] = "wrong original native request fingerprint"
        with self.assertRaisesRegex(ValueError, "Native group request fingerprint"):
            audit.effective_inputs(journal, envelope, [], grouping, sessions, True, env)

    def test_plain_group_history_is_original_previous_ten(self):
        env = audit.prompt_environment()
        sessions, grouping, envelope, journal, context = source_fixture(env)
        journal.pop("source_local_context")
        journal.pop("effective_request_sha256")
        history = [{"source_turn_id": str(i), "writer_message": {"role": "user", "content": str(i)}} for i in range(12)]
        journal["history_context"] = history[-10:]
        messages, effective, returned = audit.effective_inputs(journal, envelope, history, grouping, sessions, False, env)
        self.assertEqual(effective, history[-10:])
        self.assertEqual(messages, journal["writer_messages"])
        self.assertIsNone(returned)

    def test_reused_sqlite_helper_leaves_database_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE history (memory_id TEXT, event TEXT)")
            con.execute("INSERT INTO history VALUES ('m1', 'ADD')")
            con.commit()
            con.close()
            before = path.read_bytes()
            rows = audit.index_audit.rows(path, "history")
            self.assertEqual(rows, [{"memory_id": "m1", "event": "ADD"}])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
