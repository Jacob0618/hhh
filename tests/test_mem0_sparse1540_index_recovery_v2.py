"""Offline copied-index recovery contracts; never load credentials or call APIs."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools import recover_mem0_sparse1540_index_v2 as target


def entity(text, links, vector=1):
    return {"id": "irrelevant", "vector": {"": [vector, 0]},
            "payload": {"data": text, "entity_type": "TOPIC", "user_id": "u",
                        "linked_memory_ids": links, "extra": "unchanged"}}


class EntityUnionTests(unittest.TestCase):
    def test_union_changes_only_declared_target_link_and_leaves_input_untouched(self):
        entities = {"target": entity(" Video   Games ", ["old"]),
                    "unrelated": entity("movies", ["elsewhere"], 2)}
        before = copy.deepcopy(entities)
        eid, after = target.intended_entities(entities, "orphan")
        self.assertEqual(eid, "target")
        self.assertEqual(entities, before)
        expected = copy.deepcopy(before)
        expected["target"]["payload"]["linked_memory_ids"] = ["old", "orphan"]
        self.assertEqual(after, expected)

    def test_fact_already_linked_elsewhere_is_ineligible_partial_loss(self):
        entities = {"target": entity("video games", ["old"]),
                    "other": entity("movies", ["orphan"])}
        with self.assertRaisesRegex(ValueError, "not a fully orphaned"):
            target.intended_entities(entities, "orphan")

    def test_two_normalized_exact_destinations_rejected(self):
        entities = {"x": entity("Video Games", ["one"]), "y": entity(" video   games ", ["two"])}
        with self.assertRaisesRegex(ValueError, "unique"):
            target.intended_entities(entities, "orphan")

    def test_no_exact_target_cannot_invent_entity_or_use_near_name(self):
        entities = {"x": entity("video game", ["old"])}
        with self.assertRaisesRegex(ValueError, "unique"):
            target.intended_entities(entities, "orphan")


class SourceAndSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_inventory_binds_nested_store_bytes_ignores_only_lock(self):
        folder = self.root / "qdrant"; (folder / "collection/facts").mkdir(parents=True)
        source = folder / "collection/facts/storage.sqlite"; source.write_bytes(b"original")
        (folder / ".lock").write_text("owner")
        before = target.inventory(folder)
        self.assertEqual(len(before), 1)
        (folder / ".lock").write_text("different owner")
        self.assertEqual(target.inventory(folder), before)
        source.write_bytes(b"changed")
        self.assertNotEqual(target.inventory(folder), before)

    def install_journals(self):
        conv = self.root / "conv_3"
        groups = [{"session_id": "sparse_topic_%04d" % i, "observation_date": "2023-01-01",
                   "messages": [{"role": "user", "content": "A: source %d" % i}],
                   "source_turn_ids": ["D1:%d" % i]} for i in range(2)]
        self.write(conv / "group_inputs.json", {"envelopes": groups})
        for i, group in reversed(list(enumerate(groups))):
            row = {**group, "status": "success", "fact_records": [{"id": "fact%d" % i, "text": "fact%d" % i}]}
            self.write(conv / "native/session_audit/user_scope" / ("%d.json" % i), row)
        return conv

    def test_nested_success_journals_rebuild_in_group_input_order(self):
        conv = self.install_journals()
        with patch.object(target.base, "conv_path", return_value=conv):
            facts, journals = target.facts_and_journals("regroup_rank")
        self.assertEqual([f["id"] for f in facts], ["fact0", "fact1"])
        self.assertEqual([Path(j["path"]).name for j in journals], ["0.json", "1.json"])

    def test_duplicate_success_journal_not_silently_selected(self):
        conv = self.install_journals()
        old = conv / "native/session_audit/user_scope/0.json"
        new = conv / "native/session_audit/duplicate_scope/copy.json"
        new.parent.mkdir(); new.write_bytes(old.read_bytes())
        with patch.object(target.base, "conv_path", return_value=conv), self.assertRaisesRegex(ValueError, "Duplicate"):
            target.facts_and_journals("regroup_rank")

    def test_changed_date_or_source_messages_rejected(self):
        conv = self.install_journals()
        p = conv / "native/session_audit/user_scope/0.json"
        row = json.loads(p.read_text()); row["observation_date"] = "2030-01-01"; self.write(p, row)
        with patch.object(target.base, "conv_path", return_value=conv), self.assertRaisesRegex(ValueError, "scope changed"):
            target.facts_and_journals("regroup_rank")

    def test_seed_uses_provider_request_digest_and_exact_existing_response(self):
        source = self.root / "source"; destination = self.root / "destination"; destination.mkdir()
        (destination / "native").mkdir()
        question = "Which event?"
        signature = target.base.query_signature(question)
        digest = target.base.request_digest(signature)
        cached = {"request": signature, "stage": "mem0_embedding_search",
                  "response": {"embeddings": [{"values": [1.] + [0.] * 767}]}}
        src = source / "api_cache" / (digest + ".json"); self.write(src, cached)
        self.write(destination / "repair_result.json", {"source_dir": str(source)})
        with patch.object(target, "folder", return_value=destination), \
                patch.object(target.base, "questions", return_value=[{"question": question}]), \
                patch("mem0.memory.main.extract_entities", return_value=[]), \
                patch.object(target, "ModelAPI", side_effect=AssertionError("No provider allowed")):
            rows = target.seed_requests("session_rank")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_hash"], digest)
        self.assertEqual((destination / "native/api_cache" / src.name).read_bytes(), src.read_bytes())
        self.assertEqual(json.loads(src.read_text()), cached)


class NativeGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.history = Path(self.temp.name) / "history.sqlite"
        db = sqlite3.connect(self.history)
        db.execute("CREATE TABLE history(memory_id TEXT,new_memory TEXT,event TEXT,is_deleted INTEGER)")
        db.execute("INSERT INTO history VALUES('f','A source fact.','ADD',0)")
        db.commit(); db.close()
        self.facts = [{"id": "f", "text": "A source fact."}]

    def gate(self, entities, extracted=None):
        expected = [[("TOPIC", "video games")]] if extracted is None else extracted
        with patch("mem0.memory.main.extract_entities_batch", return_value=expected):
            return target.native_gate(self.facts, entities, self.history)

    def test_repaired_link_and_original_add_history_pass(self):
        output = self.gate({"entity": entity("video games", ["f"])})
        self.assertEqual(output["history_adds_verified"], 1)
        self.assertEqual(output["entity_facts_without_any_link"], [])

    def test_foreign_fact_link_rejected(self):
        with self.assertRaisesRegex(ValueError, "scope gate"):
            self.gate({"entity": entity("video games", ["f", "foreign"] )})

    def test_missing_required_link_rejected(self):
        with self.assertRaisesRegex(ValueError, "scope gate"):
            self.gate({"entity": entity("video games", [])})

    def test_wrong_history_or_duplicate_add_rejected(self):
        db = sqlite3.connect(self.history)
        db.execute("INSERT INTO history VALUES('f','A source fact.','ADD',0)")
        db.commit(); db.close()
        with self.assertRaisesRegex(ValueError, "history"):
            self.gate({"entity": entity("video games", ["f"])})

    def test_unavailable_nlp_cannot_be_mistaken_for_entityless_bank(self):
        with patch("mem0.utils.spacy_models.get_nlp_full", return_value=None), \
                self.assertRaisesRegex(ValueError, "NLP unavailable"):
            self.gate({"entity": entity("video games", ["f"])})

    def test_incomplete_nlp_batch_cannot_hide_required_entities(self):
        with self.assertRaisesRegex(ValueError, "cardinality"):
            self.gate({"entity": entity("video games", ["f"])}, extracted=[])


if __name__ == "__main__":
    unittest.main()
