"""Fixed prior numeric leaves and invocation-local path validation caching.

Only temporary files and an explicitly synthetic in-memory closure are used.
The complete actual evidence graph is checked by the separate final gate.
"""
import copy
from pathlib import Path
import tempfile
import unittest

from tools import baseline_evidence_closure_v1 as typed
from tools.baselines_common_v1 import read, sha

ROOT = Path(__file__).resolve().parents[1]


class PriorNumericLeafTests(unittest.TestCase):
    def test_exact_table_matches_actual_root_scope_and_document_schemas(self):
        decision = ROOT / typed.PRIOR_DECISION
        self.assertEqual(sha(decision), typed.PRIOR_DECISION_SHA)
        self.assertEqual(read(decision)['exact_reference_documents'],
            {str(ROOT / p): h for p, (h, _) in typed.PRIOR_DOCUMENTS.items()})
        self.assertEqual(sha(ROOT / typed.PRIOR_REVIEW), typed.PRIOR_REVIEW_SHA)
        for name, (h, schema) in typed.PRIOR_DOCUMENTS.items():
            self.assertEqual(sha(ROOT / name), h)
            self.assertEqual(read(ROOT / name)['schema'], schema)

    def virtual(self):
        # Explicitly synthetic byte/SHA reader: actual bytes are covered above
        # and the final independent complete production-closure gate.
        self.virtual_docs = {
            ROOT / p: (h, {'schema': schema,
                          'old_reference': {'path': 'C:/external/old.json', 'sha256': 'a' * 64}})
            for p, (h, schema) in typed.PRIOR_DOCUMENTS.items()
        }
        self.virtual_docs[ROOT / typed.PRIOR_DECISION] = (typed.PRIOR_DECISION_SHA, {})
        self.virtual_docs[ROOT / typed.PRIOR_REVIEW] = (typed.PRIOR_REVIEW_SHA, {})
        self.checked = []
        def safe(p):
            p = Path(p)
            self.checked.append(p)
            if not p.is_relative_to(ROOT):
                raise ValueError('foreign')
            if p not in self.virtual_docs:
                raise FileNotFoundError(str(p))
            return p
        self.safe = safe
        return next(iter(self.virtual_docs))

    def collect(self, p, expected=None, annotations=None):
        return typed.collect([(str(p), expected or self.virtual_docs[p][0])],
            root=ROOT, safe_path=self.safe,
            read=lambda p: copy.deepcopy(self.virtual_docs[p][1]),
            sha=lambda p: self.virtual_docs[p][0], annotations=annotations)

    def test_one_leaf_retains_all_four_and_authorities_without_optional_journal(self):
        p = self.virtual()
        result = self.collect(p)
        self.assertEqual(set(result), set(map(str, self.virtual_docs)))
        self.assertTrue(all(p.is_relative_to(ROOT) for p in self.checked))
        journal = []
        self.collect(p, annotations=journal)
        self.assertEqual(len(journal), 4)
        self.assertEqual({x['document'] for x in journal},
                         {str(ROOT / n) for n in typed.PRIOR_DOCUMENTS})
        self.assertTrue(all(x['json_pointer'] == '$' for x in journal))

    def test_known_path_wrong_expected_actual_bytes_schema_and_changed_authority_rejected(self):
        p = self.virtual()
        with self.assertRaisesRegex(ValueError, 'expected SHA'):
            self.collect(p, expected='0' * 64)
        h, doc = self.virtual_docs[p]
        self.virtual_docs[p] = ('0' * 64, doc)
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            self.collect(p, expected=h)
        self.virtual_docs[p] = (h, {**doc, 'schema': 'different'})
        with self.assertRaisesRegex(ValueError, 'schema differs'):
            self.collect(p)
        p = self.virtual()
        self.virtual_docs[ROOT / typed.PRIOR_DECISION] = ('0' * 64, {})
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            self.collect(p)

    def test_same_schema_or_same_bytes_at_unknown_path_do_not_become_leaves(self):
        p = self.virtual()
        other = ROOT / 'research/ordinary_current_evidence.json'
        self.virtual_docs[other] = copy.deepcopy(self.virtual_docs[p])
        with self.assertRaisesRegex(ValueError, 'foreign'):
            self.collect(other)


class ExactPathCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.file = self.root / "source.txt"
        self.file.write_text("unique bytes", encoding="utf8")
        self.checked = []
    def safe(self, p):
        self.checked.append(Path(p))
        p = Path(p).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("foreign")
        return p
    def collect(self, entries):
        return typed.collect(entries, root=self.root, safe_path=self.safe, read=read, sha=sha)
    def test_cache_exact_original_string_only_and_never_cross_invocation(self):
        h = sha(self.file)
        self.collect([("source.txt", h)] * 5)
        self.assertEqual(len(self.checked), 1)
        self.collect([("source.txt", h)] * 5)
        self.assertEqual(len(self.checked), 2)
        self.checked.clear()
        self.collect([("source.txt", h), (str(self.file), h), ("./source.txt", h)])
        self.assertEqual(len(self.checked), 3)
    def test_references_still_reject_conflict_foreign_and_first_visit_failures(self):
        h = sha(self.file)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.collect([("source.txt", "0" * 64), ("source.txt", h)])
        with self.assertRaisesRegex(ValueError, "foreign"):
            self.collect([(str(self.root.parent / "foreign.txt"), h)])
        called = []
        def reject(p):
            called.append(p)
            raise ValueError("Reparse point")
        with self.assertRaisesRegex(ValueError, "Reparse"):
            typed.collect([("source.txt", h)] * 5, root=self.root,
                safe_path=reject, read=read, sha=sha)
        self.assertEqual(len(called), 1)


if __name__ == "__main__":
    unittest.main()
