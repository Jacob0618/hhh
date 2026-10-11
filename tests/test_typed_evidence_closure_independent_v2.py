"""Independent invocation-cache and exact delivery-annotation counterexamples."""
from pathlib import Path
import tempfile
import unittest
from tools import baseline_evidence_closure_v1 as typed
from tools.baselines_common_v1 import read, sha


class IndependentInvocationPathCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.target = self.root/'target.py'
        self.target.write_text('pass', encoding='utf-8')
        self.calls = []

    def safe(self, value):
        self.calls.append(str(value))
        resolved = Path(value).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError('foreign fixture path')
        return resolved

    def collect(self, initial):
        return typed.collect(initial, root=self.root, safe_path=self.safe, read=read, sha=sha)

    def test_exact_string_reuses_only_this_invocations_checked_path(self):
        initial = [(str(self.target), sha(self.target))]*9
        for count in (1, 2):
            self.assertEqual(self.collect(initial), {str(self.target): sha(self.target)})
            self.assertEqual(len(self.calls), count)

    def test_alias_is_checked_separately_and_conflicting_digest_still_rejects(self):
        (self.root/'sub').mkdir()
        alias = str(self.root)+'/sub/../target.py'
        digest = sha(self.target)
        self.collect([(str(self.target), digest), (alias, digest)])
        self.assertEqual(len(self.calls), 2)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.collect([(str(self.target), '0'*64), (alias, digest)])
        with self.assertRaisesRegex(ValueError, 'foreign fixture'):
            self.collect([(str(self.target), digest), (str(self.root)+'/../outside.py', digest)])

    def test_every_invocation_revalidates_first_path_before_digest_read(self):
        digest_calls = []
        def deny(path):
            raise ValueError('reparse or changed containment')
        with self.assertRaisesRegex(ValueError, 'reparse'):
            typed.collect([(str(self.target), sha(self.target))], root=self.root,
                          safe_path=deny, read=read,
                          sha=lambda p: digest_calls.append(p))
        self.assertEqual(digest_calls, [])


class IndependentPriorReferenceBoundary(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.expected = {str(self.root/p): value[0] for p, value in typed.PRIOR_DOCUMENTS.items()}
        self.schemas = {str(self.root/p): value[1] for p, value in typed.PRIOR_DOCUMENTS.items()}
        self.expected[str(self.root/typed.PRIOR_DECISION)] = typed.PRIOR_DECISION_SHA
        self.expected[str(self.root/typed.PRIOR_REVIEW)] = typed.PRIOR_REVIEW_SHA
        self.first = self.root/next(iter(typed.PRIOR_DOCUMENTS))
        self.foreign = str(Path('C:/outside-do-not-read/historical-cache.json'))
        self.safe_calls = []
        self.sha_calls = []
        self.read_calls = []

    def safe(self, path):
        path = Path(path).resolve()
        self.safe_calls.append(str(path))
        if not path.is_relative_to(self.root):
            raise ValueError('foreign path is not an operative input')
        return path

    def digest(self, path):
        self.sha_calls.append(str(path))
        if str(path) not in self.expected:
            raise FileNotFoundError(str(path))
        return self.expected[str(path)]

    def document(self, path):
        self.read_calls.append(str(path))
        if str(path) in self.schemas:
            return {'schema': self.schemas[str(path)],
                    'old_cache': {'path': self.foreign, 'sha256': 'a'*64}}
        return {}

    def collect(self, initial=None, **kwargs):
        return typed.collect(initial or [(str(self.first), self.expected[str(self.first)])],
                             root=self.root, safe_path=self.safe,
                             sha=kwargs.pop('sha', self.digest),
                             read=kwargs.pop('read', self.document), **kwargs)

    def test_exact_leaf_retains_all_four_documents_and_both_decisions(self):
        annotations = []
        actual = self.collect(annotations=annotations)
        self.assertEqual(actual, self.expected)
        self.assertEqual(len(annotations), 4)
        self.assertNotIn(self.foreign, self.safe_calls+self.sha_calls+self.read_calls)
        self.assertEqual(set(self.read_calls), set(self.expected))
        self.assertEqual(self.collect(), self.expected)

    def test_known_path_cannot_receive_new_expected_or_actual_sha(self):
        with self.assertRaises(ValueError):
            self.collect([(str(self.first), '0'*64)], sha=lambda p: '0'*64)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            self.collect(sha=lambda p: '0'*64)

    def test_schema_mismatch_is_not_silently_an_opaque_leaf(self):
        with self.assertRaises(ValueError):
            self.collect(read=lambda p: {'schema': 'different'})

    def test_same_schema_and_bytes_at_another_path_still_recurse_strictly(self):
        other = self.first.with_name('unknown-reference.json')
        self.expected[str(other)] = self.expected[str(self.first)]
        self.schemas[str(other)] = self.schemas[str(self.first)]
        with self.assertRaisesRegex(ValueError, 'foreign path'):
            self.collect([(str(other), self.expected[str(other)])])


if __name__ == '__main__':
    unittest.main()

