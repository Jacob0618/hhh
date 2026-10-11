"""Independent narrow evidence-reference gates. No production writes or API."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from tools import baseline_evidence_closure_v1 as typed
from tools import archive_reviewed_chain_typed_closure_v1 as entry
from tools.baselines_common_v1 import read, sha


class IndependentTypedReferences(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def safe(self, value):
        value = Path(value).resolve()
        if not value.is_relative_to(self.root):
            raise ValueError('out of fixture root')
        return value

    def check(self, payload):
        source = self.root / 'review.json'
        source.write_text(json.dumps(payload), encoding='utf-8')
        return typed.collect([(str(source), sha(source))], root=self.root,
                             safe_path=self.safe, read=read, sha=sha)

    def test_every_complete_pair_rejects_non_string_path_or_digest(self):
        for pk, hk in typed.PAIRS:
            for path, digest in [('x.py', None), ('x.py', 17), ('x.py', True),
                                 (None, 'a'*64), (17, 'a'*64),
                                 ('x.py', 'A'*64), ('x.py', 'a'*63)]:
                with self.subTest(pair=(pk, hk), path=path, digest=digest):
                    with self.assertRaisesRegex(ValueError, 'Explicit evidence'):
                        self.check({pk: path, hk: digest})

    def test_unknown_scalar_nested_scalar_and_foreign_paths_hard_fail(self):
        for key in typed.MAP_KEYS:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, 'Unknown non-map'):
                    self.check({'nested': {key: 'a'*64}})
        with self.assertRaisesRegex(ValueError, 'out of fixture root'):
            self.check({'path': '../outside.py', 'sha256': 'a'*64})

    def test_bound_target_bytes_are_verified_even_if_duplicate_reference(self):
        target = self.root / 'target.py'
        target.write_text('pass', encoding='utf-8')
        digest = sha(target)
        result = self.check({'path': str(target), 'sha256': digest,
                             'nested': {'files_sha256': {str(target): digest}}})
        self.assertEqual(result[str(target)], digest)
        target.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            self.check({'path': str(target), 'sha256': digest})

    def test_initial_bindings_cannot_bypass_digest_type_or_scope(self):
        for path, digest in [(str(self.root/'x'), None), (None, 'a'*64),
                             (str(self.root/'x'), 'A'*64)]:
            with self.subTest(path=path, digest=digest):
                with self.assertRaisesRegex(ValueError, 'Initial evidence'):
                    typed.collect([(path, digest)], root=self.root,
                                  safe_path=self.safe, read=read, sha=sha)


class IndependentArchiveBoundary(unittest.TestCase):
    def test_admission_rejection_never_installs_hook_or_calls_builder(self):
        original_a = entry.amem.engine.closure
        original_l = entry.lifecycle.engine.evidence_closure
        for method in ['amem', *entry.lifecycle.METHODS]:
            with self.subTest(method=method):
                with patch.object(entry, 'verify_review', side_effect=ValueError('reject review')):
                    with patch.object(entry.amem, 'archive') as amem_archive:
                        with patch.object(entry.lifecycle, 'archive') as later_archive:
                            with self.assertRaisesRegex(ValueError, 'reject review'):
                                entry.archive(method, Path('result'), Path('review'))
                            amem_archive.assert_not_called()
                            later_archive.assert_not_called()
            self.assertIs(entry.amem.engine.closure, original_a)
            self.assertIs(entry.lifecycle.engine.evidence_closure, original_l)

    def test_builder_exception_restores_exact_original_hook_for_all_methods(self):
        for method in ['amem', *entry.lifecycle.METHODS]:
            target = entry.amem if method == 'amem' else entry.lifecycle
            old_a = entry.amem.engine.closure
            old_l = entry.lifecycle.engine.evidence_closure
            with self.subTest(method=method):
                with patch.object(entry, 'verify_review', return_value={}):
                    with patch.object(target, 'archive', side_effect=RuntimeError('builder refused')):
                        with self.assertRaisesRegex(RuntimeError, 'builder refused'):
                            entry.archive(method, Path('result'), Path('review'))
            self.assertIs(entry.amem.engine.closure, old_a)
            self.assertIs(entry.lifecycle.engine.evidence_closure, old_l)


class IndependentHistoricalAnnotation(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.document = self.root/typed.HISTORICAL_CONFIG
        self.edge = read(self.document)['baseline']['summary']

    def test_exact_coordinates_and_reappearing_file_cannot_widen_exception(self):
        good = (self.edge, self.document, typed.HISTORICAL_CONFIG_SHA, '$.baseline.summary', self.root)
        self.assertTrue(typed._historical_annotation(*good))
        for offset, replacement in [(1, self.document.with_name('foreign.json')),
                                    (2, '0'*64), (3, '$.dataset.source'),
                                    (0, {**self.edge, 'path': str(self.root/'missing.json')}),
                                    (0, {**self.edge, 'sha256': '0'*64})]:
            args = list(good)
            args[offset] = replacement
            with self.subTest(offset=offset, replacement=replacement):
                self.assertFalse(typed._historical_annotation(*args))
        with patch.object(Path, 'exists', return_value=True):
            with self.assertRaisesRegex(ValueError, 'reappeared'):
                typed.collect([(str(self.document), typed.HISTORICAL_CONFIG_SHA)], root=self.root,
                              safe_path=lambda p: Path(p).resolve(), read=lambda p: {'baseline': {'summary': self.edge}},
                              sha=lambda p: typed.HISTORICAL_CONFIG_SHA)

    def test_classification_evidence_is_mandatory_even_without_optional_journal(self):
        # Virtual digest/read callbacks isolate traversal. The final real-data
        # closure separately verifies every actual decision and evidence byte.
        expected = {str(self.document): typed.HISTORICAL_CONFIG_SHA,
                    str(self.root/typed.HISTORICAL_DECISION): typed.HISTORICAL_DECISION_SHA,
                    str(self.root/typed.HISTORICAL_REVIEW): typed.HISTORICAL_REVIEW_SHA}
        seen = []
        def digest(path):
            seen.append(str(path))
            return expected[str(path)]
        def document(path):
            return {'baseline': {'summary': self.edge}} if path == self.document else {}
        result = typed.collect([(str(self.document), typed.HISTORICAL_CONFIG_SHA)], root=self.root,
                               safe_path=lambda p: Path(p).resolve(), read=document, sha=digest)
        self.assertEqual(result, expected)
        self.assertNotIn(str(self.root/typed.HISTORICAL_TARGET), seen)
        damaged = dict(expected)
        damaged[str(self.root/typed.HISTORICAL_REVIEW)] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            typed.collect([(str(self.document), typed.HISTORICAL_CONFIG_SHA)], root=self.root,
                          safe_path=lambda p: Path(p).resolve(), read=document,
                          sha=lambda p: damaged[str(p)])


if __name__ == '__main__':
    unittest.main()
