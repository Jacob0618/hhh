"""Pure offline harness gates only; never executes real-source reconstruction."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.baselines_common_v1 import save_new, sha

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'research/baseline_execution_2026-10-08/heartbeat_20261009T0949/verify_hypermem_native_v4_prefix_candidate_v1.py'
spec = importlib.util.spec_from_file_location('independent_v4_harness_under_review', PATH)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


class HarnessBoundaries(unittest.TestCase):
    def fixture(self, ci, position):
        sig = {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'operation': 'generateContent', 'body': {'fixture': ci}}
        context = {'method': 'hypermem', 'conversation': ci, 'phase': 'build'}
        stage = 'hypermem_episode_roles'
        row = {'signature': sig, 'stage': stage, 'request_hash': h.digest(sig), 'context': context,
               'mode': 'replace_once' if ci in h.admission.REPLACEMENT_CONVERSATIONS else 'inherit'}
        proof = {'conversations': [{'prefix': []} for _ in range(10)], 'producer_registry': {row['request_hash']: {}}}
        proof['conversations'][ci]['prefix'] = [row]
        cls = type('PureFixtureAPI', (), {'admission': {'progress': {ci: position}}})
        api = cls(); api.ci = ci; api.context = context
        return api, proof, row

    def test_four_boundaries_require_original_full_identity_and_last_slot(self):
        for ci in h.admission.REPLACEMENT_CONVERSATIONS:
            api, proof, row = self.fixture(ci, 0)
            self.assertEqual(h.expected_boundary(api, proof, row['signature'], row['stage'])[0],
                             'before_exact_ungranted_replacement')
            sig = deepcopy(row['signature']); sig['body']['changed'] = True
            with self.assertRaises(BaseException):
                h.expected_boundary(api, proof, sig, row['stage'])
            with self.assertRaises(BaseException):
                h.expected_boundary(api, proof, row['signature'], 'hypermem_topic_match')
            api.context = {**api.context, 'phase': 'wrong'}
            with self.assertRaises(BaseException):
                h.expected_boundary(api, proof, row['signature'], row['stage'])

    def test_only_six_complete_prefixes_stop_at_new_key_known_keys_are_not_stops(self):
        for ci in h.admission.COMPATIBLE_CONVERSATIONS:
            api, proof, row = self.fixture(ci, 1)
            self.assertIsNone(h.expected_boundary(api, proof, row['signature'], row['stage']))
            new = {**row['signature'], 'body': {'new': ci}}
            self.assertEqual(h.expected_boundary(api, proof, new, 'hypermem_topic_match')[0],
                             'after_exact_prefix_before_unseen')
            type(api).admission['progress'][ci] = 0
            self.assertIsNone(h.expected_boundary(api, proof, row['signature'], row['stage']))
        api, proof, row = self.fixture(2, 1)
        with self.assertRaises(BaseException):
            h.expected_boundary(api, proof, {**row['signature'], 'body': {'new': 'forbidden'}}, row['stage'])

    def test_review_must_match_exact_harness_core_and_nonproduction_scope(self):
        # Pure review-shape boundary: final real source inventory is checked
        # separately after the candidate launcher and all sources are frozen.
        code = {'synthetic_candidate.py': 'a' * 64}
        dependencies = {'parent_manifest': {'path': 'synthetic_manifest', 'sha256': 'b' * 64}, 'files_sha256': code}
        value = {'schema': 'hypermem-native-v4-offline-harness-review-v1', 'status': 'passed',
                 'scope': 'zero_api_source_prefix_preflight_only', 'production_launch_authorized': False,
                 'code_sha256': code, 'frozen_dependencies': dependencies}
        with tempfile.TemporaryDirectory(dir=ROOT / 'research', prefix='independent_v4_harness_') as folder:
            patcher = patch.object(h, 'code_hashes', return_value=code)
            patcher.start(); self.addCleanup(patcher.stop)
            dep_patch = patch.object(h, 'frozen_dependencies', return_value=dependencies)
            dep_patch.start(); self.addCleanup(dep_patch.stop)
            folder = Path(folder)
            good = save_new(folder / 'good.json', value)
            self.assertEqual(h.validate_harness_review(good['path']), good)
            changes = [('schema', 'another-schema'), ('status', 'needs_review'),
                       ('scope', 'production'), ('production_launch_authorized', True), ('code_sha256', {}),
                       ('frozen_dependencies', {})]
            for i, (key, changed) in enumerate(changes):
                bad = save_new(folder / f'bad_{i}.json', {**value, key: changed})
                with self.assertRaises(BaseException):
                    h.validate_harness_review(bad['path'])

    def test_frozen_dependency_bytes_and_parent_manifest_both_reject_drift(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'research', prefix='independent_v4_dependencies_') as folder:
            folder = Path(folder)
            source = folder / 'frozen_science.py'; source.write_bytes(b'# original fixture\n')
            manifest = save_new(folder / 'run_manifest.json',
                                {'code_sha256': {str(source): sha(source)}, 'upstream_sha256': {}})
            with patch.object(h.admission, 'OLD', folder), patch.object(h, 'PARENT_MANIFEST_SHA', manifest['sha256']):
                result = h.frozen_dependencies()
                self.assertEqual(result['parent_manifest'], manifest)
                self.assertEqual(result['files_sha256'], {str(source): sha(source)})
                source.write_bytes(b'# changed fixture\n')
                with self.assertRaises(BaseException):
                    h.frozen_dependencies()
                source.write_bytes(b'# original fixture\n')
                path = Path(manifest['path']); path.write_bytes(path.read_bytes() + b' ')
                with self.assertRaises(BaseException):
                    h.frozen_dependencies()


if __name__ == '__main__':
    unittest.main()
