"""Independent PrefixAPI boundaries, synthetic artifacts, zero real transport."""
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from unittest.mock import patch
import json
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import requests
from tools import hypermem_prefix_api_v4 as p
from tools.baselines_common_v1 import BaselineFatal, save_new, read, sha
from jev_mem import providers

ROOT = Path(__file__).resolve().parents[1]


def response(text):
    return {'candidates': [{'index': 0, 'finishReason': 'STOP', 'content': {'role': 'model', 'parts': [{'text': text}]}}],
            'modelVersion': 'gemini-2.5-flash', 'usageMetadata': {'promptTokenCount': 4, 'candidatesTokenCount': 3, 'totalTokenCount': 7}}


class PrefixV4(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=ROOT / 'research', prefix='independent_prefix_v4_')
        self.addCleanup(tmp.cleanup); self.folder = Path(tmp.name)
        self.new = self.folder / 'new'; self.old = self.folder / 'old'
        self.proof = {'schema': p.PROOF_SCHEMA, 'status': 'candidate_evidence_only', 'producer_registry': {},
                      'conversations': [], 'compatibility_cases': {}, 'replacement_cases': {}, 'role_cases': [],
                      'questions': [{'conversation': i, 'query_id': f'c{i}_q0'} for i in range(10)],
                      'source_bindings': {'fixture': 'not-production'}}
        for ci in range(10):
            replacing = ci in (2, 3, 6, 9)
            stage = 'hypermem_episode_roles' if replacing else 'hypermem_boundary'
            sig = {'provider': 'gemini', 'model': 'gemini-2.5-flash', 'operation': 'generateContent',
                   'body': p.generation_body(f'fixture-prefix-{ci}', json_output=True, max_output=16384)}
            key = p.digest(sig); raw = self.old / 'api_cache' / (key + '.json')
            save_new(raw, {'request': sig, 'stage': stage, 'response': response('{"episode_roles":[]}' if replacing else '{"should_end":false}')})
            context = {'method': 'hypermem', 'conversation': ci, 'phase': 'build'}
            name = ('topic' if replacing else 'episode') + f'_000001_{stage}.json'
            native = self.old / f'native/c{ci}/calls' / name
            save_new(native, {'fixture_only': True})
            row = {'ordinal': 1, 'context': context, 'signature': sig, 'request_hash': key, 'stage': stage,
                   'raw': p.binding(raw), 'native_call': p.binding(native), 'mode': 'replace_once' if replacing else 'inherit'}
            self.proof['conversations'].append({'conversation': ci, 'prefix': [row]})
            self.proof['producer_registry'][key] = {'signature': sig, 'raw': p.binding(raw), 'stages': [stage]}
            if replacing:
                self.proof['replacement_cases'][key] = {'kind': 'known_complete_invalid_role_uuid_set', 'retry_limit': 1,
                    'conversation': ci, 'request_hash': key, 'signature': sig, 'context': context, 'stage': stage,
                    'raw': p.binding(raw), 'native_call': p.binding(native), 'prefix_ordinal': 1, 'human_authorized': False}
        self.manifest = save_new(self.new / 'run_manifest.json', {'fixture_only': True})
        self.proof_binding = save_new(self.folder / 'proof.json', self.proof)
        vault = patch.object(providers, 'load_credentials', side_effect=AssertionError('No credential access in fixture'))
        vault.start(); self.addCleanup(vault.stop)

    def klass(self, **kwargs):
        return p.admitted_api(self.proof, self.new, manifest=self.manifest, proof_binding=self.proof_binding, **kwargs)

    def api(self, cls, ci=0):
        return cls(self.new / 'work/api_method', context={'method': 'hypermem', 'conversation': ci, 'phase': 'build'},
                   keys={'GEMINI_API_KEY': 'synthetic-never-network'})

    def invoke(self, api, ci=0):
        e = self.proof['conversations'][ci]['prefix'][0]; s = e['signature']
        return api.request(s['provider'], s['model'], s['body'], operation=s['operation'], stage=e['stage'])

    @contextmanager
    def hypothetical_grant(self):
        # A temporary fixed-path substitution models root's later publication;
        # it never writes the real fixed authorization path or conveys approval.
        targets = []
        for cert in self.proof['replacement_cases'].values():
            targets.append({**{k: cert[k] for k in ('conversation', 'request_hash', 'signature', 'context', 'stage')},
                            'original_raw': {'source_path': cert['raw']['path'], 'observed_sha256': cert['raw']['sha256']}})
        proposal = save_new(self.folder / 'grant_proposal.json', {'targets': targets})
        refs = {'proposal': proposal}
        self.proof['source_bindings'] = refs
        fixedproof = save_new(self.folder / 'fixed_candidate_proof.json', self.proof)
        code = {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)}
        fixedreview = save_new(self.folder / 'fixed_candidate_review.json', {
            'schema': 'hypermem-native-v4-independent-candidate-review-v1',
            'status': 'passed_implementation_and_zero_api_preflight_only', 'production_launch_authorized': False,
            'code_sha256': code})
        replacements = [{**{k: c[k] for k in ('conversation', 'request_hash', 'signature', 'context', 'stage', 'raw')},
                         'logical_limit': 1} for c in self.proof['replacement_cases'].values()]
        message = 'Synthetic fixture only, not a real user instruction.'
        grant = save_new(self.folder / 'fixed_synthetic_grant.json', {
            'schema': 'hypermem-native-v4-explicit-human-grant-v1',
            'status': 'authorized_by_explicit_user_instruction', 'issued_by': 'root_after_trusted_user_confirmation',
            'source_bindings': refs, 'logical_limit_per_key': 1,
            'old_consumed_key': p.admission.CONSUMED_OLD_KEY, 'old_consumed_key_permission': False,
            'human_confirmation': {'actor': 'user', 'source': 'direct_user_message', 'message_id': 'synthetic-only',
                                   'text': message, 'text_sha256': p.adapter.text_sha(message)},
            'terminal_proof': fixedproof, 'implementation_review': fixedreview, 'code_sha256': code,
            'replacements': replacements, 'accepted_compatibility_request_hashes': []})
        with ExitStack() as stack:
            stack.enter_context(patch.object(p, 'HUMAN_GRANT_PATH', Path(grant['path'])))
            stack.enter_context(patch.object(p, 'PROOF_PATH', Path(fixedproof['path'])))
            stack.enter_context(patch.object(p, 'REVIEW_PATH', Path(fixedreview['path'])))
            stack.enter_context(patch.object(p.admission, 'fixed_bindings', lambda: refs))
            yield grant

    def test_missing_or_forged_grant_rejected_without_work_or_credentials(self):
        for grant in (None, {'path': str(self.folder / 'nonexistent.json'), 'sha256': '0' * 64}):
            with self.assertRaises(BaselineFatal):
                self.klass(human_grant=grant)
        self.assertFalse((self.new / 'work').exists())
        self.assertFalse(p.HUMAN_GRANT_PATH.exists())
        denied_class = type('DirectConstructionMustStillDeny', (p.PrefixAPI,),
                            {'admission': {'research_only': False, 'human_grant': None, 'proof': self.proof}})
        with self.assertRaises(BaselineFatal):
            self.api(denied_class)
        self.assertFalse((self.new / 'work').exists())

    def test_research_prefix_only_imports_and_stops_before_both_transport_boundaries(self):
        cls = self.klass(research_only=True)
        with patch.object(requests.Session, 'post', side_effect=AssertionError('No physical transport')):
            api = self.api(cls)
            self.invoke(api)
            with self.assertRaises(BaselineFatal):
                api.generate('unknown-suffix', json_output=True, stage='hypermem_boundary')
            with self.assertRaises(BaselineFatal):
                self.invoke(self.api(cls, 2), 2)
        self.assertEqual(cls.admission['progress'][0], 1)
        self.assertEqual(cls.admission['progress'][2], 0)
        self.assertFalse((self.new / 'work/replacement_v4').exists())
        self.assertFalse((self.new / 'work/api_method/request_intents').exists())
        rows = [json.loads(x) for x in (self.new / 'work/api_method/api_calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1); self.assertTrue(rows[0]['cache_hit'])

    def test_changed_known_request_or_source_never_falls_back_to_transport(self):
        cls = self.klass(research_only=True); api = self.api(cls)
        with patch.object(requests.Session, 'post', side_effect=AssertionError('No physical transport')):
            with self.assertRaises(BaselineFatal):
                api.generate('changed-prefix', json_output=True, stage='hypermem_boundary')
            raw = Path(self.proof['conversations'][0]['prefix'][0]['raw']['path'])
            raw.write_bytes(raw.read_bytes() + b' ')  # Private synthetic history only.
            with self.assertRaises(BaselineFatal):
                self.invoke(api)
        self.assertFalse((self.new / 'work/api_method/request_intents').exists())

    def test_research_construction_without_keys_never_loads_credentials(self):
        cls = self.klass(research_only=True)
        with patch.object(providers, 'load_credentials', side_effect=AssertionError('Credential boundary reached')) as vault:
            try:
                cls(self.new / 'work/api_method', context={'method': 'hypermem', 'conversation': 0, 'phase': 'build'})
            except BaselineFatal:
                pass
        self.assertEqual(vault.call_count, 0)

    def test_research_lower_level_request_cannot_bypass_no_transport_capability(self):
        from jev_mem.sdk20_cached_api import CachedLegacyAPI
        cls = self.klass(research_only=True)
        api = self.api(cls)
        class ReachedFakeTransport(BaseException):
            pass
        body = p.generation_body('unseen-direct-parent-request', json_output=True, max_output=16384)
        with patch.object(requests.Session, 'post', side_effect=ReachedFakeTransport('never a real network call')) as fake:
            for lower in (p.BaselineAPI.request, CachedLegacyAPI.request, providers.ModelAPI.request):
                try:
                    lower(api, 'gemini', 'gemini-2.5-flash', body, stage='hypermem_boundary')
                except BaseException:
                    pass
            # Reusing the research object's exported configuration in the old
            # cache class still has no usable credential capability.
            configured_inner = CachedLegacyAPI(self.folder / 'configured_inner', keys=api.keys,
                                               retries=api.retries, context=api.context)
            try:
                configured_inner.request('gemini', 'gemini-2.5-flash', body, stage='hypermem_boundary')
            except BaseException:
                pass
            self.assertEqual(fake.call_count, 0, 'An inherited parent request bypassed research-only transport denial')

    def test_research_owner_rejects_production_and_dotdot_alias_without_writes(self):
        targets = (p.admission.NEW, p.admission.NEW / 'work',
                   ROOT / 'research' / '..' / 'runs' / 'forbidden_v4_independent_fixture')
        for target in targets:
            self.assertFalse(target.exists())
            try:
                cls = p.admitted_api(self.proof, target, manifest=self.manifest,
                                     proof_binding=self.proof_binding, research_only=True)
                with self.assertRaises((BaselineFatal, ValueError)):
                    cls(target / 'work/api_method', context={'method': 'hypermem', 'conversation': 0, 'phase': 'build'},
                        keys={'GEMINI_API_KEY': 'synthetic-never-network'})
            except (BaselineFatal, ValueError):
                pass
            self.assertFalse(target.exists())

    def test_certificate_lookup_binds_exact_native_ordinal_and_original_raw(self):
        e = self.proof['conversations'][0]['prefix'][0]
        cert = {'conversation': 0, 'request_hash': e['request_hash'], 'signature': e['signature'], 'context': e['context'],
                'stage': e['stage'], 'raw': e['raw'], 'native_call': e['native_call'], 'prefix_ordinal': 1,
                'response_text_sha256': p.adapter.text_sha('{"should_end":false}')}
        self.proof['compatibility_cases'][e['request_hash']] = cert
        cls = self.klass(research_only=True); api = self.api(cls)
        request = {'prompt': 'fixture-prefix-0', 'response_format': {'type': 'json_object'}, 'stage': e['stage'],
                   'json_output': True, 'max_output': 16384, 'system': None}
        original = self.new / 'work/native/c0/calls' / Path(e['native_call']['path']).name
        self.assertEqual(api.compatibility_certificate(stage=e['stage'], request=request, native_call_path=str(original)), cert)
        wrong = original.with_name('episode_999999_hypermem_boundary.json')
        with self.assertRaises(BaselineFatal):
            api.compatibility_certificate(stage=e['stage'], request=request, native_call_path=str(wrong))
        raw = Path(e['raw']['path']); raw.write_bytes(raw.read_bytes() + b' ')
        with self.assertRaises(BaselineFatal):
            api.compatibility_certificate(stage=e['stage'], request=request, native_call_path=str(original))

    def test_captured_false_and_syntax_values_use_real_prefix_native_and_cached_authority(self):
        from tests.test_hypermem_native_v4_independent import exact_case, finish
        # Actual captured response/request bytes; caller objects are synthetic.
        # This is an interface test, not a claim of actual full-source replay.
        for ci in (0, 8):
            row, native, text, cert = exact_case(ci)
            original_native = {'path': row['native_call']['source_path'], 'sha256': row['native_call']['observed_sha256']}
            copied_raw = self.old / 'api_cache' / (row['request_hash'] + '.json')
            copied_raw.write_bytes(Path(row['raw']['captured']['path']).read_bytes())
            self.assertEqual(sha(copied_raw), row['raw']['captured']['sha256'])
            raw = p.binding(copied_raw)
            e = {'ordinal': 1, 'context': row['context'], 'signature': row['signature'], 'request_hash': row['request_hash'],
                 'stage': row['stage'], 'raw': raw, 'native_call': original_native, 'mode': 'inherit'}
            self.proof['conversations'][ci]['prefix'] = [e]
            self.proof['producer_registry'][e['request_hash']] = {'signature': e['signature'], 'raw': raw, 'stages': [e['stage']]}
            cert.update(signature=e['signature'], context=e['context'], raw=raw, native_call=original_native, prefix_ordinal=1)
            self.proof['compatibility_cases'][e['request_hash']] = cert
            cls = self.klass(research_only=True)
            native_dir = self.new / f'work/native/c{ci}'
            filename = Path(original_native['path']).name
            ordinal = int(filename.split('_')[1])
            def one(api):
                generation = p.adapter._Generation(api, native_dir, 'topic')
                generation.count = ordinal - 1
                async def _llm_match_topics_batch():
                    simple_to_real = deepcopy(row['diagnosis']['candidate_mapping'])
                    topics_batch = [SimpleNamespace(to_dict=lambda ident=ident: {'topic_id': ident}) for ident in simple_to_real.values()]
                    episode = SimpleNamespace(to_dict=lambda: {'event_id': 'synthetic-episode'})
                    return await generation.generate(native['request']['prompt'], response_format={'type': 'json_object'})
                async def _update_existing_topic():
                    topic = SimpleNamespace(to_dict=lambda: {'topic_id': 'synthetic-topic'})
                    new_episode = SimpleNamespace(to_dict=lambda: {'event_id': 'synthetic-new-episode'})
                    return await generation.generate(native['request']['prompt'], response_format={'type': 'json_object'})
                return finish(_llm_match_topics_batch() if ci == 0 else _update_existing_topic())
            with patch.object(requests.Session, 'post', side_effect=AssertionError('No physical transport')):
                first = one(self.api(cls, ci))
                second = one(self.api(cls, ci))  # New instance, sealed native + durable original consumer.
            self.assertEqual(first, second)
            retained = read(native_dir / 'calls' / filename)
            self.assertEqual(retained['raw_response'], json.loads(text) if ci == 0 else text)
            self.assertEqual(sha(self.new / 'work/api_method/api_cache' / (e['request_hash'] + '.json')), raw['sha256'])
            self.assertEqual(cls.admission['progress'][ci], 1)
            events = [json.loads(x) for x in (self.new / 'work/api_method/logical_events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(x['request_hash'] == e['request_hash'] for x in events), 1)
            self.assertFalse((self.new / 'work/api_method/request_intents').exists())

    def test_prefix_wait_does_not_hold_conversation_lock_but_reentry_is_blocked(self):
        cls = self.klass(research_only=True); api = self.api(cls)
        entered, release = threading.Event(), threading.Event()
        original = p.BaselineAPI.request
        def delayed(instance, *args, **kwargs):
            entered.set()
            if not release.wait(4):
                raise AssertionError('Fixture stalled')
            return original(instance, *args, **kwargs)
        with patch.object(requests.Session, 'post', side_effect=AssertionError('No physical transport')):
            with patch.object(p.BaselineAPI, 'request', delayed), ThreadPoolExecutor(max_workers=1) as pool:
                task = pool.submit(self.invoke, api)
                try:
                    self.assertTrue(entered.wait(2))
                    held = cls.admission['locks'][0].acquire(blocking=False)
                    self.assertTrue(held)
                    if held:
                        cls.admission['locks'][0].release()
                    with self.assertRaises(BaselineFatal):
                        self.invoke(self.api(cls))
                finally:
                    release.set()
                task.result(timeout=4)
        self.assertEqual(cls.admission['progress'][0], 1)
        self.assertIsNone(cls.admission['inflight'][0])

    def test_hypothetical_grant_fixture_marker_precedes_fake_transport_and_bad_result_has_no_second_send(self):
        e = self.proof['conversations'][2]['prefix'][0]
        marker = self.new / 'work/replacement_v4' / (e['request_hash'] + '.json')
        sends = []
        test = self
        class FakeResponse:
            status_code = 200
            ok = True
            headers = {}
            def json(self):
                return response('{"still_invalid":')
        def post(*args, **kwargs):
            test.assertTrue(marker.exists())
            test.assertEqual(read(marker)['signature'], e['signature'])
            test.assertTrue(read(marker)['permission_consumed_before_transport'])
            sends.append(1)
            return FakeResponse()
        with self.hypothetical_grant() as grant:
            cls = self.klass(human_grant=grant)
            with patch.object(requests.Session, 'post', post):
                with self.assertRaises(BaselineFatal):
                    self.api(cls, 2).generate('fixture-prefix-2', json_output=True, stage='hypermem_episode_roles')
                with self.assertRaises(BaselineFatal):
                    self.invoke(self.api(cls, 2), 2)
        self.assertEqual(len(sends), 1)
        self.assertEqual(sha(e['raw']['path']), e['raw']['sha256'])
        self.assertEqual(read(marker)['logical_replacements'], 1)
        self.assertFalse(p.HUMAN_GRANT_PATH.exists())


if __name__ == '__main__':
    unittest.main()
