"""Conditional new representation only; no credentials or production work."""
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import uuid

from tools import baselines_hypermem_native_v3 as a
from tools.baselines_common_v1 import BaselineFatal


class RoleRepresentation(unittest.TestCase):
    def setUp(self):
        self.mapping = {f'episode_{i+1}': str(uuid.uuid5(uuid.NAMESPACE_URL, f'fixture-{i}')) for i in range(2)}
        self.value = {'episode_roles': [
            {'episode_id': real, 'role': 'developing', 'weight': 0.5, 'extra': [i, '原文']}
            for i, real in enumerate(reversed(list(self.mapping.values())))],
            'coherence_score': 0.8, 'reasoning': 'kept exactly', 'unknown': {'nested': [True, None]}}

    def test_exact_uuid_inverse_and_all_non_id_fields_unchanged(self):
        before = deepcopy(self.value)
        derived, journal = a.represent_episode_ids(self.value, self.mapping)
        self.assertEqual(self.value, before)
        self.assertEqual([x['episode_id'] for x in derived['episode_roles']], ['episode_2', 'episode_1'])
        self.assertTrue(journal['inverse_equal'])
        for row, old in zip(derived['episode_roles'], before['episode_roles']):
            self.assertEqual({k:v for k,v in row.items() if k != 'episode_id'}, {k:v for k,v in old.items() if k != 'episode_id'})
        self.assertEqual({k:v for k,v in derived.items() if k != 'episode_roles'}, {k:v for k,v in before.items() if k != 'episode_roles'})

    def test_original_alias_branch_keeps_original_frozen_contract(self):
        original = deepcopy(self.value)
        for i, row in enumerate(original['episode_roles']):
            row['episode_id'] = f'episode_{i+1}'
        value, journal = a.represent_episode_ids(original, {'episode_1':'native-non-uuid-1','episode_2':'native-non-uuid-2'})
        self.assertIs(value, original)
        self.assertIsNone(journal)

    def test_mixed_missing_unknown_duplicate_and_nonstring_ids_rejected(self):
        cases = []
        for replacement in ('episode_1', str(uuid.uuid4()), None, 1, self.value['episode_roles'][1]['episode_id']):
            value = deepcopy(self.value); value['episode_roles'][0]['episode_id'] = replacement; cases.append(value)
        value = deepcopy(self.value); value['episode_roles'].pop(); cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(a.HyperMemError):
                a.represent_episode_ids(value, self.mapping)

    def test_mapping_and_role_weight_invalid_still_fail(self):
        for field, bad in (('role','invented'),('weight',True),('weight',1.1),('weight',float('nan'))):
            value = deepcopy(self.value); value['episode_roles'][0][field] = bad
            with self.subTest(field=field,bad=bad), self.assertRaises(a.HyperMemError):
                a.represent_episode_ids(value, self.mapping)
        for mapping in ({'episode_2':list(self.mapping.values())[0],'episode_1':list(self.mapping.values())[1]},
                        {'episode_1':'not-a-uuid','episode_2':'also-invalid'},
                        {'episode_1':list(self.mapping.values())[0],'episode_2':list(self.mapping.values())[0]}):
            with self.assertRaises(a.HyperMemError):a.represent_episode_ids(self.value,mapping)

    def test_only_two_insertions_science_and_missing_wait_unchanged(self):
        inserted = [line for line in a._derived_source.splitlines() if 'value = self._represent(' not in line]
        self.assertEqual('\n'.join(inserted)+'\n', a._original)
        self.assertEqual(a.describe(), a._v2.describe())
        self.assertIsNone(a._Generation._validate({'should_end':False},'hypermem_boundary',{}))
        for bad in (None, '', 0):
            with self.assertRaises(a.HyperMemError):a._Generation._validate({'should_end':False,'should_wait':bad},'hypermem_boundary',{})

    def run_native_generate(self, api, folder=None):
        if folder is None:
            temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'research')
            self.addCleanup(temp.cleanup)
            folder = Path(temp.name)
        gen = a._Generation(api, folder, 'topic')
        async def _assign_episode_roles_and_weights():
            simple_to_real = self.mapping
            episodes = [SimpleNamespace(event_id=real,to_dict=lambda r=real:{'event_id':r,'content':'fixture'}) for real in simple_to_real.values()]
            topic = SimpleNamespace(to_dict=lambda:{'topic_id':'fixture-topic'})
            return await gen.generate('exact fixture prompt',response_format={'type':'json_object'})
        return folder, _assign_episode_roles_and_weights

    def test_real_generation_without_authority_stops_and_preserves_original(self):
        api = SimpleNamespace(generate=lambda *args,**kwargs:deepcopy(self.value))
        folder, native = self.run_native_generate(api)
        with self.assertRaises(BaselineFatal):asyncio.run(native())
        record = a._runtime._read_json(folder/'calls/topic_000001_hypermem_episode_roles.json')
        self.assertEqual(record['raw_response'],self.value)
        self.assertEqual(record['status'],'failed')
        self.assertFalse((folder/'representations').exists())

    def test_real_generation_records_original_and_derived_separately(self):
        receipts=[]
        def authorize(evidence,*,original,derived):
            receipts.append(deepcopy((evidence,original,derived)))
            return {'status':'authorized','fixture_only':True}
        api = SimpleNamespace(generate=lambda *args,**kwargs:deepcopy(self.value),authorize_episode_representation=authorize)
        folder, native = self.run_native_generate(api)
        asyncio.run(native())
        record = a._runtime._read_json(folder/'calls/topic_000001_hypermem_episode_roles.json')
        self.assertEqual(record['status'],'success')
        self.assertEqual(record['raw_response'],self.value)
        self.assertEqual(record['parsed_response'],receipts[0][2])
        self.assertEqual(len(receipts),1)
        journal = folder/'representations/topic_000001_hypermem_episode_roles.json'
        a._runtime._verify_seal(journal)
        self.assertEqual(a._runtime._read_json(journal)['local_episodes'][0]['event_id'],self.mapping['episode_1'])

    def test_native_cache_rechecks_original_raw_mapping_and_derivation_journal(self):
        receipt = {'status':'authorized','fixture_only':True}
        api = SimpleNamespace(generate=lambda *args,**kwargs:deepcopy(self.value),
            authorize_episode_representation=lambda *args,**kwargs:dict(receipt))
        folder, native = self.run_native_generate(api); asyncio.run(native())
        def no_generate(*args,**kwargs):raise AssertionError('Native cache may not generate')
        api.generate = no_generate
        _, native = self.run_native_generate(api, folder); asyncio.run(native())
        values = list(self.mapping.values())
        self.mapping = {'episode_1':values[1],'episode_2':values[0]}
        _, native = self.run_native_generate(api, folder)
        with self.assertRaises(a.HyperMemError):asyncio.run(native())
        self.mapping = {'episode_1':values[0],'episode_2':values[1]}
        journal = folder/'representations/topic_000001_hypermem_episode_roles.json'
        journal.unlink()  # Only the uniquely owned temporary test artifact.
        _, native = self.run_native_generate(api, folder)
        with self.assertRaises(a.HyperMemError):asyncio.run(native())


if __name__ == '__main__':unittest.main()
