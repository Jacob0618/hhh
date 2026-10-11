"""Offline strict quote-token bridge and frozen-science equivalence checks."""
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from tools import baselines_amem_v3 as v3
from tools import baselines_amem_v4 as v4
from tools.check_experiment_health import shared_read
from tests.test_baselines_amem_v3 import raw_fixture
from tests.test_baselines_amem_v1 import FakeAPI, FakeEncoder, conversation, evolution

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/five_baselines_gemini25_v1_20261008/amem_native_v3'
KEY = '4a67f80cbb747d131e51949d111adb2451f00b12bed600417bfac1feadb20965'
RAW_SHA = '28e9ab6f4901e6f556707dd31f65b9c1997f9190afccb3c9ecbd332b07c58a87'


def decode(raw):
    return v4.decode_generation_json(raw, raw_artifact_sha256=hashlib.sha256(json.dumps(raw).encode()).hexdigest())


class QuoteBridgeTests(unittest.TestCase):
    def test_token_values_and_apostrophes_preserved_with_exact_inverse(self):
        text = '{"tags":["John\'s actions",\'gratitude\',"Maria\'s values",\'中文🙂\'],"x":\'null\'}'
        raw = raw_fixture(text); original = copy.deepcopy(raw)
        value, journal = decode(raw)
        self.assertEqual(value, {'tags': ["John's actions", 'gratitude', "Maria's values", '中文🙂'], 'x': 'null'})
        self.assertEqual(raw, original); self.assertEqual(len(journal['changes']), 6)
        derived, changes, _ = v4._single_quote_tokens(text)
        reverse = list(derived)
        for item in changes:
            self.assertEqual(item['utf8_byte_offset'], len(text[:item['character_offset']].encode()))
            self.assertEqual(text[item['character_offset']], "'")
            self.assertEqual(derived[item['character_offset']], '"')
            reverse[item['character_offset']] = "'"
        self.assertEqual(''.join(reverse), text)
        self.assertEqual(journal['derived_text_utf8_sha256'], hashlib.sha256(derived.encode()).hexdigest())

    def test_prior_v3_success_and_tab_journals_are_identical(self):
        for text in ('{"tags":["John\'s actions", "ok"]}', '{"x":"a\tb"}'):
            raw = raw_fixture(text); h = hashlib.sha256(json.dumps(raw).encode()).hexdigest()
            self.assertEqual(v4.decode_generation_json(raw, raw_artifact_sha256=h),
                             v3.decode_generation_json(raw, raw_artifact_sha256=h))

    def test_ambiguous_single_quotes_and_nonstring_python_forms_rejected(self):
        invalid = ["{'key':'value'}", '{"x":\'a\\n\'}', '{"x":\'a"b\'}', '{"x":\'a\nb\'}',
                   '{"x":\'a\tb\'}', '{"x":\'John\'s\'}', '{"x":\'unclosed}',
                   '{"x":\'a\' \"b\"}', '{"x":\'a\',"y":True}', '{"x":\'a\',"y":None}',
                   '{"x":\'a\',"y":(1,2)}', '{"x":\'a\',}', '{"x":\'a\'} trailing',
                   '{"x":\'a\',"x":1}', '{"x":\'a\',"y":NaN}', '{"x":\'a\',"y":Infinity}',
                   '{"x":\'a\',"y":"has\ttab"}', '{"x":\'a\'', '{"x":\'a\',"y":__import__("os")}',
                   '{"x":\'\'\'a\'\'\'}', '{"x":\'a\'+\'b\'}', "['a']"]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises((v3.AMemV3Error, json.JSONDecodeError)):
                decode(raw_fixture(text))

    def test_metadata_restrictions_apply_before_accepting_quote_derivation(self):
        fixtures = []
        for key, value in (('finishReason', 'MAX_TOKENS'), ('index', 1)):
            raw = raw_fixture('{"x":\'a\'}'); raw['response']['candidates'][0][key] = value; fixtures.append(raw)
        raw = raw_fixture('{"x":\'a\'}'); raw['response']['candidates'].append(copy.deepcopy(raw['response']['candidates'][0])); fixtures.append(raw)
        raw = raw_fixture('{"x":\'a\'}'); raw['response']['modelVersion'] = 'other-model'; fixtures.append(raw)
        raw = raw_fixture('{"x":\'a\'}'); raw['response']['promptFeedback'] = {'blockReason': 'SAFETY'}; fixtures.append(raw)
        raw = raw_fixture('{"x":\'a\'}'); raw['response']['candidates'][0]['content']['parts'][0]['thought'] = True; fixtures.append(raw)
        raw = raw_fixture('{"x":\'a\'}'); raw['request']['body']['generationConfig']['temperature'] = 0; fixtures.append(raw)
        raw = raw_fixture('{"x":\'a\'}'); raw['request']['body']['generationConfig']['thinkingConfig']['thinkingBudget'] = 1; fixtures.append(raw)
        for raw in fixtures:
            with self.subTest(raw=raw), self.assertRaises(v3.AMemV3Error): decode(raw)

    def test_private_science_prompt_actions_and_memory_are_unchanged(self):
        before = (v3.SCHEMA, v3._private.SCHEMA, v3._runtime.SCHEMA, v3._Controller)
        response = evolution(should_evolve=True, actions=['strestrengthen', 'update_neighbor'], suggested_connections=['0'],
                             tags_to_update=['not-applied'], new_context_neighborhood=['Updated'], new_tags_neighborhood=[['tag']])
        old_api, new_api = FakeAPI([evolution(), response]), FakeAPI([evolution(), response])
        with tempfile.TemporaryDirectory() as directory, patch.object(v3._runtime, '_get_encoder', return_value=FakeEncoder()), \
             patch.object(v4._runtime, '_get_encoder', return_value=FakeEncoder()):
            old = v3.build(conversation(), Path(directory) / 'v3', old_api)
            new = v4.build(conversation(), Path(directory) / 'v4', new_api)
            for key in ('notes', 'corpus', 'memory_sha256', 'evo_cnt'):
                self.assertEqual(old[key], new[key])
            self.assertEqual(old_api.calls, new_api.calls)
        self.assertEqual(before, (v3.SCHEMA, v3._private.SCHEMA, v3._runtime.SCHEMA, v3._Controller))

    @unittest.skipUnless((RUN / 'work/api_method/api_cache' / (KEY + '.json')).exists(), 'Original evidence archived')
    def test_real_failure_has_exactly_seven_literals_and_native_schema_accepts(self):
        blob = shared_read(RUN / 'work/api_method/api_cache' / (KEY + '.json'))
        self.assertEqual(hashlib.sha256(blob).hexdigest(), RAW_SHA)
        value, journal = v4.decode_generation_json(json.loads(blob), raw_artifact_sha256=RAW_SHA)
        self.assertEqual(len(journal['changes']), 14)
        self.assertEqual([x['value'] for x in journal['single_quoted_value_tokens']],
                         ['gratitude', 'collaboration', 'commitment', 'spreading positivity', 'global impact', 'recognition of impact', 'compassion'])
        self.assertEqual(journal['derived_text_utf8_sha256'], '48f83282d5521f19b2fa00fa686faf6f258a9943c853cbd0ac344a4460a183d3')
        self.assertEqual(journal['parsed_payload_sha256'], '387807fa997d19dcd3880fc0afaf8d7a983734ef1b2e15dd470eb09477454552')
        call = json.loads(shared_read(RUN / 'work/native/c2/calls/note_000454_amem_evolution.json'))
        controller = v4._Controller(None, RUN / 'work/native/c2'); controller.memory_count = 453
        context = call['request']['prompt'].split('The nearest neighbors memories:', 1)[-1].split('Based on this information', 1)[0]
        controller.neighbor_indices = [int(x) for x in re.findall(r'memory index:(\d+)', context)]
        derived, _ = controller._derive(value, call['request']['response_format']['json_schema']['schema'], 'amem_evolution')
        self.assertEqual(derived['new_tags_neighborhood'], value['new_tags_neighborhood'])

    @unittest.skipUnless((RUN / 'work/native/c2/calls').exists(), 'Original evidence archived')
    def test_actual_907_successful_prefix_payloads_and_journals_identical(self):
        count = 0
        for path in sorted((RUN / 'work/native/c2/calls').glob('*.json')):
            call = json.loads(shared_read(path))
            if call['status'] != 'success': continue
            lineage = call['transport_lineage']; blob = shared_read(lineage['cache_path'])
            h = hashlib.sha256(blob).hexdigest(); self.assertEqual(h, lineage['cache_sha256'])
            raw = json.loads(blob)
            result = v4.decode_generation_json(raw, raw_artifact_sha256=h)
            self.assertEqual(result, v3.decode_generation_json(raw, raw_artifact_sha256=h))
            self.assertEqual(result[0], call['raw_response']); count += 1
        self.assertEqual(count, 907)

    def test_original_c5_c7_bad_json_is_still_rejected(self):
        previous = RUN.parent / 'amem_format_v2/work/api_method/api_cache'
        keys = ['3154f0663543a20a85224a2523a98b453acf3d1e7e0753aa90a10463d28cc414',
                '471f602bbb3fd27c2bc803bab98940f20aca4b86a887c1e035869d820ec4ce1a']
        for key in keys:
            path = previous / (key + '.json')
            if not path.exists(): continue
            blob = shared_read(path)
            with self.subTest(key=key), self.assertRaises((v3.AMemV3Error, json.JSONDecodeError)):
                v4.decode_generation_json(json.loads(blob), raw_artifact_sha256=hashlib.sha256(blob).hexdigest())


if __name__ == '__main__': unittest.main()
