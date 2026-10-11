"""Independent quote-only boundary and eligibility probes; no transport."""
import copy
import json
import random
import unittest
from tools import baselines_amem_v4 as v4
from tools import baselines_amem_v3 as v3
from tests.test_baselines_amem_v3 import raw_fixture

class IndependentV4DecoderTests(unittest.TestCase):
    def test_literal_values_survive_delimiters_unicode_and_numeric_spelling(self):
        rng = random.Random(23001)
        alphabet = 'abc XYZ0123456789[]{}:,;-/🙂中文'
        values = ['', 'null', 'false', 'True', 'None', 'NaN', '1e99', '[]{}:,', '中文🙂']
        values += [''.join(rng.choice(alphabet) for _ in range(rng.randrange(0,50))) for _ in range(200)]
        for literal in values:
            text = '{"same":"John\'s \\\"quoted\\\" words", "value":\'' + literal + '\',"next": [1,true,null]}'
            raw = raw_fixture(text); before = copy.deepcopy(raw)
            value, journal = v4.decode_generation_json(raw, raw_artifact_sha256='a'*64)
            self.assertEqual(value, {'same':'John\'s "quoted" words','value':literal,'next':[1,True,None]})
            self.assertEqual(raw,before)
            self.assertEqual(len(journal['changes']),2)
            self.assertEqual(journal['single_quoted_value_tokens'][0]['value'],literal)
            derived,changes,_ = v4._single_quote_tokens(text)
            inverse = list(derived)
            for change in changes: inverse[change['character_offset']]="'"
            self.assertEqual(''.join(inverse),text)

    def test_control_escape_and_metadata_cannot_be_hidden_by_quote_bridge(self):
        for char in [chr(i) for i in range(32)] + ['\\','"']:
            raw = raw_fixture('{"value":\'a' + char + 'b\'}')
            with self.subTest(character=ord(char)), self.assertRaises((v3.AMemV3Error,json.JSONDecodeError)):
                v4.decode_generation_json(raw,raw_artifact_sha256='a'*64)
        mutations = [
            lambda r:r['response']['candidates'][0]['content'].update(role='user'),
            lambda r:r['response']['candidates'][0]['content']['parts'].append({'text':'x'}),
            lambda r:r['response']['candidates'][0].update(safetyRatings=[{'blocked':True}]),
            lambda r:r['request']['body'].update(tools=[]),
            lambda r:r['request']['body']['contents'][0].update(role='model'),
        ]
        for change in mutations:
            raw=raw_fixture('{"value":\'abc\'}');change(raw)
            with self.assertRaises(v3.AMemV3Error):
                v4.decode_generation_json(raw,raw_artifact_sha256='a'*64)

if __name__ == '__main__': unittest.main()
