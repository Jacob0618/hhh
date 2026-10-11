"""Pure compatibility tests; no files from the live run, gold or API are used."""
from copy import deepcopy
import math
import unittest
from jev_mem import source_set_r11 as strict
from jev_mem import source_set_rounding_r11 as compat
from jev_mem import source_append_r05 as source


def memories():
    return [{'id': str(i), 'text': f'Memory {i}', 'observation_date': None} for i in range(20)]


def candidates():
    out = []
    for i in range(4):
        view = {'session_id': 's', 'source_observation_date': '2023-01-01',
            'turns': [{'source_turn_id': str(i), 'speaker_text': f'A: source {i}', 'machine_captions': []}]}
        text = source.attachment(view)
        out.append({'packet_id': source.digest(view), 'packet_view': view, 'attachment': text,
            'attachment_utf8_bytes': len(text.encode('utf-8')),
            'origin': {'anchor_slot': i, 'anchor_id': str(i)}})
    return out


def signature():
    protocol = {'p': {'instructions': 'Only compare each pair with the memories.',
        'criteria_template': 'Candidates {first} and {second}.',
        'presentation_seed': strict.POLICY['presentation_seed']}}
    return strict.build_p_request('c0_q0', 'What happened?', None, memories(), candidates(), protocol)


def response(values, choice='p00', confidence=0):
    return {'model': strict.MODEL, 'answers': {'pair_choice': {'type': 'choice',
        'probabilities': dict(zip(strict.PAIR_KEYS, values)), 'choice': choice, 'confidence': confidence}}}


class RoundingTests(unittest.TestCase):
    def test_strict_results_identical_including_non_cent_probabilities(self):
        r = response([0.54321, 0.25679, 0.1, 0.1, 0, 0]); sig = signature()
        self.assertEqual(compat.parse_p_compatible(r, sig), strict.parse_p(r, sig))
        certificate = compat.rounding_certificate(r, sig)
        self.assertEqual(certificate['mode'], 'strict')
        self.assertFalse(certificate['rounded_mass_compatible'])

    def test_point99_and_point101_retained_without_normalization(self):
        for values in ([0.93, 0.05, 0.01, 0, 0, 0], [0.91, 0.05, 0.02, 0.01, 0.01, 0.01]):
            r = response(values); original = deepcopy(r); sig = signature()
            with self.assertRaises(ValueError): strict.parse_p(r, sig)
            parsed = compat.parse_p_compatible(r, sig)
            self.assertEqual(parsed['probabilities'], r['answers']['pair_choice']['probabilities'])
            self.assertEqual(parsed['choice'], 'p00')
            self.assertEqual(r, original)
            self.assertEqual(compat.rounding_certificate(r, sig)['mode'], 'rounded_mass')
            d = compat.choose_p_compatible(memories(), candidates(), parsed, signature=sig)
            self.assertEqual(d['candidate_indices'], sig['body']['state']['bundle_options']['p00'])

    def test_float_roundtrip_tail_accepted_but_epsilon_not_widened(self):
        r = response([0.5700000000000001, 0.28, 0.14, 0, 0, 0])
        self.assertEqual(compat.rounding_certificate(r, signature())['mode'], 'rounded_mass')
        r['answers']['pair_choice']['probabilities']['p00'] = 0.57 + 2e-12
        self.assertFalse(compat.rounding_certificate(r, signature())['accepted'])
        with self.assertRaises(ValueError): compat.parse_p_compatible(r, signature())

    def test_clamped_zero_interval_and_exact_integer_certificate(self):
        r = response([0.98, 0, 0, 0, 0, 0]); cert = compat.rounding_certificate(r, signature())
        self.assertEqual(cert['lower_numerators']['p01'], 0)
        self.assertEqual(cert['upper_numerators']['p01'], 1)
        self.assertTrue(cert['accepted'])
        self.assertEqual(cert['lower_sum_numerator'], 195)
        self.assertEqual(cert['upper_sum_numerator'], 202)

    def test_interval_impossible_and_non_lattice_not_rescued_by_small_mass_error(self):
        for values in ([0.94, 0, 0, 0, 0, 0], [0.5005, 0.2995, 0.19, 0, 0, 0]):
            r = response(values); cert = compat.rounding_certificate(r, signature())
            self.assertFalse(cert['accepted'])
            with self.assertRaises(ValueError): compat.parse_p_compatible(r, signature())

    def test_tied_max_still_baseline_even_rounded_mass(self):
        r = response([0.27, 0.27, 0.14, 0.13, 0.11, 0.07]); sig = signature()
        parsed = compat.parse_p_compatible(r, sig)
        self.assertTrue(parsed['tied_max'])
        d = compat.choose_p_compatible(memories(), candidates(), parsed, signature=sig)
        self.assertEqual(d['reason'], 'probability_maximum_tie_fallback')
        self.assertEqual(strict.selected_facts(memories(), candidates(), d), memories())

    def test_nonmax_choice_not_replaced_even_when_gap_one_cent(self):
        r = response([0.29, 0.30, 0.18, 0.16, 0.05, 0.02], choice='p00')
        for function in (compat.rounding_certificate, compat.parse_p_compatible):
            with self.assertRaises(ValueError): function(r, signature())

    def test_all_other_original_contract_checks_retained(self):
        base = response([0.93, 0.05, 0.01, 0, 0, 0])
        changes = [lambda r: r.update(model='wrong'),
            lambda r: r['answers'].update(extra={}),
            lambda r: r['answers']['pair_choice'].update(type='noul'),
            lambda r: r['answers']['pair_choice']['probabilities'].pop('p05'),
            lambda r: r['answers']['pair_choice']['probabilities'].update(p00=True),
            lambda r: r['answers']['pair_choice']['probabilities'].update(p00=math.nan),
            lambda r: r['answers']['pair_choice'].update(confidence=True),
            lambda r: r['answers']['pair_choice'].update(confidence=1.1)]
        for change in changes:
            r = deepcopy(base); change(r)
            with self.assertRaises(ValueError): compat.parse_p_compatible(r, signature())

    def test_original_parser_and_policy_not_mutated(self):
        self.assertEqual(strict.POLICY['probability_sum_tolerance'], 1e-6)
        self.assertIs(strict.choose_p.__globals__['parse_p'], strict.parse_p)
        r = response([0.93, 0.05, 0.01, 0, 0, 0])
        compat.parse_p_compatible(r, signature())
        with self.assertRaises(ValueError): strict.parse_p(r, signature())


if __name__ == '__main__':
    unittest.main()
