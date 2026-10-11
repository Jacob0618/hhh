"""Offline rounded-mass representation sensitivity; retain raw values/choice.

This is a post-scoring two-decimal compatibility hypothesis, not a provider
precision assertion. The frozen strict parser and scientific results remain.
"""
from __future__ import annotations
import math
from types import FunctionType
from jev_mem import source_set_r11 as strict

POLICY = {'version': 'source-set-r11-rounded-mass-compat-v1',
    'grid_denominator': 100, 'grid_machine_epsilon': 1e-12,
    'interval_denominator': 200, 'raw_probabilities_unchanged': True,
    'raw_choice_unchanged': True, 'normalization': False,
    'strict_primary_preserved': True, 'model_calls': 0}

# Clone the exact strict parser with only its mass test disabled locally. Every
# schema/value/confidence/choice/argmax check is still the original bytecode.
# The new acceptance certificate below supplies the narrowly bounded mass rule.
_scope = {**strict.parse_p.__globals__, 'POLICY': {**strict.POLICY,
    'probability_sum_tolerance': math.inf}}
_parse_other_contracts = FunctionType(strict.parse_p.__code__, _scope,
    '_parse_other_contracts', strict.parse_p.__defaults__, strict.parse_p.__closure__)


def rounding_certificate(response, signature):
    """Validate every other strict field, then certify strict/rounded/rejected.

Malformed models/schema/values/confidence/nonmax choice raise ValueError.
Rejected mass representations return accepted=False plus diagnostic bounds.
No probability, maximum, tie or returned choice is repaired or substituted.
"""
    parsed = _parse_other_contracts(response, signature)
    probs = parsed['probabilities']
    raw_sum = math.fsum(probs.values())
    strict_valid = abs(raw_sum - 1) <= strict.POLICY['probability_sum_tolerance']
    centers = {key: round(value * 100) for key, value in probs.items()}
    grid_error = {key: abs(value - centers[key] / 100) for key, value in probs.items()}
    lattice = all(error <= POLICY['grid_machine_epsilon'] for error in grid_error.values())
    # Individual nearest-cent intervals are [k/100-.005,k/100+.005],
    # intersected with [0,1]. Integer numerators avoid a loose .03 mass cutoff.
    lower = {key: max(0, 2 * k - 1) for key, k in centers.items()}
    upper = {key: min(200, 2 * k + 1) for key, k in centers.items()}
    lower_sum, upper_sum = sum(lower.values()), sum(upper.values())
    interval_feasible = lower_sum <= 200 <= upper_sum
    rounded_compatible = lattice and interval_feasible
    accepted = strict_valid or rounded_compatible
    mode = 'strict' if strict_valid else 'rounded_mass' if rounded_compatible else 'rejected'
    reason = None if accepted else ('probabilities_not_two_decimal_lattice' if not lattice
        else 'rounding_intervals_cannot_sum_to_one')
    return {'schema': POLICY['version'] + '-certificate', 'mode': mode, 'accepted': accepted,
        'strict_valid': strict_valid, 'rounded_mass_compatible': rounded_compatible,
        'raw_sum': raw_sum, 'grid_denominator': 100,
        'grid_machine_epsilon': POLICY['grid_machine_epsilon'],
        'integer_cent_values': centers, 'grid_errors': grid_error,
        'interval_denominator': 200, 'lower_numerators': lower, 'upper_numerators': upper,
        'lower_sum_numerator': lower_sum, 'upper_sum_numerator': upper_sum,
        'interval_sum_one_feasible': interval_feasible,
        'original_choice': parsed['choice'], 'tied_max': parsed['tied_max'], 'reason': reason,
        'representation_hypothesis_not_provider_contract': True, 'model_calls': 0}


def parse_p_compatible(response, signature):
    certificate = rounding_certificate(response, signature)
    if not certificate['accepted']:
        raise ValueError('Probability mass is neither strict nor rounded-mass compatible')
    return _parse_other_contracts(response, signature)


_choose_scope = {**strict.choose_p.__globals__, 'parse_p': parse_p_compatible}
choose_p_compatible = FunctionType(strict.choose_p.__code__, _choose_scope,
    'choose_p_compatible', strict.choose_p.__defaults__, strict.choose_p.__closure__)
choose_p_compatible.__kwdefaults__ = strict.choose_p.__kwdefaults__.copy()
