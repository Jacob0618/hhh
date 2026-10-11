"""Replay mature physical-request fixtures against the refinement dispatcher."""
import unittest
from unittest.mock import patch
from tests import test_module_stack_dispatch_v1 as fixtures
from tools import prepare_r03_r05_refinement_v1 as prep
from tools import score_r03_r05_refinement_v1 as score


class DispatcherTests(unittest.TestCase):
    def run_original(self, name):
        # Only test-local module routing changes; production old code stays untouched.
        with patch.object(fixtures,'prepare',prep),patch.object(fixtures,'score',score):
            case=fixtures.DispatchTests(name)
            getattr(case,name)()

    def test_exact_cached_producer_consumer_raw(self):
        self.run_original('test_real_exact_cache_producer_and_consumer_raw_replay')

    def test_exhausted_transport_known_and_uniform_recovery_gate(self):
        self.run_original('test_exhausted_transport_consumer_without_ledger_event_is_known_failure')

    def test_partial_or_missing_raw_unknown(self):
        self.run_original('test_partial_transport_and_http200_without_raw_stay_unknown')

    def test_failed_raw_replay_changed_signature_rejected(self):
        self.run_original('test_failed_raw_replays_parser_and_changed_signature_rejected')

    def test_consumer_event_required(self):
        self.run_original('test_complete_result_requires_its_consumer_event')

    def test_exact_hash_context_pair_required(self):
        self.run_original('test_ledger_hash_and_context_must_be_admitted_as_a_pair')

    def test_only_known_failed_once_same_retries(self):
        self.run_original('test_recovery_uses_original_retries_and_failed_subset_once')

    def test_original_bytecode_is_reused_new_globals_bound(self):
        for name in ('context','classify_failure','one','verify_result'):
            a,b=getattr(score,name),getattr(score.prior,name)
            self.assertIs(a.__code__,b.__code__)
            self.assertEqual(a.__globals__['STAGE'],score.STAGE)
        self.assertIs(score.one.__globals__['core'],score.core)
        self.assertIs(score.verify_result.__globals__['classify_failure'],score.classify_failure)


if __name__=='__main__':unittest.main()
