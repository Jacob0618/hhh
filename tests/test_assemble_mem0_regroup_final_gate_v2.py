"""Gate assembly boundary checks, without source clients or gate publication."""
import copy
import unittest
from unittest.mock import patch
from tools import assemble_mem0_regroup_final_qa_gate_v2 as a


class GateAssemblyTests(unittest.TestCase):
    def test_missing_review_stops_before_source_assembly(self):
        with patch.object(a.sources,'assemble_rows') as assembled:
            with self.assertRaises(FileNotFoundError):a.build({'development':'absent.json'})
            assembled.assert_not_called()

    def test_real_two_quality_protocols_bind_their_actual_targets(self):
        snap=a.Snapshot()
        for eof in (False,True):
            core=a.load(snap,a.DEFAULTS['eof_core' if eof else 'core'])
            quality=a.load(snap,a.DEFAULTS['eof_quality' if eof else 'quality'])
            kwargs={'eof':True,'bank':core['banks'][0]['bank'],'core_ref':a.pointer(snap,a.DEFAULTS['eof_core'])} if eof else {}
            a.validate_quality(snap,quality,a.quality_targets(core,include_eof=eof),**kwargs)
        self.assertEqual(snap.unchanged(),[])

    def test_missing_or_unread_quality_target_is_not_accepted(self):
        snap=a.Snapshot();core=a.load(snap,a.DEFAULTS['eof_core']);quality=a.load(snap,a.DEFAULTS['eof_quality'])
        kwargs={'eof':True,'bank':core['banks'][0]['bank'],'core_ref':a.pointer(snap,a.DEFAULTS['eof_core'])}
        changed=copy.deepcopy(quality);changed['groups'].pop()
        with self.assertRaisesRegex(ValueError,'Every targeted'):a.validate_quality(snap,changed,a.quality_targets(core,include_eof=True),**kwargs)
        changed=copy.deepcopy(quality);changed['groups'][0]['all_extracted_records_read']=False
        with self.assertRaisesRegex(ValueError,'completed review'):a.validate_quality(snap,changed,a.quality_targets(core,include_eof=True),**kwargs)

    def test_preflight_rejection_publishes_no_persistent_gate(self):
        calls=[]
        def reject(path):calls.append(path);raise ValueError('Review not accepted')
        with patch.object(a.sources,'admit_gate',side_effect=reject):
            with self.assertRaisesRegex(ValueError,'not accepted'):a.preflight({'banks':[]},{'banks':[]})
        self.assertEqual(len(calls),1)
        self.assertFalse(calls[0].exists());self.assertFalse(calls[0].parent.exists())

    def test_publication_destination_checked_before_any_source_reads(self):
        with patch.object(a,'build') as build:
            with self.assertRaises(ValueError):a.publish(a.ROOT/'runs/forbidden.json',a.ROOT/'research/new.json')
            build.assert_not_called()


if __name__=='__main__':unittest.main()
