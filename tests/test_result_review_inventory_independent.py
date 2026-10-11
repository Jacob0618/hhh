"""A passed summary alone must not approve a different writer/source snapshot."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import archive_amem_continuation_v3 as amem
from tools import baseline_lifecycle_v1 as single


class IndependentResultSnapshotTests(unittest.TestCase):
    def test_amem_summary_only_review_cannot_authorize_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);roots={k:root/k for k in ('v1','v2','v3')};run=roots['v3']
            summary=run/'results/summary.json';summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({'audit_status':'no_invalid_artifacts',
                'effective':{'slots':1540,'reader_valid':1540,'judge_valid':1540}}))
            review=root/'review.json';review.write_text(json.dumps({'status':'passed','summary_sha256':amem.sha(summary)}))
            with patch.object(amem,'ROOT',root),patch.object(amem,'ROOTS',roots):
                with self.assertRaises(ValueError):amem.verify_result_review(review)

    def test_single_root_summary_only_review_cannot_authorize_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runs=root/'runs';run=runs/'simplemem'
            summary=run/'results/summary.json';summary.parent.mkdir(parents=True)
            summary.write_text(json.dumps({'schema':'five-baseline-result-v1','method':'simplemem',
                'audit_status':'no_invalid_artifacts','effective':{'slots':1540,'reader_valid':1540,'judge_valid':1540}}))
            (run/'status.json').write_text(json.dumps({'stage':'completed_awaiting_result_review','summary_sha256':single.sha(summary)}))
            review=root/'review.json';review.write_text(json.dumps({'status':'passed','method':'simplemem','summary_sha256':single.sha(summary)}))
            manifest={'method':'simplemem','questions':[{'query_id':str(i)} for i in range(1540)]}
            with patch.object(single,'ROOT',root),patch.object(single,'RUN',runs),patch.object(single.base,'manifest',return_value=manifest):
                with self.assertRaises(ValueError):single.verify_result('simplemem',review)


if __name__=='__main__':unittest.main()
