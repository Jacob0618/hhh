"""Changed writer evidence must fail before creating any archive output."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import archive_amem_continuation_v3 as a


class IndependentNativeSealTests(unittest.TestCase):
    def test_changed_writer_audit_blocks_archive_before_output_creation(self):
        from tools import run_five_baselines_amem_native_v3_admission_v2 as frozen
        for changed in ('raw','native'):
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);roots={name:root/name for name in ('v1','v2','v3')};new=roots['v3']
                (new/'results').mkdir(parents=True)
                proof=root/'proof.json';proof.write_text('{}')
                (new/'run_manifest.json').write_text(json.dumps({'code_sha256':{},'upstream_sha256':{},
                    'technical_continuation':{'proof':{'path':str(proof),'sha256':a.sha(proof)}}}))
                (new/'results/raw_request_audit.json').write_text('{"status":"passed"}')
                (new/'results/native_output_audit.json').write_text('{"status":"passed"}')
                raw_check=lambda:{'status':'changed' if changed=='raw' else 'passed'}
                native_check=lambda m:{'status':'changed' if changed=='native' else 'passed'}
                with patch.object(a,'ROOT',root),patch.object(a,'ROOTS',roots),patch.object(a,'stopped'), \
                     patch.object(a,'verify_result_review'),patch.object(a,'verify_tool_review'), \
                     patch.object(frozen,'audit_new_requests',raw_check),patch.object(frozen,'audit_native_outputs',native_check):
                    with self.assertRaisesRegex(ValueError,'differs from sealed'):
                        a.archive(root/'result_review.json',root/'tool_review.json')
                self.assertFalse((new/'results/continuation_archive').exists())


if __name__=='__main__':unittest.main()
