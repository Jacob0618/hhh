"""Actual immutable old evidence, temporary ZIP only; no production mutation.

This is separate from the synthetic current-native / 1540 QA fixture. It
replays the full hash-locked historical 6,214-native-call qualification using
only temporary archive bytes, without opening any model, index or DB.
"""
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools import archive_hypermem_native_v3 as a
from tools.baselines_common_v1 import read,sha,save_new


class ActualAncestorArchive(unittest.TestCase):
    @unittest.skipUnless(a.PARENT.joinpath('work').is_dir(),'Actual historical work was already cleaned')
    def test_full_original_and_parent_terminal_collector_from_zip_only(self):
        original=read(a.ROOT/a.COHORT_REVIEW)
        self.assertEqual(sha(a.ROOT/a.COHORT_REVIEW),a.COHORT_REVIEW_SHA)
        wanted={**original['original_inventory'],**original['current_inventory']}
        for name in ('code_sha256','upstream_sha256'):
            wanted.update({str(a.ROOT/p):h for p,h in original[name].items()})
        for record in (original['original_terminal_proof'],original['implementation_review'],original['role_qualification'],original['malformed_json_diagnosis']):wanted[record['path']]=record['sha256']
        roles=read(original['role_qualification']['path'])
        for name in ('code_sha256','evidence_sha256'):
            wanted.update({str(a.ROOT/p):h for p,h in roles[name].items()})
        wanted.update({str(a.ROOT/p):h for p,h in a.code_hashes().items()})
        for p in (a.ROOT/a.COHORT_REVIEW,a.ROOT/a.COLLECTOR,a.ROOT/'data/locomo10.json',
                  a.ROOT/'configs/sdk20_full1540_locked_v1.json',
                  Path(original['role_qualification']['path']).with_suffix('.py')):
            wanted[str(p)]=sha(p)
        proof=a._core().admission.normalize(original)
        # Normalization is in memory only. The published report and all
        # production files retain their original bytes and bindings.
        with tempfile.TemporaryDirectory(dir=a.ROOT/'research') as td:
            root=Path(td);mp=root/'manifest.json';zp=root/'evidence.zip';records=[]
            for p,h in sorted(wanted.items()):
                reason=a.disposition(p)
                self.assertIsNone(reason,'Historical complete audit needs all its original qualified bytes')
                records.append({'path':p,'sha256':h,'size':Path(p).stat().st_size,'retained':True,'member':'blobs/'+h})
            save_new(mp,{'schema':a.SCHEMA,'records':records})
            with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                z.writestr('archive_manifest.json',mp.read_bytes());seen=set()
                for row in records:
                    if row['member'] not in seen:
                        data=Path(row['path']).read_bytes()
                        self.assertEqual(a.hashlib.sha256(data).hexdigest(),row['sha256'])
                        z.writestr(row['member'],data);seen.add(row['member'])
            with a.Resolver(zp,mp) as view:
                runtime=a.frozen.metric.prepare_metric_runtime()
                with a._read_context(),a.frozen.metric.only_zip_workspace_reads(view,runtime):
                    result=a.audit_ancestors(view,proof)
                self.assertEqual(result['native_calls'],6214)
                self.assertEqual(result['original_file_count'],380)
                self.assertEqual(result['parent_file_count'],25252)
                self.assertEqual(result['unknown_or_inflight'],[])
        # One complete after comparison; never mutate, delete or recertify
        # original run evidence or repeatedly rescan during individual reads.
        self.assertEqual(a.hashes(a.inventory(a.OLD)),proof['original_inventory'])
        self.assertEqual(a.hashes(a.inventory(a.PARENT)),proof['current_inventory'])


if __name__=='__main__':unittest.main()
