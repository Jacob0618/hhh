"""Exact historical-version binding; all files read only."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from tools import archive_hypermem_native_v3 as a


class HistoricalSources(unittest.TestCase):
    def test_exact_failed_report_preserves_every_noncode_field_and_old_bytes(self):
        path=a.ROOT/a.FAILED_PREFIX_REPORT;value=a.read(path);before=a.sha(path);journal=[]
        result=a.historical_failed_source_projection(value,path,journal)
        self.assertEqual({k:v for k,v in result.items() if k not in ('code_sha256','historical_source_snapshot_lineage')},
            {k:v for k,v in value.items() if k!='code_sha256'})
        self.assertEqual(len(result['code_sha256']),4);self.assertEqual(len(journal),1)
        for p,h in result['code_sha256'].items():self.assertEqual(a.sha(p),h)
        self.assertEqual(a.sha(path),before)
        self.assertEqual(set(result['code_sha256'].values()),set(value['code_sha256'].values()))

    def test_unknown_document_changed_report_capsule_or_mapping_hard_reject(self):
        path=a.ROOT/a.FAILED_PREFIX_REPORT;value=a.read(path)
        with self.assertRaises(ValueError):a.historical_failed_source_projection(value,path.with_name('other.json'))
        for changes in ({'status':'passed'},{'production_launch_authorized':True},{'code_sha256':{}}):
            with self.assertRaises(ValueError):a.historical_failed_source_projection({**value,**changes},path)
        lineage=a.read(a.ROOT/a.FAILED_SOURCE_LINEAGE);capsule=Path(lineage['code_snapshots'][0]['snapshot']['path']);original_sha=a.sha
        with patch.object(a,'sha',side_effect=lambda p:'0'*64 if Path(p)==capsule else original_sha(p)):
            with self.assertRaisesRegex(ValueError,'capsule'):a.historical_failed_source_projection(value,path)
        original_read=a.read;bad=deepcopy(lineage);bad['code_snapshots'][0]['original_path']=str(a.ROOT/'unknown.py')
        with patch.object(a,'read',side_effect=lambda p:bad if Path(p)==a.ROOT/a.FAILED_SOURCE_LINEAGE else original_read(p)):
            with self.assertRaises(ValueError):a.historical_failed_source_projection(value,path)

    def test_actual_failed_report_full_recursive_closure_uses_capsules_only_for_exact_four_versions(self):
        path=a.ROOT/a.FAILED_PREFIX_REPORT;journal=[]
        records=a.evidence_closure([(str(path),a.FAILED_PREFIX_REPORT_SHA)],annotations=journal)
        self.assertEqual(records[str(path)],a.FAILED_PREFIX_REPORT_SHA)
        self.assertEqual(records[str(a.ROOT/a.FAILED_SOURCE_LINEAGE)],a.FAILED_SOURCE_LINEAGE_SHA)
        for row in a.read(a.ROOT/a.FAILED_SOURCE_LINEAGE)['code_snapshots']:
            self.assertEqual(records[row['snapshot']['path']],row['original_sha256'])
            if row['original_path'] in records:
                self.assertEqual(records[row['original_path']],a.sha(row['original_path']))
        self.assertTrue(any(r.get('json_pointer')=='$.code_sha256' and r.get('document')==str(path) for r in journal))


if __name__=='__main__':unittest.main()
