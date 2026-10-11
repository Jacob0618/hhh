"""Offline source-admission adapter tests; no real namespace/model/store opened."""
import copy
from contextlib import ExitStack
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_mem0_regroup_plain_qa_v1 as fixtures
from tools import mem0_regroup_plain_qa_sources_v2 as src
from tools import run_mem0_regroup_plain_qa_v2 as run
from tools.run_sdk20_modules import save_new, verify
from tools.run_pilot400_experiments_v2 import publish_json


class SourceV2Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PlainQATests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.origins, self.parents, old_gate = self.fixture.fake_twenty_gate()
        old = verify(old_gate)
        self.rows = old['banks']
        self.export = self.root / 'export'
        self.eof = self.root / 'eof'
        self.export_results = {}
        for row in self.rows:
            ci, arm = row['conversation'], row['arm']
            row['source_kind'] = src.expected_kind(ci, arm)
            row['native_index_limitations'] = []
            if ci == 3:
                # Metadata export preserves the original failed state. Its pure
                # validator is delegated; common input/fact/native checks remain real.
                sp, op = Path(row.pop('write_status')['path']), Path(row.pop('outcome')['path'])
                status, outcome = src.read(sp), verify(op)
                status['status'] = 'failed'; publish_json(sp, status)
                outcome.pop('content_sha256'); outcome.update(status='failed', write_status=src.pointer(sp))
                op.unlink(); save_new(op, outcome)
                row['original_write_status'], row['original_outcome'] = src.pointer(sp), src.pointer(op)
                ep = self.export / f'conv_{ci}' / arm / 'reviewed_export.json'
                save_new(ep, {'mock': 'reviewed original-failed metadata'})
                row['reviewed_export'] = src.pointer(ep)
                row['source_review'] = row['audits']['index'][0]
                row['native_index_limitations'] = ['One original missing entity link remains']
                self.export_results[str(ep)] = {'bank': verify(row['bank']['path']),
                    'native': self.origins[2] / f'conv_{ci}' / arm / 'native',
                    'review': {k: row[k] for k in ('bank', 'native_inventory', 'source_kind', 'original_write_status',
                                                  'original_outcome', 'reviewed_export', 'source_manifest', 'source_review')}}
            if (ci, arm) == (8, 'dated_merged'):
                # Separate new complete reconstruction, with old failure pointers.
                sp, op = Path(row['write_status']['path']), Path(row['outcome']['path'])
                status, outcome, bank = src.read(sp), verify(op), verify(row['bank']['path'])
                old_status = copy.deepcopy(status); old_status.update(status='failed', completed_groups=0)
                publish_json(sp, old_status)
                old_outcome = {k:v for k,v in outcome.items() if k != 'content_sha256'}
                old_outcome.update(status='failed', write_status=src.pointer(sp)); op.unlink(); save_new(op, old_outcome)
                row['original_write_status'], row['original_outcome'] = src.pointer(sp), src.pointer(op)
                folder = self.eof / f'conv_{ci}' / arm; native = folder / 'native'
                (native / 'qdrant').mkdir(parents=True)
                (native / 'history.sqlite').write_bytes(b'closed recovery history')
                (native / 'qdrant/meta.json').write_text('{}')
                bank.pop('content_sha256'); bank.update(native_dir=str(native.resolve()), protocol_sha256='eof')
                save_new(folder / 'bank.json', bank)
                status.update(protocol_sha256='eof', bank=src.pointer(folder / 'bank.json'))
                publish_json(folder / 'write_status.json', status)
                outcome.pop('content_sha256'); outcome.update(protocol_sha256='eof', bank=status['bank'], write_status=src.pointer(folder / 'write_status.json'))
                save_new(folder / 'eof_recovery_outcome.json', outcome)
                self.eof_parent = copy.deepcopy(self.parents[str(self.origins[2].resolve())])
                self.eof_parent['protocol_sha256'] = 'eof'
                self.eof_parent['eof_recovery'] = {'resources': self.eof_parent['remaining_expansion']['resources']}
                publish_json(self.eof / 'run_manifest.json', self.eof_parent)
                for key, name in [('bank','bank.json'),('write_status','write_status.json'),('outcome','eof_recovery_outcome.json')]:
                    row[key] = src.pointer(folder / name)
                row['source_manifest'] = src.pointer(self.eof / 'run_manifest.json')
                row['native_inventory'] = src.inventory(native)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for key, val in zip(('DEVELOPMENT','FOURTH','REMAINING'), self.origins):
            self.stack.enter_context(patch.object(src.legacy,key,val))
        self.stack.enter_context(patch.object(src,'EXPORT_ROOT',self.export))
        self.stack.enter_context(patch.object(src,'EOF_ROOT',self.eof))
        self.stack.enter_context(patch.object(src.legacy,'source_manifests',return_value=self.parents))
        self.stack.enter_context(patch.object(src.legacy,'validate_resources'))
        self.stack.enter_context(patch.object(src.eof,'manifest',return_value=self.eof_parent))
        self.complete = self.stack.enter_context(patch.object(src.eof,'validate_complete',return_value={}))
        self.stack.enter_context(patch.object(src.exports,'check_export_bank',side_effect=lambda p:self.export_results[p['path']]))
        self.stack.enter_context(patch.object(run.frozen.native,'open_memory',side_effect=AssertionError('Original client forbidden')))

    def gate(self, rows=None, covered=None):
        rows = copy.deepcopy(rows or self.rows)
        index = len(list(self.root.glob('gatev2*')))
        uniform = self.root / f'uniform{index}.json'
        save_new(uniform, {'banks': covered if covered is not None else rows})
        gate = self.root / f'gatev2_{index}.json'
        save_new(gate, {'schema':src.GATE_SCHEMA,'status':'passed','checks':{k:True for k in src.CHECKS},
                        'banks':rows,'uniform_twenty_bank_review':src.pointer(uniform),'limitations':['Native defects retained; not perfect entity links']})
        return gate

    def test_exact_twenty_different_source_kinds_admitted_without_changing_failures(self):
        gate,banks,_ = src.admit_gate(self.gate())
        self.assertEqual(len(banks),20); self.complete.assert_called_once()
        for arm in src.ARMS:
            row = banks[3,arm]['review']
            self.assertNotIn('write_status',row)
            self.assertEqual(src.read(row['original_write_status']['path'])['status'],'failed')
        self.assertEqual(banks[8,'dated_merged']['native'],self.eof/'conv_8/dated_merged/native')

    def test_wrong_exception_scope_rejected_before_recovery_read(self):
        rows=copy.deepcopy(self.rows);rows[0]['source_kind']='reviewed_export'
        with self.assertRaisesRegex(ValueError,'source exception'):src.admit_gate(self.gate(rows))
        self.complete.assert_not_called()

    def test_uniform_review_must_cover_current_twenty_not_original_partial(self):
        covered=copy.deepcopy(self.rows);covered[-3]['native_inventory']={'different':'hash'}
        with self.assertRaisesRegex(ValueError,'another source'):src.admit_gate(self.gate(covered=covered))

    def test_partial_recovery_never_admitted(self):
        self.complete.side_effect=ValueError('partial29 bank')
        with self.assertRaisesRegex(ValueError,'partial29'):src.admit_gate(self.gate())

    def test_export_cannot_spoof_original_complete_status(self):
        rows=copy.deepcopy(self.rows);r=next(v for v in rows if v['conversation']==3)
        r['write_status']=r['original_write_status']
        with self.assertRaisesRegex(ValueError,'preserve original failed'):src.admit_gate(self.gate(rows))

    def test_failed_source_provenance_preserved_for_recovery(self):
        rows=copy.deepcopy(self.rows);r=next(v for v in rows if (v['conversation'],v['arm'])==(8,'dated_merged'))
        r['original_write_status']=r['write_status']
        with self.assertRaisesRegex(ValueError,'another lineage'):src.admit_gate(self.gate(rows))

    def test_runtime_binding_restores_and_uses_top_level_spawn_function(self):
        old=(run.frozen.sources,run.frozen.worker,run.frozen.CODE)
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with run.source_binding():
                self.assertIs(run.frozen.sources,src);self.assertIs(run.frozen.worker,run.worker)
                self.assertTrue(set(src.VALIDATOR_CODE)<=set(run.frozen.CODE))
                with self.assertRaisesRegex(RuntimeError,'nested'):
                    with run.source_binding():pass
                raise RuntimeError('synthetic')
        self.assertEqual((run.frozen.sources,run.frozen.worker,run.frozen.CODE),old)

    def test_missing_gate_rejected_before_namespace_or_cache_creation(self):
        with patch.object(run.frozen,'ROOT',self.root),patch.object(src.legacy,'protocol_config',return_value={}),patch.object(run.frozen.embedding_pack,'audit_sources') as packs:
            with self.assertRaises(FileNotFoundError):run.prepare(self.root/'runs/newqa',self.root/'absent.json')
            self.assertFalse((self.root/'runs').exists());packs.assert_not_called()

    def test_row_assembler_returns_provenance_not_a_passed_gate(self):
        audits={(r['conversation'],r['arm']):r['audits'] for r in self.rows}
        limits={(r['conversation'],r['arm']):r['native_index_limitations'] for r in self.rows}
        original=self.export_results
        with patch.object(src.exports,'check_export_bank',side_effect=lambda p:original[str(Path(p)/'reviewed_export.json')]):
            rows=src.assemble_rows(audits,limits)
        self.assertEqual(len(rows),20)
        self.assertEqual({(r['conversation'],r['arm']) for r in rows},set(audits))
        self.assertEqual(sum(r['source_kind']=='reviewed_export' for r in rows),2)
        self.assertFalse(any(r.get('status')=='passed' for r in rows))
        self.assertEqual(src.eof.__name__,'tools.recover_mem0_regroup_eof_v2')


if __name__=='__main__':unittest.main()
