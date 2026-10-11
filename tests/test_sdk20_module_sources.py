import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tools import sdk20_module_sources as src


class SourceCopyTests(unittest.TestCase):
    def setup_sources(self,root):
        result={}
        for kind in ('session','preseg'):
            native=root/'originals'/kind
            (native/'qdrant/collection/facts').mkdir(parents=True)
            (native/'qdrant/collection/facts/storage.sqlite').write_bytes(b'original-'+kind.encode())
            (native/'qdrant/meta.json').write_text('{}',encoding='utf8')
            (native/'qdrant/.lock').write_bytes(b'oldlock')
            (native/'history.sqlite').write_bytes(b'history')
            (native/'api_calls.jsonl').write_bytes(b'must not copy ledger')
            result[kind]={'facts':[],'user_id':kind,'native_source':str(native)}
        return result

    def test_new_copy_is_exact_without_ledger_or_lock_and_no_second_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);banks=self.setup_sources(root)
            before={k:src._native_inventory(v['native_source']) for k,v in banks.items()}
            with patch.object(src,'load_banks',return_value=banks):
                out=src.prepare_conv(root/'new_run',0)
                with self.assertRaises(FileExistsError):src.prepare_conv(root/'new_run',0)
            for kind in banks:
                native=Path(out['banks'][kind]['native_dir'])
                self.assertEqual(src._native_inventory(native),before[kind])
                self.assertEqual(src._native_inventory(banks[kind]['native_source']),before[kind])
                self.assertFalse((native/'api_calls.jsonl').exists())
                self.assertFalse((native/'qdrant/.lock').exists())
                bank=json.loads(Path(out['banks'][kind]['bank']['path']).read_text(encoding='utf8'))
                self.assertEqual(bank['source_inventory'],before[kind])
            self.assertEqual(out['inherited_ledger_bytes'],0)
            self.assertTrue((root/'new_run/conv_0/source_manifest.json').is_file())

    def test_cannot_target_old_or_parent_runs(self):
        for path in (src.BASE,src.BASE/'child',src.BASE.parent):
            with self.assertRaises(ValueError):src.prepare_conv(path,0)

    def test_source_mutation_during_copy_rejected_and_no_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);banks=self.setup_sources(root);copy=src.shutil.copyfile
            def mutate(s,d):
                result=copy(s,d)
                if str(s).endswith('history.sqlite'):Path(s).write_bytes(b'changed mid copy')
                return result
            with patch.object(src,'load_banks',return_value=banks),patch.object(src.shutil,'copyfile',side_effect=mutate):
                with self.assertRaises(ValueError):src.prepare_conv(root/'new_run',0)
            self.assertFalse((root/'new_run/conv_0/source_manifest.json').exists())
            self.assertTrue((root/'new_run/conv_0/prepare_intent.json').exists())

    def test_wrong_question_is_rejected_before_source_read(self):
        q={'id':'c0_q0','question':'original','reference':'gold','category':1}
        with patch.object(src,'_old_conv',return_value=([],{}, {'c0_q0':q},'2023-01-01')):
            for key,val in [('question','altered'),('reference','altered'),('category',4)]:
                with self.assertRaises(ValueError):src.baseline_row(0,{**q,key:val})

    def test_read_only_history_validation_requires_exact_adds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.sqlite';db=sqlite3.connect(path)
            db.execute('CREATE TABLE history (memory_id TEXT,new_memory TEXT,event TEXT,is_deleted INTEGER)')
            db.execute("INSERT INTO history VALUES ('f1','fact','ADD',0)");db.commit();db.close()
            before=src.sha(path);src._history_valid(path,[{'id':'f1','text':'fact'}]);self.assertEqual(src.sha(path),before)
            with self.assertRaises(ValueError):src._history_valid(path,[{'id':'f1','text':'changed fact'}])

if __name__=='__main__':unittest.main()
