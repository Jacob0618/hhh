"""Independent retention-policy counterexamples for four-root ZIP readers."""
import json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from tools import archive_amem_continuation_v4 as a

class RetentionPolicyTests(unittest.TestCase):
    def check_rejected(self,path_tail,reason,retained=False):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);roots={f'v{i}':root/f'v{i}' for i in range(1,5)}
            path=roots['v3']/path_tail
            manifest={'schema':a.SCHEMA,'roots':{k:str(v) for k,v in roots.items()},'records':[
                {'path':str(path),'sha256':'a'*64,'size':111,'retained':retained,'disposal_reason':reason}]}
            mp=root/'manifest.json';mp.write_text(json.dumps(manifest),encoding='utf-8');zp=root/'evidence.zip'
            with zipfile.ZipFile(zp,'x') as z:z.writestr('archive_manifest.json',mp.read_bytes())
            with patch.object(a,'ROOT',root),patch.object(a,'ROOTS',roots),self.assertRaises(ValueError):
                with a.Resolver(zp,mp):pass
    def test_cannot_omit_session_json_or_unknown_cache_text_even_with_valid_crc(self):
        for tail in ('work/native/c2/sessions/session_1/note_000001.json','work/native/c2/cache/notes.json','work/native/c2/cache/error.txt'):
            with self.subTest(path=tail):self.check_rejected(tail,'reproducible MiniLM numeric vectors/index; SHA and size retained')
    def test_numeric_omission_requires_exact_reason_and_boolean_flag(self):
        self.check_rejected('work/native/c2/cache/embeddings.npy','made-up permission')
        self.check_rejected('work/native/c2/cache/embeddings.npy','reproducible MiniLM numeric vectors/index; SHA and size retained',retained=0)

if __name__=='__main__':unittest.main()
