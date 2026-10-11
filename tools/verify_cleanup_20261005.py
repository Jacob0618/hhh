"""Verify preserved bytes and self-contained review archive after authorized cleanup."""
import json,hashlib,zipfile,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(r'D:\jev mem');OUT=ROOT/'research/cleanup_2026-10-05'
entries=json.loads((OUT/'preserved_files.json').read_text(encoding='utf8'))
def check(x):
    p=Path(x['path']);assert p.is_file(),str(p)
    raw=p.read_bytes();assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256'],str(p)
    return len(raw)
with ThreadPoolExecutor(max_workers=8) as pool: sizes=list(pool.map(check,entries))
proof=json.loads((ROOT/'archives/retained_rounds_2026-10-05/verification.json').read_text(encoding='utf8'))
zpath=Path(proof['archive']);assert hashlib.sha256(zpath.read_bytes()).hexdigest()==proof['sha256']
with zipfile.ZipFile(zpath) as z:assert z.testzip() is None
deletions=json.loads((OUT/'delete_manifest.json').read_text(encoding='utf8'))['deletions']
remaining=[x['path'] for x in deletions if Path(x['path']).exists()]
result={'status':'passed' if not remaining else 'deletion_incomplete','preserved_files':len(entries),'preserved_bytes':sum(sizes),'all_preserved_sha256_unchanged':True,'archive_sha256':proof['sha256'],'archive_crc_passed':True,'remaining_delete_targets':remaining,'api_calls':0,'disk_free_bytes':shutil.disk_usage(ROOT).free}
(OUT/'postcleanup_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False),flush=True)
