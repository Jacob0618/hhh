"""Read-only cleanup planner/hash inventory. Filesystem removal is separate PowerShell."""
import os,json,hashlib,stat
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(r'D:\jev mem'); OUT=ROOT/'research/cleanup_2026-10-05'; OUT.mkdir(exist_ok=True)
R=ROOT/'research/jev_autoresearch_2026-10-04'
keep={}; required={}
def add(p,sha=None):
    p=Path(p); assert p.is_file(),str(p)
    assert p.resolve().is_relative_to(ROOT.resolve()),str(p)
    k=str(p).lower();keep[k]=p
    if sha:
        assert k not in required or required[k]==sha
        required[k]=sha
def refs(o):
    if isinstance(o,dict):
        if isinstance(o.get('path'),str) and o.get('sha256'):add(o['path'],o['sha256'])
        for v in o.values():refs(v)
    elif isinstance(o,list):
        for v in o:refs(v)
for name in ('source_append_r05_v1','source_append_qa_r05_v1'):
    for p in (ROOT/'runs'/name).rglob('*'):
        if p.is_file():add(p)
s=json.loads((R/'r05_final_summary.json').read_text(encoding='utf8'))
for p,sha in s['input_sha256'].items():add(p,sha)
m=json.loads((ROOT/'runs/source_append_r05_v1/run_manifest.json').read_text(encoding='utf8'))
refs(m['protocol'])
rp=json.loads((R/'R05_source_append_protocol_final_v1.json').read_text(encoding='utf8'))
refs(rp.get('input_refs',{}))
for c in range(10):
    ps=list((ROOT/f'runs/gemini38_rebuild_v1_20261003/retrieval/conv_{c}/session/selections').glob('*.json'))
    for p in ps:add(p)
add(ROOT/'runs/evidence_exchange_r01_v1/run_manifest.json')
add(ROOT/'runs/evidence_selection_qa_r03_v1/run_manifest.json','a4a8028beb0b455e9d0763e75588ac50b2628103ae153adf0bcafa5b1d043ac4')
# Preserve compact reports/code and all R05 research, discard large other-round diagnostics/caches.
for p in (ROOT/'research').rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(ROOT/'research')
    if rel.parts[0]=='cleanup_2026-10-05':continue
    size=p.stat().st_size
    r05=any('r05' in part.lower() for part in rel.parts)
    if r05 or p.suffix.lower() in ('.md','.py','.bib') or (size<=262144 and p.suffix.lower() in ('.json','.csv','.txt','.jsonl')):add(p)
print(f'keep candidates {len(keep)}; hash verification starting',flush=True)
def hash_one(item):
    k,p=item; b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
    assert k not in required or sha==required[k],f'original SHA mismatch: {p}'
    return {'path':str(p),'sha256':sha,'bytes':len(b),'original_sha_verified':k in required}
with ThreadPoolExecutor(max_workers=8) as ex: inventory=list(ex.map(hash_one,keep.items()))
(OUT/'preserved_files.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf8')
ancestors=set()
for p in keep.values():
    for parent in p.parents:
        if parent==ROOT:break
        ancestors.add(str(parent).lower())
deletions=[]; totals={'bytes':0,'files':0}; reparse=[]
def scan(p):
    # No traversal of junctions/symlinks is permitted, even in whole deleted trees.
    a=p.lstat()
    if a.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        reparse.append(str(p));return (0,0)
    if p.is_file():return(a.st_size,1)
    size=n=0
    with os.scandir(p) as it:
        for e in it:
            b,f=scan(Path(e.path));size+=b;n+=f
    return size,n
def plan(p):
    key=str(p).lower()
    if key in keep:return
    if key in ancestors:
        for ch in p.iterdir():plan(ch)
    else:
        size,n=scan(p)
        deletions.append({'path':str(p),'bytes':size,'files':n,'directory':p.is_dir()})
        totals['bytes']+=size;totals['files']+=n
for base in ('runs','research'):
    for p in (ROOT/base).iterdir():
        if p==OUT:continue
        plan(p)
failed_env=ROOT/'.venv-anaconda-failed'
if failed_env.exists():plan(failed_env)
assert not reparse, f'reparse targets need review: {reparse}'
manifest={'schema':'explicit-authorized-cleanup-v1','workspace':str(ROOT),'roots':['runs','research','.venv-anaconda-failed'],
 'preserved_file_count':len(inventory),'preserved_bytes':sum(x['bytes'] for x in inventory),'original_sha_verified':len(required),
 'deletions':deletions,'delete_totals':totals,'reparse_targets':reparse,'archive_required':'archives/retained_rounds_2026-10-05/verification.json',
 'initial_disk_free_bytes':17779388416,'delete_executed':False}
(OUT/'delete_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in manifest.items() if k!='deletions'},ensure_ascii=False),flush=True)
