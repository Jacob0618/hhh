"""Derive a second immutable archive from the first, adding metric provenance.

No ancestor inventory, native initialization, model dispatch, or deletion is
performed here. The original ZIP, manifest, plan and logical records survive.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from collections import Counter
import builtins
import copy
import encodings.utf_8_sig
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import zipfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import archive_amem_retrieval_continuation_v1 as old
from tools import archive_reviewed_chain_typed_closure_v1 as typed_entry
from tools.baselines_common_v1 import read,sha,save_new,now

SOURCE_PLAN=old.DEFAULT_PLAN
DEFAULT_PLAN=old.FINAL/'cleanup_plan_retrieval_v2.json'
OUTPUT=old.FINAL/'results/continuation_archive_v2'
METRIC='research/sources/locomo_evaluation_cbfbc1.py'
METRIC_SHA='8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd'
EXTRA=('tools/archive_amem_metric_closure_v2.py','tools/cleanup_amem_metric_closure_v2.ps1',
       'tools/run_five_baselines_reviewed_chain_v4.py')
REVIEW_SCHEMA='amem-metric-archive-derivation-independent-review-v2'
DERIVATION_SCHEMA='amem-metric-source-archive-derivation-v2'
SOURCE_SHA256={'plan':'56a620bc39e6da13893c56aaba342eb04bf5da576b7a8aaa81b85281580fc319',
    'archive':'20d5b7372eaa2f75616696fcf8dfdac8096d7c0a7c805a91ba929edde769f6e8',
    'archive_manifest':'890da956338d0db87e62e7f0ad4a10222c6dc02919bd42e0fb6bbf8869174a88',
    'archive_replay':'0e62171801fd6c8b7dacd993e3c24182b3409bf83100fae935b5652752d4ed2d'}


def code_hashes():
    return {**typed_entry.code_hashes(),**{p:sha(ROOT/p) for p in EXTRA},METRIC:METRIC_SHA}


def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def canonical(path):
    p=Path(path).absolute()
    if not p.is_relative_to(ROOT) or '..' in p.parts:raise ValueError('Foreign/traversing logical evidence')
    return str(p)


def source_bindings(plan_path=None):
    if plan_path is None:plan_path=SOURCE_PLAN
    if Path(plan_path)!=SOURCE_PLAN:raise ValueError('Only the original archive may be augmented')
    plan=read(old.safe_path(plan_path))
    values={'plan':binding(plan_path),**{name:plan[name] for name in ('archive','archive_manifest','archive_replay')}}
    expected_paths={'plan':SOURCE_PLAN,'archive':old.output_dir()/'evidence.zip',
        'archive_manifest':old.output_dir()/'archive_manifest.json','archive_replay':old.output_dir()/'archive_replay.json'}
    if any(v!={'path':str(expected_paths[k].resolve()),'sha256':SOURCE_SHA256[k]} for k,v in values.items()):
        raise ValueError('Only the four exact original packaging objects are admitted')
    return values


def verify_review(path):
    value=read(old.safe_path(path))
    if (value.get('schema')!=REVIEW_SCHEMA or value.get('status')!='passed'
            or value.get('scope')!='archive_derivation_only_no_scientific_change'
            or any(value.get('code_sha256',{}).get(p)!=h for p,h in code_hashes().items())):
        raise ValueError('Exact original eighteen plus three new sources and official metric review required')
    if value.get('source_archive')!=source_bindings():raise ValueError('Original archive authority differs')
    for bound in value['source_archive'].values():old.verify_binding(bound)
    for p,h in value.get('evidence_sha256',{}).items():
        if sha(old.safe_path(ROOT/p))!=h:raise ValueError('Derivation review evidence changed')
    if not value.get('evidence_sha256'):raise ValueError('Independent derivation evidence required')
    return value


def frontier(initial, original_records, controls, original_zip):
    """Expand only new evidence; exact immutable original records are resolved leaves.

    Original packaging objects are retained in keep, never appended to the
    original v4 logical inventory. No original work file is opened here.
    """
    previous={r['path']:r for r in original_records}
    pending=list(initial);expected={};added={};payloads={};kept={}
    def enqueue(p,h):
        if not isinstance(p,str) or not typed_entry.typed._digest(h):raise ValueError('Malformed new evidence binding')
        pending.append((p,h))
    def visit(value):
        if isinstance(value,list):
            for x in value:visit(x)
        elif isinstance(value,dict):
            for pk,hk in typed_entry.typed.PAIRS:
                if pk in value and hk in value:enqueue(value[pk],value[hk])
            for key in typed_entry.typed.MAP_KEYS:
                if key in value:
                    if not isinstance(value[key],dict):raise ValueError('New evidence map must be a dictionary')
                    for p,h in value[key].items():enqueue(p,h)
            for x in value.values():visit(x)
    while pending:
        name,h=pending.pop()
        if not isinstance(name,str) or not typed_entry.typed._digest(h):raise ValueError('Malformed initial binding')
        p=canonical(ROOT/name)
        if p in expected:
            if expected[p]!=h:raise ValueError('Conflicting augmentation SHA')
            continue
        expected[p]=h
        if p in controls:
            if controls[p]!=h:raise ValueError('Original packaging binding changed')
            old.verify_binding({'path':p,'sha256':h});kept[p]=h;continue
        if p in previous:
            record=previous[p]
            if record['sha256']!=h:raise ValueError('New reference conflicts with original record')
            # Full source ZIP members are verified/copied once by derive().
            continue
        if any(Path(p).is_relative_to(folder) for folder in old.ROOTS.values()):
            raise ValueError('New evidence may not alter any original run logical inventory')
        physical=old.safe_path(p);raw=physical.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('New evidence SHA differs')
        payloads[h]=raw
        added[p]={'path':p,'sha256':h,'size':len(raw),'retained':True,'member':'blobs/'+h}
        kept[p]=h
        if physical.suffix.lower()=='.json':visit(json.loads(raw.decode('utf-8-sig')))
    return list(added.values()),payloads,kept


def prepare_metric_runtime():
    """Initialize only the original metric's declared lexical dependencies."""
    import regex
    import nltk
    from nltk.stem import PorterStemmer
    PorterStemmer().stem('provenance')
    modules={}
    roots=(ROOT/'runtime',ROOT/'.tools/python',ROOT/'.venv/Lib/site-packages')
    for name,module in tuple(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and any(path.is_relative_to(r) for r in roots):modules[str(path)]=sha(path)
    return {'versions':{n:importlib.metadata.version(n) for n in ('regex','nltk')},
            'initialized':'original regex and NLTK PorterStemmer plus utf_8_sig encoding; no .venv directory exemption',
            'runtime_modules_sha256':modules}


@contextmanager
def only_zip_workspace_reads(view, runtime):
    """No unrecorded scientific source or artifact may fall back to live files."""
    originals={'builtin':builtins.open,'io':io.open,
        'loader_data':importlib.machinery.SourceFileLoader.get_data,
        'loader_stats':importlib.machinery.SourceFileLoader.path_stats}
    allowed={os.path.normcase(p):h for p,h in runtime.get('runtime_modules_sha256',{}).items()};counts=Counter()
    def runtime_bytes(p):
        key=os.path.normcase(str(p))
        with originals['builtin'](p,'rb') as f:data=f.read()
        if hashlib.sha256(data).hexdigest()!=allowed[key]:raise AssertionError('Initialized runtime module changed')
        counts['exact_initialized_runtime_read']+=1
        return data
    def stream(data,mode,encoding,errors,newline):
        if 'b' in mode:return io.BytesIO(data)
        return io.TextIOWrapper(io.BytesIO(data),encoding=encoding or 'utf8',errors=errors,newline=newline)
    def wrapper(original):
        def opened(file,mode='r',buffering=-1,encoding=None,errors=None,newline=None,closefd=True,opener=None):
            if any(c in mode for c in 'wax+'):raise AssertionError('Archive replay is read-only')
            if isinstance(file,(str,os.PathLike)):
                p=Path(file).absolute();key=os.path.normcase(str(p))
                if key in view.records:
                    counts['archive_routed_open']+=1;data=view.bytes(p)
                    return stream(data,mode,encoding,errors,newline)
                if key in allowed:return stream(runtime_bytes(p),mode,encoding,errors,newline)
                if p.is_relative_to(ROOT):
                    counts['forbidden_live_workspace_open']+=1
                    raise AssertionError('Unarchived live workspace fallback: '+str(p))
            return original(file,mode,buffering,encoding,errors,newline,closefd,opener)
        return opened
    def loader_data(loader,path):
        p=Path(path).absolute();key=os.path.normcase(str(p))
        if key in view.records:counts['archive_routed_source_loader']+=1;return view.bytes(p)
        if key in allowed:return runtime_bytes(p)
        if p.is_relative_to(ROOT):
            if p.suffix=='.pyc':
                try:source=importlib.util.source_from_cache(str(p))
                except ValueError:source=''
                if os.path.normcase(source) in view.records or os.path.normcase(source) in allowed:
                    counts['unarchived_bytecode_cache_skipped']+=1
                    raise FileNotFoundError('Use retained source instead of live bytecode')
            counts['forbidden_live_source_loader']+=1
            raise AssertionError('Unarchived live source-loader fallback: '+str(p))
        return originals['loader_data'](loader,path)
    def loader_stats(loader,path):
        p=Path(path).absolute();key=os.path.normcase(str(p));record=view.records.get(key)
        if record is not None:return {'mtime':0,'size':record['size']}
        if key in allowed:return {'mtime':0,'size':len(runtime_bytes(p))}
        if p.is_relative_to(ROOT):raise AssertionError('Unarchived live source-loader stat: '+str(p))
        return originals['loader_stats'](loader,path)
    previous_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        with patch('builtins.open',wrapper(originals['builtin'])),patch('io.open',wrapper(originals['io'])), \
             patch.object(importlib.machinery.SourceFileLoader,'get_data',loader_data), \
             patch.object(importlib.machinery.SourceFileLoader,'path_stats',loader_stats):
            yield counts
    finally:sys.dont_write_bytecode=previous_bytecode


def derive(review_path):
    old.engine.stopped();review=verify_review(review_path)
    if OUTPUT.exists() or DEFAULT_PLAN.exists():raise FileExistsError('Fresh v2 archive and plan required')
    source=review['source_archive'];plan=read(SOURCE_PLAN);manifest=read(source['archive_manifest']['path'])
    if (manifest.get('schema')!=old.SCHEMA or manifest['work_inventories']!=plan['work_inventories']
            or manifest['code_sha256']!=plan['code_sha256']
            or plan['code_sha256']!={p:sha(ROOT/p) for p in old.TOOL_FILES}):
        raise ValueError('Original manifest/plan frozen contract differs')
    records=manifest['records'];paths=[r['path'] for r in records]
    if len(set(paths))!=len(paths):raise ValueError('Duplicate original logical path')
    controls={v['path']:v['sha256'] for v in source.values()}
    with zipfile.ZipFile(source['archive']['path']) as original:
        if original.read('archive_manifest.json')!=Path(source['archive_manifest']['path']).read_bytes():
            raise ValueError('Original embedded/external manifest mismatch')
        members={'archive_manifest.json'}|{r['member'] for r in records if r['retained']}
        if set(original.namelist())!=members or len(original.namelist())!=len(members):raise ValueError('Original ZIP member set differs')
        added,payloads,kept=frontier([(str(Path(review_path).resolve()),sha(review_path)),(str(ROOT/METRIC),METRIC_SHA)],records,controls,original)
        if {r['path'] for r in added}&set(paths):raise ValueError('Augmentation overlaps original records')
        new=copy.deepcopy(manifest);new['records']=[*records,*sorted(added,key=lambda r:r['path'])]
        new['created_utc']=now();new['tool_review']=binding(review_path)
        new['derivation']={'schema':DERIVATION_SCHEMA,'source_archive':source,'original_records':len(records),
            'original_records_sha256':hashlib.sha256(json.dumps(records,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'added_paths':[r['path'] for r in sorted(added,key=lambda r:r['path'])],
            'scientific_change':False,'source_inventory_rescanned':False}
        OUTPUT.mkdir(parents=True,exist_ok=False)
        mp=OUTPUT/'archive_manifest.json';save_new(mp,new);target=OUTPUT/'evidence.zip'
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as dest:
            dest.writestr('archive_manifest.json',mp.read_bytes());written=set()
            for member in sorted(members-{'archive_manifest.json'}):
                raw=original.read(member)
                if not member.startswith('blobs/') or hashlib.sha256(raw).hexdigest()!=member[6:]:raise ValueError('Original blob SHA differs')
                dest.writestr(member,raw);written.add(member)
            for h,raw in payloads.items():
                if 'blobs/'+h not in written:dest.writestr('blobs/'+h,raw);written.add('blobs/'+h)
    runtime=prepare_metric_runtime()
    with old.Resolver(target,mp) as view:
        if view.manifest['records'][:len(records)]!=records:raise ValueError('Original logical record changed')
        with only_zip_workspace_reads(view,runtime) as counts:replay=old.replay_all(view)
        if replay!=read(source['archive_replay']['path']):raise ValueError('Metric-source-only replay changed result')
    save_new(OUTPUT/'archive_replay.json',replay)
    save_new(OUTPUT/'strict_runtime_replay.json',{'status':'passed','runtime':runtime,'read_counts':dict(counts),
        'original_records_unchanged':True,'new_summary_or_scores':False,'model_calls':0,'credential_reads':0})
    for b in source.values():old.verify_binding(b)
    for p,h in kept.items():old.verify_binding({'path':p,'sha256':h})
    new_plan=copy.deepcopy(plan)
    new_plan.update(created_utc=now(),archive=binding(target),archive_manifest=binding(mp),
        archive_replay=binding(OUTPUT/'archive_replay.json'),tool_review=binding(review_path),derivation=new['derivation'])
    new_plan['keep_sha256']={**plan['keep_sha256'],**controls,**kept,
        **{str(p.resolve()):sha(p) for p in OUTPUT.iterdir() if p.is_file()}}
    save_new(DEFAULT_PLAN,new_plan)
    return {'status':'derived_archive_verified','plan':binding(DEFAULT_PLAN),'archive':binding(target),
        'original_records':len(records),'new_records':len(new['records']),'added_records':len(added),'old_artifacts_unchanged':True}


def verify_derivation(plan_path=None):
    if plan_path is None:plan_path=DEFAULT_PLAN
    if Path(plan_path)!=DEFAULT_PLAN:raise ValueError('New v2 plan is mandatory')
    plan=read(old.safe_path(plan_path));review=verify_review(plan['tool_review']['path'])
    if plan.get('derivation',{}).get('source_archive')!=review['source_archive']:raise ValueError('Derivation ancestry differs')
    for p in (*EXTRA,METRIC):
        h=code_hashes()[p]
        if plan['keep_sha256'].get(str((ROOT/p).resolve()))!=h:raise ValueError('New technical/scientific source absent from keep map')
    for b in review['source_archive'].values():
        if plan['keep_sha256'].get(b['path'])!=b['sha256']:raise ValueError('Original packaging not retained')
    return plan


def verify_completion():
    verify_derivation()
    return old.verify_completion(DEFAULT_PLAN)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('derive','verify-derivation','verify-completion'))
    parser.add_argument('--review',type=Path);args=parser.parse_args()
    if args.command=='derive':
        if args.review is None:parser.error('--review required')
        value=derive(args.review)
    elif args.command=='verify-derivation':value=verify_derivation()
    else:value=verify_completion()
    print(json.dumps(value,ensure_ascii=True))


if __name__=='__main__':main()
