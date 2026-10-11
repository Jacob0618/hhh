"""Offline R11 rounded-mass sensitivity overlay; strict originals remain unchanged."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import prepare_source_set_r11 as strict
from jev_mem import source_set_r11 as core
from jev_mem import source_set_rounding_r11 as compatible
Reader,save_new,file_sha,now,digest=strict.Reader,strict.save_new,strict.file_sha,strict.now,strict.digest
SCHEMA='source-set-r11-rounding-view-manifest-v1'
SOURCE_SCHEMA='source-set-r11-rounding-selected-inputs-v1'
STRICT_SOURCE=ROOT/'runs/source_set_r11_v1/selected_inputs.json'
POLICY={'version':'r11-rounded-mass-sensitivity-v1','primary':'original_strict_R11_unchanged',
    'basis':'all_1540_P_raw_responses_uniform_offline_parse_before_QA_no_gold',
    'grid':0.01,'grid_epsilon':1e-12,'integer_denominator':200,
    'certificate':'k=round(100p); L=max(0,2k-1),U=min(200,2k+1); sumL<=200<=sumU',
    'original_values_choice_argmax_tie_preserved':True,'renormalization':False,'new_JEV_calls':0,
    'assumption':'Compatible with two-decimal rounding; provider precision is not established',
    'strict_QA':'Strict failure returnsN0; reuse actual exactN0 Reader/Judge. Other strict requests equal effective requests.',
    'qa_requests':'Execute only effective baseline/g/p. Report strict primary and rounded sensitivity separately.'}
CODE=tuple(dict.fromkeys((*strict.CODE,'jev_mem/source_set_rounding_r11.py',
    'tools/prepare_source_set_rounding_r11.py','tools/run_source_set_rounding_qa_r11.py')))
CACHE={}

def codes():return {p:file_sha(ROOT/p) for p in CODE}

def derive(reader,src,psha):
    items=[];counts=Counter()
    for oldjob in src['jobs']:
        job=deepcopy(oldjob);q=job['query_id'];a=job['method'];record=None;certificate=None
        if a=='p':
            original=reader.bound(job['selection']);rawref=original['raw_response']
            if rawref:
                raw=reader.bound(rawref,sealed=False)
                try:
                    certificate=compatible.rounding_certificate(raw['response'],raw['request'])
                    parsed=compatible.parse_p_compatible(raw['response'],raw['request'])
                except (ValueError,KeyError,TypeError):
                    # Already authenticated known strict failure remains unchanged.
                    if original['decision']['score_status']!='failed': raise
                    certificate={'mode':'rejected','accepted':False};parsed=None
                if parsed is not None and original['decision']['score_status']=='failed':
                    prop=reader.bound(original['proposal']);fs=prop['original_job']['selected_facts'];cs=prop['locator']['candidates']
                    decision=compatible.choose_p_compatible(fs,cs,parsed,signature=raw['request'])
                    facts=core.selected_facts(fs,cs,decision)
                    record=deepcopy({k:v for k,v in original.items() if k!='content_sha256'})
                    record.update(schema='source-set-r11-rounding-selection-v1',protocol_sha256=psha,
                        strict_selection=job['selection'],rounding_certificate=certificate,
                        raw_scores=parsed,decision=decision,selected_facts=facts,selected_sha256=digest(facts),
                        unchanged_ordered_reader_facts=facts==fs,
                        attachments=[deepcopy(cs[i]) for i in decision['candidate_indices']])
                    job.update(selected_facts=facts,selected_sha256=digest(facts),unchanged_ordered_reader_facts=facts==fs)
                    counts['new_pair_inputs']+=facts!=original['selected_facts']
            counts['P_'+str((certificate or {}).get('mode','no_raw'))]+=1
        job.update(strict_selection=oldjob['selection'],strict_protocol_sha256=oldjob['protocol_sha256'],protocol_sha256=psha)
        items.append((job,record,certificate))
    return items,dict(counts)

def effective_counts(src,rows):
    counts=deepcopy(src['selection_counts'])
    original={j['job_id']:j for j in src['jobs']}
    for job,record,certificate in rows:
        if record is None:continue
        if job['method']!='p' or not original[job['job_id']]['unchanged_ordered_reader_facts']:
            raise ValueError('Only an original failed P/N0 view can acquire compatibility selection')
        counts['p_score_failed']-=1;counts['p_score_complete']+=1
        counts['p_changed_queries']+=not job['unchanged_ordered_reader_facts']
        counts['p_attachments']+=len(record['attachments'])
    return counts

def prepare(run_dir):
    run=Path(run_dir).resolve()
    if run.parent!=(ROOT/'runs').resolve() or run.name!='source_set_r11_rounding_view_v1' or run.exists():
        raise FileExistsError('Unique fresh rounding sensitivity namespace required')
    src=strict.validate_source(STRICT_SOURCE);reader=Reader();ref=reader.raw_pointer(STRICT_SOURCE)
    p={'schema':SCHEMA,'strict_source':ref,'policy':POLICY,'code_sha256':codes()};psha=digest(p)
    rows,counts=derive(reader,src,psha)
    run.mkdir(parents=True,exist_ok=False);jobs=[];certs={}
    for job,record,cert in rows:
        if record is not None:job['selection']=save_new(run/'selections'/f'{job["query_id"]}.json',record)
        if cert is not None:certs[job['query_id']]=cert
        jobs.append(job)
    mref=save_new(run/'run_manifest.json',{'schema':SCHEMA,'created_utc':now(),'protocol':p,'protocol_sha256':psha,
        'counts':counts,'certificates':certs,'model_calls':0})
    value=deepcopy({k:v for k,v in src.items() if k!='content_sha256'})
    value.update(schema=SOURCE_SCHEMA,parent_run=str(run),created_utc=now(),jobs=jobs,
        selector_manifest=mref,strict_source=ref,rounding_policy=POLICY,protocol_sha256=psha,
        code_sha256=codes(),rounding_counts=counts,selection_counts=effective_counts(src,rows),
        strict_selection_counts=src['selection_counts'],primary_result_requires_strict_exact_view_projection=True)
    return {'source':save_new(run/'selected_inputs.json',value),'jobs':len(jobs),'counts':counts,'new_model_calls':0}

def validate_source(path,*,fresh=False):
    path=Path(path).resolve();cache=CACHE.get(str(path))
    if cache and not fresh and all(Path(p).is_file() and (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in cache['stats'].items()):
        return deepcopy(cache['source'])
    reader=Reader();src=reader.read(path);run=Path(src['parent_run']);m=reader.bound(src['selector_manifest']);p=m['protocol']
    if (path!=run/'selected_inputs.json' or src['schema']!=SOURCE_SCHEMA or m['schema']!=SCHEMA
        or src['status']!='sealed' or m['protocol_sha256']!=digest(p) or src['protocol_sha256']!=m['protocol_sha256']
        or p['policy']!=POLICY or src['rounding_policy']!=POLICY or p['code_sha256']!=codes() or src['code_sha256']!=codes()
        or p['strict_source']!=src['strict_source'] or Path(p['strict_source']['path']).resolve()!=STRICT_SOURCE.resolve()):
        raise ValueError('Rounding view identity/protocol differs')
    reader.bound(p['strict_source']);old=strict.validate_source(p['strict_source']['path'])
    rows,counts=derive(reader,old,m['protocol_sha256']);expected=[];certs={}
    for j,rec,cert in rows:
        if rec is not None:
            sp=run/'selections'/f'{j["query_id"]}.json';actual=reader.read(sp)
            if {k:v for k,v in actual.items() if k!='content_sha256'}!=rec: raise ValueError('Rounding selection raw replay differs')
            j['selection']=reader.pointer(sp)
        if cert is not None:certs[j['query_id']]=cert
        expected.append(j)
    if src['jobs']!=expected or m['counts']!=counts or src['rounding_counts']!=counts or m['certificates']!=certs:
        raise ValueError('Rounding cohort/certificate differs')
    if src['selection_counts']!=effective_counts(old,rows) or src['strict_selection_counts']!=old['selection_counts']:
        raise ValueError('Strict/effective selection counts differ')
    for k in ('frozen_source','frozen_qa_run_manifest','frozen_baseline_reuse_proof','reuse_proof_qa_manifest',
              'source_dataset_sha256','official_metric_provenance','source_set_protocol','r05_source','arms','questions_per_arm'):
        if src[k]!=old[k]:raise ValueError('Scientific evaluation/source contract changed: '+k)
    if Counter(j['method'] for j in expected)!={a:1540 for a in ('baseline','g','p')}:raise ValueError('Full cohort required')
    stats=deepcopy(strict._VALIDATION_CACHE[str(STRICT_SOURCE.resolve())]['stats'])
    stats.update({str(q):(Path(q).stat().st_size,Path(q).stat().st_mtime_ns) for q in reader.refs})
    for rel in CODE:
        q=str((ROOT/rel).resolve());stats[q]=(Path(q).stat().st_size,Path(q).stat().st_mtime_ns)
    CACHE[str(path)]={'source':deepcopy(src),'stats':stats}
    return src

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('prepare','check'))
    p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    out=prepare(a.run_dir) if a.stage=='prepare' else {'valid':True,'jobs':len(validate_source(a.run_dir/'selected_inputs.json')['jobs']),'model_calls':0}
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
