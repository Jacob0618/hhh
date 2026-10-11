"""Versioned technical dispatch: frozen runner, A-Mem decimal-ID compatibility.

Only A-Mem uses a fresh namespace and its forty audited exact raw requests.
Other methods keep v1 implementations and namespaces; prior-method gate resolves
the effective A-Mem namespace. No checkpoint/index or successful fact is copied.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_five_baselines_gemini25_v1 as base
from tools.baselines_common_v1 import BaselineAPI, BaselineFatal, read, sha, save_new, now
from tools.audit_amem_format_failure_v2 import OLD, MANIFEST_SHA
from jev_mem.sdk20_cached_api import digest

NEW = base.RUN / 'amem_format_v2'
PROOF = base.RESEARCH / 'amem_format_v2/original_prefix_admission.json'
EXTRA = ('tools/run_five_baselines_format_v2.py', 'tools/audit_amem_format_failure_v2.py',
         'tools/baselines_amem_v2.py')
_paths, _adapter, _audit, _seal = base.paths, base.adapter, base.audit_requests, base.seal
_manifest = base.manifest
_proof = None

def proof(recheck=True):
    value = read(PROOF)
    if value.get('status') != 'passed' or value.get('manifest_sha256') != MANIFEST_SHA:
        raise ValueError('Original prefix admission is not passed')
    if value.get('physical_http200') != 40 or value.get('known_representation_failures') != 10 or value.get('unknown') != 0:
        raise ValueError('Original prefix counts differ')
    if recheck:
        inventory={str(p.relative_to(OLD)):sha(p) for p in OLD.rglob('*') if p.is_file()}
        if inventory != value['original_inventory']:
            raise ValueError('Frozen original failed-run inventory changed')
    return value

class PrefixAPI(BaselineAPI):
    """Reuse only the reviewed same-conversation four-call prefix, then v1 API."""
    def __init__(self, run_dir, *, cache_roots=(), **kwargs):
        context = kwargs.get('context') or {}
        self.prefix_admitted = context.get('method') == 'amem' and context.get('phase') == 'build'
        # Lexical absolute paths are stable while ten workers create the shared
        # API directory. Never interpret a failed location comparison as permit
        # to regenerate an already returned prefix.
        if self.prefix_admitted and (os.path.normcase(os.path.abspath(run_dir)) !=
                os.path.normcase(os.path.abspath(NEW / 'work/api_method'))):
            raise BaselineFatal('A-Mem build outside the admitted continuation API directory')
        expected = ((OLD / 'work/api_method/api_cache').resolve(),) if self.prefix_admitted else ()
        if tuple(Path(p).resolve() for p in cache_roots) not in ((), expected):
            raise ValueError('Unapproved historical cache root')
        super().__init__(run_dir, cache_roots=(), **kwargs)
        self.cache_roots = expected
        self.prefix_position = 0

    def scoped(self, **context):
        return type(self)(self.run_dir, keys=self.keys, retries=self.retries, timeout=self.timeout,
                          context={**self.context, **context})

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        key = digest({'provider':provider, 'model':model, 'operation':operation, 'body':body})
        if self.prefix_admitted:
            if _proof is None:
                raise BaselineFatal('Prefix proof must be admitted before provider use')
            sequence = _proof['prefixes'][str(self.context['conversation'])]
            if self.prefix_position < len(sequence) and key != sequence[self.prefix_position]:
                raise BaselineFatal('Offline prefix reconstruction differs; no replacement request')
            candidate = OLD / 'work/api_method/api_cache' / (key + '.json')
            if self.prefix_position < len(sequence) or key in _proof['records'] or candidate.exists():
                record = _proof['records'].get(key)
                if (not candidate.is_file() or record is None or record['context'] != self.context or record['stage'] != stage
                        or sha(candidate) != record['raw_sha256']):
                    raise BaselineFatal('Prefix context/stage/raw binding differs')
        value = super().request(provider, model, body, operation=operation, stage=stage, cache=cache)
        if self.prefix_admitted and self.prefix_position < 4:
            self.prefix_position += 1
        return value

def audit_requests(folder):
    if Path(folder).resolve() != NEW.resolve():
        return _audit(folder)
    bound = proof()
    inventory, errors, inherited = [], [], []
    for root in sorted((folder / 'work').glob('api_*')):
        ledger = [json.loads(x) for x in (root/'api_calls.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (root/'api_calls.jsonl').exists() else []
        events = [json.loads(x) for x in (root/'logical_events.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()] if (root/'logical_events.jsonl').exists() else []
        intents = {p.stem:p for p in (root/'request_intents').glob('*.json')}
        lineages = {p.stem:p for p in (root/'cache_lineage').glob('*.json')}
        raws = {p.stem:p for p in (root/'api_cache').glob('*.json')}
        if set(intents) & set(lineages) or set(raws) != set(intents)|set(lineages):
            errors.append('Raw/intent/lineage inventory mismatch: '+root.name)
        if {e['request_hash'] for e in events} != set(raws) or {r['request_hash'] for r in ledger} != set(raws):
            errors.append('Producer/consumer scope mismatch: '+root.name)
        for key,p in raws.items():
            raw=read(p); signature=raw.get('request')
            if digest(signature) != key:
                errors.append('Raw request digest mismatch: '+key)
            consumers=[e for e in events if e['request_hash']==key]
            calls=[r for r in ledger if r['request_hash']==key]
            if any(e.get('error_type') or e.get('cache_sha256') != sha(p) or Path(e['cache_path']).resolve()!=p.resolve() for e in consumers):
                errors.append('Logical event/raw binding mismatch: '+key)
            if key in intents:
                intent=read(intents[key]); physical=[r for r in calls if not r.get('cache_hit')]
                if intent.get('signature') != signature or intent.get('request_hash') != key or not any(type(r.get('status')) is int and 200<=r['status']<300 for r in physical):
                    errors.append('Local producer/signature mismatch: '+key)
                inventory.append({'api':root.name,'request_hash':key,'raw_sha256':sha(p),
                    'intent_sha256':sha(intents[key]),'operation':signature['operation']})
            elif key in lineages:
                lineage=read(lineages[key]); rec=bound['records'].get(key)
                expected={'source_path':str((OLD/'work/api_method/api_cache'/p.name).resolve()),
                    'source_sha256':rec['raw_sha256'] if rec else None,'target_path':str(p.resolve()),
                    'target_sha256':rec['raw_sha256'] if rec else None,'request_hash':key}
                if (root.name!='api_method' or rec is None or lineage!=expected or sha(p)!=rec['raw_sha256']
                        or any(not r.get('cache_hit') for r in calls)
                        or any(e['context']!=rec['context'] or e['stage']!=rec['stage'] for e in consumers)):
                    errors.append('Inherited prefix lineage mismatch: '+key)
                inherited.append(key)
                inventory.append({'api':root.name,'request_hash':key,'raw_sha256':sha(p),
                    'lineage_sha256':sha(lineages[key]),'operation':signature['operation'],
                    'source':'audited_original_prefix'})
    if set(inherited)!=set(bound['records']) or len(inherited)!=40:
        errors.append('All forty original prefix requests must be reused exactly once as sources')
    return {'status':'passed' if not errors else 'needs_review','errors':errors,'inventory':inventory,
            'inherited_unique_prefix_requests':len(inherited),'original_prefix_proof_sha256':sha(PROOF)}

def seal(method):
    result=_seal(method)
    if method=='amem':
        result_copy={'schema':'amem-format-continuation-usage-v2','current_summary_sha256':sha(NEW/'results/summary.json'),
            'original_physical_attempts':40, 'original_status_counts':{'200':40},
            'original_not_rebilled':True,'reused_original_unique_requests':40,
            'note':'Current summary physical requests exclude the original 40; its cache count includes exact-prefix reuse.',
            'cost_usd':None,'original_proof_sha256':sha(PROOF)}
        save_new(NEW/'results/continuation_usage_v2.json',result_copy)
    return result

def manifest(method):
    value = _manifest(method)
    if method == 'amem':
        expected = {'path':str(PROOF),'sha256':sha(PROOF)}
        continuation = value.get('technical_continuation', {})
        review = read(value['review']['path'])
        if (continuation.get('prefix_proof') != expected
                or continuation.get('original_manifest_sha256') != MANIFEST_SHA
                or continuation.get('original_run') != str(OLD)
                or review.get('prefix_proof_sha256') != expected['sha256']):
            raise ValueError('Prepared continuation/proof binding differs')
    return value

def install(method='amem'):
    global _proof
    base.paths=lambda method: NEW if method=='amem' else _paths(method)
    # Later methods only resolve the accepted prior-method result. They neither
    # need the retired prefix cache nor gain permission to import it.
    if method != 'amem':
        return
    _proof=proof()
    from tools import baselines_amem_v2
    base.adapter=lambda method: baselines_amem_v2 if method=='amem' else _adapter(method)
    base.BaselineAPI=PrefixAPI
    base.audit_requests=audit_requests
    base.seal=seal
    base.manifest=manifest

def prepare_amem(review):
    if NEW.exists():
        raise FileExistsError('Fresh format-v2 namespace required')
    checked=read(review)
    if checked.get('status')!='passed' or checked.get('prefix_proof_sha256')!=sha(PROOF):
        raise ValueError('Independent continuation review/proof required')
    old=read(OLD/'run_manifest.json')
    code={**old['code_sha256'],**{p:sha(ROOT/p) for p in EXTRA}}
    if checked.get('code_sha256')!=code:
        raise ValueError('Independent review must bind all unchanged and added technical code')
    proof()
    from tools import baselines_amem_v2
    m=deepcopy(old)
    m.update(schema='five-baselines-amem-format-v2',created_utc=now(),code_sha256=code,
        review={'path':str(Path(review).resolve()),'sha256':sha(review)},
        historical_cache='Only the forty full-signature original format-failure prefix requests; all other history forbidden',
        specification=baselines_amem_v2.describe(),
        technical_continuation={'original_manifest_sha256':MANIFEST_SHA,'original_run':str(OLD),
            'prefix_proof':{'path':str(PROOF),'sha256':sha(PROOF)},
            'change':'Canonical decimal connection IDs normalized losslessly; raw JSON and scientific prompts unchanged.'})
    save_new(NEW/'run_manifest.json',m)
    return {'manifest_sha256':sha(NEW/'run_manifest.json'),'model_calls':0}

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','check','archive'])
    p.add_argument('--method',choices=base.METHODS,required=True);p.add_argument('--review',type=Path)
    args=p.parse_args()
    if args.command in ('prepare','archive') and args.review is None:p.error('--review required')
    install(args.method)
    if args.command=='prepare':
        result=prepare_amem(args.review) if args.method=='amem' else base.prepare(args.method,args.review)
    elif args.command=='run':
        if (base.paths(args.method)/'execution_intent.json').exists():
            raise FileExistsError('Existing execution intent is immutable; no controller reentry')
        try: result=base.run(args.method)
        except BaseException as e:
            base.status(base.paths(args.method),'needs_review',error=base.error_record(e));raise
    elif args.command=='check':
        m=base.manifest(args.method);result={'status':'passed','manifest_sha256':sha(base.paths(args.method)/'run_manifest.json')}
    else: result=base.archive_results(args.method,args.review)
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
