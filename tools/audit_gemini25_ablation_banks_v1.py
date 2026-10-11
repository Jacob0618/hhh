"""Offline session-only source/raw/ADD/index audit and bound admission.

Reuses the existing inspector and review publisher with isolated model/schema
bindings. Semantic review must be supplied separately; no scores or model calls.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import audit_gemini38_banks_v1 as frozen
from tools import publish_gemini38_bank_review_v1 as publisher
from tools import run_gemini25_ablation_writer_v1 as writer
from jev_mem.gemini25_ablation_provider_v1 import MODEL,generation_body,response_text,WRITER_MAX_OUTPUT

SCHEMA = 'gemini25-ablation-terminal-source-index-audit-v1'
SEMANTIC_SCHEMA = 'gemini25-ablation-writer-source-semantic-review-v1'
DEVELOPMENT_SCHEMA = 'gemini25-ablation-development-bank-admission-v1'
FULL_SCHEMA = 'gemini25-ablation-complete-bank-admission-v1'
idx = frozen.idx

check_generation = writer.clone(frozen.check_generation,MODEL=MODEL,generation_body=generation_body,
                                response_text=response_text,WRITER_MAX_OUTPUT=WRITER_MAX_OUTPUT)
inspect_session = writer.clone(frozen.inspect_session,check_generation=check_generation)
audit = writer.clone(frozen.audit,writer=writer,inspect_session=inspect_session,SCHEMA=SCHEMA)


def _publisher_adapter():
    # Same instructions and validation branches; only literal schema identities
    # and two cohort-size diagnostics differ from the 40-bank publisher.
    replacements = {
        'gemini38-terminal-source-index-audit-v1': SCHEMA,
        'gemini38-writer-source-semantic-review-v1': SEMANTIC_SCHEMA,
        'gemini38-development-bank-admission-v1': DEVELOPMENT_SCHEMA,
        'gemini38-complete-bank-admission-v1': FULL_SCHEMA,
        'Only development8/full40 admission supported': 'Only development2/full10 admission supported',
    }
    fn = publisher.publish
    code = fn.__code__.replace(co_consts=tuple(replacements.get(x,x) if isinstance(x,str) else x for x in fn.__code__.co_consts))
    result = FunctionType(code,{**fn.__globals__,'writer':writer},fn.__name__,fn.__defaults__,fn.__closure__)
    result.__kwdefaults__ = dict(fn.__kwdefaults__ or {})
    return result


publish = _publisher_adapter()


def validate_review(run,m=None,*,development=True):
    """Revalidate the published audit/semantic/native bindings before expansion."""
    run = Path(run).resolve()
    m = writer.manifest(run) if m is None else m
    convs = [0,1] if development else list(range(10))
    gate = idx.verify(run/('development_review.json' if development else 'all_bank_review.json'))
    if (gate.get('schema') != (DEVELOPMENT_SCHEMA if development else FULL_SCHEMA)
        or gate.get('status') != 'passed' or gate.get('protocol_sha256') != m['protocol_sha256']
        or gate.get('writer_manifest') != idx.pointer(run/'run_manifest.json') or gate.get('exceptions') != []):
        raise ValueError('Bound current Writer admission required')
    if len(gate.get('source_date_audits',[])) != 1 or gate.get('index_audits') != gate['source_date_audits']:
        raise ValueError('One shared source/date/index audit required')
    audit_ref = gate['source_date_audits'][0]
    value = idx.bound(audit_ref)
    if (value.get('schema') != SCHEMA or value.get('strict_passed') is not True
        or value.get('protocol_sha256') != m['protocol_sha256'] or value.get('conversations') != convs
        or value.get('exceptions') != [] or value.get('writer_manifest') != gate['writer_manifest']):
        raise ValueError('Source/index audit differs from admission')
    semantic = idx.bound(gate['semantic_review'])
    if (semantic.get('schema') != SEMANTIC_SCHEMA or semantic.get('status') != 'reviewed_with_limitations'
        or semantic.get('protocol_sha256') != m['protocol_sha256'] or semantic.get('structural_audit') != audit_ref
        or semantic.get('qa_gold_used') is not False):
        raise ValueError('Source-only semantic review differs from admission')
    expected = {(ci,'session') for ci in convs}
    if (len(value.get('banks',[])) != len(expected) or len(semantic.get('banks',[])) != len(expected)
        or {(b['conversation'],b['arm']) for b in value['banks']} != expected
        or {(b['conversation'],b['arm']) for b in semantic['banks']} != expected):
        raise ValueError('Incomplete session review population')
    expected_banks = [r['bank'] for r in value['banks']]
    actual_banks = gate.get('banks',[]) if development else [r['bank'] for r in gate.get('banks',[])]
    if actual_banks != expected_banks:
        raise ValueError('Gate banks differ from audited banks')
    for path,hashed in value['source_file_hashes'].items():
        if idx.sha(path) != hashed:
            raise ValueError('Audited source changed after review')
    semantic_banks={(b['conversation'],b['arm']):b for b in semantic['banks']}
    for row in value['banks']:
        ci = row['conversation']
        folder = run/f'conv_{ci}/session'
        bank = idx.bound(row['bank'])
        review=semantic_banks[(ci,'session')]
        if (not row.get('source_generation_persistence_passed') or review.get('bank')!=row['bank']
            or not review.get('notes') or review.get('blocking_implementation_error') is not False):
            raise ValueError('Unreviewed source/semantic implementation finding')
        required={x['journal']['path'] for x in row.get('semantic_review_targets',[])}
        if not row.get('journals'):
            raise ValueError('Audited session journals missing')
        required.update(row['journals'][i]['journal']['path'] for i in (0,-1))
        refs={x['path']:x for x in bank['journal_files']}
        reviewed=review.get('reviewed_journals',[])
        if not required<={p['path'] for p in reviewed}:
            raise ValueError('Required source/date/empty/history review targets missing')
        for pointer in reviewed:
            if refs.get(pointer['path'])!=pointer:
                raise ValueError('Semantic review cites a journal outside this bank')
            idx.bound(pointer)
        out = idx.verify(folder/'write_outcome.json')
        state = idx.read(folder/'write_status.json')
        if (row.get('strict_passed') is not True or row.get('orphan_fact_ids')
            or row['bank'] != idx.pointer(folder/'bank.json') or bank.get('complete') is not True
            or (bank['conversation'],bank['arm'],bank['protocol_sha256']) != (ci,'session',m['protocol_sha256'])
            or out.get('status') != 'complete' or state.get('status') != 'complete'
            or out.get('bank') != row['bank'] or state.get('bank') != row['bank']
            or idx.native_inventory(folder/'native') != row['native_inventory']):
            raise ValueError('Reviewed session bank/native changed')
    return gate


def admit_review(writer_run,gate_path=None,conversations=range(10)):
    """Return the checked Writer manifest, review and admitted bank rows."""
    run=Path(writer_run).resolve()
    convs=list(conversations)
    if convs not in ([0,1],list(range(10))):
        raise ValueError('Only development two/full ten admission supported')
    development=convs==[0,1]
    expected=run/('development_review.json' if development else 'all_bank_review.json')
    if gate_path is not None and Path(gate_path).resolve()!=expected:
        raise ValueError('Gate must be the current Writer admission')
    m=writer.manifest(run)
    gate=validate_review(run,m,development=development)
    rows=(idx.bound(gate['source_date_audits'][0])['banks'] if development else gate['banks'])
    return {'manifest':m,'review':gate,'banks':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('audit','publish','check'))
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--conversations',type=int,nargs='+',default=[0,1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--semantic-review',type=Path)
    args=parser.parse_args()
    if args.stage=='audit':
        if args.output is None: parser.error('--output required')
        value=audit(args.run_dir,args.conversations)
        writer.grouped.save_new(args.output,value)
        result={'audit':str(args.output),'status':value['status'],'banks':len(value['banks']),'model_calls':0}
    elif args.stage=='publish':
        if args.audit is None or args.semantic_review is None: parser.error('--audit and --semantic-review required')
        result=publish(args.run_dir,args.audit,args.semantic_review)
    else:
        result={'valid':bool(validate_review(args.run_dir,development=args.conversations==[0,1])),'model_calls':0}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
