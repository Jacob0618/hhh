"""Offline single-arm fixed400 QA audit using the frozen plain replay/metrics.

No prepare, provider, credential, Memory or Qdrant client is invoked. This does
not compare arms: paired statistics are explicitly unavailable for one arm.
"""
from __future__ import annotations
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import summarize_sdk20_plain_qa_v1 as base
from tools import run_regroup_refinement_qa_v1 as driver_v1
from tools import run_regroup_refinement_qa_v2 as driver_v2
from tools import run_sdk20_plain_qa_v1 as plain
from tools.summarize_mem0_sparse1540 import Snapshot, measure_usage
from tools.summarize_sdk20_modules import InvalidArtifact
from tools.run_sdk20_modules import save_new, now, digest
from jev_mem.local_bm25_context_v2 import forbid_network

ARMS = ('source_local_context', 'jev_refined')
DENOMINATOR = 400
CODE = ('tools/summarize_regroup_refinement_qa_v1.py', 'tools/summarize_sdk20_plain_qa_v1.py',
        'tools/summarize_mem0_regroup_plain_qa_v1.py', 'tools/summarize_mem0_sparse1540.py',
        'tools/replay_plain_judge_compat_v1.py', 'tools/summarize_pilot400.py')


def scoped(function, **bindings):
    result = FunctionType(function.__code__, {**function.__globals__, **bindings},
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    return result


def driver_for(run):
    header = plain.read(Path(run)/'run_manifest.json')
    drivers = {d.SCHEMA:d for d in (driver_v1, driver_v2)}
    if header.get('schema') not in drivers:
        raise InvalidArtifact('Only frozen refinement QA v1/v2 namespaces supported')
    return drivers[header['schema']]


def require_closed_files(run):
    """Prevent the driver's existing seal helper from ever creating a file."""
    run = Path(run).resolve()
    if not (run/'run_outcome.json').is_file():
        raise InvalidArtifact('Use health checks while active; a terminal QA outcome is required')
    header = plain.read(run/'run_manifest.json')
    if header['phase_kind'] == 'firstpass':
        if not (run/'source_jobs.json').is_file():
            raise InvalidArtifact('Existing sealed source_jobs required; summary never seals jobs')
    elif header['phase_kind'] == 'technical_recovery':
        parent = Path(header['parent']['run_dir']).resolve()
        if parent == run or run != parent.with_name(parent.name+'_recovery_v1'):
            raise InvalidArtifact('Only the one prescribed recovery namespace is supported')
        p = plain.read(parent/'run_manifest.json')
        if p['phase_kind'] != 'firstpass':
            raise InvalidArtifact('Recovery of recovery is not permitted')
        require_closed_files(parent)
    else:
        raise InvalidArtifact('Unknown phase kind')


class RunnerFacade:
    """Only adapt terminal's signature; stage/payload/metric functions are plain."""
    def __init__(self, driver):
        self.driver = driver

    def __getattr__(self, name):
        return getattr(plain, name)

    def terminal(self, run, _source=None):
        require_closed_files(run)
        m, out = self.driver.terminal(run)
        if _source is not None and plain.verify(m['source_manifest']['path']) != _source:
            raise InvalidArtifact('Phase source differs from the firstpass sealed source')
        return m, out


class PhaseLedgerView:
    """Filter ledger bytes in memory; SHA and recorded files remain original."""
    def __init__(self, snapshot, phase):
        self.snapshot, self.phase = snapshot, phase

    def __getattr__(self, name):
        return getattr(self.snapshot, name)

    def raw(self, path):
        data = self.snapshot.raw(path)
        if Path(path).name != 'api_calls.jsonl':
            return data
        kept = []
        for line in data.decode('utf-8-sig').splitlines():
            if not line.strip():continue
            row = json.loads(line)
            phase = row.get('context', {}).get('phase')
            if phase not in ('qa', 'retrieval'):
                raise InvalidArtifact('Unexpected phase in shared retrieval/QA ledger')
            if phase == self.phase:kept.append(line)
        return ('\n'.join(kept)+'\n').encode('utf-8')


def audit_phase(run, source, *, parent=None):
    driver = driver_for(run)
    facade = RunnerFacade(driver)
    qa_ledger = scoped(base.ledger, runner=facade)
    def ledger(snapshot, folder, m):
        return qa_ledger(PhaseLedgerView(snapshot, 'qa'), folder, m)
    # Same existing Reader/Judge replay, invalid-label handling, compatibility,
    # inherited Reader and combined-record validation; only terminal/ledger adapt.
    replay = scoped(base.audit_phase, runner=facade, ledger=ledger)
    return replay(run, source, parent=parent)


def aggregate(rows, arm):
    """Keep the original per-arm aggregation AST; omit its two-arm tail only."""
    if arm not in ARMS or {r['method'] for r in rows} != {arm}:
        raise InvalidArtifact('Exactly one declared arm required')
    tree = ast.parse(inspect.getsource(base.existing.aggregate))
    function = tree.body[0]
    tail = function.body[-2:]
    if (not isinstance(tail[0], ast.Assign) or ast.unparse(tail[0].targets[0]) != 'paired'
            or not isinstance(tail[1], ast.Return)):
        raise InvalidArtifact('Original aggregation tail changed; review the adapter')
    function.body = function.body[:-2] + ast.parse(
        "return {'arms': arms, 'paired': None, 'interpretation': 'Single-arm fixed400 report; no paired comparison has been performed.'}").body
    globals_ = {**base.existing.aggregate.__globals__, 'ARMS':(arm,)}
    exec(compile(ast.fix_missing_locations(tree), __file__+'::single_arm_aggregate', 'exec'), globals_)
    return globals_['aggregate'](rows, denominator=DENOMINATOR, bootstrap_samples=2000)


def audit_retrieval_and_usage(phase, source):
    run = Path(phase['run']); m = phase['manifest']; snapshot = Snapshot()
    cohort = {str(ci):[{'id':j['query_id']} for j in source['jobs'] if j['conversation']==ci] for ci in range(10)}
    retrieval_ledger = scoped(base.existing.ledger_index,
        driver=SimpleNamespace(SCHEMA=m['schema']), ARMS=(m['arm'],))
    rows, _, refs, missing = retrieval_ledger(PhaseLedgerView(snapshot,'retrieval'), run, cohort)
    if m['phase_kind']=='technical_recovery' and rows:
        raise InvalidArtifact('Recovery may not perform new retrieval')
    active_convs={j['conversation'] for j in m['jobs']} if m['phase_kind']=='technical_recovery' else set(range(10))
    missing=[p for p in missing if int(Path(p).parent.parent.name.split('_')[1]) in active_convs]
    all_rows = phase['ledger']+rows
    # This existing lineage auditor sees all physical rows, avoiding a false
    # unlogged-dispatch flag for a QA failure in a shared request_intents folder.
    lineage = base.existing.usage_and_lineage(snapshot,run,m,all_rows,refs,missing)
    if snapshot.unchanged():raise InvalidArtifact('Source changed during retrieval accounting')
    return {'retrieval_usage':measure_usage(rows),'qa_usage':phase['usage'],
            'all_phase_usage':measure_usage(all_rows),'combined_lineage':lineage,
            'retrieval_records':rows,'input_sha256':{str(k):v for k,v in snapshot.files.items()}}


def writer_accounting(path, arm, first):
    if path is None:
        return {'included_in_current_qa_usage':False, 'usage':None,
                'source_gate':first['manifest']['gate'], 'reason':'Writer accounting supplied separately; not a measured zero'}
    report = plain.read(path)
    if report.get('schema') != 'regroup-refinement-all20-writer-call-audit-v1' or report.get('closed_bank_count') != 20:
        raise InvalidArtifact('Expected separately audited complete20 writer ledger report')
    selected = [a for a in report['arms'] if a['arm']==arm]
    if len(selected)!=1:raise InvalidArtifact('Writer accounting arm differs')
    a=selected[0]
    gate = plain.verify(first['manifest']['gate']['path'])
    if gate.get('arm') != arm:raise InvalidArtifact('Writer gate arm differs')
    # Bind the report to this QA gate's original writer manifests, not merely an arm label.
    if 'uniform_twenty_bank_review' in gate:
        uniform=plain.verify(gate['uniform_twenty_bank_review']['path'])
        parents={json.dumps(r['source_manifest'],sort_keys=True) for r in uniform['banks'] if r['arm']==arm}
    else:
        parents={json.dumps(gate['source_manifest'],sort_keys=True)}
    if parents != {json.dumps(a['manifest'],sort_keys=True)}:
        raise InvalidArtifact('Writer report belongs to other source banks')
    for bank in a['banks']:
        for ref in (bank['ledger'],bank['terminal_state'],bank['terminal_outcome']):
            if plain.pointer(ref['path']) != ref:raise InvalidArtifact('Writer accounting source drift')
    return {'included_in_current_qa_usage':False,'report':plain.pointer(path),'arm':arm,
            'usage':a['usage'],'groups':a['groups'],'facts':a['facts'],
            'inherited_grouping_and_segmentation_costs_included':False,'total_cold_start_cost_usd':None}


def summarize(firstpass, recovery=None, *, writer_call_audit=None):
    run=Path(firstpass).resolve();require_closed_files(run)
    initial=plain.read(run/'run_manifest.json')
    if initial['phase_kind']!='firstpass' or initial['arm'] not in ARMS:
        raise InvalidArtifact('Main input must be a declared arm firstpass')
    source=plain.verify(run/'source_jobs.json');jobs=source['jobs'];arm=initial['arm']
    if (len(jobs)!=DENOMINATOR or len({j['query_id'] for j in jobs})!=DENOMINATOR
            or len({j['job_id'] for j in jobs})!=DENOMINATOR or {j['method'] for j in jobs}!={arm}
            or {j['conversation'] for j in jobs}!=set(range(10))):
        raise InvalidArtifact('All original400 slots across ten conversations must remain')
    first=audit_phase(run,source)
    second=audit_phase(recovery,source,parent=first) if recovery else None
    if second:
        p=second['manifest'].get('parent',{})
        if (second['manifest']['phase_kind']!='technical_recovery' or second['manifest']['arm']!=arm
                or p.get('manifest')!=plain.pointer(run/'run_manifest.json')
                or p.get('outcome')!=plain.pointer(run/'run_outcome.json')):
            raise InvalidArtifact('Recovery belongs to another firstpass')
        expected=driver_for(run).recovery_jobs(first['manifest'],first['outcome'])
        if second['manifest']['jobs']!=expected:
            raise InvalidArtifact('Recovery population is not the uniform bounded technical set')
    views={'strict_firstpass':base.record_rows(jobs,first),
           'compatible_firstpass':base.record_rows(jobs,first,compatible=True),
           'final':base.record_rows(jobs,first,second,compatible=True,final=True)}
    phases=[first]+([second] if second else [])
    accounts={p['run']:audit_retrieval_and_usage(p,source) for p in phases}
    all_qa=[r for p in phases for r in p['ledger']]
    all_retrieval=[r for a in accounts.values() for r in a['retrieval_records']]
    errors=[{'phase':p['run'],**e} for p in phases for e in p['errors']]
    combined_usage=measure_usage(all_qa+all_retrieval)
    combined_usage['scope_complete']=all(a['combined_lineage']['scope_complete'] for a in accounts.values())
    if any(a['combined_lineage']['new_namespace_only']['total_cost_usd'] is None for a in accounts.values()):
        combined_usage['total_cost_usd']=None
    return {'schema':'regroup-refinement-single-arm400-qa-summary-v1','created_utc':now(),'arm':arm,
        'planned_per_arm':DENOMINATOR,'planned_total':DENOMINATOR,'source_manifest':plain.pointer(run/'source_jobs.json'),
        'views':{name:aggregate(rows,arm) for name,rows in views.items()},'slots_by_view':views,
        'paired_comparison':None,'recovery_supplied':second is not None,
        'final_interpretation':'uniform bounded recovery applied' if second else 'compatible firstpass; no recovery supplied',
        'audit_status':'no_invalid_artifacts' if not errors else 'invalid_artifacts_retained_as_missing','audit_errors':errors,
        'phase_sources':[{'manifest':plain.pointer(Path(p['run'])/'run_manifest.json'),
                          'outcome':plain.pointer(Path(p['run'])/'run_outcome.json'),'ledgers':p['ledger_refs']} for p in phases],
        'phase_counts':{p['run']:p['outcome']['counts'] for p in phases},
        'qa_usage':measure_usage(all_qa),'retrieval_usage':measure_usage(all_retrieval),
        'current_qa_plus_retrieval_usage':combined_usage,'phase_accounting':accounts,
        'writer_accounting':writer_accounting(writer_call_audit,arm,first),
        'answer_length':{name:{'valid':len(v:=[r for r in rows if r['reader_answer_characters'] is not None]),
            'mean_characters':sum(r['reader_answer_characters'] for r in v)/len(v) if v else None,
            'mean_whitespace_words':sum(r['reader_answer_whitespace_words'] for r in v)/len(v) if v else None}
            for name,rows in views.items()},
        'input_sha256':{k:v for p in phases for k,v in p['inputs'].items()},
        'summary_code_sha256':{p:plain.sha(ROOT/p) for p in CODE},'model_calls_in_summary':0,'native_clients':0,
        'limitations':['One arm only; no paired or historical comparison is inferred.',
            'Official LoCoMo F1, custom token F1 and Judge each retain their independent valid denominator and fixed400 missingness sensitivity.',
            'Final without recovery is compatible firstpass, not a claim that technical failures were retried.',
            'Compatibility reinterprets original Judge text only; firstpass/compatible/recovery remain separate.',
            'Retrieval and QA share a physical ledger but are partitioned by validated context phase, with original full-file SHA retained.',
            'Writer and upstream grouping/segmentation costs are separate; cache reuse is not free cold-start computation.',
            'Known ten-conversation cohort is not blind validation; twenty facts do not impose equal token counts.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firstpass',type=Path,required=True)
    parser.add_argument('--recovery',type=Path)
    parser.add_argument('--writer-call-audit',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh summary output required')
    with forbid_network():value=summarize(args.firstpass,args.recovery,writer_call_audit=args.writer_call_audit)
    save_new(args.output,value)
    print(json.dumps({'output':str(args.output),'audit_status':value['audit_status'],'model_calls':0}))


if __name__=='__main__':main()
