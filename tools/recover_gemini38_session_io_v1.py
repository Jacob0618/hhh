"""Candidate, gated one-shot c3/session rebuild; import/check never construct an API.

No frozen file is patched. The original write_sessions and add_session code
objects execute with isolated globals. Only journal publication gains bounded
PermissionError retries. This is not an effective40 source admission tool.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_gemini38_writer_v1 as frozen
from tools.check_experiment_health import shared_read
from jev_mem import mem0_adapter
from jev_mem.gemini38_provider_v1 import Gemini38API, check_request, response_text
from jev_mem.sdk20_cached_api import digest as request_digest, _publish_bytes

SCHEMA = 'gemini38-session-io-exact-prefix-recovery-v1'
REVIEW_SCHEMA = 'gemini38-stage-end-failure-review-v1'
CODE_REVIEW_SCHEMA = 'gemini38-session-io-candidate-code-review-v1'
SELF = Path(__file__).resolve()
SLOT = (3, 'session')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(shared_read(path)).hexdigest()


def pointer(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def read_bound(p):
    raw = shared_read(p['path'])
    require(hashlib.sha256(raw).hexdigest() == p['sha256'], 'Bound file SHA differs')
    return json.loads(raw)


def new_json(path, value):
    _publish_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def exclusive_intent(path, value):
    """Atomic OS create, not exists-then-replace; partial intent is retained."""
    with Path(path).open('x',encoding='utf-8') as h:
        json.dump(value,h,ensure_ascii=False,allow_nan=False)
        h.write('\n'); h.flush(); os.fsync(h.fileno())


@contextlib.contextmanager
def recovery_execution_lock(namespace):
    """Exclusive byte lock only in the new namespace, separate from parent read lock."""
    import msvcrt
    with (Path(namespace)/'recovery_execution.lock').open('a+b') as h:
        h.seek(0,2)
        if h.tell() == 0:
            h.write(b'0'); h.flush(); os.fsync(h.fileno())
        h.seek(0)
        try:
            msvcrt.locking(h.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as ex:
            raise RuntimeError('Another recovery executor owns this new namespace') from ex
        try:
            yield
        finally:
            h.seek(0); msvcrt.locking(h.fileno(),msvcrt.LK_UNLCK,1)


def journal_json(path, value, *, replace=os.replace, sleeper=time.sleep):
    """Write once, fsync once, retry only the same journal-byte publication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8') as h:
        json.dump(value, h, ensure_ascii=False, indent=2, allow_nan=False)
        h.write('\n'); h.flush(); os.fsync(h.fileno())
    for attempt in range(8):
        try:
            replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            sleeper(0.1*(attempt+1))


def clone(function, **replacements):
    space = dict(function.__globals__)
    space.update(replacements)
    result = FunctionType(function.__code__, space, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class JournalRetryAdapter(mem0_adapter.Mem0SessionAdapter):
    add_session = clone(mem0_adapter.Mem0SessionAdapter.add_session, _atomic_json=journal_json)


class CacheDrift(BaseException):
    """Never swallowed by SDK catch Exception; never a provider retry."""


@contextlib.contextmanager
def closed_parent_lock(parent):
    """Read-only byte lock on existing original controller lock; no file writes."""
    import msvcrt
    path = Path(parent)/'writer_execution.lock'
    require(path.is_file() and path.stat().st_size >= 1, 'Original controller lock missing')
    with path.open('rb') as h:
        h.seek(0)
        try:
            msvcrt.locking(h.fileno(), msvcrt.LK_NBRLCK, 1)
        except OSError as ex:
            raise RuntimeError('Original Writer still owns its execution lock') from ex
        try:
            yield
        finally:
            h.seek(0); msvcrt.locking(h.fileno(), msvcrt.LK_UNLCK, 1)


def pid_alive(pid):
    require(type(pid) is int and pid > 0, 'Missing original process identity')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.GetExitCodeProcess.restype = wintypes.BOOL
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            require(ctypes.get_last_error() == 87, 'Cannot establish process termination')
            return False
        try:
            code = wintypes.DWORD()
            require(bool(api.GetExitCodeProcess(handle, ctypes.byref(code))), 'Process status unreadable')
            return code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def validate_review(review_pointer, namespace, *, alive=pid_alive):
    r = read_bound(review_pointer)
    require(r.get('schema') == REVIEW_SCHEMA and r.get('status') == 'reviewed_after_firstpass'
            and r.get('qa_gold_used') is False, 'Uniform stage-end review required')
    m = read_bound(r['parent_manifest'])
    parent = Path(r['parent_manifest']['path']).resolve().parent
    require(m == frozen.manifest(parent) and r['protocol_sha256'] == m['protocol_sha256'],
            'Original manifest/protocol binding differs')
    policy = read_bound(r['policy']) if str(r['policy']['path']).endswith('.json') else None
    if policy is None:
        require(sha(r['policy']['path']) == r['policy']['sha256'], 'Policy SHA differs')
    controller = json.loads(shared_read(parent/'controller_status.json'))
    require(controller.get('terminal') is True and controller.get('status') in ('complete', 'failed')
            and controller.get('protocol_sha256') == m['protocol_sha256'], 'Controller not terminal')
    require(not alive(controller['pid']), 'Original controller process remains alive')
    expected = {(ci, arm) for ci in range(10) for arm in frozen.ARMS}
    require(len(r.get('banks', [])) == 40, 'Exactly forty stage-end bank records required')
    seen, failures = set(), set()
    for row in r['banks']:
        slot = (row['conversation'], row['arm'])
        require(slot in expected and slot not in seen, 'Duplicate or foreign bank slot')
        seen.add(slot)
        folder = parent/f'conv_{slot[0]}'/slot[1]
        require(Path(row['write_status']['path']).resolve() == (folder/'write_status.json').resolve()
                and Path(row['outcome']['path']).resolve() == (folder/'write_outcome.json').resolve(),
                'Bank status/outcome path differs')
        state, outcome = read_bound(row['write_status']), read_bound(row['outcome'])
        frozen.verify(row['outcome']['path'])
        for value in (state, outcome):
            require((value.get('conversation'),value.get('arm'),value.get('protocol_sha256'))
                    == (*slot,m['protocol_sha256']), 'Terminal bank identity differs')
            require(value.get('status') in ('complete','failed') and value.get('finished_utc'),
                    'Original bank lacks a terminal outcome')
        require(state['status'] == outcome['status'] == row['original_status']
                and outcome['write_status'] == row['write_status'], 'Terminal status mismatch')
        inp = frozen.verify(folder/'group_inputs.json')
        require(state.get('expected_groups') == len(inp['envelopes'])
                and type(state.get('completed_groups')) is int
                and 0 <= state['completed_groups'] <= state['expected_groups']
                and type(state.get('facts')) is int and state['facts'] >= 0,
                'Terminal progress counts differ')
        if state['status'] == 'complete':
            bank = read_bound(outcome['bank'])
            frozen.verify(outcome['bank']['path'])
            require(state['completed_groups'] == state['expected_groups']
                    and bank.get('complete') is True and len(bank['facts']) == state['facts']
                    and (bank['conversation'],bank['arm'],bank['protocol_sha256']) == (*slot,m['protocol_sha256']),
                    'Complete bank/count binding differs')
        require(not alive(state['pid']), 'Original bank process remains alive')
        if state['status'] != 'complete':
            failures.add(slot)
    require(seen == expected, 'Missing stage-end bank')
    fs = r.get('failures', [])
    require(len(fs) == len(failures) and {(x['conversation'],x['arm']) for x in fs} == failures,
            'Failure list must exactly equal original non-complete banks')
    for item in fs:
        require(type(item.get('approved_for_once')) is bool, 'Failure approval not explicit')
        read_bound(item['diagnosis'])
        require(item.get('action') in ('exact_prefix_io_rebuild','readonly_native_export_with_declared_link_deficit')
                and item['approved_for_once'] is True, 'Unreviewed failure blocks recovery')
    items = [x for x in fs if (x['conversation'],x['arm']) == SLOT]
    require(len(items) == 1 and items[0]['action'] == 'exact_prefix_io_rebuild'
            and Path(items[0]['recovery_namespace']).resolve() == Path(namespace).resolve(),
            'Approved one-shot namespace differs')
    return r, m, parent, items[0]


def prefix_sources(diagnosis_pointer, parent):
    proof = read_bound(diagnosis_pointer)
    require(proof.get('schema') == 'gemini38-c3-session-failure-readonly-audit-61-v1'
            and proof['terminal_status']['status'] == 'failed'
            and proof['terminal_status']['completed_groups'] == 3
            and proof['terminal_status']['facts'] == 21, 'Wrong IO failure diagnosis')
    evidence = proof['evidence_files']
    for p, binding in evidence.items():
        require(sha(p) == binding['sha256'], 'Original IO evidence changed')
    native = Path(parent)/'conv_3/session/native'
    ledger_path = native/'api_calls.jsonl'
    require(str(ledger_path.resolve()) in evidence, 'Original ledger not sealed')
    ledger = [json.loads(x) for x in shared_read(ledger_path).splitlines() if x.strip()]
    require(len(ledger) == 18 and len({x['request_hash'] for x in ledger}) == 18,
            'Expected eighteen unique original ended requests')
    keys = {x['request_hash']+'.json' for x in ledger}
    require({p.name for p in (native/'api_cache').glob('*.json')} == keys
            and {p.name for p in (native/'request_intents').glob('*.json')} == keys,
            'Extra/missing original raw or unresolved intent')
    rows = []
    for event in ledger:
        key = event['request_hash']
        rp = native/'api_cache'/f'{key}.json'
        ip = native/'request_intents'/f'{key}.json'
        require(str(rp.resolve()) in evidence and str(ip.resolve()) in evidence,
                'Raw/intent evidence missing')
        raw = read_bound({'path':str(rp),'sha256':evidence[str(rp.resolve())]['sha256']})
        intent = read_bound({'path':str(ip),'sha256':evidence[str(ip.resolve())]['sha256']})
        require(raw['request'] == intent['signature'] and request_digest(raw['request']) == key
                and intent['context'] == event['context'] and intent['stage'] == event['stage']
                and event.get('status') == 200 and event.get('attempt') == 1
                and event.get('timestamp_utc'), 'Original request lacks complete success proof')
        signature = raw['request']
        check_request(signature['provider'],signature['model'],signature['body'],
                      signature['operation'],event['stage'])
        rows.append({'request_hash':key,'signature':signature,'context':event['context'],
                     'stage':event['stage'],'raw':pointer(rp),'intent':pointer(ip),
                     'original_attempt':copy.deepcopy(event)})
    pins = proof['generation_cache_chain']
    generations = [x for x in rows if x['stage'] == 'mem0_extract']
    require([x['context']['session_id'] for x in generations]
            == ['session_1','session_2','session_3','session_4'] and len(pins) == 4,
            'Exactly four original generations required')
    for row, pin, count in zip(generations,pins,(9,6,6,5)):
        value = read_bound(row['raw'])
        require(row['request_hash'] == pin['request_hash']
                and len(json.loads(response_text(value['response']))['memory']) == pin['parsed_count'] == count,
                'Generation pin/content differs')
    return rows, {p:x['sha256'] for p,x in evidence.items()}


class PrefixGuard:
    def __init__(self, rows, event_path):
        self.rows, self.event_path = rows, Path(event_path)
        self.position = 0
        self.fourth_returned = False

    def log(self, event):
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open('a',encoding='utf-8') as h:
            h.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n'); h.flush(); os.fsync(h.fileno())

    def before(self, context, signature, stage):
        try:
            check_request(signature['provider'],signature['model'],signature['body'],signature['operation'],stage)
            sid = context.get('session_id')
            if self.position < len(self.rows):
                row = self.rows[self.position]
                require(row['signature'] == signature and row['context'] == context and row['stage'] == stage,
                        'Known complete request/context/order drift; no fresh generation')
                read_bound(row['raw']); read_bound(row['intent'])
                return row
            require(self.fourth_returned and sid not in ('session_1','session_2','session_3'),
                    'Original prefix incomplete or earlier session repeated')
            require(sid != 'session_4' or stage != 'mem0_extract', 'Fourth generation must never be re-extracted')
            require(sid in {f'session_{n}' for n in range(4,30)}, 'Undeclared future session')
            return None
        except Exception as ex:
            raise CacheDrift(str(ex)) from ex

    def returned(self, row):
        if row is not None:
            self.position += 1
            if row['stage'] == 'mem0_extract' and row['context']['session_id'] == 'session_4':
                self.fourth_returned = True


class RecoveryAPI(Gemini38API):
    def __init__(self, run_dir, *, prefix_guard, **kwargs):
        self.prefix_guard = prefix_guard
        super().__init__(run_dir, **kwargs)

    def scoped(self, **context):
        return type(self)(self.run_dir,prefix_guard=self.prefix_guard,keys=self.keys,retries=self.retries,
                          timeout=self.timeout,context={**self.context,**context},cache_roots=(),
                          gate_path=self.gate_path,interval_seconds=self.interval_seconds,
                          cooldown_seconds=self.cooldown_seconds)

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        signature = {'provider':provider,'model':model,'operation':operation,'body':body}
        key = request_digest(signature)
        try:
            row = self.prefix_guard.before(self.context,signature,stage)
        except CacheDrift as ex:
            self.prefix_guard.log({'event':'hard_guard_rejection','request_hash':key,'stage':stage,
                                   'context':copy.deepcopy(self.context),'finished_utc':frozen.now(),
                                   'error_type':'CacheDrift','message':str(ex)[:1500],'physical_attempts':0})
            raise
        event = {'request_hash':key,'context':copy.deepcopy(self.context),'stage':stage,
                 'started_utc':frozen.now(),'source':'inherited_exact_request' if row else 'new_recovery_request',
                 'original_raw':row['raw'] if row else None,'new_physical_generation':False if row else None}
        self.prefix_guard.log({**event,'event':'request_started'})
        try:
            if row:
                target = self.cache_dir/f'{key}.json'
                require(sha(target) == row['raw']['sha256'], 'Imported raw SHA differs')
            response = super().request(provider,model,body,operation=operation,stage=stage,cache=cache)
            if row:
                require(response == read_bound(row['raw'])['response'], 'Replayed response differs')
            self.prefix_guard.returned(row)
            self.prefix_guard.log({**event,'event':'request_returned','finished_utc':frozen.now(),
                                  'consumed_raw':pointer(self.cache_dir/f'{key}.json')})
            return response
        except Exception as ex:
            self.prefix_guard.log({**event,'event':'request_failed','finished_utc':frozen.now(),
                                  'error_type':type(ex).__name__})
            if row:
                raise CacheDrift('Known prefix import/consume failed: '+type(ex).__name__) from ex
            raise


def import_prefix(native, rows):
    for row in rows:
        key = row['request_hash']; source = row['raw']
        read_bound(source)
        target = Path(native)/'api_cache'/f'{key}.json'
        lineage = {'source_path':source['path'],'source_sha256':source['sha256'],
                   'target_path':str(target.resolve()),'target_sha256':source['sha256'],'request_hash':key,
                   'original_intent':row['intent'],'original_physical_attempt':row['original_attempt']}
        new_json(Path(native)/'cache_lineage'/f'{key}.json',lineage)
        raw = shared_read(source['path'])
        require(hashlib.sha256(raw).hexdigest() == source['sha256'], 'Import source raced')
        _publish_bytes(target,raw)


def inputs_for_review(review_pointer, namespace, code_review_pointer, probe_review_pointer):
    review,m,parent,item = validate_review(review_pointer,namespace)
    cr = read_bound(code_review_pointer)
    require(cr.get('schema') == CODE_REVIEW_SCHEMA and cr.get('status') == 'passed'
            and cr.get('qa_gold_used') is False and cr.get('code_sha256',{}).get(str(SELF)) == sha(SELF),
            'Independent candidate code review missing or stale')
    for p,h in cr['code_sha256'].items():
        require(sha(p) == h, 'Candidate reviewed code changed')
    pr = read_bound(probe_review_pointer)
    require(pr.get('schema') == 'gemini38-prefix-probe-independent-review-61-v1'
            and pr.get('status') == 'passed_with_limited_scope', 'Independent prefix review missing')
    probe = read_bound(pr['result'])
    require(sha(pr['helper']['path']) == pr['helper']['sha256'], 'Reviewed prefix helper changed')
    rows,evidence = prefix_sources(item['diagnosis'],parent)
    require(probe.get('status') == 'passed' and probe.get('model_calls') == 0
            and probe.get('network_attempts') == 0 and probe.get('source_unchanged') is True
            and probe.get('successful_sessions') == 3 and probe.get('facts') == 21
            and probe.get('fourth_generation_verified_before_ADD_stop') is True
            and probe.get('source_evidence_sha256') == evidence
            and [x['request_hash'] for x in probe['trace']] == [x['request_hash'] for x in rows],
            'Prefix probe and IO diagnosis do not bind the same exact evidence')
    return review,m,parent,rows,evidence


def prepare(namespace, review_pointer, code_review_pointer, probe_review_pointer):
    namespace = Path(namespace).resolve()
    require(not namespace.exists(), 'New empty once-only namespace required')
    review = read_bound(review_pointer)
    parent = Path(review['parent_manifest']['path']).resolve().parent
    require(namespace != parent and parent not in namespace.parents and namespace not in parent.parents,
            'Recovery namespace must be separate from original run')
    with closed_parent_lock(parent):
        r,m,parent,rows,evidence = inputs_for_review(review_pointer,namespace,code_review_pointer,probe_review_pointer)
        manifest = {'schema':SCHEMA,'created_utc':frozen.now(),'parent_manifest':r['parent_manifest'],
                    'protocol_sha256':m['protocol_sha256'],'failure_review':review_pointer,
                    'candidate_code_review':code_review_pointer,'prefix_independent_review':probe_review_pointer,
                    'code_sha256':{str(SELF):sha(SELF)},'original_evidence_sha256':evidence,
                    'prefix_requests':rows,'namespace':str(namespace),'model_calls_at_prepare':0,
                    'policy':{'once_only':True,'journal_replace_attempts':8,'journal_retry_only':'PermissionError',
                              'partial_native_copy':False,'original_prefix_fresh_calls':False,
                              'scientific_protocol_unchanged':True,'source_admission_requires_new_full_review':True}}
        manifest['execution_recovery_sha256'] = frozen.digest(manifest)
        namespace.mkdir(parents=True,exist_ok=False)
        raw = shared_read(parent/'conv_3/session/group_inputs.json')
        _publish_bytes(namespace/'conv_3/session/group_inputs.json',raw)
        new_json(namespace/'recovery_manifest.json',manifest)
    return manifest


def check(namespace):
    namespace = Path(namespace).resolve()
    m = json.loads(shared_read(namespace/'recovery_manifest.json'))
    require(m.get('schema') == SCHEMA and m.get('namespace') == str(namespace)
            and m.get('execution_recovery_sha256') == frozen.digest({k:v for k,v in m.items() if k != 'execution_recovery_sha256'})
            and m.get('code_sha256') == {str(SELF):sha(SELF)}, 'Recovery manifest/code changed')
    r,parent_manifest,parent,rows,evidence = inputs_for_review(m['failure_review'],namespace,
                                                              m['candidate_code_review'],m['prefix_independent_review'])
    require(m['parent_manifest'] == r['parent_manifest'] and m['protocol_sha256'] == parent_manifest['protocol_sha256']
            and rows == m['prefix_requests'] and evidence == m['original_evidence_sha256'], 'Prefix binding changed')
    require(shared_read(namespace/'conv_3/session/group_inputs.json') == shared_read(parent/'conv_3/session/group_inputs.json'),
            'Recovery source sessions changed')
    return m,parent_manifest,parent


def run_core(namespace, m, parent_manifest, guard, *, api_factory=None, core=None):
    """Injectable seam for mock tests; original scientific code object is retained."""
    if api_factory is None:
        api_factory = lambda *a,**kw: RecoveryAPI(*a,prefix_guard=guard,**kw)
    core = core or frozen.write_sessions
    function = clone(core,Gemini38API=api_factory,Mem0SessionAdapter=JournalRetryAdapter,
                     publish_json=journal_json)
    folder = Path(namespace)/'conv_3/session'
    try:
        state = function(namespace,3,parent_manifest)
    except CacheDrift as ex:
        prior = json.loads(shared_read(folder/'write_status.json'))
        state = {**prior,'status':'blocked_request_drift','finished_utc':frozen.now(),
                 'error':{'type':'CacheDrift','message':str(ex)[:1500]},'store_reentry_forbidden':True}
        journal_json(folder/'write_status.json',state)
    except BaseException as ex:
        state = {'schema':frozen.SCHEMA,'protocol_sha256':parent_manifest['protocol_sha256'],
                 'conversation':3,'arm':'session','status':'interrupted','finished_utc':frozen.now(),
                 'error':{'type':type(ex).__name__,'message':'Unclassified hard stop; no automatic reentry'},
                 'store_reentry_forbidden':True}
        journal_json(folder/'write_status.json',state)
        new_json(folder/'recovery_outcome.json',{'schema':SCHEMA,'status':'interrupted',
                 'execution_recovery_sha256':m['execution_recovery_sha256'],'write_status':pointer(folder/'write_status.json')})
        raise
    if state['status'] == 'complete' and guard.position != len(guard.rows):
        state.update(status='blocked_request_drift',error={'type':'IncompletePrefix',
                     'message':'Native core claimed complete without consuming exact original prefix'},
                     store_reentry_forbidden=True)
        journal_json(folder/'write_status.json',state)
    result = {'schema':SCHEMA,'status':state['status'],'execution_recovery_sha256':m['execution_recovery_sha256'],
              'protocol_sha256':parent_manifest['protocol_sha256'],'original_status':'failed',
              'write_status':pointer(folder/'write_status.json'),'finished_utc':frozen.now(),
              'prefix_requests_consumed':guard.position,'prefix_requests_expected':len(guard.rows),
              'cost_accounting':'Original18 physical calls retained; imported usage is reused, not newly billed. New physical attempts stay in native/api_calls.jsonl.',
              'admission':'not_admitted_pending_full29session_structural_and_semantic_review'}
    if state['status'] == 'complete':
        result['bank'] = state['bank']
    scientific_outcome = {'schema':frozen.SCHEMA,'protocol_sha256':parent_manifest['protocol_sha256'],
                          'conversation':3,'arm':'session','status':state['status'],
                          'write_status':result['write_status'],'finished_utc':result['finished_utc'],
                          'external_cache_roots':[],'execution_recovery_sha256':m['execution_recovery_sha256'],
                          'cache_import_policy':'eighteen_original_same_bank_exact_requests_only',
                          'original_write_status':'failed','post_recovery_admission':False}
    if state['status'] == 'complete':
        scientific_outcome['bank'] = state['bank']
    else:
        scientific_outcome['error'] = state.get('error')
    new_json(folder/'write_outcome.json',frozen.grouped.sealed(scientific_outcome))
    result['scientific_write_outcome'] = pointer(folder/'write_outcome.json')
    new_json(folder/'recovery_outcome.json',result)
    return result


def execute(namespace):
    namespace = Path(namespace).resolve()
    pre = json.loads(shared_read(namespace/'recovery_manifest.json'))
    parent = Path(pre['parent_manifest']['path']).resolve().parent
    with recovery_execution_lock(namespace), closed_parent_lock(parent):
        m,pm,parent = check(namespace)
        folder = namespace/'conv_3/session'
        require(not any((folder/x).exists() for x in ('native','write_status.json','recovery_outcome.json')),
                'Recovery store/outcome exists; no reentry')
        intent = namespace/'recovery_intent.json'
        exclusive_intent(intent,{'schema':SCHEMA,'execution_recovery_sha256':m['execution_recovery_sha256'],
                         'pid':os.getpid(),'created_utc':frozen.now(),'once_only':True})
        guard = PrefixGuard(m['prefix_requests'],folder/'recovery_request_events.jsonl')
        try:
            import_prefix(folder/'native',guard.rows)
            with frozen.local_bm25_context(pm['resources']['snapshot'],pm['resources']['files_sha256']):
                return run_core(namespace,m,pm,guard)
        except BaseException as ex:
            if not (folder/'recovery_outcome.json').exists():
                new_json(folder/'recovery_outcome.json',{'schema':SCHEMA,'status':'failed_before_core',
                         'execution_recovery_sha256':m['execution_recovery_sha256'],
                         'finished_utc':frozen.now(),'error_type':type(ex).__name__,'store_reentry_forbidden':True})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','check','execute'))
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--failure-review',type=Path)
    p.add_argument('--candidate-code-review',type=Path)
    p.add_argument('--prefix-independent-review',type=Path)
    a = p.parse_args()
    if a.stage == 'prepare':
        require(all((a.failure_review,a.candidate_code_review,a.prefix_independent_review)), 'All review paths required')
        result = prepare(a.run_dir,pointer(a.failure_review),pointer(a.candidate_code_review),pointer(a.prefix_independent_review))
    elif a.stage == 'check':
        m,_,parent = check(a.run_dir)
        with closed_parent_lock(parent):
            check(a.run_dir)
        result = {'valid':True,'model_calls':0,'execution_recovery_sha256':m['execution_recovery_sha256']}
    else:
        result = execute(a.run_dir)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
