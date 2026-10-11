"""Independent read-only checks for the single c8 EOF reconstruction.

No recovery-runner import, API, Memory/Qdrant client or database writes.
The caller supplies hash-tracked readers; SQLite is handled by the audit core.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = 'mem0-writer-single-eof-root-close-v1'
SCHEMA = 'dated-mem0-eof-empty-reconstruction-v1'
ENTRY_SCHEMA = 'dated-mem0-eof-empty-reconstruction-entry-v2'


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key')
        out[key] = value
    return out


def independent_parse(response, http_status):
    """Decode original array before appending; never call author compatibility.

    A single independently decoded complete memory array and whitespace must
    occupy everything after the initial root key. This excludes unclosed inner
    values without relying on the author's delimiter-stack implementation.
    """
    candidates = response.get('candidates', [])
    if not candidates or candidates[0].get('finishReason') != 'STOP':
        raise ValueError('Generation not STOP')
    parts = candidates[0].get('content', {}).get('parts', [])
    original = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
    try:
        parsed = json.loads(original)
    except json.JSONDecodeError as error:
        if error.pos != len(original):
            raise ValueError('Non-EOF JSON error') from error
        error_info = {'type': type(error).__name__, 'message': error.msg,
                      'position': error.pos, 'length': len(original)}
    else:
        return parsed, {'protocol': PROTOCOL, 'action': 'unchanged',
                        'original_text_sha256': text_sha(original)}
    if (http_status != 200 or len(candidates) != 1 or len(parts) != 1
            or set(parts[0]) != {'text'} or not isinstance(parts[0]['text'], str)):
        raise ValueError('EOF repair requires one plain HTTP200/STOP response')
    decoder = json.JSONDecoder(object_pairs_hook=_unique)
    remaining = original.lstrip(' \t\r\n')
    if not remaining.startswith('{'):
        raise ValueError('Root object absent')
    remaining = remaining[1:].lstrip(' \t\r\n')
    key, end = decoder.raw_decode(remaining)
    remaining = remaining[end:].lstrip(' \t\r\n')
    if key != 'memory' or not remaining.startswith(':'):
        raise ValueError('Only memory root is admissible')
    remaining = remaining[1:].lstrip(' \t\r\n')
    memory, end = decoder.raw_decode(remaining)
    if not isinstance(memory, list) or remaining[end:].strip(' \t\r\n'):
        raise ValueError('Complete terminal memory array required')
    if any(not isinstance(m, dict) or not isinstance(m.get('text'), str)
           or not m['text'].strip() for m in memory):
        raise ValueError('Invalid original extraction record')
    derived = original + '}'
    parsed = decoder.decode(derived)
    if parsed != {'memory': memory} or list(parsed) != ['memory']:
        raise ValueError('Changed original records')
    return parsed, {'protocol': PROTOCOL, 'action': 'append_single_root_brace',
        'original_error': error_info, 'original_text_sha256': text_sha(original),
        'derived_text_sha256': text_sha(derived), 'appended_utf8_hex': '7d',
        'original_prefix_byte_identical': True, 'records': len(memory),
        'memory_sha256': text_sha(json.dumps(memory, ensure_ascii=False, sort_keys=True)),
        'http_status': http_status, 'finish_reason': 'STOP'}


def terminal_preflight(run, read, bound, verify, digest):
    """Must execute before opening any native SQLite file."""
    run = Path(run)
    folder = run / 'conv_8/dated_merged'
    state = read(folder / 'write_status.json')
    if (state.get('status') != 'complete' or state.get('completed_groups') != 29
            or state.get('expected_groups') != 29):
        raise ValueError('Recovery not terminal complete; no native SQLite opened')
    outcome = verify(folder / 'eof_recovery_outcome.json')
    bank = bound(outcome['bank'])
    manifest = read(run / 'run_manifest.json')
    if (outcome.get('status') != 'complete' or not bank.get('complete')
            or len(bank.get('journal_files', [])) != 29
            or bound(outcome['write_status']) != state or state['bank'] != outcome['bank']
            or outcome.get('original_source_status') != 'failed'
            or outcome.get('source_partial_store_copied') is not False
            or outcome.get('compatibility_protocol') != PROTOCOL):
        raise ValueError('Recovery outcome is incomplete or provenance changed')
    protocol = manifest['protocol_sha256']
    if protocol != digest({k: v for k, v in manifest.items() if k != 'protocol_sha256'}):
        raise ValueError('Manifest protocol hash mismatch')
    if any(v['protocol_sha256'] != protocol for v in (state, bank, outcome)):
        raise ValueError('Protocol chain mismatch')
    info = manifest['eof_recovery']
    if (info['schema'] != ENTRY_SCHEMA or outcome['schema'] != ENTRY_SCHEMA
            or outcome['source_binding_sha256'] != digest(info['binding'])):
        raise ValueError('Source binding mismatch')
    return manifest, state, outcome, bank


def check_manifest_bindings(run, manifest, read, bound, sha, digest):
    """Bind the 14 original first-generation pins and unchanged failure seal."""
    info = manifest['eof_recovery']; binding = info['binding']
    original = Path(binding['source_run'])
    old = original / 'conv_8/dated_merged'
    if original.resolve() == Path(run).resolve():
        raise ValueError('Recovery cannot alias original run')
    original_manifest = bound(binding['source_manifest'])
    failure_audit = bound(binding['source_audit']); case_proof = bound(binding['case_proof'])
    seal = bound(info['terminal_seal'])
    if (seal['status'] != 'all16_terminal_original_failures_preserved'
            or seal['counts']['banks'] != 16 or len(seal['process_snapshots']) != 2):
        raise ValueError('Original firstpass seal not terminal')
    for p in seal['process_snapshots']:
        if bound(p)['processes'] != []:
            raise ValueError('Original writer was active in sealed snapshot')
    for p in [seal['source_manifest'], *seal['original_code'], *seal['file_inventory']]:
        if sha(p['path']) != p['sha256'] or Path(p['path']).stat().st_size != p['bytes']:
            raise ValueError('Original sealed file changed')
    actual_inventory = {str(p.resolve()): sha(p) for p in old.rglob('*')
                        if p.is_file() and p.name != '.lock'}
    if actual_inventory != binding['source_inventory'] or actual_inventory != failure_audit['target_inventory_before']:
        raise ValueError('Original partial source changed')
    if (read(old / 'write_status.json')['status'] != 'failed'
            or read(old / 'write_outcome.json')['status'] != 'failed' or (old / 'bank.json').exists()):
        raise ValueError('Original failure status must remain preserved')
    source_input = bound(binding['source_input'])
    if sha(old / 'group_inputs.json') != sha(Path(run) / 'conv_8/dated_merged/group_inputs.json'):
        raise ValueError('Recovery inputs not byte-identical')
    if len(source_input['envelopes']) != 29:
        raise ValueError('Original group population changed')
    pins = binding['pins']
    if len(pins) != 14 or [p['sequence'] for p in pins] != list(range(14)):
        raise ValueError('Fourteen original prefix pins required')
    for i, pin in enumerate(pins):
        j = bound(pin['journal']); raw = bound(pin['raw'])
        if (pin['group_id'] != source_input['envelopes'][i]['group_id']
                or j['sequence'] != i or j['group_id'] != pin['group_id']
                or raw['request'] != pin['signature']
                or hashlib.sha256(json.dumps(raw['request'], sort_keys=True, ensure_ascii=False).encode()).hexdigest() != pin['request_hash']
                or Path(pin['raw']['path']).resolve() != (old/'native/api_cache'/(pin['request_hash']+'.json')).resolve()):
            raise ValueError('Original pin sequence/request mismatch')
        parsed, proof = independent_parse(raw['response'], 200)
        if parsed['memory'] != pin['parsed_memory'] or proof != pin['parse_proof']:
            raise ValueError('Original pin parse mismatch')
        if i < 13:
            if proof['action'] != 'unchanged' or parsed['memory'] != j['llm_calls'][0]['parsed_memory']:
                raise ValueError('Original successful prefix changed')
        elif proof['action'] != 'append_single_root_brace' or pin['raw'] != case_proof['original_raw']:
            raise ValueError('Reviewed original EOF pin changed')
    for key in ('code_sha256', 'runtime_source_sha256'):
        for p, expected in manifest['remaining_expansion'][key].items():
            if sha(Path(__file__).resolve().parents[1]/p) != expected:
                raise ValueError('Inherited frozen code changed')
    root = Path(__file__).resolve().parents[1]
    for p, expected in info['code_sha256'].items():
        if sha(root/p) != expected:
            raise ValueError('Recovery source code changed')
    if sha(info['policy_document']['path']) != info['policy_document']['sha256']:
        raise ValueError('Policy changed')
    probe = bound(info['prefix_probe'])
    if (probe['status'] != 'passed' or probe['source_binding_sha256'] != digest(binding)
            or probe['schema'] != ENTRY_SCHEMA+'-prefix-probe'
            or probe['network_attempts'] != 0 or probe['pinned_generation_requests'] != 14
            or probe['code_sha256'] != info['code_sha256']):
        raise ValueError('Offline prefix probe mismatch')
    prepared = info['prepared_v1']
    if (prepared['protocol_sha256'] != '7304728753b1e78e61bb36329564fb1e21c540b5c02b94470f19444a9300b876'
            or probe['prepared_v1_binding_sha256'] != digest(prepared)
            or prepared['failure']['model_calls'] != 0 or prepared['failure']['bank_intents'] != 0
            or prepared['failure']['native_stores'] != 0):
        raise ValueError('Prepared-only v1 binding changed')
    actual_prepared = {str(p.resolve()) for p in Path(prepared['run_dir']).rglob('*') if p.is_file()}
    if actual_prepared != {p['path'] for p in prepared['files']}:
        raise ValueError('Prepared v1 contains additional files')
    for p in prepared['files']:
        if sha(p['path']) != p['sha256'] or Path(p['path']).stat().st_size != p['bytes']:
            raise ValueError('Prepared v1 artifact changed')
    for key, expected in {'expected_groups':29, 'pinned_extractions':14, 'successful_prefix_groups':13,
            'bank_attempts':1, 'generation_and_compatibility_changed':False,
            'store':'new_empty_store_no_partial_copy',
            'compat_scope':'all_extractions_uniform_without_qa_or_correctness'}.items():
        if info['policy'][key] != expected:
            raise ValueError('Recovery admission policy differs')
    if original_manifest['dataset_sha256'] != manifest['dataset_sha256']:
        raise ValueError('Dataset binding changed')
    return binding


def check_lineage(folder, journal, rawpath, raw, binding, read, bound, verify, sha, digest, ledger):
    folder = Path(folder); gid = journal['group_id']; seq = journal['sequence']
    record_path = folder/'native/eof_parse_lineage'/(gid+'.json')
    record = verify(record_path)
    key = hashlib.sha256(json.dumps(raw['request'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    pin = next((p for p in binding['pins'] if p['group_id'] == gid), None)
    if (record['schema'] != SCHEMA or record['group_id'] != gid or record['request_hash'] != key
            or Path(record['original_raw']['path']).resolve() != Path(rawpath).resolve()
            or bound(record['original_raw']) != raw
            or record['derived_response_persisted_to_cache'] is not False
            or record['source_pin'] != (pin['raw'] if pin else None)
            or record['compat_source_sha256'] != sha(Path(__file__).resolve().parents[1]/'jev_mem/writer_eof_compat_v1.py')):
        raise ValueError('Recorded raw lineage differs')
    if bool(pin) != (seq < 14):
        raise ValueError('Pinned-prefix membership differs')
    if pin and (raw['request'] != pin['signature'] or sha(rawpath) != pin['raw']['sha256']):
        raise ValueError('Original raw pin not byte-identical')
    evidence = record['physical_ledger']
    expected_ledger = (Path(binding['source_run'])/'conv_8/dated_merged/native/api_calls.jsonl'
                       if pin else folder/'native/api_calls.jsonl')
    if Path(evidence['path']).resolve() != expected_ledger.resolve():
        raise ValueError('Physical evidence uses another ledger')
    sha(expected_ledger); data = expected_ledger.read_bytes(); size = evidence['prefix_bytes']
    if type(size) is not int or not 0 < size <= len(data) or hashlib.sha256(data[:size]).hexdigest() != evidence['prefix_sha256']:
        raise ValueError('HTTP ledger prefix changed')
    prefix_rows = [json.loads(s) for s in data[:size].decode().splitlines()]
    event = prefix_rows[evidence['physical_event_line']-1]
    if event != evidence['physical_event'] or digest(event) != evidence['physical_event_sha256']:
        raise ValueError('HTTP event proof changed')
    physical = [x for x in [json.loads(s) for s in data.decode().splitlines()]
                if x.get('stage') == 'mem0_extract' and x.get('request_hash') == key
                and x.get('context', {}).get('group_id') == gid and not x.get('cache_hit')]
    if (len([x for x in physical if x.get('status') == 200]) != 1 or physical[-1] != event
            or event.get('status') != 200 or event.get('provider') != 'gemini'
            or event.get('model') != 'gemini-2.5-flash' or event.get('operation') != 'generateContent'
            or event.get('context', {}).get('conversation') != 8 or event['context'].get('method') != 'dated_merged'
            or event.get('usage') != raw['response'].get('usageMetadata', {})
            or event.get('resolved_model') != raw['response'].get('modelVersion')):
        raise ValueError('Physical first-generation evidence differs')
    current_events = [x for x in ledger if x.get('stage') == 'mem0_extract'
                      and x.get('context', {}).get('group_id') == gid]
    if pin and (not current_events or any(not x.get('cache_hit') for x in current_events)):
        raise ValueError('Pinned extraction was physically redispatched')
    parsed, proof = independent_parse(raw['response'], event['status'])
    call = journal['llm_calls'][0]
    if (proof != record['parse_proof'] or json.loads(call['response']) != parsed
            or call['parsed_memory'] != parsed['memory']):
        raise ValueError('Independent original/derived parse differs from journal')
    return parsed, {'lineage': {'path': str(record_path.resolve()), 'sha256': sha(record_path)},
                    'action': proof['action'], 'physical_ledger': evidence,
                    'source_pin': pin['raw'] if pin else None, 'proof': proof}
