"""Read-only exact native writing/input audit; never invokes a provider or store."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_mem0_sparse1540 as run
from tools.run_pilot400_experiments_v2 import read
from jev_mem.data import load_data

def audit(conversations):
    from mem0.memory import main as native
    manifest=run.verify_manifest();items=[]
    for ci in conversations:
        folder=run.conv_path(ci);inputs=read(folder/'group_inputs.json')
        original=read(inputs['grouping_path'])
        if run.sha(inputs['grouping_path'])!=inputs['grouping_sha256']:
            raise ValueError('Original grouping changed')
        expected_inputs=run.group_envelopes(load_data()[ci],original)
        if inputs['envelopes']!=expected_inputs:raise ValueError('Native writing inputs are not exact original-message subsets')
        bank,digest=run.load_bank(ci,run.ARMS[1],manifest)
        if len(bank['journal_files'])!=len(expected_inputs):raise ValueError('Journal count differs from all groups')
        history=[];facts=[];warning_rows=[];empty=[];hashes=[]
        for envelope,entry in zip(expected_inputs,bank['journal_files']):
            if run.sha(entry['path'])!=entry['sha256']:raise ValueError('Native ADD journal changed')
            r=read(entry['path'])
            if r['status']!='success' or any(r[k]!=v for k,v in envelope.items()):
                raise ValueError('Native ADD input differs from source partition')
            if r['current_date']!=run.CURRENT_DATE or r['history_context']!=history[-10:]:
                raise ValueError('Native ADD date or exact ordered historical messages differ')
            if r['builder_call_count']!=1 or len(r['llm_calls'])!=1:
                raise ValueError('Expected one native extraction for each full group')
            call=r['llm_calls'][0];messages=call['kwargs']['messages']
            existing=[{'id':str(i),'text':f['text']} for i,f in enumerate(r['existing_memory_context'])]
            prompt=native.generate_additive_extraction_prompt(existing_memories=existing,
                new_messages=native.parse_messages(r['messages']),
                last_k_messages=[x['message'] for x in history[-10:]],
                timestamp=r['observation_date'],current_date=run.CURRENT_DATE,custom_instructions=None)
            if messages!=[{'role':'system','content':native.ADDITIVE_EXTRACTION_PROMPT},
                          {'role':'user','content':prompt}]:
                raise ValueError('Native extraction prompt differs beyond declared date/history adapter')
            if any(row['event']!='ADD' for row in r['results']):raise ValueError('Unexpected native non-ADD result')
            if r['persistence_check']['missing_texts'] or r['persistence_check']['unexpected_texts']:
                raise ValueError('Native persistence check failed')
            if not call['parsed_memory']:empty.append(envelope['session_id'])
            facts.extend(r['fact_records'])
            warning_rows.extend({'group':envelope['session_id'],**w} for w in r['sdk_warnings'])
            history.extend({'message':m,'session_id':envelope['session_id'],'source_turn_id':tid}
                           for m,tid in zip(envelope['messages'],envelope['source_turn_ids']))
            hashes.append(entry)
        if facts!=[{k:v for k,v in f.items() if k!='embedding'} for f in bank['facts']]:
            raise ValueError('Final native bank differs from exact successful ADD sequence')
        if bank['native_index_audit']['entity_facts_without_any_link']:
            raise ValueError('Unresolved native entity-index failure')
        items.append({'conversation':ci,'groups':len(expected_inputs),'facts':len(facts),
            'bank_sha256':digest,'source_turns':len(history),'explicit_empty_extractions':empty,
            'native_system_prompt_sha256':run.fingerprint(native.ADDITIVE_EXTRACTION_PROMPT),
            'journal_files':hashes,'sdk_warnings':warning_rows,
            'native_index_audit':bank['native_index_audit'],'passed':True})
    return {'schema':'mem0-sparse1540-native-writing-source-audit-v1','passed':True,'created_utc':run.now(),
        'manifest_sha256':run.sha(run.OUT/'manifest.json'),'rows':items,'model_calls':0,
        'verified_scope':'Exact group messages/dates/history/native prompts/ADD journals and exported bank linkage. Not per-fact semantic truth.',
        'qa_or_gold_used_in_audit':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--conversations',default='0,1,2,3,4,5,6,7,8,9');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit([int(x) for x in args.conversations.split(',')]);run.save_once(args.output,r)
    print(json.dumps({'passed':True,'conversations':len(r['rows']),'groups':sum(x['groups'] for x in r['rows']),
        'facts':sum(x['facts'] for x in r['rows']),'warnings':sum(len(x['sdk_warnings']) for x in r['rows'])}))
