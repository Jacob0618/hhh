"""Versioned Gemini 3.8 generation seam; import and body builders are offline.

Reuse durable intents, exact raw caching, transport retries and embedding pacing.
New runs must use independent cache namespaces; old model requests are rejected.
"""
from __future__ import annotations

import json
from jev_mem.paced_embedding_api import PacedEmbeddingAPI
from jev_mem.evaluation_plain_text_v1 import READER_SYSTEM

PROTOCOL = 'gemini38-generate-content-low-v1'
MODEL = 'gemini-3.8-flash'
THINKING_LEVEL = 'low'
WRITER_MAX_OUTPUT = 16384
QA_MAX_OUTPUT = 8192
READER_STAGE = 'qa_reader_gemini38_plain_v1'
JUDGE_STAGE = 'reference_judge_gemini38_plain_v1'


def generation_config(max_output, json_output=False):
    if type(max_output) is not int or not 1 <= max_output <= 65536:
        raise ValueError('Invalid Gemini 3.8 output cap')
    config = {'maxOutputTokens': max_output,
              'thinkingConfig': {'thinkingLevel': THINKING_LEVEL}}
    if json_output:
        config['responseMimeType'] = 'application/json'
    return config


def generation_body(prompt, *, system=None, json_output=False, max_output=QA_MAX_OUTPUT):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('One nonempty user text is required')
    body = {'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': generation_config(max_output, json_output)}
    if system:
        body['systemInstruction'] = {'parts': [{'text': system}]}
    return body


def check_request(provider, model, body, operation, stage):
    if operation != 'generateContent':
        if provider == 'gemini' and model != 'gemini-embedding-001':
            raise ValueError('Only the frozen embedding model may use nongeneration Gemini operations')
        return
    if provider != 'gemini':
        # JEV score_many uses its own provider/body and is not Gemini generation.
        if provider == 'jev':
            return
        raise ValueError('Unsupported generation provider')
    if model != MODEL:
        raise ValueError('New generation namespace only permits Gemini 3.8 Flash')
    writer = stage == 'mem0_extract'
    if stage not in ('mem0_extract', READER_STAGE, JUDGE_STAGE):
        raise ValueError('Undeclared Gemini 3.8 generation stage')
    expected = generation_config(WRITER_MAX_OUTPUT if writer else QA_MAX_OUTPUT, writer)
    if body.get('generationConfig') != expected:
        raise ValueError('Frozen Gemini 3.8 generation configuration differs')
    contents = body.get('contents')
    if (not isinstance(contents, list) or len(contents) != 1
        or set(contents[0]) != {'role', 'parts'} or contents[0]['role'] != 'user'
        or len(contents[0]['parts']) != 1 or set(contents[0]['parts'][0]) != {'text'}
        or not isinstance(contents[0]['parts'][0]['text'], str)
        or not contents[0]['parts'][0]['text'].strip()):
        raise ValueError('One nonempty user text part required')
    fields = {'contents', 'generationConfig'}
    if writer or stage == READER_STAGE:
        fields.add('systemInstruction')
        if not body.get('systemInstruction', {}).get('parts'):
            raise ValueError('Writer/Reader system instruction required')
    if writer:
        from mem0.configs.prompts import ADDITIVE_EXTRACTION_PROMPT
        if body.get('systemInstruction') != {'parts': [{'text': ADDITIVE_EXTRACTION_PROMPT}]}:
            raise ValueError('Pinned native ADDITIVE Writer system instruction differs')
    if stage == READER_STAGE and body.get('systemInstruction') != {'parts': [{'text': READER_SYSTEM}]}:
        raise ValueError('Reader system prompt differs')
    if set(body) != fields:
        raise ValueError('Undeclared generation body fields')


def response_text(response):
    candidates = response.get('candidates', [])
    if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
        raise RuntimeError('Missing, multiple, or incomplete generation candidates')
    if response.get('modelVersion') not in (None, MODEL):
        raise RuntimeError('Resolved Gemini model differs')
    parts = candidates[0].get('content', {}).get('parts', [])
    if any('functionCall' in p or 'functionResponse' in p for p in parts):
        raise RuntimeError('Unexpected tool output is retained and never executed')
    text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
    if not text.strip():
        raise RuntimeError('Empty ordinary generation text')
    return text


class Gemini38API(PacedEmbeddingAPI):
    def __init__(self, run_dir, *, cache_roots=(), **kwargs):
        if cache_roots:
            raise ValueError('Cross-run cache roots forbidden in the Gemini 3.8 retest')
        super().__init__(run_dir, cache_roots=(), **kwargs)

    def request(self, provider, model, body, *, operation='generateContent', stage='unspecified', cache=True):
        check_request(provider, model, body, operation, stage)
        return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)

    def generate(self, prompt, *, model=MODEL, system=None, json_output=False,
                 max_output=QA_MAX_OUTPUT, stage='unspecified'):
        body = generation_body(prompt, system=system, json_output=json_output, max_output=max_output)
        response = self.request('gemini', model, body, stage=stage)
        text = response_text(response)
        return json.loads(text) if json_output else text


class Gemini38ExtractionLLM:
    """Original native extraction input serialization and memory-list parser."""
    def __init__(self, api):
        self.api = api
        self.config = type('Config', (), {'model': MODEL})()

    def generate_response(self, messages, **kwargs):
        system = '\n'.join(m['content'] for m in messages if m['role'] == 'system')
        prompt = '\n'.join(m['content'] for m in messages if m['role'] != 'system')
        data = self.api.generate(prompt, system=system, model=MODEL, json_output=True,
                                 max_output=WRITER_MAX_OUTPUT, stage='mem0_extract')
        if not isinstance(data, dict) or not isinstance(data.get('memory'), list):
            raise RuntimeError('Invalid extraction JSON: memory list required')
        if any(not isinstance(m, dict) or not isinstance(m.get('text'), str) or not m['text'].strip()
               for m in data['memory']):
            raise RuntimeError('Invalid extracted memory record')
        return json.dumps(data, ensure_ascii=False)
