"""Independent Gemini 2.5 Flash adapter for the three-arm fresh-bank ablation.

Frozen 3.8 body validation, extraction serialization and parsing are reused
with private globals. Imports do not construct an API or load credentials.
Transport, durable intents, retries, embedding and JEV gates are inherited.
"""
from __future__ import annotations

import json
from types import FunctionType

from jev_mem import gemini38_provider_v1 as _frozen
from jev_mem.paced_embedding_api import PacedEmbeddingAPI

PROTOCOL = 'gemini25-ablation-generate-content-thinking-off-v1'
MODEL = 'gemini-2.5-flash'
THINKING_BUDGET = 0
WRITER_MAX_OUTPUT = _frozen.WRITER_MAX_OUTPUT
QA_MAX_OUTPUT = _frozen.QA_MAX_OUTPUT
READER_STAGE = 'qa_reader_gemini25_ablation_plain_v1'
JUDGE_STAGE = 'reference_judge_gemini25_ablation_plain_v1'
READER_SYSTEM = _frozen.READER_SYSTEM


def _bind_function(function, scope):
    """Reuse a frozen code object without mutating its module globals."""
    bound = FunctionType(function.__code__, scope, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = dict(function.__kwdefaults__ or {})
    bound.__annotations__ = dict(function.__annotations__)
    bound.__module__ = __name__
    bound.__doc__ = function.__doc__
    return bound


def generation_config(max_output, json_output=False):
    if type(max_output) is not int or not 1 <= max_output <= 65536:
        raise ValueError('Invalid Gemini 2.5 output cap')
    config = {'maxOutputTokens': max_output,
              'thinkingConfig': {'thinkingBudget': THINKING_BUDGET}}
    if json_output:
        config['responseMimeType'] = 'application/json'
    return config


_scope = {**vars(_frozen), 'MODEL': MODEL, 'READER_STAGE': READER_STAGE,
          'JUDGE_STAGE': JUDGE_STAGE, 'generation_config': generation_config}
generation_body = _bind_function(_frozen.generation_body, _scope)
_check_request = _bind_function(_frozen.check_request, _scope)
_response_text = _bind_function(_frozen.response_text, _scope)


def check_request(provider, model, body, operation, stage):
    try:
        return _check_request(provider, model, body, operation, stage)
    except ValueError as error:
        # Only the diagnostic model label changes; validation code is frozen.
        raise ValueError(str(error).replace('Gemini 3.8', 'Gemini 2.5')) from None


def response_text(response):
    """Accept only the requested stable model, including its resource spelling.

Missing modelVersion keeps the frozen parser's behavior. A present value must
be exactly the requested model or its ``models/`` resource name: previews,
Lite/Pro, invented numeric versions and prefix matches are never admitted.
The original response retained in the raw cache is not modified.
"""
    if response.get('modelVersion') == 'models/' + MODEL:
        response = {**response, 'modelVersion': MODEL}
    return _response_text(response)


class Gemini25API(PacedEmbeddingAPI):
    def __init__(self, run_dir, *, cache_roots=(), **kwargs):
        if cache_roots:
            raise ValueError('Cross-run cache roots forbidden in the Gemini 2.5 ablation')
        super().__init__(run_dir, cache_roots=(), **kwargs)

    def request(self, provider, model, body, *, operation='generateContent',
                stage='unspecified', cache=True):
        check_request(provider, model, body, operation, stage)
        return super().request(provider, model, body, operation=operation,
                               stage=stage, cache=cache)

    def generate(self, prompt, *, model=MODEL, system=None, json_output=False,
                 max_output=QA_MAX_OUTPUT, stage='unspecified'):
        body = generation_body(prompt, system=system, json_output=json_output,
                               max_output=max_output)
        response = self.request('gemini', model, body, stage=stage)
        text = response_text(response)
        return json.loads(text) if json_output else text


class Gemini25ExtractionLLM:
    """Unchanged native SDK extraction serializer and memory-list parser."""
    __init__ = _bind_function(_frozen.Gemini38ExtractionLLM.__init__, _scope)
    generate_response = _bind_function(
        _frozen.Gemini38ExtractionLLM.generate_response, _scope)
