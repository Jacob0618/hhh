"""Frozen 3.8 evaluation semantics with independent Gemini 2.5 bindings.

Reader serialization, prompts, full Judge reference and label parser are the
original code objects. Only model/stage and evaluation identifiers change.
"""
from __future__ import annotations

from jev_mem import evaluation_gemini38_v1 as _frozen
from jev_mem.gemini25_ablation_provider_v1 import (
    MODEL, QA_MAX_OUTPUT, READER_STAGE, JUDGE_STAGE, _bind_function,
)

PROTOCOL = 'gemini25-ablation-plain-reader-paper-era-mem0-judge-format-variant-v1'
READER_PROTOCOL = 'brief_answer_plain_text_gemini25_thinking_off_ablation_v1'
JUDGE_PROTOCOL = _frozen.JUDGE_PROTOCOL
READER_MODEL = JUDGE_MODEL = MODEL
READER_SYSTEM = _frozen.READER_SYSTEM
JUDGE_SOURCE_COMMIT = _frozen.JUDGE_SOURCE_COMMIT
JUDGE_TEMPLATE_PATH = _frozen.JUDGE_TEMPLATE_PATH
JUDGE_TEMPLATE_SHA256 = _frozen.JUDGE_TEMPLATE_SHA256
JUDGE_TEMPLATE = _frozen.JUDGE_TEMPLATE
InvalidJudgeLabel = _frozen.InvalidJudgeLabel
reader_input = _frozen.reader_input
lexical_metrics = _frozen.lexical_metrics
make_judge_prompt = _frozen.make_judge_prompt
parse_judge_label = _frozen.parse_judge_label

_scope = {**vars(_frozen), **{name: value for name, value in globals().items()
                           if not name.startswith('_')}}
for _name in ('read_answer', 'judge_answer', 'answer_and_judge'):
    _function = _bind_function(getattr(_frozen, _name), _scope)
    _function.__module__ = __name__
    globals()[_name] = _scope[_name] = _function
