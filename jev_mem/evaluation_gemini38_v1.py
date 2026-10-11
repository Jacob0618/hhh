"""Gemini 3.8 Reader + paper-era Mem0 semantics, plain-label FORMAT VARIANT.

No dispatch or API constructor on import. Judge gold remains complete; output
format only is adapted. Terminal unanimous labels are accepted on the first
attempt, not rejudged in search of a strict or more favorable answer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jev_mem.evaluation_plain_text_v1 import (
    READER_SYSTEM, reader_input, lexical_metrics, InvalidJudgeLabel as _OldInvalidJudgeLabel,
)
from jev_mem.judge_terminal_label_compat_v1 import derive_label
from jev_mem.gemini38_provider_v1 import MODEL, QA_MAX_OUTPUT, READER_STAGE, JUDGE_STAGE

PROTOCOL = 'gemini38-plain-reader-paper-era-mem0-judge-format-variant-v1'
READER_PROTOCOL = 'brief_answer_plain_text_gemini38_low_v1'
JUDGE_PROTOCOL = 'mem0-paper-era-plain-label-format-variant-strict-or-terminal-v1'
READER_MODEL = JUDGE_MODEL = MODEL
JUDGE_SOURCE_COMMIT = 'cdee6a4ff0c0cdf1757b21c9792fe00299bb80ed'
JUDGE_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / 'research/gemini38_retest_2026-10-03/mem0_paper_era_plain_label_FORMAT_VARIANT_61.txt'
JUDGE_TEMPLATE_SHA256 = '01f99e87d454148cc3408e690366b579b75522a94b9d6a0a22978f510a4eedbf'
JUDGE_TEMPLATE = JUDGE_TEMPLATE_PATH.read_text(encoding='utf-8')
if hashlib.sha256(JUDGE_TEMPLATE.encode('utf-8')).hexdigest() != JUDGE_TEMPLATE_SHA256:
    raise RuntimeError('Frozen paper-era Judge format-variant text differs')


class InvalidJudgeLabel(_OldInvalidJudgeLabel):
    def __init__(self, raw_response):
        self.raw_response = raw_response
        ValueError.__init__(self, 'Judge output has no exact or unanimous exact terminal [[yes]]/[[no]] label')


def make_judge_prompt(question, reference, response_text):
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError('Judge requires a nonempty Reader answer string')
    return JUDGE_TEMPLATE.format(question=question, gold_answer=reference, generated_answer=response_text)


def parse_judge_label(raw):
    if not isinstance(raw, str):
        raise InvalidJudgeLabel(raw)
    interpreted = derive_label(raw)
    if interpreted['status'] not in ('strict_valid', 'compat_valid'):
        raise InvalidJudgeLabel(raw)
    return {'correct': interpreted['correct'], 'label': interpreted['label'], 'raw': raw,
            'parse_mode': 'strict' if interpreted['status'] == 'strict_valid' else 'terminal',
            'parse_protocol': JUDGE_PROTOCOL, 'raw_text_sha256': interpreted['raw_text_sha256']}


def read_answer(api, question, reference, facts, *, reference_date):
    payload = reader_input(question, facts, reference_date=reference_date)
    raw = api.generate(json.dumps(payload, ensure_ascii=False), system=READER_SYSTEM,
                       model=MODEL, json_output=False, max_output=QA_MAX_OUTPUT, stage=READER_STAGE)
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError('Empty generation: Reader did not return a nonempty ordinary answer string')
    return {'prediction': {'answer': raw, 'raw': raw, 'used_fact_ids': None, 'used_local_fact_ids': None},
            **lexical_metrics(raw, reference), 'selected_count': len(facts),
            'evidence_characters': len(json.dumps(payload['facts'], ensure_ascii=False)),
            'reader_protocol': READER_PROTOCOL, 'reader_model': MODEL, 'source_evidence_metrics': None,
            'source_metrics_note': 'Model evidence citations were not collected; support provenance is not scored.'}


def judge_answer(api, question, reference, response_text):
    raw = api.generate(make_judge_prompt(question, reference, response_text), model=MODEL,
                       json_output=False, max_output=QA_MAX_OUTPUT, stage=JUDGE_STAGE)
    return parse_judge_label(raw)


def answer_and_judge(api, question, reference, facts, *, reference_date):
    result = read_answer(api, question, reference, facts, reference_date=reference_date)
    result['judgment'] = judge_answer(api, question, reference, result['prediction']['answer'])
    return {**result, 'evaluation_protocol': PROTOCOL, 'judge_protocol': JUDGE_PROTOCOL, 'judge_model': MODEL}
