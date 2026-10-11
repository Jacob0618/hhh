"""Independent plain-text Reader and user-specified binary Judge protocol.

No requests run on import. Persist ``read_answer`` before calling ``judge_answer``
when separate Reader/F1 and Judge failure denominators are required. JSON is only
the unchanged input serialization, never a required model response format.
"""
from __future__ import annotations

import json

from jev_mem.evaluation import READER_SYSTEM as LEGACY_READER_SYSTEM, lexical_metrics


PROTOCOL = "plain-text-answer-and-reference-judge-v1"
READER_PROTOCOL = "brief_answer_plain_text_v1"
JUDGE_PROTOCOL = "user_reference_contains_binary_v1"
READER_STAGE = "qa_reader_plain_text_v1"
JUDGE_STAGE = "reference_judge_plain_text_v1"
READER_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-3.5-flash-lite"

# Preserve the legacy answer instructions verbatim; replace only output format.
_FORMAT_MARKER = "\nReturn JSON with answer"
if LEGACY_READER_SYSTEM.count(_FORMAT_MARKER) != 1:
    raise RuntimeError("Frozen legacy Reader format boundary differs")
READER_SEMANTICS = LEGACY_READER_SYSTEM.split(_FORMAT_MARKER, 1)[0]
READER_SYSTEM = READER_SEMANTICS + "\nReturn only the plain-text answer."

# User-provided text. Its intermediate-steps clause is an acceptance condition
# for Judge, not an instruction for Reader to produce reasoning.
JUDGE_INSTRUCTIONS = (
    "I will give you a question, a reference answer, and a response from a model. "
    "Please answer [[yes]] if the response contains the reference answer. "
    "Otherwise, answer [[no]]. If the response is equivalent to the correct "
    "answer or contains all the intermediate steps to get the reference answer, "
    "you should also answer [[yes]]. If the response only contains a subset of "
    "the information required by the answer, answer [[no]]."
)
JUDGE_END = "Is the model response correct? Answer [[yes]] or [[no]] only."
JUDGE_TEMPLATE = (
    JUDGE_INSTRUCTIONS + "\n\n[User Question]\n{question}\n\n"
    "[The Start of Reference Answer]\n{answer}\n[The End of Reference Answer]\n\n"
    "[The Start of Model’s Response]\n{response}\n[The End of Model’s Response]\n\n"
    + JUDGE_END
)


class InvalidJudgeLabel(ValueError):
    """Technical failure with retained output; never equivalent to ``False``."""

    def __init__(self, raw_response):
        self.raw_response = raw_response
        super().__init__("Judge output must be exactly [[yes]] or [[no]] after stripping whitespace")


def make_judge_prompt(question, reference, response_text):
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Judge requires a nonempty Reader answer string")
    return JUDGE_TEMPLATE.format(question=question, answer=reference, response=response_text)


def reader_input(question, facts, *, reference_date):
    """The legacy input representation, with the same 20-unique-fact budget."""
    if len(facts) > 20:
        raise ValueError("The reader budget is at most 20 facts")
    if any(not isinstance(f.get("id"), str) or not isinstance(f.get("text"), str) for f in facts):
        raise ValueError("Reader facts require string IDs and text")
    if len({f["id"] for f in facts}) != len(facts):
        raise ValueError("Reader facts must have unique IDs")
    evidence = [{"id": f"f{index+1}", "observation_date": f.get("observation_date"), "fact": f["text"]}
                for index, f in enumerate(facts)]
    return {"question": question, "conversation_reference_date": reference_date, "facts": evidence}


def read_answer(api, question, reference, facts, *, reference_date):
    """Return a persistable Reader result and custom lexical metrics independently.

    ``reference`` is used only locally to score the answer; it is never sent to
    Reader. Output text is preserved exactly, including whitespace and any text
    resembling JSON. Model-generated evidence citations were not collected.
    """
    payload = reader_input(question, facts, reference_date=reference_date)
    raw = api.generate(json.dumps(payload, ensure_ascii=False), system=READER_SYSTEM,
                       model=READER_MODEL, json_output=False, max_output=2048, stage=READER_STAGE)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Reader did not return a nonempty answer string")
    return {
        "prediction": {"answer": raw, "raw": raw,
                       "used_fact_ids": None, "used_local_fact_ids": None},
        **lexical_metrics(raw, reference),
        "selected_count": len(facts),
        "evidence_characters": len(json.dumps(payload["facts"], ensure_ascii=False)),
        "reader_protocol": READER_PROTOCOL, "reader_model": READER_MODEL,
        "source_evidence_metrics": None,
        "source_metrics_note": "Model evidence citations were not collected; support provenance is not scored.",
    }


def judge_answer(api, question, reference, response_text):
    """Return the user's binary label only, or raise a technical failure."""
    raw = api.generate(make_judge_prompt(question, reference, response_text),
                       model=JUDGE_MODEL, json_output=False, max_output=2048, stage=JUDGE_STAGE)
    label = raw.strip() if isinstance(raw, str) else None
    if label not in ("[[yes]]", "[[no]]"):
        raise InvalidJudgeLabel(raw)
    return {"correct": label == "[[yes]]", "label": label, "raw": raw}


def answer_and_judge(api, question, reference, facts, *, reference_date):
    """Convenience only; stage-aware drivers should persist ``read_answer`` first."""
    result = read_answer(api, question, reference, facts, reference_date=reference_date)
    result["judgment"] = judge_answer(api, question, reference, result["prediction"]["answer"])
    return {**result, "evaluation_protocol": PROTOCOL,
            "judge_protocol": JUDGE_PROTOCOL, "judge_model": JUDGE_MODEL}
