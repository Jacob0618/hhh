"""One fixed reader and strict judge shared across all experimental arms."""
import collections
import re
import string
import math


READER_SYSTEM = """Answer a question about the named people using the supplied memories.
Memories may describe either human speaker. Preserve attribution, negation, plans,
and changes over time. Observation dates date the conversation, not necessarily the
event. Combine complementary facts. For lists/counts include all supported relevant
items and avoid duplicates. Reasonable inferences must be grounded in the memories;
do not invent biographical events, dates, names, or links. If evidence is insufficient,
say so. Return only the shortest complete answer: a name, date, phrase, or list as
appropriate. Do not add explanation, background, introductory sentences, or related
but unasked facts. Do not shorten a required list by omitting its items.
Return JSON with answer (a string) and used_fact_ids (an array containing only the
exact short evidence IDs supplied, such as f1 or f2; an empty array is permitted)."""

JUDGE_SYSTEM = """Evaluate the predicted answer against the question and reference answer.
Return JSON: correct (boolean), completeness (number 0..1), reason (short string).
Be strict about required entities, relationships, time, negation and event status.
Accept equivalent meanings, valid aliases and equivalent date formats. For a question
requesting multiple items or a count, one correct item is not sufficient: all required
items/count must be correct. Extra incompatible or unsupported asserted details count
against correctness. Do not accept a broad partial answer as fully correct. Reference
variants separated by semicolons may be alternative answers, not necessarily a list;
use the question to determine this. Do not use the system name or ranking method."""


def normalize(text):
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def lexical_metrics(prediction, reference):
    a, b = normalize(prediction), normalize(reference)
    left, right = a.split(), b.split()
    common = sum((collections.Counter(left) & collections.Counter(right)).values())
    if not left or not right:
        f1 = float(left == right)
    elif not common:
        f1 = 0.0
    else:
        p, r = common / len(left), common / len(right)
        f1 = 2*p*r/(p+r)
    return {"token_f1": f1, "exact_match": float(a == b)}


def answer_and_judge(api, question, reference, facts, *, reference_date):
    import json
    if len(facts) > 20:
        raise ValueError("The reader budget is at most 20 facts")
    if any(not isinstance(f.get("id"), str) or not isinstance(f.get("text"), str) for f in facts):
        raise ValueError("Reader facts require string IDs and text")
    if len({f["id"] for f in facts}) != len(facts):
        raise ValueError("Reader facts must have unique IDs")
    local_to_actual = {f"f{index+1}": fact["id"] for index, fact in enumerate(facts)}
    evidence = [{"id": f"f{index+1}", "observation_date": f.get("observation_date"), "fact": f["text"]}
                for index, f in enumerate(facts)]
    reader_input = {"question": question, "conversation_reference_date": reference_date, "facts": evidence}
    prediction = api.generate(json.dumps(reader_input, ensure_ascii=False), system=READER_SYSTEM,
                              json_output=True, max_output=2048, stage="qa_reader")
    if not isinstance(prediction, dict) or not isinstance(prediction.get("answer"), str):
        raise ValueError("Reader did not return an answer string")
    allowed_ids = set(local_to_actual)
    ids = prediction.get("used_fact_ids", [])
    if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids) or not set(ids) <= allowed_ids:
        raise ValueError("Reader cited IDs outside its selected facts")
    prediction["used_local_fact_ids"] = list(ids)
    prediction["used_fact_ids"] = [local_to_actual[x] for x in ids]
    judgment = api.generate(json.dumps({"question": question, "reference_answer": reference,
                                        "predicted_answer": prediction["answer"]}, ensure_ascii=False),
                             system=JUDGE_SYSTEM, model="gemini-3.5-flash-lite", json_output=True,
                             max_output=2048, stage="strict_judge")
    if not isinstance(judgment, dict) or not isinstance(judgment.get("correct"), bool):
        raise ValueError("Judge did not return boolean correctness")
    completeness = judgment.get("completeness")
    if (isinstance(completeness, bool) or not isinstance(completeness, (int, float))
            or not math.isfinite(completeness) or not 0 <= completeness <= 1
            or not isinstance(judgment.get("reason"), str)):
        raise ValueError("Judge returned invalid completeness or reason")
    return {"prediction": prediction, "judgment": judgment,
            **lexical_metrics(prediction["answer"], reference),
            "selected_count": len(facts), "evidence_characters": len(json.dumps(evidence, ensure_ascii=False)),
            "reader_protocol": "brief_answer_local_evidence_ids_v2",
            "reader_model": "gemini-2.5-flash", "judge_model": "gemini-3.5-flash-lite",
            "source_evidence_metrics": None,
            "source_metrics_note": "Not reported before per-fact support provenance validation."}
