"""Proposed, opt-in Reader output compatibility; not installed in any experiment.

This pure function changes no request, model, source cache or old parser.  It
accepts a string unchanged or joins a nonempty list of strings with one newline.
The audit record retains the exact original value.  Citation validation remains
the responsibility of the unchanged evaluator after applying this adapter.
"""
from __future__ import annotations

from copy import deepcopy

PROTOCOL = "reader-answer-string-or-string-list-newline-v1"


def normalize_reader_prediction(prediction):
    """Return a copied prediction and lossless provenance for a fixed type rule.

    No stripping, reordering, deduplication, answer selection or scalar/object
    coercion is allowed. Empty lists are rejected. String elements, including
    empty strings, are preserved exactly; the legacy parser already permits an
    empty string answer. This function never validates answer correctness.
    """
    if not isinstance(prediction, dict):
        raise ValueError("Compatibility layer requires a Reader JSON object")
    answer = prediction.get("answer")
    if isinstance(answer, str):
        normalized, source_type, rule = answer, "string", "identity"
    elif isinstance(answer, list) and answer and all(isinstance(item, str) for item in answer):
        normalized, source_type, rule = "\n".join(answer), "nonempty_list_of_strings", "join_with_single_newline"
    else:
        raise ValueError("Compatibility layer accepts only string or nonempty list[str] answers")
    output = deepcopy(prediction)
    output["answer"] = normalized
    return output, {"protocol": PROTOCOL, "rule": rule, "source_type": source_type,
                    "transformed": source_type != "string", "original_answer": deepcopy(answer),
                    "normalized_answer": normalized, "answer_quality_evaluated": False}
