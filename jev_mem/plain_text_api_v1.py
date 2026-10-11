"""Opt-in request guard for the new plain-text evaluation protocol.

Generation, response-text extraction, raw caching, durable unknown-request guards,
transport retries and embedding pacing are inherited unchanged. Stage names aid
audit attribution; cache identity is the full request body, not the stage name.
"""
from __future__ import annotations

from jev_mem.evaluation_plain_text_v1 import (
    JUDGE_END, JUDGE_INSTRUCTIONS, JUDGE_MODEL, JUDGE_STAGE,
    READER_MODEL, READER_STAGE, READER_SYSTEM,
)
from jev_mem.paced_embedding_api import PacedEmbeddingAPI


PROTOCOL = "guarded-plain-text-evaluation-transport-v1"


def check_plain_text_request(provider, model, body, *, operation="generateContent", stage="unspecified"):
    """Reject protocol drift before any cache lookup, intent or model request."""
    if stage in ("qa_reader", "strict_judge"):
        raise ValueError("Legacy JSON evaluation stages are not allowed by this plain-text provider")
    if stage not in (READER_STAGE, JUDGE_STAGE):
        return
    reader = stage == READER_STAGE
    config = {"temperature": 0, "maxOutputTokens": 2048}
    if reader:
        config["thinkingConfig"] = {"thinkingBudget": 0}
    if (provider != "gemini" or operation != "generateContent"
            or model != (READER_MODEL if reader else JUDGE_MODEL)
            or not isinstance(body, dict) or body.get("generationConfig") != config):
        raise ValueError("Plain-text evaluation provider, model or generation configuration differs")
    fields = {"contents", "generationConfig"}
    if reader:
        fields.add("systemInstruction")
        if body.get("systemInstruction") != {"parts": [{"text": READER_SYSTEM}]}:
            raise ValueError("Plain-text Reader system instruction differs")
    if set(body) != fields:
        raise ValueError("Plain-text evaluation request contains undeclared fields or a response schema")
    contents = body.get("contents")
    if (not isinstance(contents, list) or len(contents) != 1
            or not isinstance(contents[0], dict) or set(contents[0]) != {"role", "parts"}
            or contents[0]["role"] != "user"
            or not isinstance(contents[0]["parts"], list) or len(contents[0]["parts"]) != 1
            or not isinstance(contents[0]["parts"][0], dict)
            or set(contents[0]["parts"][0]) != {"text"}
            or not isinstance(contents[0]["parts"][0]["text"], str)):
        raise ValueError("Plain-text evaluation requires one user text part")
    text = contents[0]["parts"][0]["text"]
    if not reader:
        markers = ("\n\n[The Start of Reference Answer]\n", "\n[The End of Reference Answer]\n\n",
                   "[The Start of Model’s Response]\n", "\n[The End of Model’s Response]\n\n")
        if (not text.startswith(JUDGE_INSTRUCTIONS + "\n\n[User Question]\n")
                or not text.endswith(JUDGE_END) or not all(marker in text for marker in markers)):
            raise ValueError("Plain-text Judge instructions or input markers differ")


class PlainTextEvaluationAPI(PacedEmbeddingAPI):
    """Use inherited ``generate`` and type-preserving ``scoped`` unchanged."""

    def request(self, provider, model, body, *, operation="generateContent", stage="unspecified", cache=True):
        check_plain_text_request(provider, model, body, operation=operation, stage=stage)
        return super().request(provider, model, body, operation=operation, stage=stage, cache=cache)
