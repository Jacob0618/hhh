"""Pure source replayer for a future terminal-sealed Judge compatibility view.

No CLI, provider construction, credential read, file mutation or network call.
The production overlay remains gated by a separately reviewed full3080 seal.
"""
from __future__ import annotations

from pathlib import Path

from tools.summarize_mem0_regroup_plain_qa_v1 import PlainReplay, InvalidArtifact, Snapshot
from tools.run_sdk20_modules import digest, sha
from jev_mem.sdk20_cached_api import digest as request_digest
from jev_mem import evaluation_plain_text_v1 as strict
from jev_mem.judge_terminal_label_compat_v1 import derive_label, eligible_raw_response, PROTOCOL


class SourceReplay(PlainReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_evidence = None

    def request(self, provider,model,body,*,operation="generateContent",stage="unspecified",**kwargs):
        response = super().request(provider,model,body,operation=operation,stage=stage,**kwargs)
        if stage != strict.JUDGE_STAGE:
            raise InvalidArtifact("Compatibility is limited to the frozen plain Judge stage")
        signature={"provider":provider,"model":model,"operation":operation,"body":body}
        key=request_digest(signature)
        producer=self._physical(key,stage,signature,success=True)[-1]
        path=self.api_dir/"api_cache"/(key+".json")
        self.response_evidence={"request_hash":key,"stage":stage,
            "raw":{"path":str(path),"sha256":sha(path)},"response":response,
            "physical_producer":producer,"physical_producer_sha256":digest(producer)}
        return response


def replay_source(api_dir, strict_row, context, ledger, *, question, reference, reader_text, snapshot=None):
    """Verify an existing strict success/InvalidJudgeLabel and derive another view.

    The future orchestrator must additionally bind terminal source manifests,
    all3080 slots and exact Reader/selection/question identities before calling.
    Other failure categories are never eligible for this output-format policy.
    """
    if not isinstance(strict_row,dict) or strict_row.get("status") not in ("complete","failed"):
        raise InvalidArtifact("Only terminal plain-Judge records can be replayed")
    if strict_row["status"] == "failed" and (strict_row.get("error") or {}).get("type") != "InvalidJudgeLabel":
        raise InvalidArtifact("Only the explicit strict-label failure can use this compatibility rule")
    snapshot=snapshot or Snapshot()
    api=SourceReplay(api_dir,strict_row.get("api_events"),context,ledger,snapshot=snapshot)
    result=failure=None
    try:
        result=strict.judge_answer(api,question,reference,reader_text)
    except strict.InvalidJudgeLabel as exc:
        failure=exc
    api.assert_consumed()
    if strict_row["status"] == "complete":
        if failure is not None or strict_row.get("error") is not None or result != strict_row.get("result"):
            raise InvalidArtifact("Original strict success differs from exact source replay")
        outcome=derive_label(result["raw"])
        if outcome["status"] != "strict_valid":
            raise InvalidArtifact("Original strict success is not an exact label")
    else:
        if (failure is None or strict_row.get("result") is not None
                or strict_row.get("raw_invalid_label") != failure.raw_response):
            raise InvalidArtifact("Original invalid-label raw cannot be reproduced")
        outcome=derive_label(failure.raw_response)
        if outcome["status"] == "strict_valid":
            raise InvalidArtifact("An exact label was incorrectly recorded as a strict failure")
        evidence=api.response_evidence
        eligible,reason=eligible_raw_response(evidence["response"],failure.raw_response,
                                             http_status=evidence["physical_producer"]["status"])
        if not eligible:
            outcome={**outcome,"status":"rejected","label":None,"correct":None,"rejection_reason":reason}
    if snapshot.unchanged():
        raise InvalidArtifact("Original raw/intent changed during offline compatibility replay")
    evidence=api.response_evidence
    return {"protocol":PROTOCOL,"original_status":strict_row["status"],
            "strict_stage_content_sha256":digest({k:v for k,v in strict_row.items() if k!="content_sha256"}),
            "interpretation":outcome,"source_evidence":{k:v for k,v in evidence.items() if k!="response"},
            "reader_recomputed":False,"model_calls":0,"original_raw_or_result_modified":False}
