"""Record the user's live-run authorization and the fixed initial protocol."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_mem.providers import dump_json


def main():
    path = ROOT / "configs/experiment_plan.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(status="live_authorized_initial_mem0_ingestion_running", execution_enabled=True,
                  live_calls_authorized_in_this_turn=True, launchers_created=True)
    config["authorization"] = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_instruction": "可以先按这个测；这些测完后增加JEV话题重组写入实验",
        "scope": "local Mem0 and independent LoCoMo experiments with specified models; later separate topic regrouping",
        "credential_storage": "existing current-user Windows DPAPI vault; never copied into project artifacts",
        "api_probe_report": "runs/20260927_api_probe/summary.json"}
    config["ingestion"].update(temporal_adapter_implemented=True, history_adapter_implemented=True,
        temporal_protocol="explicit_observation_date_adapter_primary_no_native_date_result_claimed",
        history_order_policy="sidecar_source_order_last_10_successfully_committed_messages")
    config["models"]["note"] = "All four specified APIs passed actual requests. No automatic model substitutions. Gemini uses Developer API, not Vertex service-account credentials."
    config["local_environment"].update(live_provider_auth_verified=True,bm25_inference_verified=True)
    config["evaluation"].update(split_status="conv0_and_conv1_development; conv2_to_conv9_frozen_evaluation; prior_history_not_claimed_blind",
        report_upstream_and_strict_judges_separately=False,
        judge_protocol="strict_research_judge; upstream_lenient_score_not_yet_implemented",
        f1_protocol="normalized_token_overlap_initial; not_claimed_identical_to_official_category_specific_LoCoMo_metric")
    config["current_first_stage_methods"] = ["sdk20", "rank", "mmr", "jev_single", "conditional"]
    config["concurrency"] = {"query_workers":4,"selector_workers":8,"jev_http_attempts_per_second_workspace":15,
        "rate_limit_source":"https://docs.typesafe.ai/models","retry_after_honored":True}
    config["selection"]["experimental"] = "8_seed_plus_6_plus_6_conditional_additions_fixed_20_facts_no_replacement_in_v1"
    config["after_current_experiments"] = {"topic_regrouping":True,"schedule":"separate_followup_arm",
        "input":"raw conversation turns or context-preserving units before extraction, without questions/gold",
        "groups":"JEV judges same-topic membership; deterministic code groups and concatenates",
        "preserve":["original_turn_id","speaker","observation_date","within_group_source_order"],
        "controls":["original_sessions","matched_length_contiguous","embedding_topics","JEV_topics"],
        "causality":"whole_conversation_offline_preprocessing; not an online memory claim"}
    config.pop("model_calls_executed",None)
    config["call_count_source"] = "runs/**/api_calls.jsonl; live counts are not hard-coded into config"
    dump_json(path,config)
    print(json.dumps({"status":config["status"],"execution_enabled":True}))


if __name__ == "__main__":
    main()
