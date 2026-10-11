"""Small synthetic midpoint experiment; NO JEV, credentials, or network access.

An ideal oracle answers whether two indexed turns have the same topic. A midpoint
is compared with its parent's midpoint, not with the entire parent's interval.
Only the oracle/evaluator sees the synthetic ground-truth topic labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "midpoint-parent-child-oracle-probe-v1"
CASES = {
    "one_topic": ["A"] * 21,
    "one_switch": ["A"] * 10 + ["B"] * 11,
    "short_B_then_return_A": ["A"] * 7 + ["B"] + ["A"] * 13,
    "multiple_switches": ["A"] * 5 + ["B"] * 4 + ["C"] * 4 + ["A"] * 4 + ["D"] * 4,
    "alternating_topics": ["A" if i % 2 == 0 else "B" for i in range(21)],
}


def _size(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("num_turns must be a positive integer")


def _answer(oracle, left, right):
    answer = oracle(left, right)
    if not isinstance(answer, bool):
        raise TypeError("The synthetic same-topic oracle must return bool")
    return answer


def midpoint_probe(num_turns, oracle, *, prune_same=False):
    """Balanced index tree, pivot-excluding children, strict finite recursion.

    For [lo, hi), mid=(lo+hi)//2. Compare each nonempty child's midpoint with
    mid. Full recursion explores every child. Pruned recursion stops a child
    when that comparison is SAME and *assumes* its whole interval matches mid.
    These interval assumptions are separate from actual oracle observations.
    """
    _size(num_turns)
    if not isinstance(prune_same, bool):
        raise TypeError("prune_same must be bool")
    calls, assumptions, stops, visited = [], [], [], set()
    max_depth = 0

    def visit(lo, hi, parent=None, parent_span=None, depth=0):
        nonlocal max_depth
        if lo >= hi:
            return
        mid = (lo + hi) // 2
        visited.add(mid)
        max_depth = max(max_depth, depth)
        if parent is not None:
            same = _answer(oracle, parent, mid)
            calls.append({"call": len(calls) + 1, "left_index": parent,
                          "right_index": mid, "same_topic": same,
                          "parent_span": list(parent_span), "child_span": [lo, hi],
                          "depth": depth, "pruned": bool(prune_same and same)})
            if prune_same and same:
                # This is the disputed extrapolation, never disguised as evidence.
                assumptions.append({"reference_index": parent, "span": [lo, hi],
                                    "sampled_midpoint": mid,
                                    "claim": "all turns in this span share the reference topic"})
                stops.append({"span": [lo, hi], "reason": "same_midpoints_early_prune"})
                return
        if hi - lo == 1:
            stops.append({"span": [lo, hi], "reason": "singleton_leaf"})
            return
        visit(lo, mid, mid, (lo, hi), depth + 1)
        visit(mid + 1, hi, mid, (lo, hi), depth + 1)

    visit(0, num_turns)
    return {"strategy": "midpoint_prune_same" if prune_same else "midpoint_full_recursion",
            "num_turns": num_turns, "oracle_calls": len(calls),
            "visited_midpoints": sorted(visited), "unvisited_turn_count": num_turns - len(visited),
            "max_node_depth": max_depth, "calls": calls, "interval_assumptions": assumptions,
            "stops": stops, "terminated": True,
            "termination_argument": "Each nonempty child excludes its parent pivot and is strictly shorter; leaf or same-topic prune stops."}


def adjacent_probe(num_turns, oracle):
    _size(num_turns)
    calls = [{"call": i + 1, "left_index": i, "right_index": i + 1,
              "same_topic": _answer(oracle, i, i + 1)} for i in range(num_turns - 1)]
    return {"strategy": "all_adjacent_gaps", "num_turns": num_turns,
            "oracle_calls": len(calls), "visited_midpoints": None,
            "unvisited_turn_count": 0, "max_node_depth": None,
            "calls": calls, "interval_assumptions": [], "stops": [], "terminated": True}


def infer_gaps(probe, *, use_pruning_assumptions=False):
    """Three-valued inference. DIFFERENT is not transitive.

    Equality components follow observed SAME edges. Observed DIFFERENT edges
    separate components. Any adjacent pair not established by those constraints
    is UNKNOWN, not a silently inferred absence of a boundary.
    """
    n = probe["num_turns"]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    for call in probe["calls"]:
        if call["same_topic"]:
            union(call["left_index"], call["right_index"])
    if use_pruning_assumptions:
        for assumption in probe["interval_assumptions"]:
            lo, hi = assumption["span"]
            for index in range(lo, hi):
                union(assumption["reference_index"], index)
    different = set()
    contradictions = []
    for call in probe["calls"]:
        if call["same_topic"]:
            continue
        pair = frozenset((find(call["left_index"]), find(call["right_index"])))
        if len(pair) == 1:
            contradictions.append(call["call"])
        else:
            different.add(pair)
    gaps = []
    for index in range(n - 1):
        left, right = find(index), find(index + 1)
        label = "same" if left == right else "different" if frozenset((left, right)) in different else "unknown"
        gaps.append({"after_turn": index + 1, "left_id": f"t{index + 1:02d}",
                     "right_id": f"t{index + 2:02d}", "inference": label})
    return {"gaps": gaps, "contradictory_observed_call_ids": contradictions,
            "uses_unverified_interval_assumptions": use_pruning_assumptions}


def evaluate_inference(topics, inference):
    true_cuts = {i + 1 for i in range(len(topics) - 1) if topics[i] != topics[i + 1]}
    predicted = {gap["after_turn"] for gap in inference["gaps"] if gap["inference"] == "different"}
    unknown = {gap["after_turn"] for gap in inference["gaps"] if gap["inference"] == "unknown"}
    false_same = {gap["after_turn"] for gap in inference["gaps"]
                  if gap["inference"] == "same" and gap["after_turn"] in true_cuts}
    segments = []
    lo = 0
    for hi in sorted(predicted) + [len(topics)]:
        segments.append({"start_index": lo, "end_index_exclusive": hi,
                         "turn_ids": [f"t{i + 1:02d}" for i in range(lo, hi)]})
        lo = hi
    return {"true_boundary_after_turns": sorted(true_cuts),
            "certified_or_assumed_boundary_after_turns": sorted(predicted),
            "missed_boundary_after_turns": sorted(true_cuts - predicted),
            "spurious_boundary_after_turns": sorted(predicted - true_cuts),
            "unknown_gap_after_turns": sorted(unknown),
            "unknown_true_boundary_after_turns": sorted(unknown & true_cuts),
            "falsely_inferred_same_gap_after_turns": sorted(false_same),
            "true_positive_boundaries": len(predicted & true_cuts),
            "boundary_recall": len(predicted & true_cuts) / len(true_cuts) if true_cuts else None,
            "exact_boundary_set_matches_truth": predicted == true_cuts,
            "all_gaps_identified_correctly": not unknown and predicted == true_cuts and not false_same
                and not inference["contradictory_observed_call_ids"],
            "segments_if_only_different_gaps_are_cut": segments,
            "cut_policy_warning": "UNKNOWN gaps remain undecided; this illustrative segmentation is not a certificate of no boundary."}


def run_case(name, topics):
    topics = list(topics)
    if not topics or any(not isinstance(topic, str) or not topic for topic in topics):
        raise ValueError("Synthetic topics must be nonempty strings")
    oracle = lambda left, right: topics[left] == topics[right]
    strategies = [adjacent_probe(len(topics), oracle),
                  midpoint_probe(len(topics), oracle),
                  midpoint_probe(len(topics), oracle, prune_same=True)]
    for probe in strategies:
        observed = infer_gaps(probe)
        assumed = infer_gaps(probe, use_pruning_assumptions=True)
        probe["observed_evidence_only"] = {**observed, "evaluation": evaluate_inference(topics, observed)}
        probe["with_pruning_assumptions"] = {**assumed, "evaluation": evaluate_inference(topics, assumed)}
        probe["incorrect_interval_assumptions"] = [
            {**item, "counterexample_turn_ids": [f"t{i + 1:02d}" for i in range(*item["span"])
                                                 if topics[i] != topics[item["reference_index"]]]}
            for item in probe["interval_assumptions"]
            if any(topics[i] != topics[item["reference_index"]] for i in range(*item["span"]))]
    return {"case": name, "num_turns": len(topics),
            "source_turn_ids": [f"t{i + 1:02d}" for i in range(len(topics))],
            "ground_truth_topic_sequence": topics, "strategies": strategies}


def build_report():
    return {"schema": SCHEMA, "experiment_kind": "synthetic_perfect_oracle_not_real_JEV",
            "model_api_calls": 0, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "oracle_contract": "One logical call compares two turn topic labels; no known topic vocabulary/number is supplied to the algorithm.",
            "index_convention": "Internal indices zero-based, intervals [lo,hi), boundary after_turn is one-based.",
            "cost_convention": "Report total pair-oracle calls, not just recursion depth. No HTTP count, tokenizer, latency or dollar claim is made; batching and contextual JEV judgments could change deployment costs.",
            "scope_limitations": ["Only five constructed 21-turn cases, not representative corpus frequencies.",
                                  "Ideal labels exclude semantic ambiguity, coreference and JEV errors.",
                                  "A SAME midpoint pair is not an interval homogeneity oracle.",
                                  "Termination and correct boundary identification are distinct."],
            "cases": [run_case(name, topics) for name, topics in CASES.items()]}


def concise_rows(report):
    rows = []
    for case in report["cases"]:
        for probe in case["strategies"]:
            evaluation = probe["with_pruning_assumptions"]["evaluation"]
            rows.append({"case": case["case"], "strategy": probe["strategy"],
                         "calls": probe["oracle_calls"], "depth": probe["max_node_depth"],
                         "true_boundaries": evaluation["true_boundary_after_turns"],
                         "missed": evaluation["missed_boundary_after_turns"],
                         "spurious": evaluation["spurious_boundary_after_turns"],
                         "unknown": evaluation["unknown_gap_after_turns"],
                         "all_gaps_correct": evaluation["all_gaps_identified_correctly"]})
    return rows


def markdown(report):
    lines = ["# Parent–child midpoint segmentation：21-turn理想oracle验证", "",
             "**这是合成序列与完美同话题oracle的算法检查，不是JEV实测。** 总调用指两turn的一次逻辑比较，不代表真实HTTP次数、延迟或费用。", "",
             "每区间取mid，左右子区间排除该pivot。子节点mid与parent mid比较。full无条件递归到叶子；pruned遇SAME即停止该子树，并显式记录其整区间同话题这一额外假设。五个案例的完整21-turn序列、逐调用轨迹和剪枝反例均保留在JSON。", "",
             "| 序列 | 方法 | 总oracle调用 | 最大深度 | 漏边界 | 假边界 | 未确定间隙 | 全间隙正确确定 |",
             "|---|---|---:|---:|---|---|---|---|"]
    for row in concise_rows(report):
        lines.append("| {case} | {strategy} | {calls} | {depth} | {missed} | {spurious} | {unknown} | {all_gaps_correct} |".format(**row))
    lines += ["", "表中pruned行使用其整区间同类假设；JSON另给出仅观察到的oracle关系能支持的三态推断。UNKNOWN不会被当成已证明无边界；不相同关系也不能传递。", "",
              "## 可得结论", "",
              "- 21节点完整中点树深度只有4，但比较数是20=N−1，与全相邻gap扫描一样；树深度小不等于总调用小。", 
              "- 同话题提前剪枝确实可以减少这些合成例的逻辑比较，但隐藏的短B可在两个探针均为A时被整个跳过，错误在结果中保留。",
              "- full和pruned都因子区间严格缩小而终止。full即使访问全部turn，非相邻pair约束也未必唯一确定所有相邻边界；相同/不同两类关系不支持任意二分外推。",
              "- 当前完美oracle只说明可能的调用量与反例，不验证真实JEV的效果、速度或成本。真正区间同质性判断是另一种更强、输入更长的判据，不能假装由midpoint pair免费得到。", "",
              "## 后续真实小样本应保持的边界", "",
              "固定相同原始session和后续writer，对照全gap、full midpoint和pruned midpoint；记录所有剪枝区间、未评估gap、漏检与假边界。真实JEV输入/上下文及同话题门槛须另行冻结；总HTTP、token、延迟、重试与缓存分列。此次不启动模型、QA或自动化。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path,
                        default=ROOT / "research/midpoint_segmentation_probe_2026-09-28")
    parser.add_argument("--no-write", action="store_true", help="Print deterministic summary without writing reports")
    args = parser.parse_args(argv)
    report = build_report()
    if not args.no_write:
        paths = [Path(str(args.output_prefix) + extension) for extension in (".json", ".md")]
        if any(path.exists() for path in paths):
            raise FileExistsError("Refusing to overwrite an earlier probe")
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[1].write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"experiment_kind": report["experiment_kind"], "model_api_calls": 0,
                      "results": concise_rows(report)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
