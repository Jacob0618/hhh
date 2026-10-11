"""C1/C2 narrow boundary variants; all scoring is injected, never started on import.

C1 executes the frozen sparse segmenter's function body in independent globals,
changing only its proposer. C2 spends the same two logical reserve slots on
unassessed immediate neighbours of initially suspicious gaps. Neither arm
certifies unassessed gaps or uses QA/gold.
"""
from __future__ import annotations

import copy
from types import FunctionType

from jev_mem import sparse_topic_pipeline as old
from jev_mem.topic_boundaries import _session_ranges, boundary_requests
from jev_mem.topic_boundary_budget import propose_sparse_boundaries
from jev_mem.topic_grouping import _validate_units, verify_partition

VERSION = "boundary-variants-v1"
ARMS = ("c1_multiscale", "c2_neighbour")
WINDOWS = (1, 2, 4)
PRIORITY = "max_cosine_change_w1_w2_w4_then_ascending_gap"
C2_POLICY = "initial_shift_ge_keep_parent_shift_desc_neighbour_priority_desc_gap_asc_two_w2_unassessed_neighbours_no_expansion"


def propose_multiscale_boundaries(raw_turns, embeddings, *, initial_budget=4,
                                  refinement_budget=2, window_turns=2):
    if (initial_budget, refinement_budget, window_turns) != (4, 2, 2):
        raise ValueError("Frozen C1 budget/window differs")
    proposals = {w: propose_sparse_boundaries(raw_turns, embeddings,
                 initial_budget=4, refinement_budget=2, window_turns=w) for w in WINDOWS}
    result = copy.deepcopy(proposals[2])
    requests = {r["right_source_index"]: r for r in boundary_requests(raw_turns, window_turns=2)}
    for si, session in enumerate(result["sessions"]):
        scales = {str(w): proposals[w]["sessions"][si]["embedding_change_scores"] for w in WINDOWS}
        priority = {g: max(scales[str(w)][g] for w in WINDOWS) for g in session["embedding_change_scores"]}
        # Anchor positions/rounding and initial source-order evaluation are inherited.
        anchors = sorted(int(g) for g, reason in session["selection_reasons"].items()
                         if reason == "fractional_coverage_anchor")
        selected = list(anchors)
        for gap in sorted(map(int, priority), key=lambda g: (-priority[str(g)], g)):
            if len(selected) >= 4:
                break
            if gap not in selected:
                selected.append(gap)
        session.update(initial_requests=[requests[g] for g in sorted(selected)],
            selection_reasons={str(g): "fractional_coverage_anchor" if g in anchors else PRIORITY for g in selected},
            multiscale_change_scores=scales, multiscale_priority_scores=priority,
            unassessed_gap_indices=[int(g) for g in priority if int(g) not in selected])
    result.update(version=VERSION, fixed_windows=list(WINDOWS), peak_priority=PRIORITY,
                  sparse_scorer_integration_pending=False)
    return result


_c1_globals = dict(vars(old), propose_sparse_boundaries=propose_multiscale_boundaries)
_c1 = FunctionType(old.segment_sparse_sessions.__code__, _c1_globals,
                   "segment_c1", old.segment_sparse_sessions.__defaults__)
_c1.__kwdefaults__ = old.segment_sparse_sessions.__kwdefaults__


def segment_boundaries(raw_turns, *, arm, judge, embeddings, cut_threshold,
                       keep_threshold, dependency_threshold):
    if arm not in ARMS:
        raise ValueError("Undeclared boundary arm")
    if arm == ARMS[0]:
        result = _c1(raw_turns, judge=judge, embeddings=embeddings,
                     cut_threshold=cut_threshold, keep_threshold=keep_threshold,
                     dependency_threshold=dependency_threshold)
        result["trace"].update(boundary_variant=arm, boundary_variant_version=VERSION)
        return result
    for key, value in (("cut_threshold", cut_threshold), ("keep_threshold", keep_threshold),
                       ("dependency_threshold", dependency_threshold)):
        old._threshold(key, value)
    if keep_threshold >= cut_threshold:
        raise ValueError("keep_threshold must be below cut_threshold")
    units = _validate_units(raw_turns)
    vectors = old._vectors(embeddings, len(units))
    proposal = propose_multiscale_boundaries(units, vectors)
    request_map = {r["right_source_index"]: r for r in boundary_requests(units, window_turns=2)}
    trace = old._trace("sparse_boundaries", units, vectors)
    trace.update(boundary_variant=arm, boundary_variant_version=VERSION,
        thresholds={"cut": cut_threshold, "keep": keep_threshold, "dependency": dependency_threshold},
        initial_budget_per_session=4, shared_extra_budget_per_session=2,
        initial_window_per_side=2, expanded_window_per_side=4,
        refinement_policy=C2_POLICY, proposer=proposal, sessions=[], decisions=[], unassessed_gap_indices=[])
    cuts = set()
    for session in proposal["sessions"]:
        assessed = {}
        def ask(request, stage, parent=None):
            g = request["right_source_index"]
            scores = old._ask(judge, request["state"], request["criteria"], stage, trace,
                session_id=session["session_id"], right_source_index=g, pass_number=1,
                localization_parent_gap=parent)
            assessed[g] = {"right_source_index": g, "session_id": session["session_id"],
                "boundary_after": request["state"]["boundary_after"], "boundary_before": request["state"]["boundary_before"],
                "evaluations": [scores], "request_hashes": [trace["requests"][-1]["request_sha256"]],
                "refinement": "initial" if parent is None else "neighbour_w2", "localization_parent_gap": parent}
        for request in session["initial_requests"]:
            ask(request, "sparse_topic_boundary_initial")
        initial_count = len(assessed)
        all_gaps = set(map(int, session["multiscale_priority_scores"]))
        nominations = {}
        for parent, decision in assessed.items():
            shift = decision["evaluations"][0]["topic_shift"]
            if shift < keep_threshold:
                continue
            for g in (parent - 1, parent + 1):
                if g in all_gaps and g not in assessed:
                    key = (-shift, -session["multiscale_priority_scores"][str(g)], g, parent)
                    if g not in nominations or key < nominations[g][0]:
                        nominations[g] = (key, parent)
        ordered = sorted(nominations, key=lambda g: nominations[g][0])
        trace.setdefault("localization_candidates", []).append({"session_id": session["session_id"],
             "ordered_gaps": ordered, "parent_gaps": {str(g): nominations[g][1] for g in ordered}})
        for g in ordered[:2]:
            ask(request_map[g], "boundary_variants_neighbour", nominations[g][1])
        for g in sorted(all_gaps):
            if g not in assessed:
                trace["unassessed_gap_indices"].append(g)
                trace["decisions"].append({"right_source_index": g, "session_id": session["session_id"],
                    "status": "unassessed", "cut": False, "scores": None})
                continue
            decision = assessed[g]
            scores = decision["evaluations"][-1]
            shift, dependency = scores["topic_shift"], scores["right_depends_on_left"]
            cut = shift >= cut_threshold and dependency < dependency_threshold
            veto = shift >= cut_threshold and dependency >= dependency_threshold
            decision.update(status="assessed", scores=scores, cut=cut, dependency_veto=veto,
                unresolved_shift=keep_threshold <= shift < cut_threshold,
                decision="cut" if cut else "dependency_veto" if veto else "gray_joined" if shift >= keep_threshold else "low_shift_joined")
            trace["decisions"].append(decision)
            if cut:
                cuts.add(g)
        extra = len(assessed) - initial_count
        trace["sessions"].append({"session_id": session["session_id"], "initial_calls": initial_count,
            "refinement_calls": extra, "unused_reserve": 2-extra, "total_logical_calls": len(assessed)})
        assert len(assessed) <= 6
    segments = []
    for start, end in _session_ranges(units):
        points = [start] + sorted(g for g in cuts if start < g < end) + [end]
        for left, right in zip(points, points[1:]):
            record = old._group_record(units, list(range(left, right)), f"segment_{len(segments):04d}")
            record.update(segment_id=record["group_id"], singleton_requires_context_review=right-left == 1)
            segments.append(record)
    verify_partition(units, segments)
    trace.update(semantic_cut_indices=sorted(cuts), segment_count=len(segments), exactly_once_verified=True,
                 capacity_cuts_performed=False, gap_localization_performed=True)
    return {"segments": segments, "groups": copy.deepcopy(segments), "trace": trace}
