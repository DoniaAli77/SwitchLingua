"""
test_task_aware_accept.py — tests the OPT-IN task-aware acceptance filter.
Verifies: default OFF; the filter drops only task-failing sentences and keeps every
per-instance array index-aligned + re-indexed. Pure-function (no disk I/O, no API).
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import node_engine


def _state():
    return {
        "first_language": "Arabic",
        "data_generation_result": ["s0", "s1", "s2"],
        "sentence_records": [
            {"index": 0, "text": "s0", "weighted_score": 7.0, "task_passed": True},
            {"index": 1, "text": "s1", "weighted_score": 7.0, "task_passed": False},
            {"index": 2, "text": "s2", "weighted_score": 8.5, "task_passed": True},
        ],
        "fluency_results_per_instances": [{"fluency_score": 7}, {"fluency_score": 6}, {"fluency_score": 8}],
        "task_validation_results_per_instances": [{"passed": True}, {"passed": False}, {"passed": True}],
        "instance_refine_counts": [1, 1, 0],
        "sentence_scores": [7.0, 7.0, 8.5],
        "failing_sentence_indices": [0, 1],
    }


def test_default_is_on():
    # Task-aware acceptance is the default going forward; set TASK_AWARE_ACCEPT=0
    # to reproduce the historical (quality-only) corpora.
    assert node_engine.TASK_AWARE_ACCEPT is True, "TASK_AWARE_ACCEPT must default ON"
    print("PASS test_default_is_on")


def test_filter_drops_task_failing_and_realigns():
    out = node_engine._filter_state_to_task_passed(_state())
    # index 1 (task_passed False) dropped -> keep s0, s2
    assert out["data_generation_result"] == ["s0", "s2"], out["data_generation_result"]
    assert [r["text"] for r in out["sentence_records"]] == ["s0", "s2"]
    # re-indexed 0..k-1
    assert [r["index"] for r in out["sentence_records"]] == [0, 1]
    # every per-instance array filtered by the SAME kept indices
    assert out["fluency_results_per_instances"] == [{"fluency_score": 7}, {"fluency_score": 8}]
    assert out["task_validation_results_per_instances"] == [{"passed": True}, {"passed": True}]
    assert out["instance_refine_counts"] == [1, 0]
    assert out["sentence_scores"] == [7.0, 8.5]
    # failing_sentence_indices recomputed on the survivors (s0=7.0<8 -> 0 ; s2=8.5 -> not)
    assert out["failing_sentence_indices"] == [0], out["failing_sentence_indices"]
    # every surviving record actually passed the task
    assert all(r["task_passed"] is not False for r in out["sentence_records"])
    print("PASS test_filter_drops_task_failing_and_realigns")


def test_all_pass_is_noop():
    st = _state()
    for r in st["sentence_records"]:
        r["task_passed"] = True
    out = node_engine._filter_state_to_task_passed(st)
    assert out is st, "nothing to drop -> must return the same object unchanged"
    print("PASS test_all_pass_is_noop")


def test_unknown_verdict_is_kept():
    # validator disabled -> task_passed None everywhere -> keep all (never empty the corpus)
    st = _state()
    for r in st["sentence_records"]:
        r["task_passed"] = None
    out = node_engine._filter_state_to_task_passed(st)
    assert len(out["sentence_records"]) == 3
    print("PASS test_unknown_verdict_is_kept")


if __name__ == "__main__":
    print("Running task-aware acceptance tests...")
    test_default_is_on()
    test_filter_drops_task_failing_and_realigns()
    test_all_pass_is_noop()
    test_unknown_verdict_is_kept()
    print("ALL TASK-AWARE ACCEPT TESTS PASSED")
