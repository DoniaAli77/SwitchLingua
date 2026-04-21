"""
test_refiner_guardrail.py
Focused tests for RunRefinerAgent guardrail logic:
  1. quality_fail → refiner accepted  (guardrail: task still passes)
  2. quality_fail → rollback          (guardrail: task broke → keep original)
  3. task_fail    → task refiner used (guardrail: task now passes → accept)
"""
import os
import sys
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import node_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakePrompt:
    """Minimal prompt substitute that sets the LLM mode when chained via |."""
    def __init__(self, mode: str):
        self.mode = mode

    def __or__(self, llm):
        if hasattr(llm, "set_mode"):
            llm.set_mode(self.mode)
        return llm


def _make_state(task: str, sentences: list, records: list, refine_counts: list) -> dict:
    return {
        "task": task,
        "topic": "technology",
        "label": "positive",
        "first_language": "Arabic",
        "second_language": "English",
        "cs_ratio": "30%",
        "task_constraints": {"entity_types": ["PER", "ORG", "LOC"]},
        "task_validation_results_per_instances": [],
        "mcp_result": "",
        "data_generation_result": list(sentences),
        "sentence_records": records,
        "instance_refine_counts": list(refine_counts),
    }


def _record(index, text, task_passed=True, weighted_score=6.0, refine_count=0):
    return {
        "index": index,
        "text": text,
        "weighted_score": weighted_score,
        "refine_count": refine_count,
        "status": "fail" if weighted_score < 8.0 or not task_passed else "pass",
        "task_passed": task_passed,
        "task_validation": {"passed": task_passed, "feedback": "test feedback"},
        "fluency": {}, "naturalness": {}, "cs_ratio": {}, "socio_cultural": {},
    }


# ---------------------------------------------------------------------------
# Test 1: quality_fail → accepted
# ---------------------------------------------------------------------------

class _LLM_Accept:
    """Refiner returns a candidate; guardrail re-validation returns task=passed."""
    def __init__(self, *args, **kwargs):
        self.mode = "unset"
    def set_mode(self, mode):
        self.mode = mode
        return self

    def with_structured_output(self, _):
        return self

    def invoke(self, payload):
        if self.mode == "refiner":
            return {"instances": ["refined quality sentence"]}
        if self.mode in {"val_topic", "val_sentiment", "val_ner"}:
            return {"passed": True, "confidence": 0.9,
                    "notes": "ok", "predicted_label": "positive", "errors": []}
        # Quality re-scoring modes (called by _rescore_single_sentence)
        if self.mode == "fluency":
            from types import SimpleNamespace
            import json
            return SimpleNamespace(content=json.dumps({"fluency_score": 9.0, "errors": {}, "summary": "ok"}))
        if self.mode == "naturalness":
            from types import SimpleNamespace
            import json
            return SimpleNamespace(content=json.dumps({"naturalness_score": 9.0, "observations": {}, "summary": "ok"}))
        if self.mode == "cs_ratio":
            from types import SimpleNamespace
            import json
            return SimpleNamespace(content=json.dumps({"ratio_score": 9.0, "computed_ratio": "70%:30%", "notes": "ok"}))
        if self.mode == "social":
            from types import SimpleNamespace
            import json
            return SimpleNamespace(content=json.dumps({"socio_cultural_score": 9.0, "issues": "", "summary": "ok"}))
        return {}


def test_refiner_quality_fail_accepted():
    """
    Sentence 0 has quality_fail (score=6.0, task_passed=True).
    Refiner returns a candidate → guardrail confirms task still passes → candidate accepted.
    Sentence 1 is fine (score=9.0) → should not be touched.
    """
    state = _make_state(
        task="sentiment",
        sentences=["bad quality sentence", "good sentence"],
        records=[
            _record(0, "bad quality sentence", task_passed=True, weighted_score=6.0),
            _record(1, "good sentence",         task_passed=True, weighted_score=9.0),
        ],
        refine_counts=[0, 0],
    )

    with patch.object(node_engine, "ChatOpenAI", _LLM_Accept), \
         patch.object(node_engine, "REFINER_PROMPT",               _FakePrompt("refiner")), \
         patch.object(node_engine, "REFINER_TASK_TOPIC_PROMPT",    _FakePrompt("refiner_task")), \
         patch.object(node_engine, "REFINER_TASK_SENTIMENT_PROMPT",_FakePrompt("refiner_task")), \
         patch.object(node_engine, "REFINER_TASK_NER_PROMPT",      _FakePrompt("refiner_task")), \
         patch.object(node_engine, "TASK_VALIDATION_TOPIC_PROMPT",     _FakePrompt("val_topic")), \
         patch.object(node_engine, "TASK_VALIDATION_SENTIMENT_PROMPT", _FakePrompt("val_sentiment")), \
         patch.object(node_engine, "TASK_VALIDATION_NER_PROMPT",       _FakePrompt("val_ner")), \
         patch.object(node_engine, "FLUENCY_PROMPT",        _FakePrompt("fluency")), \
         patch.object(node_engine, "NATURALNESS_PROMPT",    _FakePrompt("naturalness")), \
         patch.object(node_engine, "CS_RATIO_PROMPT",       _FakePrompt("cs_ratio")), \
         patch.object(node_engine, "SOCIAL_CULTURAL_PROMPT",_FakePrompt("social")):

        result = node_engine.RunRefinerAgent(state)

    texts  = result["data_generation_result"]
    counts = result["instance_refine_counts"]

    assert texts[0] == "refined quality sentence", f"Expected refined, got: {texts[0]!r}"
    assert texts[1] == "good sentence",            "Good sentence must not be changed"
    assert counts[0] == 1, f"Refine count for sentence 0 should be 1, got: {counts[0]}"
    assert counts[1] == 0, f"Refine count for sentence 1 should be 0, got: {counts[1]}"
    print("PASS test_refiner_quality_fail_accepted")


# ---------------------------------------------------------------------------
# Test 2: quality_fail → rollback (guardrail detects task broke)
# ---------------------------------------------------------------------------

class _LLM_Rollback:
    """Refiner returns a candidate, but guardrail re-validation says task broke."""
    def __init__(self, *args, **kwargs):
        self.mode = "unset"
    def set_mode(self, mode):
        self.mode = mode
        return self

    def with_structured_output(self, _):
        return self

    def invoke(self, payload):
        if self.mode == "refiner":
            return {"instances": ["broken task sentence"]}
        if self.mode in {"val_topic", "val_sentiment", "val_ner"}:
            return {"passed": False, "confidence": 0.3,
                    "notes": "task broke", "predicted_label": "negative", "errors": []}
        # Quality re-scoring modes
        if self.mode == "fluency":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"fluency_score": 9.0, "errors": {}, "summary": "ok"}))
        if self.mode == "naturalness":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"naturalness_score": 9.0, "observations": {}, "summary": "ok"}))
        if self.mode == "cs_ratio":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"ratio_score": 9.0, "computed_ratio": "70%:30%", "notes": "ok"}))
        if self.mode == "social":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"socio_cultural_score": 9.0, "issues": "", "summary": "ok"}))
        return {}


def test_refiner_quality_fail_rollback():
    """
    Sentence 0 has quality_fail (task_passed=True, score=6.0).
    Refiner returns a candidate → guardrail detects task broke → rollback → original kept.
    """
    original = "original sentence"
    state = _make_state(
        task="sentiment",
        sentences=[original],
        records=[_record(0, original, task_passed=True, weighted_score=6.0)],
        refine_counts=[0],
    )

    with patch.object(node_engine, "ChatOpenAI", _LLM_Rollback), \
         patch.object(node_engine, "REFINER_PROMPT",               _FakePrompt("refiner")), \
         patch.object(node_engine, "REFINER_TASK_TOPIC_PROMPT",    _FakePrompt("refiner_task")), \
         patch.object(node_engine, "REFINER_TASK_SENTIMENT_PROMPT",_FakePrompt("refiner_task")), \
         patch.object(node_engine, "REFINER_TASK_NER_PROMPT",      _FakePrompt("refiner_task")), \
         patch.object(node_engine, "TASK_VALIDATION_TOPIC_PROMPT",     _FakePrompt("val_topic")), \
         patch.object(node_engine, "TASK_VALIDATION_SENTIMENT_PROMPT", _FakePrompt("val_sentiment")), \
         patch.object(node_engine, "TASK_VALIDATION_NER_PROMPT",       _FakePrompt("val_ner")), \
         patch.object(node_engine, "FLUENCY_PROMPT",        _FakePrompt("fluency")), \
         patch.object(node_engine, "NATURALNESS_PROMPT",    _FakePrompt("naturalness")), \
         patch.object(node_engine, "CS_RATIO_PROMPT",       _FakePrompt("cs_ratio")), \
         patch.object(node_engine, "SOCIAL_CULTURAL_PROMPT",_FakePrompt("social")):

        result = node_engine.RunRefinerAgent(state)

    texts  = result["data_generation_result"]
    counts = result["instance_refine_counts"]

    assert texts[0] == original, \
        f"Rollback expected original {original!r}, got: {texts[0]!r}"
    assert counts[0] == 0, \
        f"Refine count should stay 0 after rollback, got: {counts[0]}"
    print("PASS test_refiner_quality_fail_rollback")


# ---------------------------------------------------------------------------
# Test 3: task_fail → task-specific refiner → accepted
# ---------------------------------------------------------------------------

class _LLM_TaskFix:
    """Task-specific refiner fixes the sentence; guardrail confirms task now passes."""
    def __init__(self, *args, **kwargs):
        self.mode = "unset"
    def set_mode(self, mode):
        self.mode = mode
        return self

    def with_structured_output(self, _):
        return self

    def invoke(self, payload):
        if self.mode == "refiner_task":
            return {"instances": ["fixed task sentence"]}
        if self.mode in {"val_topic", "val_sentiment", "val_ner"}:
            return {"passed": True, "confidence": 0.95,
                    "notes": "task fixed", "predicted_label": "positive", "errors": []}
        # Quality re-scoring modes (not used for task_fail path, but needed for robustness)
        if self.mode == "fluency":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"fluency_score": 9.0, "errors": {}, "summary": "ok"}))
        if self.mode == "naturalness":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"naturalness_score": 9.0, "observations": {}, "summary": "ok"}))
        if self.mode == "cs_ratio":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"ratio_score": 9.0, "computed_ratio": "70%:30%", "notes": "ok"}))
        if self.mode == "social":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"socio_cultural_score": 9.0, "issues": "", "summary": "ok"}))
        return {}


def test_refiner_task_fail_accepted():
    """
    Sentence 0 has task_fail (task_passed=False).
    Refiner uses task-specific prompt → guardrail confirms task now passes → candidate accepted.
    """
    state = _make_state(
        task="sentiment",
        sentences=["wrong sentiment sentence"],
        records=[_record(0, "wrong sentiment sentence", task_passed=False, weighted_score=9.0)],
        refine_counts=[0],
    )

    with patch.object(node_engine, "ChatOpenAI", _LLM_TaskFix), \
         patch.object(node_engine, "REFINER_PROMPT",               _FakePrompt("refiner")), \
         patch.object(node_engine, "REFINER_TASK_TOPIC_PROMPT",    _FakePrompt("refiner_task")), \
         patch.object(node_engine, "REFINER_TASK_SENTIMENT_PROMPT",_FakePrompt("refiner_task")), \
         patch.object(node_engine, "REFINER_TASK_NER_PROMPT",      _FakePrompt("refiner_task")), \
         patch.object(node_engine, "TASK_VALIDATION_TOPIC_PROMPT",     _FakePrompt("val_topic")), \
         patch.object(node_engine, "TASK_VALIDATION_SENTIMENT_PROMPT", _FakePrompt("val_sentiment")), \
         patch.object(node_engine, "TASK_VALIDATION_NER_PROMPT",       _FakePrompt("val_ner")), \
         patch.object(node_engine, "FLUENCY_PROMPT",        _FakePrompt("fluency")), \
         patch.object(node_engine, "NATURALNESS_PROMPT",    _FakePrompt("naturalness")), \
         patch.object(node_engine, "CS_RATIO_PROMPT",       _FakePrompt("cs_ratio")), \
         patch.object(node_engine, "SOCIAL_CULTURAL_PROMPT",_FakePrompt("social")):

        result = node_engine.RunRefinerAgent(state)

    texts  = result["data_generation_result"]
    counts = result["instance_refine_counts"]

    assert texts[0] == "fixed task sentence", \
        f"Expected task-fixed sentence, got: {texts[0]!r}"
    assert counts[0] == 1, \
        f"Refine count should be 1 after task fix, got: {counts[0]}"
    print("PASS test_refiner_task_fail_accepted")


# ---------------------------------------------------------------------------
# Test 4: quality_fail → rollback because quality score got WORSE after refine
# ---------------------------------------------------------------------------

class _LLM_QualityRegression:
    """Refiner returns a candidate; task passes but re-score returns lower quality."""
    def __init__(self, *args, **kwargs):
        self.mode = "unset"
        self._rescore_call_count = 0

    def set_mode(self, mode):
        self.mode = mode
        return self

    def with_structured_output(self, _):
        return self

    def invoke(self, payload):
        if self.mode == "refiner":
            return {"instances": ["worse quality sentence"]}
        if self.mode in {"val_topic", "val_sentiment", "val_ner"}:
            return {"passed": True, "confidence": 0.9,
                    "notes": "ok", "predicted_label": "positive", "errors": []}
        # Quality re-scoring: return low scores (simulating quality regression)
        if self.mode == "fluency":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"fluency_score": 4.0, "errors": {}, "summary": "bad"}))
        if self.mode == "naturalness":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"naturalness_score": 4.0, "observations": {}, "summary": "bad"}))
        if self.mode == "cs_ratio":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"ratio_score": 4.0, "computed_ratio": "50%:50%", "notes": "bad"}))
        if self.mode == "social":
            from types import SimpleNamespace; import json
            return SimpleNamespace(content=json.dumps({"socio_cultural_score": 4.0, "issues": "bad", "summary": "bad"}))
        return {}


def test_refiner_quality_regression_rollback():
    """
    Sentence 0 has quality_fail (task_passed=True, score=6.0).
    Refiner returns a candidate → task guardrail passes → re-score returns 4.0 (< 6.0) → ROLLBACK.
    """
    original = "original decent sentence"
    state = _make_state(
        task="sentiment",
        sentences=[original],
        records=[_record(0, original, task_passed=True, weighted_score=6.0)],
        refine_counts=[0],
    )

    with patch.object(node_engine, "ChatOpenAI", _LLM_QualityRegression), \
         patch.object(node_engine, "REFINER_PROMPT",               _FakePrompt("refiner")), \
         patch.object(node_engine, "REFINER_TASK_TOPIC_PROMPT",    _FakePrompt("refiner_task")), \
         patch.object(node_engine, "REFINER_TASK_SENTIMENT_PROMPT",_FakePrompt("refiner_task")), \
         patch.object(node_engine, "REFINER_TASK_NER_PROMPT",      _FakePrompt("refiner_task")), \
         patch.object(node_engine, "TASK_VALIDATION_TOPIC_PROMPT",     _FakePrompt("val_topic")), \
         patch.object(node_engine, "TASK_VALIDATION_SENTIMENT_PROMPT", _FakePrompt("val_sentiment")), \
         patch.object(node_engine, "TASK_VALIDATION_NER_PROMPT",       _FakePrompt("val_ner")), \
         patch.object(node_engine, "FLUENCY_PROMPT",        _FakePrompt("fluency")), \
         patch.object(node_engine, "NATURALNESS_PROMPT",    _FakePrompt("naturalness")), \
         patch.object(node_engine, "CS_RATIO_PROMPT",       _FakePrompt("cs_ratio")), \
         patch.object(node_engine, "SOCIAL_CULTURAL_PROMPT",_FakePrompt("social")):

        result = node_engine.RunRefinerAgent(state)

    texts  = result["data_generation_result"]
    counts = result["instance_refine_counts"]

    assert texts[0] == original, \
        f"Quality regression rollback expected original, got: {texts[0]!r}"
    assert counts[0] == 0, \
        f"Refine count should stay 0 after quality-regression rollback, got: {counts[0]}"
    print("PASS test_refiner_quality_regression_rollback")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running refiner guardrail tests...")
    test_refiner_quality_fail_accepted()
    test_refiner_quality_fail_rollback()
    test_refiner_task_fail_accepted()
    test_refiner_quality_regression_rollback()
    print("ALL GUARDRAIL TESTS PASSED")
