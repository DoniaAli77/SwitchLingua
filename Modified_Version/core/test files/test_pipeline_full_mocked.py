import json
import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(REPO_ROOT, "drive_code", "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

import node_engine
from utils import weighting_scheme


class FakePrompt:
    def __init__(self, mode: str):
        self.mode = mode

    def __or__(self, llm):
        if hasattr(llm, "set_mode"):
            llm.set_mode(self.mode)
        return llm


class FakeChatOpenAI:
    def __init__(self, *args, **kwargs):
        self.mode = "unset"

    def set_mode(self, mode: str):
        self.mode = mode
        return self

    def with_structured_output(self, _schema):
        return self

    def _count_batch_sentences(self, payload: dict) -> int:
        text = payload.get("sentences_for_batch") or payload.get("sentences_with_stats") or ""
        matches = re.findall(r"(?m)^Sentence\s+\d+:", str(text))
        return len(matches) if matches else 1

    def invoke(self, payload: dict):
        if self.mode in {"gen_topic", "gen_sentiment", "gen_ner"}:
            return {
                "instances": [
                    "اليوم الجو جميل and I am happy.",
                    "أنا أحب learning because it is useful.",
                    "هذا اختبار small test for code switching.",
                ]
            }

        if self.mode in {"val_topic", "val_sentiment", "val_ner"}:
            return {
                "passed": True,
                "confidence": 0.9,
                "notes": f"ok_{self.mode}",
                "predicted_label": "ok",
                "errors": [],
            }

        n = self._count_batch_sentences(payload)

        if self.mode == "fluency":
            arr = [
                {"fluency_score": 8.0 + (i * 0.1), "errors": {}, "summary": f"fluency_{i+1}"}
                for i in range(n)
            ]
            return SimpleNamespace(content=json.dumps(arr))

        if self.mode == "naturalness":
            arr = [
                {
                    "naturalness_score": 7.5 + (i * 0.1),
                    "observations": {"1": f"obs_{i+1}"},
                    "summary": f"natural_{i+1}",
                }
                for i in range(n)
            ]
            return SimpleNamespace(content=json.dumps(arr))

        if self.mode == "cs_ratio":
            arr = [
                {
                    "ratio_score": 6 + i,
                    "computed_ratio": "60% : 40%",
                    "notes": f"ratio_{i+1}",
                }
                for i in range(n)
            ]
            return SimpleNamespace(content=json.dumps(arr))

        if self.mode == "social":
            arr = [
                {
                    "socio_cultural_score": 8.0 + (i * 0.1),
                    "issues": "",
                    "summary": f"social_{i+1}",
                }
                for i in range(n)
            ]
            return SimpleNamespace(content=json.dumps(arr))

        raise RuntimeError(f"Unhandled fake mode: {self.mode}")


def _base_state(task: str) -> dict:
    state = {
        "task": task,
        "topic": "technology",
        "label": "positive",
        "annotations": [],
        "task_constraints": {
            "intensity": "low",
            "ambiguity": "low",
            "entity_types": ["PER", "ORG", "LOC"],
            "min_entities": 1,
            "max_entities": 5,
            "must_include_types": ["PER"],
            "allow_code_switched_entities": True,
        },
        "news_dict": {},
        "news_hash": set(),
        "cs_ratio": "30%",
        "first_language": "Arabic",
        "second_language": "English",
        "tense": "Present",
        "perspective": "First Person",
        "gender": "Female",
        "age": "18-25",
        "education_level": "College",
        "conversation_type": "single_turn",
        "cs_function": "Expressive",
        "cs_type": "Intrasentential",
    }
    return state


def run_task_pipeline(task: str) -> None:
    state = _base_state(task)

    gen_out = node_engine.RunDataGenerationAgent(state)
    state.update(gen_out)

    val_out = node_engine.RunTaskValidatorAgent(state)
    state.update(val_out)

    flu_out = node_engine.RunFluencyAgent(state)
    state.update(flu_out)

    nat_out = node_engine.RunNaturalnessAgent(state)
    state.update(nat_out)

    cs_out = node_engine.RunCSRatioAgent(state)
    state.update(cs_out)

    soc_out = node_engine.RunSocialCulturalAgent(state)
    state.update(soc_out)

    sum_out = node_engine.SummarizeResult(state)
    state.update(sum_out)

    n = len(state["data_generation_result"])
    assert n > 0, "No generated instances"
    assert len(state["fluency_results_per_instances"]) == n, "Fluency per-instance mismatch"
    assert len(state["naturalness_results_per_instances"]) == n, "Naturalness per-instance mismatch"
    assert len(state["cs_ratio_results_per_instances"]) == n, "CS ratio per-instance mismatch"
    assert len(state["social_cultural_results_per_instances"]) == n, "Social per-instance mismatch"

    assert "fluency_score" in state["fluency_result"], "Missing fluency aggregate"
    assert "naturalness_score" in state["naturalness_result"], "Missing naturalness aggregate"
    assert "socio_cultural_score" in state["social_cultural_result"], "Missing social aggregate"

    expected = weighting_scheme(state)
    assert abs(state["score"] - expected) < 1e-9, "Summary score mismatch"
    assert "summary" in state and isinstance(state["summary"], str), "Missing summary"
    assert state["task_validation_result"].get("passed") is True, "Validation should pass"

    print(f"PASS task={task} instances={n} score={state['score']:.4f}")


def main() -> None:
    with patch.object(node_engine, "ChatOpenAI", FakeChatOpenAI), patch.object(
        node_engine, "DATA_GENERATION_TOPIC_PROMPT", FakePrompt("gen_topic")
    ), patch.object(
        node_engine, "DATA_GENERATION_SENTIMENT_PROMPT", FakePrompt("gen_sentiment")
    ), patch.object(
        node_engine, "DATA_GENERATION_NER_PROMPT", FakePrompt("gen_ner")
    ), patch.object(
        node_engine, "TASK_VALIDATION_TOPIC_PROMPT", FakePrompt("val_topic")
    ), patch.object(
        node_engine, "TASK_VALIDATION_SENTIMENT_PROMPT", FakePrompt("val_sentiment")
    ), patch.object(
        node_engine, "TASK_VALIDATION_NER_PROMPT", FakePrompt("val_ner")
    ), patch.object(
        node_engine, "FLUENCY_PROMPT", FakePrompt("fluency")
    ), patch.object(
        node_engine, "NATURALNESS_PROMPT", FakePrompt("naturalness")
    ), patch.object(
        node_engine, "CS_RATIO_PROMPT", FakePrompt("cs_ratio")
    ), patch.object(
        node_engine, "SOCIAL_CULTURAL_PROMPT", FakePrompt("social")
    ):
        print("Running full mocked pipeline tests...")
        for task in ["topic", "sentiment", "ner"]:
            run_task_pipeline(task)
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
