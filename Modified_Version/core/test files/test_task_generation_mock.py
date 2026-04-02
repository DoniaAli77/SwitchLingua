import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

from node_engine import RunDataGenerationAgent


def _base_state(task: str):
    base = {
        "task": task,
        "topic": "tech",
        "label": "positive",
        "annotations": [],
        "news_dict": {},
        "news_hash": set(),
    }
    
    # Topic task: no constraints needed
    if task == "topic":
        return base
    
    # Sentiment task: intensity & ambiguity constraints
    elif task == "sentiment":
        base["task_constraints"] = {
            "intensity": "low",
            "ambiguity": "low",
        }
        return base
    
    # NER task: entity-specific constraints
    elif task == "ner":
        base["task_constraints"] = {
            "entity_types": ["PER", "ORG", "LOC"],
            "min_entities": 2,
            "max_entities": 3,
            "must_include_types": ["PER", "ORG"],
            "allow_code_switched_entities": True,
        }
        return base
    
    return base


def run_mock_test():
    with patch("node_engine.RunTopicDataGenerationAgent", return_value={"instances": ["mock-topic"]}) as topic_mock, \
         patch("node_engine.RunSentimentDataGenerationAgent", return_value={"instances": ["mock-sentiment"]}) as sentiment_mock, \
         patch("node_engine.RunNERDataGenerationAgent", return_value={"instances": ["mock-ner"]}) as ner_mock:

        topic_result = RunDataGenerationAgent(_base_state("topic"))
        sentiment_result = RunDataGenerationAgent(_base_state("sentiment"))
        ner_result = RunDataGenerationAgent(_base_state("ner"))

        assert topic_result["data_generation_result"] == ["mock-topic"]
        assert sentiment_result["data_generation_result"] == ["mock-sentiment"]
        assert ner_result["data_generation_result"] == ["mock-ner"]

        assert topic_mock.call_count == 1
        assert sentiment_mock.call_count == 1
        assert ner_mock.call_count == 1

        print("PASS: task routing works without OpenAI calls")
        print("topic ->", topic_result)
        print("sentiment ->", sentiment_result)
        print("ner ->", ner_result)


if __name__ == "__main__":
    run_mock_test()
