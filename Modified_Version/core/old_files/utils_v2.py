import yaml
from pprint import pprint
from node_models import AgentRunningState
import jsonlines as jsl

def load_config(config_path: str):
    # Each config file will generate a different scenarios ~1440
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


import itertools

import itertools
from typing import Any, Dict, List

# AgentRunningState is your Union[TopicState, SentimentState, NERState]
# This function returns a list[AgentRunningState]-compatible dicts.

def generate_scenarios(pre_execute: dict) -> list:
    """
    Task-aware scenario generator (Pattern A):

    Input expected (your YAML structure):
      pre_execute:
        cs_ratio: [...]
        task: ["sentiment","ner","topic"]
        shared: {...}
        sentiment: {...}
        ner: {...}
        topic: {topics: [...]}

    Output:
      list of scenario dicts that include:
        - shared fields (topic/tense/perspective/.../cs_ratio/etc.)
        - task discriminator: task
        - task payload:
            topic/sentiment -> label
            ner -> annotations
        - task_constraints (per task)
    """

    tasks: List[str] = pre_execute["task"]
    cs_ratios: List[str] = pre_execute["cs_ratio"]

    shared: Dict[str, Any] = pre_execute["shared"]
    char: Dict[str, Any] = shared["character_setting"]

    # shared lists
    domains: List[str] = shared["topic"]
    tenses: List[str] = shared["tense"]
    perspectives: List[str] = shared["perspective"]
    genders: List[str] = char["gender"]
    ages: List[str] = char["age"]
    edu_levels: List[str] = char["education_level"]
    conversation_types: List[str] = shared["conversation_type"]
    cs_functions: List[str] = shared["cs_function"]
    cs_types: List[str] = shared["cs_type"]

    use_tools: bool = bool(shared.get("use_tools", False))
    first_language: str = char["nationality"]["first_language"]
    second_language: str = char["nationality"]["second_language"]

    # ---------
    # 1) Base context scenarios (task-agnostic)
    # ---------
    base_scenarios: List[Dict[str, Any]] = []
    for (
        domain,
        tense,
        perspective,
        gender,
        age,
        edu_level,
        cs_ratio,
        conversation_type,
        cs_function,
        cs_type,
    ) in itertools.product(
        domains,
        tenses,
        perspectives,
        genders,
        ages,
        edu_levels,
        cs_ratios,
        conversation_types,
        cs_functions,
        cs_types,
    ):
        base_scenarios.append(
            {
                # keep your original field name "topic" as domain-context
                "topic": domain,
                "tense": tense,
                "perspective": perspective,
                "gender": gender,
                "age": age,
                "education_level": edu_level,
                "cs_ratio": cs_ratio,
                "use_tools": use_tools,
                "conversation_type": conversation_type,
                "first_language": first_language,
                "second_language": second_language,
                "cs_function": cs_function,
                "cs_type": cs_type,
                # optional init fields (safe defaults)
                "data_generation_result": [],
                "response": "",
                "refine_count": 0,
            }
        )

    all_scenarios: List[Dict[str, Any]] = []

    # ---------
    # 2) Expand per task (Pattern A)
    # ---------

    # ---- Topic task
    if "topic" in tasks:
        topic_labels: List[str] = pre_execute["topic"]["topics"]
        for base in base_scenarios:
            for lbl in topic_labels:
                s = dict(base)
                s["task"] = "topic"
                s["label"] = lbl
                # Make topic generation consistent: set the context topic == label
                # (so you don't generate "finance context" but label "sports")
                s["topic"] = lbl
                # s["task_validation_result"] = {}
                all_scenarios.append(s)

    # ---- Sentiment task
    if "sentiment" in tasks:
        sent_cfg: Dict[str, Any] = pre_execute["sentiment"]
        sent_labels: List[str] = sent_cfg["labels"]
        intensities: List[str] = sent_cfg.get("intensity", ["low"])
        ambiguities: List[str] = sent_cfg.get("ambiguity", ["low"])

        for base in base_scenarios:
            for lbl, intensity, ambiguity in itertools.product(sent_labels, intensities, ambiguities):
                s = dict(base)
                s["task"] = "sentiment"
                s["label"] = lbl
                s["task_constraints"] = {
                    "intensity": intensity,
                    "ambiguity": ambiguity,
                }
                # s["task_validation_result"] = {}
                all_scenarios.append(s)

    # ---- NER task
    if "ner" in tasks:
        ner_cfg: Dict[str, Any] = pre_execute["ner"]
        entity_types: List[str] = ner_cfg["entity_types"]
        min_entities_list: List[int] = ner_cfg.get("min_entities", [2])
        max_entities_list: List[int] = ner_cfg.get("max_entities", [3])
        must_include_types: List[str] = ner_cfg.get("must_include_types", [])
        allow_cs_entities_list: List[bool] = ner_cfg.get("allow_code_switched_entities", [True])

        for base in base_scenarios:
            for min_e, max_e, allow_cs_entities in itertools.product(
                min_entities_list, max_entities_list, allow_cs_entities_list
            ):
                s = dict(base)
                s["task"] = "ner"
                s["annotations"] = []  # will be filled by NERGenerator
                s["task_constraints"] = {
                    "entity_types": entity_types,
                    "min_entities": min_e,
                    "max_entities": max_e,
                    "must_include_types": must_include_types,
                    "allow_code_switched_entities": allow_cs_entities,
                }
                # s["task_validation_result"] = {}
                all_scenarios.append(s)

    return all_scenarios



def weighting_scheme(state):
    fluency = state["fluency_result"]["fluency_score"]
    naturalness = state["naturalness_result"]["naturalness_score"]
    csratio = state["cs_ratio_result"]["ratio_score"]
    socio = state["social_cultural_result"]["socio_cultural_score"]
    return fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25


if __name__ == "__main__":
    # Here is your config dictionary (simplified for the example):
    config = {
        "character_setting": {
            "age": ["8-17", "18-25", "26-35", "56-65", "66+"],
            "education_level": ["High School", "College", "Master", "Doctor"],
            "gender": ["Male", "Female"],
            "nationality": {
                "first_language": "Cantonese",
                "second_language": "English",
            },
        },
        "cs_function": [
            "Directive",
            "Expressive",
            "Referential",
            "Phatic",
            "Metalinguistic",
            "Poetic",
        ],
        "cs_type": [
            "Intersentential",
            "Intrasentential",
            "Extra-sentential / Tag switching",
        ],
        "cs_ratio": ["Low", "Medium", "High"],
        "output_format": "json",
        "output_type": "single_turn",
        "perspective": ["First Person", "Third Person"],
        "tense": ["Past", "Present", "Future"],
        "topics": ["Tourism", "Weather", "Shopping", "Food", "Exam", "Politics"],
        "use_tools": True,
        "conversation_type": ["single_turn", "multi-turn"],
    }

    scenarios = generate_scenarios(config)
    print(f"Generated {len(scenarios)} scenario combinations.")
    # For a quick peek, let's print the first few
    for i, sc in enumerate(scenarios[:10]):
        print(f"Scenario #{i+1}:", sc)
        print("\n")
