"""
Regression tests for the generic, config-driven NER entity guidance.
Run: PYTHONPATH=Modified_Version/core python "Modified_Version/core/test files/test_ner_guidance.py"
(also pytest-discoverable). No API calls.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add core/

from node_engine import build_ner_entity_guidance, DEFAULT_ENTITY_GUIDANCE  # noqa: E402
from prompt import DATA_GENERATION_NER_PROMPT  # noqa: E402

_NER_TEMPLATE = DATA_GENERATION_NER_PROMPT.messages[0].prompt.template


def test_event_guidance_present_when_required():
    g = build_ner_entity_guidance({"must_include_types": ["EVENT"], "target_entities_script": "english"})
    assert "EVENT:" in g
    assert "English/Latin letters" in g
    # only the required type is described (PER/ORG not pulled in)
    assert "PER:" not in g and "ORG:" not in g


def test_loc_guidance_present_when_required():
    g = build_ner_entity_guidance({"must_include_types": ["LOC"], "target_entities_script": "english"})
    assert "LOC:" in g
    assert any(city in g for city in ("Cairo", "Dubai", "London"))


def test_new_tag_via_config_without_touching_prompt():
    # A brand-new tag, defined ONLY through config guidance — no prompt.py edit.
    cons = {
        "must_include_types": ["FOOD"],
        "target_entities_script": "english",
        "entity_type_guidance": {
            "FOOD": {"description": "a named dish or food item",
                     "script_rule": "must be written in English/Latin letters",
                     "examples": ["Koshari", "Falafel"]},
        },
    }
    g = build_ner_entity_guidance(cons)
    assert "FOOD:" in g
    assert "a named dish or food item" in g
    assert "Koshari" in g
    # and the prompt file itself contains no hardcoded 'FOOD' (proves config-driven, not prompt-edited)
    assert "FOOD" not in _NER_TEMPLATE


def test_missing_guidance_falls_back_safely():
    # Unknown tag with NO config guidance and NOT in DEFAULT -> generic fallback, no crash.
    assert "WORKOFART" not in DEFAULT_ENTITY_GUIDANCE
    g = build_ner_entity_guidance({"must_include_types": ["WORKOFART"], "target_entities_script": "english"})
    assert "WORKOFART:" in g
    assert "a named entity of type WORKOFART" in g
    assert "English/Latin letters" in g          # script rule from policy fallback
    # non-english policy fallback rule
    g2 = build_ner_entity_guidance({"must_include_types": ["WORKOFART"], "target_entities_script": "any"})
    assert "Arabic or English" in g2


def test_prompt_uses_placeholder_and_selfcheck():
    assert "{ner_entity_guidance}" in _NER_TEMPLATE
    assert "ner_entity_guidance" in DATA_GENERATION_NER_PROMPT.input_variables
    assert "SELF-CHECK" in _NER_TEMPLATE          # self-check instruction retained


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
