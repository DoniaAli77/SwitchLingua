"""
Regression tests for the ENGLISH-ONLY entity script test.

Pins the fix for the bug where has_latin_entity_candidate() used CAPITALISATION as a proxy for
"Latin-script entity present", which silently discarded lowercase common-noun entities (currency
names) and mislabelled them 'arabic_script_entity'.

Run: python experiments/switchlingua/test_latin_entity_span.py     (no API calls)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_ner_coverage_gen import (  # noqa: E402
    has_latin_entity_span,
    has_latin_entity_candidate,
)


def _ents(*texts):
    return [{"text": t, "type": "MISC"} for t in texts]


def test_lowercase_currency_accepted():
    # The exact spans that were being thrown away before the fix.
    for word in ("pounds", "dirhams", "euros", "riyals", "dollars"):
        assert has_latin_entity_span(_ents(word)), word


def test_capitalised_proper_nouns_still_accepted():
    for word in ("Amazon", "March", "Ramadan", "Manchester United", "H&M"):
        assert has_latin_entity_span(_ents(word)), word


def test_arabic_script_entities_rejected():
    # The policy this filter exists to enforce must still hold.
    for word in ("عمر خالد", "نيويورك", "دولار", "الجزيرة"):
        assert not has_latin_entity_span(_ents(word)), word


def test_mixed_script_span_rejected():
    # A span containing ANY Arabic is not a Latin-script entity.
    assert not has_latin_entity_span(_ents("مطعم Pizza"))


def test_empty_and_malformed_input():
    assert not has_latin_entity_span([])
    assert not has_latin_entity_span(None)
    assert not has_latin_entity_span([{}])            # no 'text' key
    assert not has_latin_entity_span([{"text": ""}])  # empty span
    assert has_latin_entity_span(["Amazon"])          # bare strings, not dicts


def test_any_latin_span_is_enough():
    # Mixed list: one Arabic entity, one Latin -> accepted (policy needs >=1 Latin entity).
    assert has_latin_entity_span(_ents("دولار", "March"))


def test_lowercase_common_nouns_still_rejected():
    # The capitalisation rule's REAL value: the LLM validator sometimes returns common nouns as
    # 'entities'. Exempting currency must not also admit these. (Observed in expN_v2 output.)
    for word in ("nutrition", "wellness", "mindfulness", "diabetes", "yoga", "water", "podcasts"):
        assert not has_latin_entity_span(_ents(word)), word


def test_multiword_lowercase_junk_rejected():
    assert not has_latin_entity_span(_ents("my mental health"))


def test_capitalised_multiword_accepted():
    assert has_latin_entity_span(_ents("Eid al-Fitr"))
    assert has_latin_entity_span(_ents("Manchester United"))


def test_documents_the_old_heuristics_blind_spot():
    # Why the fix was needed: the old text-based test rejects a sentence whose only entity is a
    # lowercase currency word, even though that entity is plainly Latin script.
    sent = "أتابع دوري كرة القدم المحلي، and the tickets are priced at one hundred pounds."
    assert not has_latin_entity_candidate(sent)       # old heuristic: rejects (the bug)
    assert has_latin_entity_span(_ents("pounds"))     # span test: accepts (correct)


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
