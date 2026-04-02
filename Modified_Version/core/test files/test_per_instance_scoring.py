import os
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORE_DIR = os.path.join(REPO_ROOT, "drive_code", "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from utils import weighting_scheme


def test_weighting_scheme_per_instance_average() -> float:
    state = {
        "fluency_results_per_instances": [{"fluency_score": 8}, {"fluency_score": 6}],
        "naturalness_results_per_instances": [
            {"naturalness_score": 9},
            {"naturalness_score": 7},
        ],
        "cs_ratio_results_per_instances": [{"ratio_score": 4}, {"ratio_score": 6}],
        "social_cultural_results_per_instances": [
            {"socio_cultural_score": 10},
            {"socio_cultural_score": 8},
        ],
    }

    score = weighting_scheme(state)

    # avg(7)*0.3 + avg(8)*0.25 + avg(5)*0.2 + avg(9)*0.25 = 7.35
    expected = 7.35
    assert abs(score - expected) < 1e-9, f"expected {expected}, got {score}"
    return score


def test_weighting_scheme_fallback_to_aggregate() -> float:
    state = {
        "fluency_result": {"fluency_score": 8},
        "naturalness_result": {"naturalness_score": 6},
        "cs_ratio_results_per_instances": [{"ratio_score": 5}, {"ratio_score": 5}],
        "social_cultural_result": {"socio_cultural_score": 9},
    }

    score = weighting_scheme(state)

    expected = 8 * 0.3 + 6 * 0.25 + 5 * 0.2 + 9 * 0.25
    assert abs(score - expected) < 1e-9, f"expected {expected}, got {score}"
    return score


def main() -> None:
    print("Running per-instance scoring tests...")

    score1 = test_weighting_scheme_per_instance_average()
    print(f"PASS test_weighting_scheme_per_instance_average -> {score1}")

    score2 = test_weighting_scheme_fallback_to_aggregate()
    print(f"PASS test_weighting_scheme_fallback_to_aggregate -> {score2}")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
