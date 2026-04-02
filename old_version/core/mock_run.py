"""Simple mock runner for SwitchLingua — no LLM calls.

Usage:
    python core/mock_run.py

This script creates a fake scenario, runs any registered MCP tools
from `core/mcp_tools.py`, computes a simple weighted score, and
prints + saves the result to `output/mock_run_result.jsonl`.
"""
import json
import os
import sys
from datetime import datetime

# Ensure repo root is on sys.path so `core` can be imported when running
# this script as `python core/mock_run.py` (sys.path[0] becomes core/).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from core.mcp_tools import get_all_tools
except Exception:
    # typing.Protocol may be unavailable on older Python versions (e.g., 3.7).
    # Fall back to an empty registry so the mock runner still works.
    def get_all_tools():
        return {}

# Import the new hybrid CS ratio calculator
from core.cs_ratio_calculator import compute_cs_ratio, calculate_ratio_score


def weighting_scheme(state: dict) -> float:
    fluency = state["fluency_result"]["fluency_score"]
    naturalness = state["naturalness_result"]["naturalness_score"]
    csratio = state["cs_ratio_result"]["ratio_score"]
    socio = state["social_cultural_result"]["socio_cultural_score"]
    return fluency * 0.3 + naturalness * 0.25 + csratio * 0.2 + socio * 0.25


def main():
    # Minimal fake scenario/state following AgentRunningState fields
    state = {
        "topic": "sports",
        "tense": "Present",
        "perspective": "Third Person",
        "cs_ratio": "30%",
        "gender": "Female",
        "age": "26-35",
        "education_level": "College",
        "first_language": "Arabic",
        "second_language": "English",
        "conversation_type": "single_turn",
        "cs_function": "Expressive",
        "cs_type": "Intersentential",
        # generated instances (code-switched samples)
        "data_generation_result": [
            "الفريق بدأ المباراة بشكل قوي جدًا وحقق تقدم كبير في البداية. But then the defense couldn't keep up and things started to fall apart.",
            "الجمهور كان متحمس جدًا في الربع الأول. The mood shifted quickly after halftime when the game got tense.",
            "الخسارة كانت صعبة على الجميع، خصوصًا بعد الأداء القوي في البداية. It's frustrating to see them lose after such a promising start."
        ],
        # fake evaluator outputs
        "fluency_result": {
            "fluency_score": 9.0,
            "errors": {},
            "summary": "High fluency; sentence-level switches are natural."
        },
        "naturalness_result": {
            "naturalness_score": 8.5,
            "observations": {},
            "summary": "Sounds natural for bilingual speakers."
        },
        # UPDATED: Use hybrid approach for CS ratio
        "cs_ratio_result": None,  # Will be computed below
        "social_cultural_result": {
            "socio_cultural_score": 9.0,
            "issues": "",
            "summary": "No cultural problems detected."
        },
        "refine_count": 0,
    }
    
    # HYBRID CS RATIO CALCULATION
    print("\n=== Hybrid CS Ratio Calculation ===")
    ratio_data = compute_cs_ratio(
        state["data_generation_result"],
        state["first_language"],
        state["second_language"]
    )
    
    ratio_score = calculate_ratio_score(
        ratio_data["lang2_percent"],
        state["cs_ratio"]
    )
    
    state["cs_ratio_result"] = {
        "ratio_score": ratio_score,
        "computed_ratio": ratio_data["computed_ratio"],
        "notes": f"Target: {state['cs_ratio']}, Actual: {ratio_data['lang2_percent']:.1f}%. {ratio_data['details']}"
    }
    
    print(f"Target ratio: {state['cs_ratio']}")
    print(f"Computed ratio: {ratio_data['computed_ratio']}")
    print(f"Ratio score: {ratio_score:.2f}")
    print(f"Details: {ratio_data['details']}\n")

    # Run MCP tools (if any) and merge results
    mcp_results = {}
    tools = get_all_tools()
    for name, tool in tools.items():
        try:
            mcp_results.update(tool.run(state))
        except Exception as e:
            mcp_results[name] = f"ERROR: {e}"

    state["mcp_result"] = mcp_results

    # compute final score
    state["score"] = weighting_scheme(state)
    state["summary"] = f"Mock run produced {len(state['data_generation_result'])} instances."

    # ensure output folder exists
    os.makedirs("output", exist_ok=True)

    out_path = os.path.join("output", "mock_run_result.jsonl")
    # append to jsonlines file
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(state, ensure_ascii=False) + "\n")

    print("Mock run complete")
    print("Score:", state["score"])
    print("MCP results:", json.dumps(mcp_results, ensure_ascii=False, indent=2))
    print(f"Saved output to {out_path}")


if __name__ == "__main__":
    main()
