**CS Ratio Changes Summary**

- **Purpose:** Fix unreliable LLM-based code-switching ratio counting by adding a deterministic calculator, clarifying prompt semantics, and integrating a hybrid agent (deterministic numbers + LLM interpretation).

**What I changed**
- **Deterministic calculator:** Added/updated `core/cs_ratio_calculator.py` to detect word language using Unicode ranges and compute an accurate embedded (second-language) percent. `computed_ratio` is formatted with the second language first (e.g., "30.0% English : 70.0% Arabic").
- **Scoring:** `calculate_ratio_score()` converts the numeric deviation to a 0–10 score programmatically.
- **Agent orchestration:** `core/node_engine.py` (`RunCSRatioAgent`) now:
  - calls `compute_cs_ratio()` to get accurate counts,
  - computes `ratio_score` deterministically using `calculate_ratio_score()` (this score overrides any LLM score),
  - passes `computed_ratio`, `actual_percent`, and explicit `target_second_percent` / `target_first_percent` into the LLM prompt so the LLM can write semantic notes (but not change the numeric score).
- **Prompt updates:** `core/prompt.py` `CS_RATIO_PROMPT` now documents that `cs_ratio` refers to the embedded / second language percentage and accepts explicit `{target_second_percent}` and `{target_first_percent}` variables to avoid misinterpretation.
- **Runner fixes:** `core/run_french.py` was patched to safely handle missing webhook and ensure `mcp_result` is present in the initial state.

**Why this design**
- Accuracy: deterministic counting avoids LLM hallucination when computing percentages.
- Explainability: LLM supplies human-readable notes and suggestions without affecting numeric scoring.
- Reproducibility: same input always yields the same numeric ratio and score.

**Commands used (reproduce)**
```powershell
# Activate venv and run full pipeline
.\.venv\Scripts\Activate.ps1
python -u core/run_french.py

# Convert JSONL to Excel (helper)
python convertToExcel.py
```

**Files produced**
- `output/Arabic.jsonl` — scenario JSON lines
- `output/Arabic_analysis.xlsx` — sentence-level rows
- `output/Arabic.xlsx` — scenario-level rows

**Notable observations from runs**
- `cs_ratio` in config is interpreted as the second-language (embedded) percentage. The agent computes `actual_percent` as the embedded-language percent and compares against that target.
- Before clarifying, notes sometimes stated the target with reversed languages; with the prompt + node changes this is now consistent.

**Next steps / options**
- If you want sentence-level cs ratios, call `compute_cs_ratio()` per sentence and save those fields into state.
- If you'd like the LLM to influence numeric scoring, modify `RunCSRatioAgent` to merge or accept `llm_response['ratio_score']` (currently overridden by deterministic score).
- I can add `target_first_percent` and `target_second_percent` as explicit fields in the saved JSONL for easier analysis.

**Contact / notes**
- I ran multiple iterations and regenerated Excel files after each patch. If you want, I can open and paste the first 12 rows from `output/Arabic_analysis.xlsx` here.

-- End of summary
