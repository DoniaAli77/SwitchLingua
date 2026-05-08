# SwitchLingua Thesis Experiments

This directory contains experiment scripts for the thesis comparison study.
Core pipeline logic lives in `Original_baseLine/core/` and `Modified_Version/core/`
and must **not** be modified by these scripts.

---

## Systems Under Comparison

| Label | Pipeline | Model | Notes |
|---|---|---|---|
| **System A** | `Original_baseLine` | GPT-4o | Original NeurIPS 2025 submission |
| **System B** | `Modified_Version` | GPT-4o-mini | Task-aware, per-sentence scoring, deterministic CS ratio |
| **System C** | `Original_baseLine` | GPT-4o-mini | Model control — same code as A, cheaper model |

System C isolates the effect of the model switch from the effect of architectural changes.
Comparing A vs C shows the model effect; comparing C vs B shows the architecture effect.

---

## Output Directory

All experiment outputs go under:

```
experiments/outputs/switchlingua/
├── system_a/          ← System A JSONL outputs
├── system_b/          ← System B JSONL outputs
├── system_c/          ← System C JSONL outputs
├── human_eval/        ← Human evaluation sheets (CSV/XLSX)
├── ablation/          ← Ablation run outputs
├── csratio/           ← CS ratio validation outputs
└── per_sentence/      ← Per-sentence vs scenario analysis outputs
```

---

## Script Overview

| Script | Purpose |
|---|---|
| `run_full_pipeline_generation.py` | Run System A, B, C and write JSONL outputs |
| `run_model_control.py` | Run System C specifically (Original code + GPT-4o-mini) |
| `create_human_eval_sheets.py` | Sample sentences from A/B/C outputs; write human eval XLSX |
| `analyze_human_eval.py` | Load completed human eval sheets; compute scores and agreement |
| `score_system_comparison.py` | Load A/B/C outputs; compute automated metric tables |
| `run_csratio_validation.py` | Compare deterministic CS stats vs LLM-reported ratio per sentence |
| `run_llm_ratio_repeats.py` | Re-run the same scenarios multiple times; measure LLM ratio consistency |
| `run_ablation.py` | Run Modified_Version with components selectively disabled |
| `score_ablation.py` | Load ablation outputs; compute score tables across ablation conditions |
| `run_refinement_strategy.py` | Compare per-sentence vs scenario-level refinement strategies |
| `analyze_per_sentence_vs_scenario.py` | Quantify benefit of per-sentence scoring granularity |
| `analyze_cost_quality.py` | API token cost vs quality score analysis across systems |
| `analyze_failures.py` | Analyse failure cases: low scores, validation failures, CS ratio mismatches |

---

## Running

```bash
# Generate outputs for all three systems
python experiments/switchlingua/run_full_pipeline_generation.py

# Score and compare systems
python experiments/switchlingua/score_system_comparison.py

# Run ablation
python experiments/switchlingua/run_ablation.py
python experiments/switchlingua/score_ablation.py
```

---

## Import Convention

Scripts add the workspace root to `sys.path` so they can reach both core packages:

```python
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Original_baseLine core
from Original_baseLine.core.agents import CodeSwitchingAgent as BaselineAgent
from Original_baseLine.core.utils  import load_config as baseline_load_config, generate_scenarios as baseline_scenarios

# Modified_Version core
from Modified_Version.core.agents import CodeSwitchingAgent as ModifiedAgent
from Modified_Version.core.utils  import load_config, generate_scenarios, compute_true_cs_stats
```
