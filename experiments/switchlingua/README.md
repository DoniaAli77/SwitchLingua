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
├── system_a_original_gpt4o/   ← System A JSONL outputs
├── system_b_modified_mini/    ← System B JSONL outputs
├── system_c_original_mini/    ← System C JSONL outputs
├── human_eval/                ← Human evaluation sheets (CSV/XLSX)
├── ablations/                 ← Ablation run outputs
├── csratio/                   ← CS ratio validation outputs
├── refinement/                ← Refinement strategy comparison outputs
├── per_sentence/              ← Per-sentence vs scenario analysis outputs
├── cost_quality/              ← Token cost vs quality score outputs
└── failure_analysis/          ← Failure case analysis outputs
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

## Design Principles

- Experiment scripts call existing pipeline code where possible and do not
  duplicate logic that already exists in `core/`.
- Core pipeline logic (`Original_baseLine/core/` and `Modified_Version/core/`)
  must **not** be changed by experiment scripts. A thin wrapper (monkey-patch
  of `node_engine.MODEL` and `node_engine.OUTPUT_DIR`) is the only permitted
  form of integration.
- Legacy/sample outputs (`Original_baseLine/output/`, `Original_baseLine/Sample/`,
  any `Modified_Version/output/` folder) are **not** used for thesis results.
  All fresh results are written under `experiments/outputs/switchlingua/`.

---

## Import Convention

Neither `Original_baseLine/core/` nor `Modified_Version/core/` contain
`__init__.py`, so package-style imports do **not** work at runtime.
Scripts must insert the target core directory onto `sys.path` and use bare
module imports:

```python
import sys
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[2]
BASELINE_CORE = ROOT / "Original_baseLine" / "core"
MODIFIED_CORE = ROOT / "Modified_Version"  / "core"

# ---- to use Original_baseLine ----
sys.path.insert(0, str(BASELINE_CORE))
from run_french import CodeSwitchingAgent as BaselineAgent   # bare import
from utils      import load_config, generate_scenarios       # bare import
import node_engine as _ne
_ne.MODEL      = "gpt-4o"          # or "gpt-4o-mini" for System C
_ne.OUTPUT_DIR = str(ROOT / "experiments" / "outputs" / "switchlingua" / "system_a_original_gpt4o")

# ---- to use Modified_Version ----
sys.path.insert(0, str(MODIFIED_CORE))
from run_french import CodeSwitchingAgent as ModifiedAgent   # bare import
from utils      import load_config, generate_scenarios, compute_true_cs_stats
import node_engine as _ne
_ne.MODEL      = "gpt-4o-mini"
_ne.OUTPUT_DIR = str(ROOT / "experiments" / "outputs" / "switchlingua" / "system_b_modified_mini")
```

See `pipeline_wrappers.py` for the `_activate_core()` helper that handles
module eviction when both cores need to be used in the same process.
