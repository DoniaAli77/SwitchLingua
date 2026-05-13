# Thesis Experiment Outputs

This folder contains **fresh thesis experiment outputs only**.

Legacy sample/test outputs located in `Original_baseLine/output/`,
`Original_baseLine/Sample/`, and `Modified_Version/output/` are **not used**
for final results. They were generated during development and should be
ignored for any analysis or comparison reported in the thesis.

All new experiment runs must write their outputs into the subfolders below.

---

## Subfolder layout

| Subfolder | Purpose |
|---|---|
| `system_a_original_gpt4o/` | System A: Original_baseLine code + GPT-4o |
| `system_b_modified_mini/` | System B: Modified_Version code + GPT-4o-mini |
| `system_c_original_mini/` | System C: Original_baseLine code + GPT-4o-mini (model control) |
| `human_eval/` | Human evaluation annotation sheets and aggregated scores |
| `csratio/` | CS ratio validation outputs (deterministic vs LLM-reported) |
| `ablations/` | Ablation study outputs (components selectively disabled) |
| `refinement/` | Refinement strategy comparison outputs |
| `per_sentence/` | Per-sentence vs scenario-level scoring analysis outputs |
| `cost_quality/` | Token cost vs quality score analysis outputs |
| `failure_analysis/` | Tagged and categorised failure case outputs |

Each subfolder contains a `.gitkeep` placeholder until real outputs are
generated. Replace or supplement `.gitkeep` with actual JSONL/CSV/XLSX
outputs produced by the scripts in `experiments/switchlingua/`.
