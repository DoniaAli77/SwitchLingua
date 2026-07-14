# Topic Classification Pipeline — Implementation Audit

Correctness & reproducibility audit of the ARENTC topic classification path in the Multi-Agent
BERT framework. Evidence-based (code + logs + artifacts). No behavior modified. Date: 2026-07-07.

## 1. Executive summary
**The topic classification path is correctly wired and previous results are valid.**
- Real LLM usage is **PROVEN** for every topic full_agentic run (client warning in logs + counted
  `POST api.openai.com … 200 OK` calls + token/cost usage JSON).
- Reported numbers are **exactly reproducible** from saved predictions (recomputed = reported for
  both primary_only and full_agentic, ARENTCV1 + ARENTCV2).
- No split leakage, correct 9-label mapping, correct XLM-R topic checkpoint, task-generic prompts
  (no sentiment/EESA/Ahmed wording), and a task-generic consensus (no hardcoded sentiment labels).
- **No bugs found.** Two minor non-blocking observations (§13) documented for safety.

## 2. Files inspected
`src/pipeline/router.py`, `src/pipeline/orchestrator.py`, `src/agents/consensus_agent.py`,
`src/agents/intent_gate_agent.py`, `src/models/primary_transformer_classifier.py`,
`src/prompts/{llm_lexical,llm_logic,contextual}_prompt.py`, `evaluate_pipeline.py`,
`src/config/default.yaml` + `src/config/topic_disambig.yaml`, `src/config/loader.py`,
`src/llm/openai_client.py`; artifacts under `experiments/outputs/multi_agent_bert/experiment_T1_*`,
`experiment_T2_*`; checkpoint `experiments/checkpoints/topic_arentcv2_xlmr/`.

## 3. Topic path architecture (verified)
`transformer primary (XLM-R, 9-class) → Router (conf < threshold ⇒ escalate) → [escalate:
LLMLexical + LLMLogic + Contextual → ConsensusAgent] → metrics`. IntentGate is **NOT** wired for
topic (sentiment-only; `intent_gate_agent=None`). Non-escalated samples take the primary's label
directly.

## 4. Dataset & label audit — PASS
| dataset | train | dev | test |
|---|---|---|---|
| ARENTCV1 | 73,976 | 10,569 | 21,137 |
| ARENTCV2 | 73,956 | 10,562 | **21,134** (matches reports) |

- Labels: exactly **9** — business, education, finance, health, medical, shopping, social, sports,
  tech. Rows are `{id, text, label}`.
- **Leakage: none.** ARENTCV2 id-overlap train∩test=0, dev∩test=0, train∩dev=0; **text-overlap
  train∩test=0**.
- No sentiment labels present in topic data.

## 5. Primary classifier audit — PASS
- Checkpoint `topic_arentcv2_xlmr/config.json`: `model_type=xlm-roberta`, **num_labels=9**,
  `id2label` = the 9 topic labels (alphabetical). `label_map.json` present and consistent.
- Decoding: `PrimaryTransformerClassifier` uses `label_map` if supplied, else **falls back to the
  checkpoint's own `config.id2label`** (code L225-230). The CLI passes **no** label_map, so the
  checkpoint mapping is used → predictions decoded with **topic** labels, not sentiment.
- **Proof of correct decoding:** primary_only accuracy = **0.9947** (a mis-ordered map would score
  ~1/9 ≈ 0.11). Confidence = softmax max-prob (L262/312).

## 6. Router audit — PASS
- `router.py`: `decision = "escalate" if primary.confidence < threshold else "accept_primary"`
  (task-generic; no sentiment inheritance). Topic runs used **threshold 0.90**.
- High-confidence samples bypass agents (get the primary label); low-confidence escalate.
- **Escalation count reproduces:** ARENTCV2 full run = **48/21,134** escalated (matches the metrics
  JSON `escalated_count=48`).

## 7. Prompt audit — PASS
- Topic user prompt: `task_name=topic_classification`; **contains all 9 topic labels**; **contains
  no sentiment labels** (positive/negative/neutral); **no EESA/Ahmed** wording.
- System prompts are **task-generic** — labels are injected from `task_config.labels`, not
  hardcoded. The reason-first system prompts (`AGENT_PROMPT_STYLE=reasoned`) contain **no**
  positive/negative/neutral and **no** eesa/ahmed/sentiment strings.
- The sentiment `semantic_v*` prompt variants are gated by `SENTIMENT_PROMPT_VARIANT` and are **not
  active** for topic; topic used `default` or the opt-in reason-first style.

## 8. Agent wiring audit — PASS
- Topic full_agentic builds the **default trio** (LLMLexical + LLMLogic + Contextual) + Consensus —
  general classifier agents, not sentiment specialists (no Polarity/Intent/gate for topic).
- Roles/prompts are appropriate and generic for classification.

## 9. LLM-call audit — PASS (real calls PROVEN)
Every topic full_agentic run logs `WARNING … Using REAL LLM client (openai, model=…)` and shows
counted OpenAI HTTP calls:

| run | model | api.openai calls | usage JSON |
|---|---|---|---|
| T2 ARENTCV2 (default agents) | gpt-4o-mini | **192** | (log-proven; pre-usage-tracking) |
| T2 @ gpt-4.1-mini | gpt-4.1-mini | **192** | calls=192 (cost $0.00 = missing price row, not missing calls) |
| T2 disambig-48 | gpt-4o-mini | — | calls=192, $0.035 |
| T2 reasoned-48 | gpt-4o-mini | — | calls=192, $0.037 |
| **T2 reasoned-FULL** | gpt-4o-mini | **187** | calls=187, 195,900 tok, **$0.036** |
| T1 ARENTCV1 (default agents) | gpt-4o-mini | **252** | (log-proven) |

- **Mock vs real is cleanly separated:** the pipeline emits an explicit REAL-client WARNING; mock is
  only used when `--llm_client` is not `openai`. All topic experiments passed `--llm_client openai`.
- Parser outputs are LLM-shaped (varied labels + reasoning), not deterministic mock echoes.
- **Real LLM usage IS proven** from logs + call counts + usage JSON. (Older runs T1/T2-default have
  no `llm_usage.json` — usage tracking was added later — but their logs prove 252 / 192 real calls.)

## 10. Consensus audit — PASS
- `ConsensusAgent` is fully **task-generic**: scores accumulate over `task_config.labels`; no
  hardcoded `positive/negative/neutral` (grep clean — the only "negative" hits are the
  weight-clamping docstring/comment).
- 9-class aggregation works (weighted vote over 9 labels). Tie-break `_select_winner` is
  non-positional (primary-anchor → most voters → highest contribution → **alphabetical**), no
  3-class assumption. Abstain/no-vote defers to the primary label (never `labels[0]`).
- IntentGate decoupled and **not used** for topic.

## 11. Evaluation audit — PASS
- Metrics computed over all **9** classes; `escalation_rate`/`escalated_count`/`escalated_accuracy`
  present and consistent.
- **Reproducibility (recomputed from saved predictions vs reported):**
  - ARENTCV2 primary_only: reported 0.9947 = **recomputed 0.9947** (n=21,134, escalated 0).
  - ARENTCV2 full_agentic (reasoned): reported 0.9947 = **recomputed 0.9947** (escalated 48).
  - ARENTCV2 full_agentic (default): reported 0.9944 = **recomputed 0.9944**.
- W→C / C→W recomputable from predictions + primary_only predictions (used in prior reports).

## 12. Output-artifact audit — PASS
Each run saved `*_metrics.json` + `*_predictions.{json,csv}` (+ `*_llm_usage.json` for recent runs)
under `experiment_T{1,2}_*`. Reported metrics match recomputation exactly (§11). Reports and run
logs are present and consistent.

## 13. Bugs found / risks / uncertainties
- **Bugs: none.**
- **Observation A (non-blocking):** the pipeline config's label *order* (business, education,
  health, shopping, medical, sports, tech, finance, social) differs from the checkpoint's
  alphabetical `id2label`. This is **safe** because the transformer decodes by the checkpoint's own
  `id2label` (label_map defaults to None) and consensus/eval operate on label *strings*, not
  indices — **verified** by the 0.9947 reproduction. *Risk only if* someone ever passes a
  config-ordered `label_map` to the transformer; currently nothing does.
- **Observation B (non-blocking):** the two oldest runs (T1, T2-default) predate `llm_usage.json`,
  so cost isn't saved for them — but their logs prove 252 / 192 real OpenAI calls, so real usage is
  still established.
- **Observation C (cosmetic):** the gpt-4.1-mini usage JSON shows `est_cost_usd=0.0` because the
  local price table lacks a 4.1-mini row; `calls=192` confirms the calls were real.

## 14. Required fixes
None required for correctness or reproducibility. *Optional hardening* (not needed): add a
gpt-4.1-mini row to the cost table (Obs C); assert `label_map is None or == checkpoint.id2label`
when loading a transformer primary (Obs A) as a guardrail.

## 15. Are the previous topic results trustworthy?
**Yes.** Primary_only 0.9947 and full_agentic 0.9944 (default) / 0.9947 (reasoned) on ARENTCV2 are
reproducible from saved predictions, computed over the correct 9 labels, on a leak-free test split,
with **proven real LLM calls** on the escalated subset. The reason-first / sharpened-description
improvements were run with real gpt-4o-mini (187 calls, $0.036).

## 16. Exact rerun commands (real LLM)
Load the OpenAI key first (bash): `set -a; source ../Modified_Version/.env; set +a`  (or export
`OPENAI_API_KEY`). Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

**A. Topic primary_only (no LLM, GPU only):**
```
python evaluate_pipeline.py \
  --config src/config/default.yaml --active_task topic_classification \
  --dataset data/Topic/processed/ARENTCV2/test.jsonl \
  --mode full_pipeline --pipeline_mode primary_only \
  --primary_model transformer \
  --transformer_checkpoint experiments/checkpoints/topic_arentcv2_xlmr \
  --transformer_device cuda \
  --output_dir experiments/outputs/multi_agent_bert/rerun_T2_primary --run_id T2_primary
```

**B. Topic full_agentic with REAL LLM (default agents):**
```
python evaluate_pipeline.py \
  --config src/config/default.yaml --active_task topic_classification \
  --dataset data/Topic/processed/ARENTCV2/test.jsonl \
  --mode both --pipeline_mode full_agentic --threshold 0.90 \
  --primary_model transformer \
  --transformer_checkpoint experiments/checkpoints/topic_arentcv2_xlmr \
  --transformer_device cuda \
  --llm_client openai --llm_model gpt-4o-mini \
  --output_dir experiments/outputs/multi_agent_bert/rerun_T2_full --run_id T2_full
```

**C. Topic full_agentic with the reason-first + sharpened improvements:**
```
export AGENT_PROMPT_STYLE=reasoned
python evaluate_pipeline.py \
  --config src/config/topic_disambig.yaml --active_task topic_classification \
  --dataset data/Topic/processed/ARENTCV2/test.jsonl \
  --mode both --pipeline_mode full_agentic --threshold 0.90 \
  --primary_model transformer \
  --transformer_checkpoint experiments/checkpoints/topic_arentcv2_xlmr \
  --transformer_device cuda \
  --llm_client openai --llm_model gpt-4o-mini \
  --output_dir experiments/outputs/multi_agent_bert/rerun_T2_reasoned --run_id T2_reasoned
```
(Proof of real calls: the run log will print `Using REAL LLM client (openai …)` and
`*_llm_usage.json` will record `calls` and `est_cost_usd`.)
