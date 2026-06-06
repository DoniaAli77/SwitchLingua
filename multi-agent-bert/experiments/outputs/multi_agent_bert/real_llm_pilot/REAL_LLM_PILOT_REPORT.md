# Real-LLM full_agentic Pilot (GPT-4o-mini)

**Real results.** Tests whether *real* LLM specialist agents help on the
low-confidence escalated subset, where the earlier `full_agentic` result used
`MockLLMClient` only. Run date: 2026-06-06.

Config: `pipeline_mode full_agentic`, `primary_model transformer`,
`llm_client openai`, `llm_model gpt-4o-mini`, **router threshold 0.6 (unchanged)**,
EESA test (818). LLM agents fire only on escalated samples via the router.
paper_style agents, the router threshold, and the fine-tuned checkpoints were
**not** changed; SwitchLingua untouched.

## Headline

| Model | primary_only | mock full_agentic | **real-LLM full_agentic** |
|---|---|---|---|
| **mBERT** acc | 0.7971 | 0.7958 | **0.8166** |
| **mBERT** macro F1 | 0.7833 | 0.7788 | **0.8038** |
| **XLM-R** acc | 0.8240 | 0.8130 | **0.8399** |
| **XLM-R** macro F1 | 0.8088 | 0.7973 | **0.8264** |

**Real LLM agents help; the mock hurt.** For both models the real-LLM pipeline
beats both `primary_only` and the mock `full_agentic`:
- mBERT: **+0.0195 acc / +0.0205 macro F1** over primary_only.
- XLM-R: **+0.0159 acc / +0.0176 macro F1** over primary_only.

## Cost / usage (actual, from the API)

| Model | escalated | OpenAI calls | prompt tok | completion tok | **cost** |
|---|---|---|---|---|---|
| mBERT | 47 | 186 | 104,279 | 12,653 | **$0.0232** |
| XLM-R | 41 | 164 | 92,076 | 11,389 | **$0.0206** |
| **Total** | 88 | **350** | 196,355 | 24,042 | **$0.0438** |

(~4 calls per escalated sample. gpt-4o-mini list price $0.15/$0.60 per 1M.)
**Structured output: 0 parse warnings** (JSON mode `response_format=json_object`)
— the explainability parse errors that plagued the mock are gone.
**1 transient connection error** (mBERT) absorbed by SDK retry → fell back to
primary for that one sample; `error_samples = 0` in both runs.

## Escalated-subset deep dive (where the agents actually act)

| Model | esc n | primary acc on esc | **real-LLM agent acc on esc** | W→C | C→W | **net** |
|---|---|---|---|---|---|---|
| mBERT | 47 | 0.4255 | **0.7660** | 21 | 5 | **+16** |
| XLM-R | 41 | 0.4146 | **0.7317** | 15 | 2 | **+13** |

For comparison, the **mock** agents on the same escalated subset scored
**0.4043 (mBERT, net −1)** and **0.2195 (XLM-R, net −8)** — i.e. worse than the
primary. The real LLM agents nearly **double** the agent accuracy on the
escalated slice and turn the net strongly positive.

## Per-class effects (the negative class is recovered)

Real-LLM full_agentic per-class F1 (test):

| Label | mBERT | XLM-R |
|---|---|---|
| positive | 0.878 | 0.906 |
| negative | 0.759 | 0.792 |
| neutral | 0.774 | 0.781 |

Escalated-subset change by true class (C→W / W→C, and where agents send each):

- **Negative — the mock's worst failure — is fixed.**
  - mBERT escalated negatives (16): agents predict `negative` 15× → **W→C 7, C→W 1, net +6**. (Mock: all→positive, W→C 0, net −9.)
  - XLM-R escalated negatives (16): `negative` 14× → **W→C 7, C→W 0, net +7**. (Mock: net −7.)
- **Neutral** improves modestly (mBERT net +2, XLM-R net +3) — no longer dumped into positive.
- **Positive** still gains (mBERT net +8, XLM-R net +3) but now through real discrimination, not a majority-class default.

The mock collapsed everything to `positive` (negative W→C was 0 in every cell);
the real LLM agents instead predict the *correct* minority labels, which is why
macro F1 rises faster than accuracy.

## Four-way comparison (threshold 0.6, EESA test)

| Mode | mBERT acc / macroF1 | XLM-R acc / macroF1 |
|---|---|---|
| primary_only | 0.7971 / 0.7833 | 0.8240 / 0.8088 |
| paper_style | 0.7958 / 0.7788 | 0.8142 / 0.7983 |
| mock full_agentic | 0.7958 / 0.7788 | 0.8130 / 0.7973 |
| **real-LLM full_agentic** | **0.8166 / 0.8038** | **0.8399 / 0.8264** |

Weighted F1 (real-LLM): mBERT 0.8167, XLM-R 0.8392.

## Interpretation

- The threshold sweep concluded the agents were the bottleneck — confirmed: with
  a *capable* contextual agent (GPT-4o-mini) instead of the mock, `full_agentic`
  goes from **hurting** to **helping**, on both primaries, for **~$0.02 each**.
- The benefit is concentrated exactly on the escalated low-confidence subset and
  on the **negative** class — the primary's weakest spot — which is the ideal
  place for a multi-agent escalation design to add value.
- This is the first result where the multi-agent architecture beats the
  fine-tuned primary on the real EESA benchmark.

## Scope / caveats

- Pilot only: threshold 0.6, both checkpoints, EESA test. **No threshold sweep
  with the real LLM** (not run, per instruction).
- Deliberation stays off (config). Only the 4 LLM agents
  (lexical/logic/contextual/explainability) used GPT-4o-mini.
- 1 sample (mBERT) fell back to primary on a transient connection error.
- Outputs per model: `real_llm_pilot/<model>/` (`*_metrics.{json,csv}`,
  `*_predictions.{json,csv}`, `pilot_<model>__llm_usage.json`); full run log
  `real_llm_pilot/real_llm_pilot.log`.

## Default unchanged
`--llm_client` defaults to `mock`; full test suite stays offline at **859
passed**. The real client is reached only via `--llm_client openai`.
