# Weak Topic Primary + Agents — the Topic Analog of C3

XLM-R trained on **generated topic-540** (weak primary) + reason-first/sharpened agents, on a
495-sample stratified subsample of real ARENTCV2 test. threshold 0.90, gpt-4o-mini. Date: 2026-07-15.

## Result
| system | accuracy | macro F1 |
|---|---|---|
| weak primary (gen-540 trained) | 0.7717 | 0.7552 |
| **+ agents (reasoned+sharpened)** | **0.9232** | **0.9232** |
| **gain** | **+0.1515** | **+0.168** |

- 100% escalation (weak primary is very unconfident) → agents decided all 495.
- Essentially clean: 15 connection errors / 1925 calls = 0.78% (transient; retry absorbed).
- Full-set T-gen540 primary (21,134) = 0.7653; subsample primary = 0.7717 (representative).

## Meaning — topic now mirrors sentiment
Agents help most when the primary is weak:
- Real-trained topic primary (0.99) → agents can't help (primary > agents' own topic ability).
- **Weak generated-topic primary (0.77) → agents help +0.15**, lifting toward the agent ceiling.
- **Topic agent-ceiling ~0.92** (agents reach 0.9232) — much higher than sentiment's ~0.77,
  because topic classification is easier for the LLM agents than code-switched sentiment.

So the weak-primary regime is where agents add value on BOTH tasks; the difference is the
ceiling (topic 0.92 vs sentiment 0.77).

## Artifacts
- `experiment_Tgen540_agentic_sub/` ; weak primary `experiment_T_gen540_xlmr/` (0.7653 full);
  subsample `data/Topic/processed/ARENTCV2/test_sub500.jsonl`.

## Complete backbone × primary-strength 2x2 (topic)
| backbone | strong (real ARENTC) | weak (generated-540) |
|---|---|---|
| XLM-R | 0.9947 → 0.9947 (+0.000) | 0.7717 → 0.9232 (+0.152) |
| mBERT | 0.9923 → 0.9931 (+0.001) | 0.6525 → 0.9212 (+0.269) |

**Backbone-independent agent ceiling ~0.92:** weak XLM-R (0.77) and weak mBERT (0.65) BOTH
reach ~0.92 with agents — the agents contribute a fixed reasoning ceiling (they use only the
primary's escalated predictions, not its weights), so at 100% escalation they reach their own
topic ability regardless of the primary. The GAIN = distance from that ceiling: mBERT +0.269 >
XLM-R +0.152 (weaker primary → bigger gain). Topic agent-ceiling ~0.92 vs sentiment ~0.77
(topic is easier for the LLM agents). mBERT-540 agentic: 33/1850 conn-errors (1.8%, ~clean).
