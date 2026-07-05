# Design G — Ahmed Failure-Case Analysis (18 escalated errors)

**Analysis only — no LLM calls.** Breaks down every failed escalated case of Design G
(Lexical + Polarity + Contextual + IntentGate) on the Ahmed frozen primary, from the saved
attribution table (labels + votes + gate + true). 84 escalated, **66 correct / 18 wrong**
(the other 40 full-set errors are non-escalated primary errors the agents never see).
Date: 2026-07-02.

## Summary breakdown of the 18 failures
| axis | count |
|---|---|
| **C→W** (primary was right, agents broke it) | **7** |
| **W→W** (primary already wrong, agents didn't fix) | **11** |
| — of all 18: **information floor** (no agent held the truth) | **14** |
| — of all 18: **consensus loss** (some agent held truth, lost) | **4** |
| — of all 18: **gate-blocked** a correct rescue | **2** |

Confusion of the 18: `negative→neutral 6`, `neutral→negative 5`, `positive→neutral 4`,
`neutral→positive 3`. **Over-neutralization (10) and over-polarizing of true-neutrals (8)
in near-balance** — the panel has no directional bias, it fails symmetrically.

## Three failure clusters (this is the real content)

### Cluster 1 — Implicit insult / praise with no explicit cue (the information floor) — ~7 cases
The base model (all agents *and* usually the primary) cannot read Egyptian colloquial
implicit stance. Neither aggregation nor the gate can recover these — **no agent held the
truth.**
- `00008` "هوة معتز مسعود دة gay ?" — true **neg** (slur-as-question); all agents → neutral.
- `00097` "…الفيديو مفيهوش ولا راجل … 😂" (implicit "no real man") — true **neg**; all neutral.
- `00182` "انت no one" — true **neg** (put-down); all neutral, primary neutral.
- `00635` "دا هيبقا مسلسل out of season" — true **pos** (implicit praise); all neutral.
- `00642` "عاش عاش اسمك ايه في لعبة Free Fire" — true **pos** ("عاش"=bravo) + a question; all neutral.
- `00220`, `00250` — aphorism / "don't idolize idiots" → literal polarity ≠ annotated stance.

**These require pragmatic / cultural knowledge the LLM lacks** — the genuine floor. This is
the *only* place a targeted improvement (an Egyptian implicit-stance / idiom cue) could
move anything, and even then it's ~7 noisy cases.

### Cluster 2 — Platform-meta "dislike" ambiguity (express-vs-mention) — ~5 cases
The word **"dislike"** (talking ABOUT the dislike button) drags the whole panel negative,
but the annotated label is neutral (meta) — the exact express-vs-mention axis the gate
exists for, and it fires **inconsistently** here.
- `00240` "على اساس ايه اللى عاملين dislike" — true **neu**; all → negative.
- `00446` "Dislikes كتير اوي" — true **neu**; all → negative.
- `00298` "…ضل راجل = لايك … الفتوة= dislike" — true **neu**; all → negative.
- `00046` "اللي عاملين dislike عم يسمعوا…😂" — true **neg** here; lexical alone got it, lost.
- `00290` "ليه كده يا رمضان؟؟ 😏 … 🙄" — true **neu**; all → negative (emoji-driven over-call).

The gate is *meant* to neutral-protect these but its label is unstable (we measured 23/84
run-to-run gate flips), so it catches some and misses others. **This is the one cluster with
a recoverable signal — but it's the same lever G already partly exploits.**

### Cluster 3 — Gate over-blocking (G-specific own goals) — 2 cases
The IntentGate forced neutral where the agents **correctly** found a stance:
- `00021` "…ممثلة تعمل دور الممحونة … مى عمر" — lex+pol+ctx all **negative** (correct), gate
  = neutral → **blocked** → final neutral. Gate hurt.
- `00706` "…permission to dance وشعار BTS… 🙃💜💜" — lex+ctx **positive** (correct), gate
  neutral → blocked → neutral. Gate hurt.

These are the only clean **G-caused** errors. But loosening the gate to fix them
re-admits the Cluster-2 over-neutralization it prevents elsewhere — which is exactly why the
selective gate (G2) only netted +1. **Net-neutral trade; not a free fix.**

## What this says
1. **14/18 (78%) are the information floor** — no agent had the truth, so **no aggregation,
   weighting, gate, or sequential reordering can recover them.** This is the hard ceiling on
   the strong primary, made concrete.
2. **The recoverable 4** split into 2 gate-overblocks (fixing them costs elsewhere) and 2
   lone-agent-correct (low-precision to chase). Consistent with every prior consensus-loss
   result: ~4/84 recoverable, not worth a rule.
3. **The two floor clusters are exactly the two axes the whole project identified:** implicit
   colloquial stance (cultural/pragmatic) and express-vs-mention platform-meta ("dislike").
   The first needs a *stronger/knowledge-augmented model*, not agent topology; the second is
   the gate's domain and is already ~half-caught.
4. **This is why G is at ceiling on Ahmed and why v1/v2 couldn't help** — the residual errors
   are not aggregation failures, they are model-knowledge failures. It also explains v2's
   harm: giving noisy pragmatic features *more* authority amplifies Cluster-1/2 misreads.

## Cross-link to the weak primary
On C3 the agents win big (+53) because there the primary is wrong on *easy* cases the agents
*can* read — not these floor cases. The Ahmed residual (Clusters 1–2) is the genuinely hard
tail that survives on any primary; it just isn't visible on C3 because C3's escalated set is
dominated by recoverable errors.

## Artifacts
- Table: `experiment_ahmed_designG_intent_gate/error_attribution/attribution_table.json`
- Case dump: `experiments/outputs/multi_agent_bert/_G_ahmed_failures.txt`
- Related: `EXPERIMENT_AGENT_SPECIALIZATION_ANALYSIS.md` (same express-vs-mention axis),
  `EXPERIMENT_ESCALATED_CEILING_ROOT_CAUSE.md` (the floor), `EXPERIMENT_G_C3_RESULTS.md`.
