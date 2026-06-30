# Design E — Intent Agent Ablation (Lexical + Intent + Polarity + Contextual)

Does adding a narrow **Intent / stance-detection** agent improve the best 3-agent
design (C = Lexical + Polarity + Contextual)? Tested on the **same** Ahmed
frozen-primary full_agentic setup (semantic_v1 prompts, threshold 0.7, Fix-2 consensus
w_primary 1.0, gpt-4o-mini, no training, no generation). Opt-in experimental variant;
default unchanged. Date: 2026-07-01.

**Design E** = Lexical + **Intent** + Polarity + Contextual (4 votes). Intent is the
4th agent (writes the generic `polarity_output` slot, consensus 4th-slot weight 1.0);
**no new consensus logic** — it reuses the existing four-slot wiring from Design D.

The **Intent agent is not a sentiment classifier**: it decides whether the *author*
expresses their own evaluation and toward what target, mapping to positive only for
clear approval, negative only for clear criticism/insult, and **neutral** for
descriptive / quoted / platform-meta / target-ambiguous / no-stance text.

---

## 1–9. Headline metrics vs all prior designs (full test 818; escalated 84)
| design | acc | macro F1 | wtd F1 | esc-acc | W→C | C→W (harmful) | **net** | neu→neg / neu→pos | total breaks | cost/calls |
|---|---|---|---|---|---|---|---|---|---|---|
| primary_only | 0.9254 | 0.9207 | 0.9254 | 0.750 | — | — | — | — | — | — |
| A Lex+Logic+Ctx | 0.9230 | 0.9183 | 0.9202 | 0.726 | 12 | 14 | −2 | 5 / 4 | 14 | $0.050 / 336 |
| B Pol+Ctx | 0.9254 | 0.9212 | 0.9254 | 0.750 | 10 | 10 | 0 | 4 / 1 | 10 | $0.035 / 252 |
| **C Lex+Pol+Ctx** | **0.9267** | 0.9226 | 0.9266 | 0.762 | 12 | 11 | **+1** | 4 / 2 | 11 | **$0.049 / 336** |
| D Lex+Logic+Ctx+Pol | 0.9254 | 0.9211 | 0.9254 | 0.750 | 11 | 11 | 0 | 4 / 2 | 11 | $0.064 / 420 |
| **E Lex+Intent+Pol+Ctx** | **0.9267** | **0.9227** | 0.9266 | **0.762** | 11 | **10** | **+1** | **4 / 1** | **10** | $0.064 / 420 |

(items 1–9 for E: acc **0.9267**, macroF1 **0.9227**, wtdF1 **0.9266**, esc-acc **0.7619**,
W→C **11**, C→W/harmful **10**, net **+1**, neu→neg **4**, neu→pos **1**.)

**E matches C exactly on accuracy and net (+1)**, with **one fewer harmful override
(10 vs 11)** and the **fewest breaks (10, tied with B)** — but at **C's cost +30%
($0.064 vs $0.049)**, being a 4-agent/5-call config.

## 10–13. Per-agent capture (84 escalated, deterministic re-run)
> Capture has ≤1-sample temp-0 drift from the headline run (capture final = 0.7500 /
> net 0; headline = 0.7619 / net +1). Per-agent rows characterise behaviour; the
> transition counts above are the official headline run.

| agent | accuracy on escalated |
|---|---|
| lexical | 0.7262 |
| polarity | 0.7262 |
| contextual | 0.7381 |
| **intent** | **0.7143 (weakest)** |
| final consensus | 0.7500 (capture) / **0.7619 (headline)** |
| (Ahmed primary) | 0.7500 |

| agreement | value |
|---|---|
| all-4-agree (item 13) | **66.7%** |
| **Intent ↔ Polarity** (item 11) | **0.8452** |
| **Intent ↔ Contextual** (item 12) | **0.7857** |
| Intent ↔ Lexical | **0.6786** |
| Polarity ↔ Contextual | 0.9405 |

- **Intent is the weakest specialist (0.714)** — adding it did *not* add a strong agent.
- **Intent is the most decorrelated agent** (Intent↔Lexical 0.679, Intent↔Contextual
  0.786) — it contributes genuine *diversity*, not accuracy. The all-4-agree rate (66.7%)
  is the lowest of any design (partly mechanical with 4 agents, partly real diversity).

## 15. Where Intent helped (vs C)
- **`ahmed-eesa-00320`** (true *neutral*): *"…شعار bts… لحظتو ولا لاة🙂💜🌚"* — pointing out
  a **BTS logo** on a shirt (a spotting/observation, not praise). Lexical → positive
  (brand + emojis), but **Intent → neutral** ("author is reporting, not evaluating") and
  Polarity → neutral → **E neutral ✓** (C's run went positive). This is the artifact /
  platform-mention case Intent was designed for.
- (Two further "helped" cases, `00310`/`00330`, coincide with the borderline samples that
  flipped under C's capture, so they partly reflect temp-0 noise rather than a clean gain.)

## 16. Where Intent hurt (vs C)
- **`ahmed-eesa-00706`** (true *positive*): an excited ARMY fan pointing out *permission
  to dance* / BTS logo details — genuine enthusiasm. **Intent → neutral** ("just
  spotting"), pulling Lexical/Contextual's positive down to **neutral ✗** (C kept
  positive). Intent's "is this just describing?" test **over-neutralised real excitement.**
- **`ahmed-eesa-00113`** (true *negative*): *"planted the flower… then cut it wtf"*.
  **Intent → neutral** (read as event description) and dragged the panel to **neutral ✗**
  (C kept negative) — it **missed the implicit negative stance carried by "wtf".**

→ Intent's neutral-leaning discipline is **double-edged**: it kills artifact/spotting
positives (good) but also **flattens implicit positive excitement and implicit negative
reactions** (bad). On this subset the two roughly cancel → net the same as C.

## 14. Cost
**420 calls / $0.0639** (4 specialists + explainability × 84). +30% over C ($0.0494) and
≈ D ($0.0635).

---

## Interpretation
- **E does not clearly beat C.** It matches C's accuracy (0.9267) and net (+1), with a
  1-sample reduction in harmful overrides (10 vs 11) and the fewest breaks (10). Every one
  of these gaps is **within the ±1–2 sample temp-0 noise** already observed.
- **Intent adds diversity, not accuracy.** It is the weakest agent (0.714) but the least
  correlated (Intent↔Lexical 0.679) — exactly the "narrow, independent" agent intended.
  Its value is qualitative (catching artifact/spotting mentions) but it introduces a
  symmetric failure (flattening implicit stance), so it does not move the net.
- **It does not behave like D.** D (Logic re-added) was net 0; E is net +1 and reduces
  breaks — so Intent is a *better* 4th agent than re-adding Logic. But it still doesn't
  exceed the 3-agent C.

## Decision (against the stated criteria)
- "Beats C clearly or reduces harmful overrides → keep as candidate": E **reduces harmful
  overrides by 1** (within noise) → **keep E available as an opt-in candidate.**
- "Ties C but costs more → C remains preferred": E **ties C on accuracy/net at +30% cost**
  → **C remains the preferred design.**
- "Behaves like D / increases harm → retire": **E does not** — it is net-positive and
  lowers breaks, so **Intent is not retired.**

**Verdict: keep C as the lead variant; keep E as an opt-in secondary candidate (not
promoted).** Intent is a legitimate, diversity-adding specialist but, on a strong primary,
its neutral-leaning discipline is a wash and its extra cost is not justified by a robust
gain. If a future regime needs to suppress artifact/spotting false-positives specifically,
E is the tool; otherwise C is the better cost/performance point.

## Caveats (do not overclaim)
- All of B/C/D/E sit **within ±1 sample of primary_only (0.9254)**; E's +1 and its 10-vs-11
  override edge over C are single-sample, noise-level differences. The robust ranking is
  **A worst; B/C/D/E all neutralise the agentic harm; C and E are the best point
  estimates (tied), C at lower cost.**
- Strong-primary regime: the agent ceiling (~0.73–0.75) caps the upside; the decisive
  value test is the **weak-primary C3 regime** (not run here).
- Per-agent rows are from the instrumentation re-run (≤1-sample drift); the E-vs-C
  helped/hurt comparison is affected by temp-0 noise on both sides.

## Recommendation / next
- **Preferred: C** (Lexical + Polarity + Contextual) — best cost/performance, no 4th agent.
- **Keep E opt-in** as a candidate for artifact-heavy inputs; do not promote to default.
- **Retire D** (dominated). **Next:** validate **C** (optionally **E**) on the **C3
  generated-primary** to confirm the polarity decomposition preserves the +0.059
  weak-primary gain. **Not run here, per instruction.**

## Artifacts
- E: `experiment_ahmed_designE_intent/` (`full/` headline + `error_attribution/` per-agent).
- Agents/prompts: `src/agents/intent_agent.py`, `src/prompts/intent_prompt.py`.
- Capture driver: `scripts/ahmed_designE_attribution.py`.
- Comparators: `experiment_ahmed_polarity/` (C), `…_designB/…_designD/…_semantic_v1/`.
- Implementation/how-to-enable: `EXPERIMENT_SENTIMENT_POLARITY_AGENT_CHANGELOG.md`.
