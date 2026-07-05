# Stronger-Model Diagnostic on Design-G Ahmed Failures (gpt-4.1-mini)

Targeted test: re-run **only the 18 Design-G escalated failures** on Ahmed through the full
G pipeline, comparing **gpt-4o-mini (control re-run)** vs **gpt-4.1-mini (stronger)**, temp 0.
Isolates whether a stronger base model fixes the compliance/knowledge failures or whether any
change is just temp-0 noise. Cost ≈ $0.03 total. Date: 2026-07-02.

## Result

| model | now-correct of the 18 prior failures |
|---|---|
| gpt-4o-mini (control re-run) | **0 / 18** |
| **gpt-4.1-mini** | **4 / 18** |
| fixed by 4.1-mini but NOT by the 4o-mini re-run | **4** |

**The noise floor is 0** — the 18 failures are stable/deterministic under 4o-mini, so all 4
of gpt-4.1-mini's fixes are a **real model effect, none attributable to noise.**

## What the stronger model fixed (4)
| id | text (gist) | true | 4o-mini | 4.1-mini | cluster |
|---|---|---|---|---|---|
| 00240 | "…اللى عاملين **dislike**" | neutral | negative | **neutral ✅** | 2 — platform-meta "dislike" |
| 00298 | "…ضل راجل=لايك … الفتوة=**dislike**" | neutral | negative | **neutral ✅** | 2 — platform-meta "dislike" |
| 00542 | "تول عومراك **جاميلا** XD XD" (misspelled "beautiful") | positive | neutral | **positive ✅** | 1 — obscured praise |
| 00642 | "**عاش عاش** … Free Fire" ("bravo") | positive | neutral | **positive ✅** | 1 — implicit praise |

So gpt-4.1-mini recovered **2 of the ~3 "dislike"-meta compliance cases** (it now applies the
"dislike = weak/meta" rule the prompt always had) and **2 obscured/implicit praise** cases.

## What it did NOT fix (14 remain wrong)
The hard **implicit Egyptian insult** cluster survives even the stronger model:
- 00008 "هوة … gay ?" (slur-as-question), 00097 "…مفيهوش ولا راجل" ("no real man"),
  00182 "انت no one", 00193, 00021, 00046 — **all still neutral** (true negative).
- 00446 "Dislikes كتير اوي" — ultra-short meta, still negative even for 4.1-mini.

**Reading:** the residual ceiling is *two* things, and the stronger model separates them:
- **Compliance + obscured-cue errors (recoverable):** a better model applies rules 4o-mini
  ignored ("dislike"=meta) and reads misspelled/implicit praise → **4.1-mini fixes these.**
- **Deep cultural-implicit insults (still floor):** slur-as-question, "no real man", "you are
  no one" — these need cultural/pragmatic knowledge **4.1-mini still lacks.**

## Implication — the ceiling is partly a MODEL ceiling, and it moves cheaply
This is the first evidence that the strong-primary ceiling can be broken: **4/18 residual
errors recovered by a $0.03 model swap, zero from noise.** It refines the earlier conclusion:
you can't *prompt-engineer* past the ceiling (v1/v2/semantic_v1 all failed), but a **stronger
model** recovers the compliance/obscured-cue slice — though not the deepest cultural slice.

**Caveat (important):** this only shows 4.1-mini fixes 4 *failing* cases. It does **not**
prove a net +4, because a full run at 4.1-mini would also re-decide the **66 currently-correct
escalated cases** — it could break some of those. The net effect requires a full run.

## Recommended next step (cheap, ~$0.13)
Run **full Design G on Ahmed at gpt-4.1-mini** (`--llm_model gpt-4.1-mini`, all 84 escalated)
to get the **net** number: does fixing these 4 (and any others) survive after 4.1-mini also
re-decides the 66 correct ones? If net-positive, it's the first real move off 0.9279 toward
the 0.930 target — and it would very likely help the **weak C3** run even more (where the
recoverable slice is larger). Then optionally the same swap on C3.

## Artifacts
- `experiment_G_ahmed_stronger_model/diagnostic.json`, `…_run.log`
- Script: `scripts/ahmed_G_failure_diagnostic.py`
- Basis: `EXPERIMENT_G_AHMED_FAILURE_ANALYSIS.md`, `EXPERIMENT_G_PROMPT_VS_FAILURE_MAPPING.md`.
