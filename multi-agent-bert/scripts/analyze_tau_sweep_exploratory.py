"""Assemble the complete EXPLORATORY threshold sweep for Topic-540 XLM-R:
primary-only -> tau=0.30 (headline) -> 0.40 -> 0.50 -> 0.60, on Silver-1163.

Every routed row's decision comes from exactly one real agentic run; no row is
ever rerun. Composition by primary confidence c:
  c <  0.30  (169) -> tau=0.30 run
  0.30<=c<0.40 (217) -> band 0.30-0.40 run (sample_00763 from its clean retry)
  0.40<=c<0.60 (460) -> band 0.40-0.60 run
  c >= tau            -> primary prediction, untouched, zero LLM calls

Read-only. No threshold is nominated as optimal.
"""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parent.parent
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

T30 = ROOT / "experiments/outputs/multi_agent_bert/experiment_gen540_agentic_silver1163_tau030"
T40 = ROOT / "experiments/outputs/multi_agent_bert/experiment_gen540_agentic_silver1163_tau040_exploratory"
T56 = ROOT / "experiments/outputs/multi_agent_bert/experiment_gen540_agentic_silver1163_tau0506_exploratory"
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]
TAUS = [0.30, 0.40, 0.50, 0.60]

prim = list(csv.DictReader(open(T30 / "pipeline_out/gen540_agentic_silver1163_tau030__primary_only_predictions.csv", encoding="utf-8")))
assert len(prim) == 1163
order = [r["sample_id"] for r in prim]
conf = {r["sample_id"]: float(r["confidence"]) for r in prim}
true = {r["sample_id"]: r["true_label"] for r in prim}
predP = {r["sample_id"]: r["predicted_label"] for r in prim}

# agentic decision pool: sample_id -> final label (one entry per routed-at-some-tau row)
pool = {}
for r in csv.DictReader(open(T30 / "pipeline_out/gen540_agentic_silver1163_tau030__full_pipeline_predictions.csv", encoding="utf-8")):
    if conf[r["sample_id"]] < 0.30:
        pool[r["sample_id"]] = r["predicted_label"]
b34 = {r["sample_id"]: r for r in csv.DictReader(open(T40 / "pipeline_out/band030_040_arentcxlmr__full_pipeline_predictions.csv", encoding="utf-8"))}
b34.update({r["sample_id"]: r for r in csv.DictReader(open(T40 / "pipeline_out_retry/band_retry_00763__full_pipeline_predictions.csv", encoding="utf-8"))})
assert b34["sample_00763"]["pipeline_error"] == ""
for k, v in b34.items():
    pool[k] = v["predicted_label"]
for r in csv.DictReader(open(T56 / "pipeline_out/band040_060_arentcxlmr__full_pipeline_predictions.csv", encoding="utf-8")):
    pool[r["sample_id"]] = r["predicted_label"]

print("=" * 92)
print("COMPOSITION / INTEGRITY")
print("=" * 92)
print(f"agentic decisions available : {len(pool)}  (169 + 217 + 460 = {169+217+460})")
expect = {s for s in order if conf[s] < 0.60}
print(f"cover exactly the c<0.60 set: {set(pool) == expect}")
calls = {0.30: 676, 0.40: 676 + 865 + 4, 0.50: 676 + 869 + 920, 0.60: 676 + 869 + 1840}

def assemble(tau):
    return {s: (pool[s] if conf[s] < tau else predP[s]) for s in order}

def met(pred):
    yt = [true[s] for s in order]; yp = [pred[s] for s in order]
    return (sum(a == b for a, b in zip(yt, yp)), accuracy_score(yt, yp),
            f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
            f1_score(yt, yp, labels=LABELS, average="weighted", zero_division=0))

preds = {t: assemble(t) for t in TAUS}
# integrity: non-routed rows identical to primary; lower-tau decisions reused verbatim
for t in TAUS:
    bad = [s for s in order if conf[s] >= t and preds[t][s] != predP[s]]
    print(f"tau={t:.2f}: non-routed rows differing from primary (must be 0): {len(bad)}")
for a, b in zip(TAUS, TAUS[1:]):
    reused = sum(1 for s in order if conf[s] < a and preds[b][s] == preds[a][s])
    n = sum(1 for s in order if conf[s] < a)
    print(f"tau={a:.2f} decisions reused verbatim inside tau={b:.2f}: {reused}/{n}")

cP, aP, mP, wP = met(predP)
print("\n" + "=" * 92)
print("EXPLORATORY THRESHOLD SWEEP (n=1163)   [tau=0.30 is the predefined headline]")
print("=" * 92)
print(f"{'condition':<26}{'routed':>7}{'cov%':>8}{'calls':>7}{'correct':>9}{'acc':>9}{'macroF1':>10}{'wF1':>9}")
print(f"{'primary only':<26}{0:>7}{0.0:>7.2f}%{0:>7}{cP:>9}{aP:>9.4f}{mP:>10.4f}{wP:>9.4f}")
rowsout = []
for t in TAUS:
    nr = sum(1 for s in order if conf[s] < t)
    c, a, m, w = met(preds[t])
    tag = f"tau={t:.2f}" + (" (HEADLINE)" if t == 0.30 else " (exploratory)")
    print(f"{tag:<26}{nr:>7}{nr/1163*100:>7.2f}%{calls[t]:>7}{c:>9}{a:>9.4f}{m:>10.4f}{w:>9.4f}")
    rowsout.append((t, nr, calls[t], c, a, m, w))

print("\ndeltas vs primary-only:")
for t, nr, cl, c, a, m, w in rowsout:
    print(f"  tau={t:.2f}: dAcc={a-aP:+.4f}  dMacroF1={m-mP:+.4f}  dWF1={w-wP:+.4f}")

print("\n" + "=" * 92)
print("TRANSITIONS + ROUTED-SUBSET ACCURACY")
print("=" * 92)
prim_err = sum(1 for s in order if predP[s] != true[s])
print(f"{'tau':<7}{'routed':>7}{'errRouted':>11}{'C->C':>7}{'W->C':>7}{'C->W':>7}{'W->W':>7}{'net':>7}{'routedAccBefore':>17}{'routedAccAfter':>16}")
trans = {}
for t in TAUS:
    routed = [s for s in order if conf[s] < t]
    wc = sum(1 for s in routed if predP[s] != true[s] and preds[t][s] == true[s])
    cw = sum(1 for s in routed if predP[s] == true[s] and preds[t][s] != true[s])
    ww = sum(1 for s in routed if predP[s] != true[s] and preds[t][s] != true[s])
    cc = sum(1 for s in routed if predP[s] == true[s] and preds[t][s] == true[s])
    er = sum(1 for s in routed if predP[s] != true[s])
    ab = accuracy_score([true[s] for s in routed], [predP[s] for s in routed])
    aa = accuracy_score([true[s] for s in routed], [preds[t][s] for s in routed])
    trans[t] = dict(routed=len(routed), err_routed=er, C_to_C=cc, W_to_C=wc, C_to_W=cw,
                    W_to_W=ww, net=wc - cw, routed_acc_before=ab, routed_acc_after=aa)
    print(f"{t:<7.2f}{len(routed):>7}{er:>11}{cc:>7}{wc:>7}{cw:>7}{ww:>7}{wc-cw:>+7}{ab:>17.4f}{aa:>16.4f}")
print(f"\n(primary errors total = {prim_err}; errRouted is how many of them the router surfaced)")

print("\nper-band incremental behaviour (rows first routed in that band):")
bands = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60)]
for lo, hi in bands:
    sel = [s for s in order if lo <= conf[s] < hi]
    wc = sum(1 for s in sel if predP[s] != true[s] and pool[s] == true[s])
    cw = sum(1 for s in sel if predP[s] == true[s] and pool[s] != true[s])
    ab = accuracy_score([true[s] for s in sel], [predP[s] for s in sel])
    aa = accuracy_score([true[s] for s in sel], [pool[s] for s in sel])
    print(f"  [{lo:.2f},{hi:.2f}): n={len(sel):>3}  W->C={wc:>3}  C->W={cw:>2}  net={wc-cw:>+4}  acc {ab:.4f} -> {aa:.4f}")

print("\n" + "=" * 92)
print("PER-CLASS F1")
print("=" * 92)
yt = [true[s] for s in order]
_, _, f1P, sup = precision_recall_fscore_support(yt, [predP[s] for s in order], labels=LABELS, zero_division=0)
f1s = {}
for t in TAUS:
    _, _, f, _ = precision_recall_fscore_support(yt, [preds[t][s] for s in order], labels=LABELS, zero_division=0)
    f1s[t] = f
print(f"{'label':<12}{'sup':>5}{'primary':>10}" + "".join(f"{'tau='+format(t,'.2f'):>10}" for t in TAUS))
for i, l in enumerate(LABELS):
    print(f"{l:<12}{sup[i]:>5}{f1P[i]:>10.4f}" + "".join(f"{f1s[t][i]:>10.4f}" for t in TAUS))

usage = {"tau030_run": json.load(open(T30 / "pipeline_out/gen540_agentic_silver1163_tau030__llm_usage.json", encoding="utf-8")),
         "band030_040": json.load(open(T40 / "pipeline_out/band030_040_arentcxlmr__llm_usage.json", encoding="utf-8")),
         "band_retry": json.load(open(T40 / "pipeline_out_retry/band_retry_00763__llm_usage.json", encoding="utf-8")),
         "band040_060": json.load(open(T56 / "pipeline_out/band040_060_arentcxlmr__llm_usage.json", encoding="utf-8"))}
tot_calls = sum(u["calls"] for u in usage.values())
tot_cost = round(sum(u["est_cost_usd"] for u in usage.values()), 4)
tot_tok = sum(u["total_tokens"] for u in usage.values())
print("\n" + "=" * 92)
print("LLM USAGE (recorded)")
print("=" * 92)
for k, u in usage.items():
    print(f"  {k:<14}: {u['calls']:>5} calls  {u['total_tokens']:>9} tokens  ${u['est_cost_usd']}")
print(f"  {'TOTAL':<14}: {tot_calls:>5} calls  {tot_tok:>9} tokens  ${tot_cost}")
print(f"  rows decided by agents: {len(pool)}  ->  {len(pool)*4} calls expected, {tot_calls} recorded")

json.dump({
    "note": "EXPLORATORY sweep. tau=0.30 is the predefined headline condition. No optimal threshold is selected.",
    "primary_only": {"correct": cP, "accuracy": round(aP, 6), "macro_f1": round(mP, 6), "weighted_f1": round(wP, 6)},
    "sweep": {f"{t:.2f}": {"routed": nr, "coverage_pct": round(nr / 1163 * 100, 2), "calls": cl,
                            "correct": c, "accuracy": round(a, 6), "macro_f1": round(m, 6),
                            "weighted_f1": round(w, 6), **{k: (round(v, 6) if isinstance(v, float) else v)
                                                             for k, v in trans[t].items()}}
              for t, nr, cl, c, a, m, w in rowsout},
    "per_class_f1": {"labels": LABELS, "support": [int(x) for x in sup],
                      "primary": [round(x, 6) for x in f1P],
                      **{f"tau_{t:.2f}": [round(x, 6) for x in f1s[t]] for t in TAUS}},
    "usage": usage, "usage_total_calls": tot_calls, "usage_total_cost_usd": tot_cost,
}, open(T56 / "tau_sweep_summary.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote -> {T56/'tau_sweep_summary.json'}")
