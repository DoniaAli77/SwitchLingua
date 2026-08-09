"""Assemble the complete EXPLORATORY tau=0.40 result for Topic-540 XLM-R and
compare against primary-only and the predefined headline tau=0.30 condition.

Composition (no row is ever rerun):
  c <  0.30  (169 rows) -> cached agentic decision from the tau=0.30 run
  0.30<=c<0.40 (217)    -> agentic decision from the incremental band run
                            (sample_00763 taken from the clean retry)
  c >= 0.40  (777)      -> primary prediction, untouched, zero LLM calls

Read-only over saved artifacts.
"""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parent.parent
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

OLD = ROOT / "experiments/outputs/multi_agent_bert/experiment_silver_topic540"
T30 = ROOT / "experiments/outputs/multi_agent_bert/experiment_gen540_agentic_silver1163_tau030"
T40 = ROOT / "experiments/outputs/multi_agent_bert/experiment_gen540_agentic_silver1163_tau040_exploratory"
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]

# ---- primary baseline (all 1163) ----
prim = list(csv.DictReader(open(T30 / "pipeline_out/gen540_agentic_silver1163_tau030__primary_only_predictions.csv", encoding="utf-8")))
assert len(prim) == 1163
P = {r["sample_id"]: r for r in prim}
order = [r["sample_id"] for r in prim]
conf = {r["sample_id"]: float(r["confidence"]) for r in prim}
true = {r["sample_id"]: r["true_label"] for r in prim}

# ---- tau=0.30 final decisions ----
f30 = {r["sample_id"]: r for r in csv.DictReader(open(T30 / "pipeline_out/gen540_agentic_silver1163_tau030__full_pipeline_predictions.csv", encoding="utf-8"))}

# ---- band 0.30-0.40 decisions (+ clean retry for sample_00763) ----
band = {r["sample_id"]: r for r in csv.DictReader(open(T40 / "pipeline_out/band030_040_arentcxlmr__full_pipeline_predictions.csv", encoding="utf-8"))}
retry = {r["sample_id"]: r for r in csv.DictReader(open(T40 / "pipeline_out_retry/band_retry_00763__full_pipeline_predictions.csv", encoding="utf-8"))}
band.update(retry)                      # replace the connection-error row
assert len(band) == 217
assert band["sample_00763"]["pipeline_error"] == ""

routed30 = {s for s in order if conf[s] < 0.30}
bandset = {s for s in order if 0.30 <= conf[s] < 0.40}
routed40 = routed30 | bandset
assert len(routed30) == 169 and len(bandset) == 217 and len(routed40) == 386
assert set(band) == bandset, "band file does not match the 0.30-0.40 confidence band"

def assemble(tau):
    out = {}
    for s in order:
        if conf[s] < 0.30:
            out[s] = f30[s]["predicted_label"] if tau >= 0.30 else P[s]["predicted_label"]
        elif conf[s] < 0.40 and tau >= 0.40:
            out[s] = band[s]["predicted_label"]
        else:
            out[s] = P[s]["predicted_label"]
    return out

pred40 = assemble(0.40)
pred30 = {s: (f30[s]["predicted_label"]) for s in order}
predP = {s: P[s]["predicted_label"] for s in order}

print("=" * 74)
print("INTEGRITY CHECKS")
print("=" * 74)
# non-routed rows must equal primary
bad40 = [s for s in order if s not in routed40 and pred40[s] != predP[s]]
bad30 = [s for s in order if s not in routed30 and pred30[s] != predP[s]]
print(f"tau=0.40 non-routed rows differing from primary (must be 0): {len(bad40)}")
print(f"tau=0.30 non-routed rows differing from primary (must be 0): {len(bad30)}")
# the 169 tau=0.30 rows must carry IDENTICAL decisions inside tau=0.40 (no rerun)
ident = sum(1 for s in routed30 if pred40[s] == pred30[s])
print(f"tau=0.30 routed rows reused verbatim inside tau=0.40: {ident}/169")
print(f"band rows disjoint from tau=0.30 routed set: {len(routed30 & bandset) == 0}")

def block(name, pred):
    yt = [true[s] for s in order]
    yp = [pred[s] for s in order]
    return {
        "correct": sum(a == b for a, b in zip(yt, yp)),
        "accuracy": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(yt, yp, labels=LABELS, average="weighted", zero_division=0),
    }

mP, m30, m40 = block("primary", predP), block("t30", pred30), block("t40", pred40)

print("\n" + "=" * 74)
print("HEADLINE COMPARISON (n=1163, Silver-1163)")
print("=" * 74)
print(f"{'condition':<22}{'routed':>8}{'cov%':>8}{'correct':>9}{'acc':>9}{'macroF1':>10}{'wF1':>9}")
for nm, m, nr in [("primary only", mP, 0), ("tau=0.30 (headline)", m30, 169), ("tau=0.40 (exploratory)", m40, 386)]:
    print(f"{nm:<22}{nr:>8}{nr/1163*100:>7.2f}%{m['correct']:>9}{m['accuracy']:>9.4f}{m['macro_f1']:>10.4f}{m['weighted_f1']:>9.4f}")
print()
for nm, m in [("tau=0.30 vs primary", m30), ("tau=0.40 vs primary", m40)]:
    print(f"{nm:<22} dAcc={m['accuracy']-mP['accuracy']:+.4f}  dMacroF1={m['macro_f1']-mP['macro_f1']:+.4f}  dWF1={m['weighted_f1']-mP['weighted_f1']:+.4f}")
print(f"{'tau=0.40 vs tau=0.30':<22} dAcc={m40['accuracy']-m30['accuracy']:+.4f}  "
      f"dMacroF1={m40['macro_f1']-m30['macro_f1']:+.4f}  dWF1={m40['weighted_f1']-m30['weighted_f1']:+.4f}")

print("\n" + "=" * 74)
print("TRANSITIONS (routed rows only)")
print("=" * 74)
def trans(routed, pred):
    wc = cw = ww = cc = 0
    for s in routed:
        t, p, f = true[s], predP[s], pred[s]
        if p != t and f == t: wc += 1
        elif p == t and f != t: cw += 1
        elif p != t and f != t: ww += 1
        else: cc += 1
    return wc, cw, ww, cc
prim_err = sum(1 for s in order if predP[s] != true[s])
rows = []
for nm, routed, pred in [("tau=0.30", routed30, pred30), ("tau=0.40", routed40, pred40)]:
    wc, cw, ww, cc = trans(routed, pred)
    er = sum(1 for s in routed if predP[s] != true[s])
    ra_b = accuracy_score([true[s] for s in routed], [predP[s] for s in routed])
    ra_a = accuracy_score([true[s] for s in routed], [pred[s] for s in routed])
    rows.append((nm, len(routed), er, wc, cw, ww, cc, wc - cw, ra_b, ra_a))
    print(f"{nm}: routed={len(routed)}  primary-errors-routed={er}/{prim_err} ({er/prim_err*100:.2f}%)")
    print(f"   C->C={cc}  W->C={wc}  C->W={cw}  W->W={ww}   NET GAIN={wc-cw:+d}")
    print(f"   routed-subset accuracy: before={ra_b:.4f}  after={ra_a:.4f}")
# incremental band only
wc_b, cw_b, ww_b, cc_b = trans(bandset, pred40)
print(f"\nINCREMENTAL BAND 0.30<=c<0.40 only (217 rows):")
print(f"   C->C={cc_b}  W->C={wc_b}  C->W={cw_b}  W->W={ww_b}   NET GAIN={wc_b-cw_b:+d}")
print(f"   band accuracy: before={accuracy_score([true[s] for s in bandset],[predP[s] for s in bandset]):.4f}  "
      f"after={accuracy_score([true[s] for s in bandset],[pred40[s] for s in bandset]):.4f}")

print("\n" + "=" * 74)
print("PER-CLASS F1")
print("=" * 74)
yt = [true[s] for s in order]
_, _, f1P, sup = precision_recall_fscore_support(yt, [predP[s] for s in order], labels=LABELS, zero_division=0)
_, _, f130, _ = precision_recall_fscore_support(yt, [pred30[s] for s in order], labels=LABELS, zero_division=0)
_, _, f140, _ = precision_recall_fscore_support(yt, [pred40[s] for s in order], labels=LABELS, zero_division=0)
print(f"{'label':<12}{'sup':>5}{'primary':>10}{'tau=0.30':>10}{'tau=0.40':>10}{'d(40-30)':>10}")
for i, l in enumerate(LABELS):
    print(f"{l:<12}{sup[i]:>5}{f1P[i]:>10.4f}{f130[i]:>10.4f}{f140[i]:>10.4f}{f140[i]-f130[i]:>+10.4f}")

usage_b = json.load(open(T40 / "pipeline_out/band030_040_arentcxlmr__llm_usage.json", encoding="utf-8"))
usage_r = json.load(open(T40 / "pipeline_out_retry/band_retry_00763__llm_usage.json", encoding="utf-8"))
usage_30 = json.load(open(T30 / "pipeline_out/gen540_agentic_silver1163_tau030__llm_usage.json", encoding="utf-8"))
print("\n" + "=" * 74)
print("LLM USAGE (recorded)")
print("=" * 74)
print(f"tau=0.30 run          : {usage_30['calls']} calls, {usage_30['total_tokens']} tokens, ${usage_30['est_cost_usd']}")
print(f"band 0.30-0.40 run    : {usage_b['calls']} calls, {usage_b['total_tokens']} tokens, ${usage_b['est_cost_usd']}")
print(f"retry (1 row)         : {usage_r['calls']} calls, {usage_r['total_tokens']} tokens, ${usage_r['est_cost_usd']}")
tot_calls = usage_30['calls'] + usage_b['calls'] + usage_r['calls']
tot_cost = round(usage_30['est_cost_usd'] + usage_b['est_cost_usd'] + usage_r['est_cost_usd'], 4)
print(f"TOTAL for tau=0.40    : {tot_calls} calls, ${tot_cost}  (vs {868+676} if rerun from scratch)")

json.dump({
    "primary": {k: round(v, 6) if isinstance(v, float) else v for k, v in mP.items()},
    "tau_030_headline": {**{k: round(v, 6) if isinstance(v, float) else v for k, v in m30.items()}, "routed": 169},
    "tau_040_exploratory": {**{k: round(v, 6) if isinstance(v, float) else v for k, v in m40.items()}, "routed": 386},
    "transitions": {r[0]: {"routed": r[1], "primary_errors_routed": r[2], "W_to_C": r[3],
                             "C_to_W": r[4], "W_to_W": r[5], "C_to_C": r[6], "net_gain": r[7],
                             "routed_acc_before": round(r[8], 6), "routed_acc_after": round(r[9], 6)} for r in rows},
    "incremental_band": {"n": 217, "W_to_C": wc_b, "C_to_W": cw_b, "W_to_W": ww_b, "C_to_C": cc_b, "net_gain": wc_b - cw_b},
    "usage_total_calls": tot_calls, "usage_total_cost_usd": tot_cost,
}, open(T40 / "tau040_summary.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote -> {T40/'tau040_summary.json'}")
