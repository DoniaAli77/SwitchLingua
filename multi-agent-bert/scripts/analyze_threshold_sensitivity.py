"""Reconstruct 0.90 / 0.95 / 0.99 threshold results for the full-ArEnTC XLM-R
full-agentic experiment on Silver-1163, from a SINGLE cached agent run over
the union-at-0.99 routed subset (221 rows). No new inference, no new agent
calls -- this script only recombines already-saved predictions.

For each row:
  - primary prediction/confidence/true label <- fullarentc_xlmr_1163_predictions.csv
  - if primary confidence < threshold: routed -> use the CACHED agentic
    (full_pipeline) prediction for that row (from the union-099 run)
  - else: not routed -> final == primary (accept_primary path)
"""
import csv
import json
import collections
import statistics as st
from pathlib import Path

from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score

ROOT = Path(__file__).parent.parent
OLD = ROOT / "experiments/outputs/multi_agent_bert/experiment_silver_topic540"
NEW = ROOT / "experiments/outputs/multi_agent_bert/experiment_silver1163_thresh_sensitivity_arentc_xlmr"
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]
THRESHOLDS = [0.90, 0.95, 0.99]

# ---- load primary predictions for all 1163 rows, keyed by segment_id ----
ordered = [json.loads(l) for l in open(OLD / "silver_full1163_ordered.jsonl", encoding="utf-8")]
primary_rows = list(csv.DictReader(open(OLD / "fullarentc_xlmr_1163_predictions.csv", encoding="utf-8")))
assert len(ordered) == len(primary_rows) == 1163
primary_by_sid = {}
for o, p in zip(ordered, primary_rows):
    primary_by_sid[o["segment_id"]] = {
        "text": o["text"], "true_label": p["true_label"],
        "primary_pred": p["predicted_label"], "primary_conf": float(p["confidence"]),
    }

# ---- load cached agentic decisions for the union-0.99 subset ----
agentic_rows = list(csv.DictReader(
    open(NEW / "agentic_union099/union099_arentcxlmr__full_pipeline_predictions.csv", encoding="utf-8")))
agentic_by_sid = {r["sample_id"]: r for r in agentic_rows}
assert len(agentic_by_sid) == 221

with open(NEW / "agentic_union099/union099_arentcxlmr__llm_usage.json", encoding="utf-8") as fh:
    llm_usage_total = json.load(fh)

print("LLM usage for the full union-0.99 cache-building run (real, one-time):")
print(json.dumps(llm_usage_total, indent=2))
calls_per_row = llm_usage_total["calls"] / len(agentic_by_sid)
tokens_per_row = llm_usage_total["total_tokens"] / len(agentic_by_sid)
cost_per_row = llm_usage_total["est_cost_usd"] / len(agentic_by_sid)
print(f"\ncalls/row = {calls_per_row}  tokens/row (avg) = {tokens_per_row:.1f}  cost/row (avg) = ${cost_per_row:.5f}")


def build_rows_for_threshold(th):
    out = []
    for sid, p in primary_by_sid.items():
        routed = p["primary_conf"] < th
        if routed:
            a = agentic_by_sid[sid]
            final_pred = a["predicted_label"]
            final_conf = float(a["confidence"])
        else:
            final_pred = p["primary_pred"]
            final_conf = p["primary_conf"]
        out.append({
            "segment_id": sid, "text": p["text"], "true_label": p["true_label"],
            "primary_pred": p["primary_pred"], "primary_conf": p["primary_conf"],
            "routed": routed, "final_pred": final_pred, "final_conf": final_conf,
        })
    return out


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    return acc, macro_f1, weighted_f1


summary = {}
for th in THRESHOLDS:
    rows = build_rows_for_threshold(th)
    n = len(rows)
    routed_rows = [r for r in rows if r["routed"]]
    n_routed = len(routed_rows)

    y_true = [r["true_label"] for r in rows]
    y_primary = [r["primary_pred"] for r in rows]
    y_final = [r["final_pred"] for r in rows]

    prim_acc, prim_mf1, prim_wf1 = metrics(y_true, y_primary)
    fin_acc, fin_mf1, fin_wf1 = metrics(y_true, y_final)

    primary_errors = [r for r in rows if r["true_label"] != r["primary_pred"]]
    primary_errors_routed = [r for r in primary_errors if r["routed"]]

    wc = sum(1 for r in rows if r["routed"] and r["primary_pred"] != r["true_label"] and r["final_pred"] == r["true_label"])
    cw = sum(1 for r in rows if r["routed"] and r["primary_pred"] == r["true_label"] and r["final_pred"] != r["true_label"])
    ww = sum(1 for r in rows if r["routed"] and r["primary_pred"] != r["true_label"] and r["final_pred"] != r["true_label"])
    cc = sum(1 for r in rows if r["routed"] and r["primary_pred"] == r["true_label"] and r["final_pred"] == r["true_label"])
    net_gain = wc - cw

    routed_true = [r["true_label"] for r in routed_rows]
    routed_primary = [r["primary_pred"] for r in routed_rows]
    routed_final = [r["final_pred"] for r in routed_rows]
    routed_acc_before = accuracy_score(routed_true, routed_primary) if n_routed else 0.0
    routed_acc_after = accuracy_score(routed_true, routed_final) if n_routed else 0.0

    p, r_, f1, support = precision_recall_fscore_support(y_true, y_final, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_final, labels=LABELS)

    calls = int(round(calls_per_row * n_routed))
    tokens = tokens_per_row * n_routed
    cost = cost_per_row * n_routed

    print(f"\n{'='*70}\nTHRESHOLD {th}\n{'='*70}")
    print(f"routed: {n_routed}/{n} ({n_routed/n*100:.2f}%)")
    print(f"primary errors routed: {len(primary_errors_routed)}/{len(primary_errors)} "
          f"({len(primary_errors_routed)/len(primary_errors)*100:.2f}% of all primary errors)")
    print(f"primary-only: acc={prim_acc:.4f} macroF1={prim_mf1:.4f} weightedF1={prim_wf1:.4f}")
    print(f"final agentic: acc={fin_acc:.4f} macroF1={fin_mf1:.4f} weightedF1={fin_wf1:.4f}")
    print(f"Delta: acc={fin_acc-prim_acc:+.4f} macroF1={fin_mf1-prim_mf1:+.4f} weightedF1={fin_wf1-prim_wf1:+.4f}")
    print(f"W->C={wc}  C->W={cw}  W->W={ww}  C->C(routed,unchanged-correct)={cc}  net_gain={net_gain}")
    print(f"routed-subset accuracy: before={routed_acc_before:.4f}  after={routed_acc_after:.4f}")
    print(f"est. LLM usage at this threshold (prorated from union-0.99 real run): "
          f"calls={calls} tokens~={tokens:.0f} cost~=${cost:.4f}")
    print("per-class F1 (final):")
    for lbl, pp, rr, ff, ss in zip(LABELS, p, r_, f1, support):
        print(f"  {lbl:<12} P={pp:.4f} R={rr:.4f} F1={ff:.4f} support={ss}")
    print("confusion matrix (final; rows=true, cols=pred):")
    print("            " + "".join(l[:4].rjust(6) for l in LABELS))
    for lbl, row in zip(LABELS, cm):
        print(f"  {lbl:<10}" + "".join(str(v).rjust(6) for v in row))

    summary[str(th)] = {
        "n": n, "n_routed": n_routed, "pct_routed": round(n_routed / n * 100, 2),
        "n_primary_errors": len(primary_errors),
        "n_primary_errors_routed": len(primary_errors_routed),
        "pct_primary_errors_routed": round(len(primary_errors_routed) / len(primary_errors) * 100, 2),
        "primary_only": {"accuracy": round(prim_acc, 6), "macro_f1": round(prim_mf1, 6), "weighted_f1": round(prim_wf1, 6)},
        "final_agentic": {"accuracy": round(fin_acc, 6), "macro_f1": round(fin_mf1, 6), "weighted_f1": round(fin_wf1, 6)},
        "delta": {"accuracy": round(fin_acc - prim_acc, 6), "macro_f1": round(fin_mf1 - prim_mf1, 6), "weighted_f1": round(fin_wf1 - prim_wf1, 6)},
        "wrong_to_correct": wc, "correct_to_wrong": cw, "wrong_to_wrong": ww, "correct_to_correct_routed": cc,
        "net_gain": net_gain,
        "routed_subset_accuracy": {"before": round(routed_acc_before, 6), "after": round(routed_acc_after, 6)},
        "per_class_f1_final": [{"label": l, "precision": round(pp, 6), "recall": round(rr, 6), "f1": round(ff, 6), "support": int(ss)}
                                 for l, pp, rr, ff, ss in zip(LABELS, p, r_, f1, support)],
        "confusion_matrix_final": {"labels": LABELS, "matrix": cm.tolist()},
        "llm_usage_prorated": {"calls": calls, "tokens_est": round(tokens), "cost_usd_est": round(cost, 4)},
    }

    # save full per-row output for this threshold
    out_path = NEW / f"threshold_{str(th).replace('.', '')}_rows.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["segment_id", "text", "true_label", "primary_pred", "primary_conf",
                                            "routed", "final_pred", "final_conf", "outcome"])
        w.writeheader()
        for r in rows:
            if not r["routed"]:
                outcome = "not_routed"
            elif r["primary_pred"] != r["true_label"] and r["final_pred"] == r["true_label"]:
                outcome = "corrected"
            elif r["primary_pred"] == r["true_label"] and r["final_pred"] != r["true_label"]:
                outcome = "harmed"
            elif r["primary_pred"] != r["true_label"] and r["final_pred"] != r["true_label"]:
                outcome = "still_wrong"
            else:
                outcome = "preserved_correct"
            row = dict(r)
            row["outcome"] = outcome
            w.writerow(row)
    print(f"wrote -> {out_path}")

summary["llm_usage_real_union099_run"] = llm_usage_total
with open(NEW / "threshold_sensitivity_summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
print(f"\nwrote -> {NEW / 'threshold_sensitivity_summary.json'}")
