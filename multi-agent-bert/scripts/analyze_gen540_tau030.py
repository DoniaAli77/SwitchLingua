"""Analysis + evidence audit for:
Topic-540 XLM-R + full agentic pipeline at the primary-calibrated threshold tau=0.30.

Read-only over the completed run's artifacts. No inference, no agent calls.
"""
import csv
import json
import collections
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)

D = ROOT / "experiments/outputs/multi_agent_bert/experiment_gen540_agentic_silver1163_tau030"
OLD = ROOT / "experiments/outputs/multi_agent_bert/experiment_silver_topic540"
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]
TAU = 0.30

prim = list(csv.DictReader(open(D / "pipeline_out/gen540_agentic_silver1163_tau030__primary_only_predictions.csv", encoding="utf-8")))
finl = list(csv.DictReader(open(D / "pipeline_out/gen540_agentic_silver1163_tau030__full_pipeline_predictions.csv", encoding="utf-8")))
audit_rows = [json.loads(l) for l in open(D / "audit_rows.jsonl", encoding="utf-8")]
calls = [json.loads(l) for l in open(D / "audit_llm_calls.jsonl", encoding="utf-8")]
saved_baseline = list(csv.DictReader(open(OLD / "xlmr_combined_1163_predictions.csv", encoding="utf-8")))

assert len(prim) == len(finl) == len(audit_rows) == len(saved_baseline) == 1163

# audit_rows contains BOTH modes (primary_only pass then full_pipeline pass) if mode=both
# -> keep only the full_agentic pass (those with routing info)
ar_full = [r for r in audit_rows if r.get("routing") is not None]
print(f"audit_rows total={len(audit_rows)}  with routing info={len(ar_full)}")

print("\n" + "=" * 72)
print("A. BASELINE CONSISTENCY")
print("=" * 72)
yt = [r["true_label"] for r in prim]
yp_prim = [r["predicted_label"] for r in prim]
yp_fin = [r["predicted_label"] for r in finl]
base_pred = [r["predicted_label"] for r in saved_baseline]
print("this run's primary == previously saved primary predictions:",
      yp_prim == base_pred)
print("primary correct: %d/1163  acc=%.6f" % (sum(a == b for a, b in zip(yt, yp_prim)),
                                                accuracy_score(yt, yp_prim)))

print("\n" + "=" * 72)
print("B. ROUTING (frozen router applied tau=0.30)")
print("=" * 72)
routed_ids = {r["sample_id"] for r in ar_full if r["routing"]["decision"] == "escalate"}
n_routed = len(routed_ids)
print("escalate=%d (%.2f%%)   accept_primary=%d (%.2f%%)"
      % (n_routed, n_routed / 1163 * 100, 1163 - n_routed, (1163 - n_routed) / 1163 * 100))
thr_used = {r["routing"]["threshold"] for r in ar_full}
print("threshold values seen in routing records:", thr_used)
# every routed row must have primary conf < TAU and vice versa
bad = [r["sample_id"] for r in ar_full
       if (r["primary"]["confidence"] < TAU) != (r["routing"]["decision"] == "escalate")]
print("rows where routing disagrees with (conf<0.30):", len(bad))

# non-routed rows must keep primary prediction untouched
sid_order = [r["sample_id"] for r in prim]
fin_by_sid = {r["sample_id"]: r for r in finl}
prim_by_sid = {r["sample_id"]: r for r in prim}
changed_unrouted = [s for s in sid_order if s not in routed_ids
                     and fin_by_sid[s]["predicted_label"] != prim_by_sid[s]["predicted_label"]]
print("NON-routed rows whose final != primary (must be 0):", len(changed_unrouted))

print("\n" + "=" * 72)
print("C. HEADLINE METRICS")
print("=" * 72)


def blk(name, yp):
    a = accuracy_score(yt, yp)
    m = f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0)
    w = f1_score(yt, yp, labels=LABELS, average="weighted", zero_division=0)
    print(f"{name:<14} correct={sum(x==y for x,y in zip(yt,yp)):>4}/1163  "
          f"acc={a:.4f}  macroF1={m:.4f}  weightedF1={w:.4f}")
    return a, m, w


pa, pm, pw = blk("primary_only", yp_prim)
fa, fm, fw = blk("full_agentic", yp_fin)
print(f"{'DELTA':<14} {'':>13}  acc={fa-pa:+.4f}  macroF1={fm-pm:+.4f}  weightedF1={fw-pw:+.4f}")

print("\n" + "=" * 72)
print("D. TRANSITIONS")
print("=" * 72)
wc = cw = ww = cc = 0
for s in sid_order:
    t = prim_by_sid[s]["true_label"]
    p = prim_by_sid[s]["predicted_label"]
    f = fin_by_sid[s]["predicted_label"]
    if s not in routed_ids:
        continue
    if p != t and f == t:
        wc += 1
    elif p == t and f != t:
        cw += 1
    elif p != t and f != t:
        ww += 1
    else:
        cc += 1
prim_err = sum(1 for s in sid_order if prim_by_sid[s]["predicted_label"] != prim_by_sid[s]["true_label"])
prim_err_routed = sum(1 for s in routed_ids if prim_by_sid[s]["predicted_label"] != prim_by_sid[s]["true_label"])
print(f"primary errors total          : {prim_err}")
print(f"primary errors routed         : {prim_err_routed} ({prim_err_routed/prim_err*100:.2f}% of all primary errors)")
print(f"primary-correct -> final-correct (C->C): {cc}")
print(f"primary-wrong   -> final-correct (W->C): {wc}")
print(f"primary-correct -> final-wrong   (C->W): {cw}")
print(f"primary-wrong   -> final-wrong   (W->W): {ww}")
print(f"NET GAIN (W->C - C->W)                 : {wc-cw:+d}")
print(f"check: {cc}+{wc}+{cw}+{ww} = {cc+wc+cw+ww} == routed {n_routed}")

rt = [prim_by_sid[s]["true_label"] for s in sid_order if s in routed_ids]
rp = [prim_by_sid[s]["predicted_label"] for s in sid_order if s in routed_ids]
rf = [fin_by_sid[s]["predicted_label"] for s in sid_order if s in routed_ids]
print(f"\nrouted-subset accuracy BEFORE (primary): {accuracy_score(rt,rp):.4f}")
print(f"routed-subset accuracy AFTER  (agentic): {accuracy_score(rt,rf):.4f}")

print("\n" + "=" * 72)
print("E. PER-CLASS (primary -> final)")
print("=" * 72)
pp, pr, pf1, sup = precision_recall_fscore_support(yt, yp_prim, labels=LABELS, zero_division=0)
fp, fr, ff1, _ = precision_recall_fscore_support(yt, yp_fin, labels=LABELS, zero_division=0)
print(f"{'label':<12}{'sup':>5} | {'P_prim':>7}{'R_prim':>8}{'F1_prim':>9} | {'P_fin':>7}{'R_fin':>8}{'F1_fin':>9} | {'dF1':>7}")
for i, l in enumerate(LABELS):
    print(f"{l:<12}{sup[i]:>5} | {pp[i]:>7.4f}{pr[i]:>8.4f}{pf1[i]:>9.4f} | "
          f"{fp[i]:>7.4f}{fr[i]:>8.4f}{ff1[i]:>9.4f} | {ff1[i]-pf1[i]:>+7.4f}")

for nm, yp in [("PRIMARY", yp_prim), ("FINAL", yp_fin)]:
    print(f"\nconfusion matrix [{nm}] (rows=true, cols=pred):")
    cm = confusion_matrix(yt, yp, labels=LABELS)
    print("            " + "".join(l[:4].rjust(6) for l in LABELS))
    for l, row in zip(LABELS, cm):
        print(f"  {l:<10}" + "".join(str(v).rjust(6) for v in row))

print("\n" + "=" * 72)
print("F. EVIDENCE AUDIT (raw agent trail)")
print("=" * 72)
by_sid = collections.defaultdict(list)
for c in calls:
    by_sid[c["sample_id"]].append(c)
print(f"total LLM calls recorded : {len(calls)}")
print(f"distinct sample_ids      : {len(by_sid)}  (expected {n_routed})")
print(f"ids in calls == routed ids: {set(by_sid) == routed_ids}")
counts = collections.Counter(len(v) for v in by_sid.values())
print(f"calls-per-row histogram  : {dict(counts)}  (expected {{4: {n_routed}}})")
agents = collections.Counter(c["agent"] for c in calls)
print(f"calls per agent          : {dict(agents)}")
per_row_agents = {frozenset(c["agent"] for c in v) for v in by_sid.values()}
print(f"distinct agent-sets per row: {[sorted(s) for s in per_row_agents]}")
print(f"models used              : {set(c['model'] for c in calls)}")
errs = [c for c in calls if c["error"]]
empty = [c for c in calls if not (c["raw_response"] or "").strip()]
print(f"calls with error         : {len(errs)}")
print(f"calls with empty response: {len(empty)}")
uniq_resp = len({c["raw_response"] for c in calls})
print(f"distinct raw responses   : {uniq_resp}/{len(calls)} (placeholder/canned would collapse)")
lat = [c["latency_sec"] for c in calls]
print(f"latency sec: min={min(lat):.2f} med={sorted(lat)[len(lat)//2]:.2f} max={max(lat):.2f}")
print(f"timestamp range: {min(c['started_utc'] for c in calls)} .. {max(c['started_utc'] for c in calls)}")

print("\n-- gold-label leakage --")
markers = ["true_label", "gold", "ground truth", "correct answer", "reference label", "silver"]
hits = {m: sum(1 for c in calls if m in c["prompt"].lower()) for m in markers}
print("leak-marker occurrences in prompts:", hits)

from src.config.loader import load_task_bundle
from src.prompts.llm_lexical_prompt import build_user_prompt as lex_up, get_system_prompt as lex_sp
b = load_task_bundle("src/config/default.yaml", active_task="topic_classification",
                     pipeline_mode="full_agentic", threshold=TAU)
tc = b.task_config
text_by_sid = {r["sample_id"]: r["input_text"] for r in ar_full}
lex_calls = [c for c in calls if c["agent"] == "lexical_agent"]
ok = 0
for c in lex_calls:
    exp = f"{lex_sp()}\n\n" + lex_up(task_name=tc.task_name, labels=tc.labels,
                                       label_descriptions=tc.label_descriptions,
                                       text=text_by_sid[c["sample_id"]], primary_signal=None)
    ok += (exp == c["prompt"])
print(f"lexical prompts byte-identical to gold-free reconstruction: {ok}/{len(lex_calls)}")
print(f"agents_use_primary_signal (frozen config): {tc.agents_use_primary_signal}")

# agent independence: do agents ever just echo the silver label?
true_by_sid = {r["sample_id"]: r["true_label"] for r in prim}
lex_lbl = {r["sample_id"]: (r["lexical_output"] or {}).get("parsed", {}).get("label") for r in ar_full if r["sample_id"] in routed_ids}
agree = sum(1 for s, l in lex_lbl.items() if l == true_by_sid[s])
print(f"\nlexical agent label == silver label on routed rows: {agree}/{len(lex_lbl)} "
      f"({agree/len(lex_lbl)*100:.1f}%) -- far below 100% => not copying gold")

summary = {
    "condition": "Topic-540 XLM-R + full agentic pipeline at the primary-calibrated threshold tau=0.30",
    "tau": TAU, "n": 1163, "routed": n_routed, "routed_pct": round(n_routed / 1163 * 100, 2),
    "primary": {"correct": sum(x == y for x, y in zip(yt, yp_prim)), "accuracy": round(pa, 6),
                 "macro_f1": round(pm, 6), "weighted_f1": round(pw, 6)},
    "final": {"correct": sum(x == y for x, y in zip(yt, yp_fin)), "accuracy": round(fa, 6),
               "macro_f1": round(fm, 6), "weighted_f1": round(fw, 6)},
    "delta": {"accuracy": round(fa - pa, 6), "macro_f1": round(fm - pm, 6), "weighted_f1": round(fw - pw, 6)},
    "transitions": {"C_to_C": cc, "W_to_C": wc, "C_to_W": cw, "W_to_W": ww, "net_gain": wc - cw},
    "primary_errors_total": prim_err, "primary_errors_routed": prim_err_routed,
    "routed_subset_accuracy": {"before": round(accuracy_score(rt, rp), 6),
                                "after": round(accuracy_score(rt, rf), 6)},
    "per_class": [{"label": l, "support": int(sup[i]),
                    "primary": {"precision": round(pp[i], 6), "recall": round(pr[i], 6), "f1": round(pf1[i], 6)},
                    "final": {"precision": round(fp[i], 6), "recall": round(fr[i], 6), "f1": round(ff1[i], 6)}}
                   for i, l in enumerate(LABELS)],
    "confusion_primary": confusion_matrix(yt, yp_prim, labels=LABELS).tolist(),
    "confusion_final": confusion_matrix(yt, yp_fin, labels=LABELS).tolist(),
    "labels": LABELS,
    "llm_usage": json.load(open(D / "pipeline_out/gen540_agentic_silver1163_tau030__llm_usage.json", encoding="utf-8")),
}
json.dump(summary, open(D / "tau030_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nwrote -> {D/'tau030_summary.json'}")

with open(D / "tau030_rows.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["sample_id", "text", "silver_label", "primary_pred",
                                        "primary_conf", "routed", "final_pred", "final_conf", "outcome"])
    w.writeheader()
    conf_by_sid = {r["sample_id"]: r["primary"]["confidence"] for r in ar_full}
    fconf = {r["sample_id"]: r["confidence"] for r in finl}
    for s in sid_order:
        t = prim_by_sid[s]["true_label"]; p = prim_by_sid[s]["predicted_label"]; f = fin_by_sid[s]["predicted_label"]
        r_ = s in routed_ids
        outcome = ("not_routed" if not r_ else
                    "corrected" if p != t and f == t else
                    "harmed" if p == t and f != t else
                    "still_wrong" if p != t else "preserved_correct")
        w.writerow({"sample_id": s, "text": prim_by_sid[s]["input_text"], "silver_label": t,
                     "primary_pred": p, "primary_conf": conf_by_sid[s], "routed": r_,
                     "final_pred": f, "final_conf": fconf[s], "outcome": outcome})
print(f"wrote -> {D/'tau030_rows.csv'}")
