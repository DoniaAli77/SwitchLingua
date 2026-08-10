"""Silver topic-adaptation experiment: full metrics for G-only / S-only / G->S.

Secondary EXPLORATORY experiment. Labels are multi-LLM consensus Silver labels, not human
gold, and the hybrid split is not fully video-disjoint (one sports-dominant video contributes
a separated 17-row test block). No significance is claimed from three seeds.

Reports, per system and seed: accuracy, macro-F1 (primary), weighted-F1, per-class
precision/recall/F1, confusion matrix, predicted-class counts, largest predicted-class share,
whether all nine classes were predicted, and completed optimizer steps parsed from the log.
Then mean +/- SAMPLE standard deviation and the paired same-seed G->S minus S-only differences.
"""
import glob, json, pathlib, re, statistics
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/outputs/multi_agent_bert/experiment_silver_adaptation"
SEEDS = [42, 43, 44]
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]
SYS = {"A_gonly": "A. G-only (Topic-540)", "B_sonly": "B. S-only (Silver 860)",
       "C_gs": "C. G->S (two-stage)"}
EXPECTED_STEPS = {"A_gonly": 136, "B_sonly": 216, "C_gs": 216}
REPRO = {42: 0.6148, 43: 0.6148, 44: 0.6139}


def analyse(d):
    pj = glob.glob(str(pathlib.Path(d) / "*_predictions.json"))
    if not pj:
        raise SystemExit(f"missing predictions: {d}")
    rows = json.load(open(pj[0], encoding="utf-8"))
    n = len(rows)
    per, sup = {}, Counter(r["true_label"] for r in rows)
    for lab in LABELS:
        tp = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] == lab)
        fp = sum(1 for r in rows if r["true_label"] != lab and r["predicted_label"] == lab)
        fn = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[lab] = {"p": p, "r": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
                    "support": sup[lab]}
    pred = Counter(r["predicted_label"] for r in rows)
    cm = {t: Counter() for t in LABELS}
    for r in rows:
        cm[r["true_label"]][r["predicted_label"]] += 1
    top_lab, top_n = pred.most_common(1)[0]
    return {"n": n, "acc": sum(r["correct"] for r in rows) / n,
            "macro_f1": sum(v["f1"] for v in per.values()) / len(LABELS),
            "weighted_f1": sum(per[l]["f1"] * per[l]["support"] for l in LABELS) / n,
            "per_class": per, "pred_counts": dict(pred), "cm": cm,
            "n_pred": len(pred), "top_label": top_lab, "top_share": top_n / n,
            "all_nine": len(pred) == 9}


def steps(d):
    log = pathlib.Path(d) / "finetune.log"
    if not log.exists():
        return None
    t = log.read_text(encoding="utf-8", errors="replace")
    bars = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", t)
    return int(bars[-1][1]) if bars else None


R = {(s, k): analyse(BASE / f"{k}_seed{s}") for s in SEEDS for k in SYS}
ST = {(s, "B_sonly"): steps(BASE / f"B_sonly_seed{s}") for s in SEEDS}
ST.update({(s, "C_gs"): steps(BASE / f"C_gs_seed{s}") for s in SEEDS})
ST.update({(s, "A_gonly"): steps(BASE / f"g540_seed{s}") for s in SEEDS})
RP = {s: analyse(BASE / f"repro1163_seed{s}")["acc"] for s in SEEDS}

L = []


def out(x=""):
    print(x)
    L.append(x)


out("# Silver topic-adaptation experiment (secondary, exploratory)")
out()
out("Does pretraining XLM-R on the generated Topic-540 corpus improve adaptation to the")
out("limited Silver-labelled target domain? Three systems x model seeds 42/43/44, evaluated")
out("on the frozen 300-row Silver hybrid test split.")
out()
out("EXPLORATORY: labels are multi-LLM consensus Silver labels, not human gold, and the split")
out("is not completely video-disjoint - one sports-dominant video contributes a separated")
out("17-row test block. No statistical significance is claimed from three seeds.")
out()
out("## Checkpoint provenance and reproduction check")
out()
out("The per-seed Topic-540 136-step checkpoints had been deleted by the matched-compute")
out("script. They were regenerated with the identical recipe and seed, then verified against")
out("the recorded Silver-1163 accuracies:")
out()
out("| seed | recorded | regenerated | match |")
out("|--:|--:|--:|---|")
for s in SEEDS:
    out(f"| {s} | {REPRO[s]:.4f} | {RP[s]:.4f} | {'YES' if abs(RP[s]-REPRO[s]) < 1e-3 else 'NO'} |")

out()
out("## Optimizer steps")
out()
out("| system | seed | expected | observed | match |")
out("|---|--:|--:|--:|---|")
for k in SYS:
    for s in SEEDS:
        obs, exp = ST[(s, k)], EXPECTED_STEPS[k]
        out(f"| {SYS[k]} | {s} | {exp} | {obs} | {'YES' if obs == exp else 'NO'} |")

out()
out("## Headline metrics, 300-row Silver hybrid test")
out()
out("| system | seed | accuracy | macro-F1 | weighted-F1 | #classes predicted | all 9 | top class | top share |")
out("|---|--:|--:|--:|--:|--:|---|---|--:|")
for k in SYS:
    for s in SEEDS:
        m = R[(s, k)]
        out(f"| {SYS[k]} | {s} | {m['acc']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} "
            f"| {m['n_pred']} | {'yes' if m['all_nine'] else 'NO'} | {m['top_label']} "
            f"| {m['top_share']:.1%} |")

out()
out("## Mean +/- sample SD over seeds 42/43/44")
out()
out("| system | accuracy | macro-F1 | weighted-F1 |")
out("|---|---|---|---|")
for k in SYS:
    cols = []
    for met in ("acc", "macro_f1", "weighted_f1"):
        xs = [R[(s, k)][met] for s in SEEDS]
        cols.append(f"{statistics.mean(xs):.4f} +/- {statistics.stdev(xs):.4f}")
    out(f"| {SYS[k]} | {cols[0]} | {cols[1]} | {cols[2]} |")
out(f"| majority baseline (always tech) | 0.2467 | 0.0440 | - |")

out()
out("## PRIMARY CONTRAST: paired same-seed G->S minus S-only")
out()
out("| seed | d accuracy | d macro-F1 | d weighted-F1 |")
out("|--:|--:|--:|--:|")
d = {k: [] for k in ("acc", "macro_f1", "weighted_f1")}
for s in SEEDS:
    a, b = R[(s, "C_gs")], R[(s, "B_sonly")]
    for k in d:
        d[k].append(a[k] - b[k])
    out(f"| {s} | {a['acc']-b['acc']:+.4f} | {a['macro_f1']-b['macro_f1']:+.4f} "
        f"| {a['weighted_f1']-b['weighted_f1']:+.4f} |")
out(f"| **mean** | **{statistics.mean(d['acc']):+.4f}** | **{statistics.mean(d['macro_f1']):+.4f}** "
    f"| **{statistics.mean(d['weighted_f1']):+.4f}** |")
out()
consistent = len({x > 0 for x in d["macro_f1"]}) == 1
direction = "positive" if statistics.mean(d["macro_f1"]) > 0 else "negative"
out(f"- macro-F1 differences: {' '.join(f'{x:+.4f}' for x in d['macro_f1'])} -> "
    f"{'CONSISTENT ' + direction if consistent else 'MIXED'}")
if consistent and direction == "positive":
    out()
    out("**CONCLUSION: generated pretraining provided a consistent adaptation benefit.** All")
    out("three paired macro-F1 differences are positive. Three seeds; no significance claimed.")
else:
    out()
    out("**CONCLUSION: Topic-540 pretraining did not provide a consistent benefit under this**")
    out("**setting** (macro-F1 differences mixed or negligible). No significance claimed.")

out()
out("## Per-class precision / recall / F1")
for k in SYS:
    out()
    out(f"### {SYS[k]}")
    out()
    out("| class | support | " + " | ".join(f"P s{s} | R s{s} | F1 s{s}" for s in SEEDS) + " | mean F1 |")
    out("|---|--:|" + "--:|" * (9 + 1))
    for lab in LABELS:
        cells = []
        for s in SEEDS:
            pc = R[(s, k)]["per_class"][lab]
            cells += [f"{pc['p']:.3f}", f"{pc['r']:.3f}", f"{pc['f1']:.3f}"]
        mf = statistics.mean(R[(s, k)]["per_class"][lab]["f1"] for s in SEEDS)
        out(f"| {lab} | {R[(SEEDS[0], k)]['per_class'][lab]['support']} | "
            + " | ".join(cells) + f" | {mf:.3f} |")

out()
out("## Predicted-class counts")
out()
out("| system | seed | " + " | ".join(LABELS) + " |")
out("|---|--:|" + "--:|" * len(LABELS))
for k in SYS:
    for s in SEEDS:
        pc = R[(s, k)]["pred_counts"]
        out(f"| {SYS[k]} | {s} | " + " | ".join(str(pc.get(l, 0)) for l in LABELS) + " |")

out()
out("## Confusion matrices (rows = true, columns = predicted)")
for k in SYS:
    for s in SEEDS:
        out()
        out(f"### {SYS[k]} - seed {s}")
        out()
        out("| true \\ pred | " + " | ".join(LABELS) + " |")
        out("|---|" + "--:|" * len(LABELS))
        for t in LABELS:
            row = R[(s, k)]["cm"][t]
            out(f"| {t} | " + " | ".join(str(row.get(p, 0)) for p in LABELS) + " |")

BASE.mkdir(parents=True, exist_ok=True)
(BASE / "SILVER_ADAPTATION_RESULTS.md").write_text("\n".join(L), encoding="utf-8")
json.dump({f"{k}_s{s}": {kk: vv for kk, vv in R[(s, k)].items() if kk != "cm"}
           for s in SEEDS for k in SYS},
          open(BASE / "silver_adaptation_summary.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {BASE / 'SILVER_ADAPTATION_RESULTS.md'}")
