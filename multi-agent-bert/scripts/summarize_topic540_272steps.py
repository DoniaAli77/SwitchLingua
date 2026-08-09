"""Topic-540 at 272 steps (8 epochs) vs the same corpus at 136 steps.

Compute control run before deciding whether to generate Topic-1080: if the existing
540 corpus is still improving with more optimizer steps, the 136-step comparisons were
budget-limited rather than data-limited.

Training duration is the only difference between the two arms. Verifies that every run
actually reached 272 steps (parsed from the finetune log) and checks prediction collapse
with the same criterion used throughout: <9 predicted classes, or a top predicted-class
share >= 50%.
"""
import glob, json, os, pathlib, re, statistics
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEW = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic540_272steps"
OLD = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic_matched_compute"
SEEDS = [42, 43, 44]
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]


def metrics(d):
    pj = glob.glob(os.path.join(str(d), "*_predictions.json"))
    if not pj:
        raise SystemExit(f"missing predictions: {d}")
    rows = json.load(open(pj[0], encoding="utf-8"))
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    f1s, sup = {}, Counter(r["true_label"] for r in rows)
    for lab in LABELS:
        tp = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] == lab)
        fp = sum(1 for r in rows if r["true_label"] != lab and r["predicted_label"] == lab)
        fn = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = 2 * p * rc / (p + rc) if p + rc else 0.0
    pred = Counter(r["predicted_label"] for r in rows)
    top_lab, top_n = pred.most_common(1)[0]
    return {"acc": acc, "macro_f1": sum(f1s.values()) / len(LABELS),
            "weighted_f1": sum(f1s[l] * sup[l] for l in LABELS) / n,
            "per_class_f1": f1s, "n_pred": len(pred),
            "top_label": top_lab, "top_share": top_n / n,
            "collapsed": len(pred) < len(LABELS) or top_n / n >= 0.50}


def steps_reached(d):
    """Parse the final progress bar and epoch from the finetune log."""
    log = pathlib.Path(d) / "finetune.log"
    if not log.exists():
        return None, None
    t = log.read_text(encoding="utf-8", errors="replace")
    bars = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", t)
    ep = re.findall(r"'epoch': ([0-9.]+)}", t)
    done, total = (int(bars[-1][0]), int(bars[-1][1])) if bars else (None, None)
    return (done, total), (float(ep[-1]) if ep else None)


new = {s: metrics(NEW / f"mc540x2_seed{s}") for s in SEEDS}
old = {s: metrics(OLD / f"mc540_seed{s}") for s in SEEDS}

lines = []


def out(s=""):
    print(s)
    lines.append(s)


out("# Topic-540 compute control: 272 steps (8 epochs) vs 136 steps (4 epochs)")
out()
out("Same fixed Topic-540 corpus, model seeds 42/43/44, identical recipe; training")
out("duration is the only difference. No dev set / load_best off in BOTH arms (the only")
out("topic dev set available is ArEnTC, excluded by this study line), so the paired")
out("difference isolates duration. Silver-1163 used for final evaluation only.")
out()
out("## Step verification")
out()
out("| seed | steps reached | final epoch | reached 272 |")
out("|--:|---|--:|---|")
all_ok = True
for s in SEEDS:
    (done, total), ep = steps_reached(NEW / f"mc540x2_seed{s}")
    ok = (done == 272 and total == 272)
    all_ok &= bool(ok)
    out(f"| {s} | {done}/{total} | {ep} | {'YES' if ok else 'NO'} |")

out()
out("## Per-run, Silver-1163")
out()
out("| budget | seed | acc | macro-F1 | weighted-F1 | #pred classes | top share | collapsed |")
out("|---|--:|--:|--:|--:|--:|--:|---|")
for tag, grp in (("136 steps", old), ("272 steps", new)):
    for s in SEEDS:
        m = grp[s]
        out(f"| {tag} | {s} | {m['acc']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} "
            f"| {m['n_pred']} | {m['top_share']:.1%} | {'YES' if m['collapsed'] else 'no'} |")

out()
out("## Mean +/- SD")
out()
out("| budget | acc | macro-F1 | weighted-F1 |")
out("|---|---|---|---|")
for tag, grp in (("136 steps", old), ("272 steps", new)):
    cols = []
    for k in ("acc", "macro_f1", "weighted_f1"):
        xs = [grp[s][k] for s in SEEDS]
        cols.append(f"{statistics.mean(xs):.4f} +/- {statistics.pstdev(xs):.4f}")
    out(f"| {tag} | {cols[0]} | {cols[1]} | {cols[2]} |")

out()
out("## Paired differences (272 - 136), same model seed")
out()
out("| seed | d acc | d macro-F1 | d weighted-F1 |")
out("|--:|--:|--:|--:|")
d = {k: [] for k in ("acc", "macro_f1", "weighted_f1")}
for s in SEEDS:
    for k in d:
        d[k].append(new[s][k] - old[s][k])
    out(f"| {s} | {new[s]['acc']-old[s]['acc']:+.4f} | "
        f"{new[s]['macro_f1']-old[s]['macro_f1']:+.4f} | "
        f"{new[s]['weighted_f1']-old[s]['weighted_f1']:+.4f} |")
out(f"| **mean** | **{statistics.mean(d['acc']):+.4f}** | **{statistics.mean(d['macro_f1']):+.4f}** "
    f"| **{statistics.mean(d['weighted_f1']):+.4f}** |")
out()
for k in d:
    out(f"- {k}: {' '.join(f'{x:+.4f}' for x in d[k])} -> "
        f"{'consistent' if len({x > 0 for x in d[k]}) == 1 else 'MIXED'}")

out()
out("## Per-class F1 (mean over seeds)")
out()
out("| class | 136 steps | 272 steps | delta |")
out("|---|--:|--:|--:|")
for lab in LABELS:
    o = statistics.mean(old[s]["per_class_f1"][lab] for s in SEEDS)
    n_ = statistics.mean(new[s]["per_class_f1"][lab] for s in SEEDS)
    out(f"| {lab} | {o:.3f} | {n_:.3f} | {n_-o:+.3f} |")

out()
out("## Checks")
out()
out(f"- all runs reached 272 steps: {'YES' if all_ok else 'NO'}")
bad = [s for s in SEEDS if new[s]["collapsed"]]
out(f"- collapsed runs at 272 steps: {bad if bad else 'NONE'}")
out(f"- reference: Original-180 corpus-draw sd at 136 steps = 0.0111 (9-run grid)")

NEW.mkdir(parents=True, exist_ok=True)
(NEW / "RESULTS_272STEPS.md").write_text("\n".join(lines), encoding="utf-8")
json.dump({"s272": {str(s): new[s] for s in SEEDS}, "s136": {str(s): old[s] for s in SEEDS}},
          open(NEW / "summary_272steps.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {NEW / 'RESULTS_272STEPS.md'}")
