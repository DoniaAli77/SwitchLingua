"""Topic-1080 vs Topic-540, both at exactly 136 optimizer steps, model seeds 42/43/44.

Corpus is the only difference; every other setting is reused from the Topic-540 136-step
matched-compute experiment. primary_only evaluation on the unchanged Silver-1163.

Reports per-seed and mean +/- SD accuracy / macro-F1 / weighted-F1, paired differences at
matching seeds, and the two collapse diagnostics (predicted-class count, largest
predicted-class share). Also verifies each run actually reached 136 steps.
"""
import glob, json, os, pathlib, re, statistics
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEW = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic1080_136steps"
OLD = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic_matched_compute"
SEEDS = [42, 43, 44]
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]


def analyse(d):
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
            "top_label": top_lab, "top_share": top_n / n}


def steps(d):
    log = pathlib.Path(d) / "finetune.log"
    if not log.exists():
        return None, None
    t = log.read_text(encoding="utf-8", errors="replace")
    bars = re.findall(r"\|\s*(\d+)/(\d+)\s*\[", t)
    ep = re.findall(r"'epoch': ([0-9.]+)}", t)
    return (bars[-1] if bars else None), (ep[-1] if ep else None)


new = {s: analyse(NEW / f"mc1080_seed{s}") for s in SEEDS}
old = {s: analyse(OLD / f"mc540_seed{s}") for s in SEEDS}

lines = []


def out(s=""):
    print(s)
    lines.append(s)


out("# Topic-1080 vs Topic-540 at 136 optimizer steps")
out()
out("Corpus is the only difference; all other settings reused from the Topic-540 136-step")
out("experiment (xlm-roberta-base, batch 16, grad_accum 1, lr 2e-5, maxlen 256, fp16,")
out("adamw_torch, no dev, load_best off, 9 labels frozen order). primary_only on Silver-1163,")
out("which is used for final evaluation only. Three model seeds; no significance claimed.")
out()
out("## Step verification")
out()
out("| corpus | seed | steps | final epoch |")
out("|---|--:|---|--:|")
for s in SEEDS:
    b, e = steps(NEW / f"mc1080_seed{s}")
    out(f"| Topic-1080 | {s} | {'/'.join(b) if b else '?'} | {e} |")
for s in SEEDS:
    b, e = steps(OLD / f"mc540_seed{s}")
    out(f"| Topic-540 | {s} | {'/'.join(b) if b else '?'} | {e} |")

out()
out("## Per-run, Silver-1163")
out()
out("| corpus | seed | acc | macro-F1 | weighted-F1 | #pred classes | top class | top share |")
out("|---|--:|--:|--:|--:|--:|---|--:|")
for tag, grp in (("Topic-540", old), ("Topic-1080", new)):
    for s in SEEDS:
        m = grp[s]
        out(f"| {tag} | {s} | {m['acc']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} "
            f"| {m['n_pred']} | {m['top_label']} | {m['top_share']:.1%} |")

out()
out("## Mean +/- SD")
out()
out("| corpus | acc | macro-F1 | weighted-F1 |")
out("|---|---|---|---|")
for tag, grp in (("Topic-540", old), ("Topic-1080", new)):
    cols = []
    for k in ("acc", "macro_f1", "weighted_f1"):
        xs = [grp[s][k] for s in SEEDS]
        cols.append(f"{statistics.mean(xs):.4f} +/- {statistics.pstdev(xs):.4f}")
    out(f"| {tag} | {cols[0]} | {cols[1]} | {cols[2]} |")

out()
out("## Paired differences (Topic-1080 - Topic-540), same model seed")
out()
out("| seed | d acc | d macro-F1 | d weighted-F1 |")
out("|--:|--:|--:|--:|")
d = {k: [] for k in ("acc", "macro_f1", "weighted_f1")}
for s in SEEDS:
    for k in d:
        d[k].append(new[s][k] - old[s][k])
    out(f"| {s} | {new[s]['acc']-old[s]['acc']:+.4f} | {new[s]['macro_f1']-old[s]['macro_f1']:+.4f} "
        f"| {new[s]['weighted_f1']-old[s]['weighted_f1']:+.4f} |")
out(f"| **mean** | **{statistics.mean(d['acc']):+.4f}** | **{statistics.mean(d['macro_f1']):+.4f}** "
    f"| **{statistics.mean(d['weighted_f1']):+.4f}** |")
out()
for k in d:
    out(f"- {k}: {' '.join(f'{x:+.4f}' for x in d[k])} -> "
        f"{'consistent' if len({x > 0 for x in d[k]}) == 1 else 'MIXED'}")

out()
out("## Per-class F1 (mean over seeds)")
out()
out("| class | Topic-540 | Topic-1080 | delta |")
out("|---|--:|--:|--:|")
for lab in LABELS:
    o = statistics.mean(old[s]["per_class_f1"][lab] for s in SEEDS)
    n_ = statistics.mean(new[s]["per_class_f1"][lab] for s in SEEDS)
    out(f"| {lab} | {o:.3f} | {n_:.3f} | {n_-o:+.3f} |")

out()
out("## Collapse check")
out()
bad = [f"1080/s{s}" for s in SEEDS if new[s]["n_pred"] < 9 or new[s]["top_share"] >= 0.50]
out(f"- Topic-1080 runs with <9 predicted classes or top share >=50%: {bad if bad else 'NONE'}")
out(f"- Topic-1080 top-share range: {min(new[s]['top_share'] for s in SEEDS):.1%} - "
    f"{max(new[s]['top_share'] for s in SEEDS):.1%}")
out()
out("Reference points at the same 136-step budget: Original-180 corpus-draw sd = 0.0111")
out("(9-run grid); Topic-540 at 272 steps = 0.6151 +/- 0.0158.")

NEW.mkdir(parents=True, exist_ok=True)
(NEW / "RESULTS_1080_136.md").write_text("\n".join(lines), encoding="utf-8")
json.dump({"t1080": {str(s): new[s] for s in SEEDS}, "t540": {str(s): old[s] for s in SEEDS}},
          open(NEW / "summary_1080_136.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {NEW / 'RESULTS_1080_136.md'}")
