"""3x3 subset x model-seed grid for the 180-sentence point at 136 optimizer steps.

Cells: subsets topic_180_seed{42,43,44} x model seeds {42,43,44}. The subset-42 row is
read from experiment_topic_arabicdominant_mc136/orig180_mseed*, the other six from
experiment_topic180_subset_variance/. Identical recipe throughout (verified by the
shared runner scripts).

Separates the two variance sources that the earlier single-subset result confounded:
  - model-seed variance : spread WITHIN a subset row
  - subset variance     : spread ACROSS subset row means
and places the Topic-360 / Topic-540 matched-compute points on the same scale.
"""
import json, glob, os, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_SV = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic180_subset_variance"
OUT_MC136 = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic_arabicdominant_mc136"
OUT_MC = ROOT / "experiments/outputs/multi_agent_bert/experiment_topic_matched_compute"
SEEDS = [42, 43, 44]
LABELS = ["business", "education", "health", "shopping", "medical",
          "sports", "tech", "finance", "social"]


def metrics(d):
    pj = glob.glob(os.path.join(str(d), "*_predictions.json"))
    if not pj:
        return None
    rows = json.load(open(pj[0], encoding="utf-8"))
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    f1s = {}
    for lab in LABELS:
        tp = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] == lab)
        fp = sum(1 for r in rows if r["true_label"] != lab and r["predicted_label"] == lab)
        fn = sum(1 for r in rows if r["true_label"] == lab and r["predicted_label"] != lab)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s[lab] = 2 * p * rc / (p + rc) if p + rc else 0.0
    from collections import Counter
    pred = Counter(r["predicted_label"] for r in rows)
    return {"acc": acc, "macro_f1": sum(f1s.values()) / len(LABELS),
            "n_pred": len(pred), "top_share": pred.most_common(1)[0][1] / n}


grid = {}
for sub in SEEDS:
    for ms in SEEDS:
        d = (OUT_MC136 / f"orig180_mseed{ms}") if sub == 42 else (OUT_SV / f"sub{sub}_mseed{ms}")
        m = metrics(d)
        if m is None:
            raise SystemExit(f"missing: {d}")
        grid[(sub, ms)] = m

lines = []


def out(s=""):
    print(s)
    lines.append(s)


out("# Topic-180 subset x model-seed variance at 136 steps")
out()
out("3 nested 180-subsets x 3 model seeds = 9 runs, identical recipe, primary_only on Silver-1163.")
out()
out("## Accuracy grid")
out()
out("| subset \\ model seed | 42 | 43 | 44 | row mean | row sd |")
out("|---|--:|--:|--:|--:|--:|")
row_means = {}
for sub in SEEDS:
    xs = [grid[(sub, ms)]["acc"] for ms in SEEDS]
    row_means[sub] = statistics.mean(xs)
    out(f"| topic_180_seed{sub} | " + " | ".join(f"{x:.4f}" for x in xs) +
        f" | **{statistics.mean(xs):.4f}** | {statistics.pstdev(xs):.4f} |")
col_means = {ms: statistics.mean(grid[(sub, ms)]["acc"] for sub in SEEDS) for ms in SEEDS}
out("| **col mean** | " + " | ".join(f"{col_means[ms]:.4f}" for ms in SEEDS) + " | | |")

all_acc = [grid[c]["acc"] for c in grid]
within = statistics.mean(statistics.pstdev([grid[(sub, ms)]["acc"] for ms in SEEDS]) for sub in SEEDS)
across = statistics.pstdev(list(row_means.values()))
out()
out("## Variance decomposition (accuracy)")
out()
out(f"| source | sd |")
out(f"|---|--:|")
out(f"| model seed, within subset (mean of row sds) | {within:.4f} |")
out(f"| **subset, across subset means** | **{across:.4f}** |")
out(f"| all 9 runs pooled | {statistics.pstdev(all_acc):.4f} |")
out()
out(f"Grand mean over the 9 runs: **{statistics.mean(all_acc):.4f}** "
    f"(min {min(all_acc):.4f}, max {max(all_acc):.4f})")

out()
out("## Macro-F1 grid")
out()
out("| subset \\ model seed | 42 | 43 | 44 | row mean |")
out("|---|--:|--:|--:|--:|")
for sub in SEEDS:
    xs = [grid[(sub, ms)]["macro_f1"] for ms in SEEDS]
    out(f"| topic_180_seed{sub} | " + " | ".join(f"{x:.4f}" for x in xs) +
        f" | **{statistics.mean(xs):.4f}** |")

out()
out("## Collapse check (all 9)")
out()
bad = [f"sub{sub}/m{ms}" for sub in SEEDS for ms in SEEDS
       if grid[(sub, ms)]["n_pred"] < 9 or grid[(sub, ms)]["top_share"] >= 0.50]
out(f"- runs predicting <9 classes or with top share >=50%: {bad if bad else 'NONE'}")
out(f"- top-share range: {min(grid[c]['top_share'] for c in grid):.1%} - "
    f"{max(grid[c]['top_share'] for c in grid):.1%}")

# comparison points at the same 136-step budget
def group(paths):
    xs = [metrics(p)["acc"] for p in paths]
    return statistics.mean(xs), statistics.pstdev(xs)

m360 = group([OUT_MC / f"mc360_seed{s}" for s in SEEDS])
m540 = group([OUT_MC / f"mc540_seed{s}" for s in SEEDS])
mad = group([OUT_MC136 / f"ad180_mseed{s}" for s in SEEDS])
out()
out("## Same 136-step budget, other corpora")
out()
out("| corpus | acc | sd |")
out("|---|--:|--:|")
out(f"| Original-180 (9 runs, 3 subsets) | {statistics.mean(all_acc):.4f} | {statistics.pstdev(all_acc):.4f} |")
out(f"| Topic-360 (3 subsets x 1 seed) | {m360[0]:.4f} | {m360[1]:.4f} |")
out(f"| Topic-540 (1 corpus x 3 seeds) | {m540[0]:.4f} | {m540[1]:.4f} |")
out(f"| ArabicDominant-180 (1 corpus x 3 seeds) | {mad[0]:.4f} | {mad[1]:.4f} |")

OUT_SV.mkdir(parents=True, exist_ok=True)
(OUT_SV / "SUBSET_VARIANCE_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
json.dump({f"sub{a}_m{b}": v for (a, b), v in grid.items()},
          open(OUT_SV / "subset_variance_summary.json", "w", encoding="utf-8"), indent=2)
print(f"\nwrote {OUT_SV / 'SUBSET_VARIANCE_RESULTS.md'}")
