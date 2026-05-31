"""
analyze_consolidated_human_eval.py
==================================
Analyze the filled consolidated human-annotation sheet(s). One sheet, four analyses:
  1. By task: label/task correctness, CS validity, fluency mean, naturalness mean, acceptability rate
  2. Masked vs non-masked: differences in acceptability, CS validity, fluency, naturalness (+ Mann-Whitney p)
  3. Pipeline agreement: false positives / false negatives / agreement vs human acceptability (+ kappa)
  4. NER: entity correctness, BIO validity, boundary correctness

No scipy (numpy + math). Auto-detects filled sheets in the human_eval/ folder.

Usage:
  python experiments/switchlingua/analyze_consolidated_human_eval.py
  python experiments/switchlingua/analyze_consolidated_human_eval.py annotator1.csv annotator2.csv
"""
import csv
import math
import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIR = ROOT / "experiments" / "outputs" / "switchlingua" / "human_eval"


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def mann_whitney_p(a, b):
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    allv = np.array(a + b, float)
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    _, counts = np.unique(allv, return_counts=True)
    n = n1 + n2; tie = (counts ** 3 - counts).sum()
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))) if n > 1 else 0
    if sigma == 0:
        return float("nan")
    return 2 * (1 - _norm_cdf(abs((u1 - n1 * n2 / 2.0) / sigma)))


def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return float("nan")
    cats = sorted({c for p in pairs for c in p})
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in pairs:
        m[idx[x], idx[y]] += 1
    tot = m.sum(); po = np.trace(m) / tot
    pe = sum(m[i].sum() * m[:, i].sum() for i in range(len(cats))) / (tot * tot)
    return float((po - pe) / (1 - pe)) if (1 - pe) else float("nan")


def yn(v):
    v = (v or "").strip().lower()
    return "yes" if v in {"yes", "y", "1", "true"} else ("no" if v in {"no", "n", "0", "false"} else None)


def num(v):
    v = (v or "").strip()
    try:
        return float(v) if v != "" else None
    except ValueError:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 3) if xs else None


def rate(flags):
    flags = [f for f in flags if f is not None]
    return round(100 * sum(f == "yes" for f in flags) / len(flags), 1) if flags else None


def load(path):
    rows = []
    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        rows.append({
            "task": (r.get("task") or "").strip(),
            "masked": yn(r.get("masked_case")),
            "pipeline_accepted": yn(r.get("pipeline_accepted")),
            "is_cs": yn(r.get("is_code_switched_yes_no")),
            "task_correct": yn(r.get("label_or_task_correct_yes_no")),
            "fluency": num(r.get("fluency_1_10")),
            "naturalness": num(r.get("naturalness_1_10")),
            "acceptable": yn(r.get("overall_acceptable_yes_no")),
            "entities_correct": yn(r.get("entities_correct_yes_no")),
            "bio_valid": yn(r.get("bio_valid_yes_no")),
            "boundary_correct": yn(r.get("boundary_correct_yes_no")),
        })
    return rows


def find_filled():
    res = []
    for p in sorted(DIR.glob("*.csv")):
        if p.name == "analysis_summary.csv":
            continue
        rows = load(p)
        if any(r["acceptable"] is not None or r["fluency"] is not None for r in rows):
            res.append(p)
    return res


def analyse(name, rows):
    print(f"\n===== {name} ({len(rows)} rows) =====")

    # 1) by task
    print("\n[1] By task")
    print(f"  {'task':10} {'n':>3} {'task_correct%':>13} {'CS_valid%':>10} {'fluency':>8} {'natural':>8} {'accept%':>8}")
    for task in ("topic", "sentiment", "ner"):
        tr = [r for r in rows if r["task"] == task]
        if not tr:
            continue
        print(f"  {task:10} {len(tr):>3} {str(rate([r['task_correct'] for r in tr])):>13} "
              f"{str(rate([r['is_cs'] for r in tr])):>10} {str(mean([r['fluency'] for r in tr])):>8} "
              f"{str(mean([r['naturalness'] for r in tr])):>8} {str(rate([r['acceptable'] for r in tr])):>8}")

    # 2) masked vs non-masked
    print("\n[2] Masked vs non-masked")
    mk = [r for r in rows if r["masked"] == "yes"]
    nm = [r for r in rows if r["masked"] == "no"]
    print(f"  acceptability:  masked={rate([r['acceptable'] for r in mk])}%  non-masked={rate([r['acceptable'] for r in nm])}%")
    print(f"  CS validity:    masked={rate([r['is_cs'] for r in mk])}%  non-masked={rate([r['is_cs'] for r in nm])}%")
    print(f"  fluency:        masked={mean([r['fluency'] for r in mk])}  non-masked={mean([r['fluency'] for r in nm])}"
          f"  (MW p={round(mann_whitney_p([r['fluency'] for r in mk],[r['fluency'] for r in nm]),4)})")
    print(f"  naturalness:    masked={mean([r['naturalness'] for r in mk])}  non-masked={mean([r['naturalness'] for r in nm])}"
          f"  (MW p={round(mann_whitney_p([r['naturalness'] for r in mk],[r['naturalness'] for r in nm]),4)})")

    # 3) pipeline agreement (pipeline_accepted vs human acceptable)
    print("\n[3] Pipeline vs human acceptability")
    fp = fn = agree = n = 0
    pa, ha = [], []
    for r in rows:
        if r["pipeline_accepted"] is None or r["acceptable"] is None:
            continue
        n += 1; pa.append(r["pipeline_accepted"]); ha.append(r["acceptable"])
        if r["pipeline_accepted"] == r["acceptable"]:
            agree += 1
        elif r["pipeline_accepted"] == "yes" and r["acceptable"] == "no":
            fp += 1     # pipeline accepted, human rejects
        elif r["pipeline_accepted"] == "no" and r["acceptable"] == "yes":
            fn += 1     # pipeline rejected, human accepts
    if n:
        print(f"  n={n}  agreement={round(100*agree/n,1)}%  false_positives={fp} ({round(100*fp/n,1)}%)  "
              f"false_negatives={fn} ({round(100*fn/n,1)}%)  kappa={round(cohen_kappa(pa,ha),3)}")
    else:
        print("  (no comparable rows)")

    # 4) NER
    ner = [r for r in rows if r["task"] == "ner"]
    print("\n[4] NER")
    if ner:
        print(f"  n={len(ner)}  entities_correct={rate([r['entities_correct'] for r in ner])}%  "
              f"bio_valid={rate([r['bio_valid'] for r in ner])}%  boundary_correct={rate([r['boundary_correct'] for r in ner])}%")
    else:
        print("  (no NER rows)")


def main():
    sheets = [pathlib.Path(a) for a in sys.argv[1:]]
    sheets = [p if p.is_absolute() else (DIR / p) for p in sheets] or find_filled()
    if not sheets:
        print(f"No FILLED sheet found in {DIR}.")
        print("Annotators fill consolidated_annotation_sheet.csv, save as annotator1.csv, then re-run.")
        return
    all_rows = {}
    for p in sheets:
        rows = load(p)
        analyse(p.name, rows)
        all_rows[p.name] = rows

    # inter-annotator agreement on overall_acceptable (if >=2)
    if len(sheets) >= 2:
        a, b = sheets[0].name, sheets[1].name
        ra = {i: r["acceptable"] for i, r in enumerate(all_rows[a])}
        rb = {i: r["acceptable"] for i, r in enumerate(all_rows[b])}
        ids = [i for i in ra if ra[i] and rb.get(i)]
        k = cohen_kappa([ra[i] for i in ids], [rb[i] for i in ids])
        print(f"\n===== inter-annotator agreement (overall_acceptable, {a} vs {b}): kappa={round(k,3)} =====")


if __name__ == "__main__":
    main()
