"""
analyze_consolidated_human_eval.py — analyze the filled consolidated human-annotation sheet(s).
Computes the 9 requested analyses. No scipy (numpy + math). Auto-detects filled sheets.

Usage:
  python experiments/switchlingua/analyze_consolidated_human_eval.py            # auto-detect
  python experiments/switchlingua/analyze_consolidated_human_eval.py annotator1.csv annotator2.csv
"""
import csv, math, pathlib, sys
import numpy as np

DIR = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "outputs" / "switchlingua" / "human_eval"
SHEET = "consolidated_human_annotation_sheet.csv"


def yn(v):
    v = (v or "").strip().lower()
    return "yes" if v in {"yes", "y", "1", "true"} else ("no" if v in {"no", "n", "0", "false"} else None)


def num(v):
    v = (v or "").strip()
    try:
        return float(v) if v != "" else None
    except ValueError:
        return None


def rate(flags):
    f = [x for x in flags if x is not None]
    return (round(100 * sum(x == "yes" for x in f) / len(f), 1), len(f)) if f else (None, 0)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 2) if xs else None


def agree(a, b):
    p = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return (round(100 * sum(x == y for x, y in p) / len(p), 1), len(p)) if p else (None, 0)


def kappa(a, b):
    p = [(x, y) for x, y in zip(a, b) if x and y]
    if not p:
        return None
    cats = sorted({c for q in p for c in q})
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in p:
        m[idx[x], idx[y]] += 1
    tot = m.sum(); po = np.trace(m) / tot
    pe = sum(m[i].sum() * m[:, i].sum() for i in range(len(cats))) / (tot * tot)
    return round(float((po - pe) / (1 - pe)), 3) if (1 - pe) else None


def mann_whitney_p(a, b):
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if not a or not b:
        return None
    allv = np.array(a + b, float); order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); sv = allv[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    n1, n2 = len(a), len(b)
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    _, c = np.unique(allv, return_counts=True); n = n1 + n2; tie = (c ** 3 - c).sum()
    sig = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))) if n > 1 else 0
    if sig == 0:
        return None
    z = (u1 - n1 * n2 / 2.0) / sig
    return round(2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))), 4)


def load(path):
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        out.append(r)
    return out


def judge_correct(r):
    """Was the AI judge 'correct' vs the target (derived from pipeline label)?"""
    t = r["task"]; lab = (r.get("pipeline_task_correct_or_judge_label") or "").strip().lower()
    tgt = (r.get("target_label") or "").strip().lower()
    if t == "sentiment":
        return "yes" if lab == tgt else "no"
    if t == "topic":
        return "yes" if lab == "relevant" else "no"
    if t == "ner":
        return "yes" if lab == "pass" else "no"
    return None


def analyse(name, rows):
    print(f"\n===== {name} ({len(rows)} rows) =====")
    filled = [r for r in rows if yn(r.get("human_overall_acceptable")) is not None or num(r.get("human_fluency_1_5")) is not None]
    if not filled:
        print("  (no human ratings filled in this sheet)"); return
    rows = filled

    # 1) human task correctness by task
    print("[1] Human task-correct % by task:")
    for task in ("topic", "sentiment", "ner"):
        tr = [r for r in rows if r["task"] == task]
        pc, n = rate([yn(r.get("human_task_correct")) for r in tr])
        print(f"    {task:9} {pc}%  (n={n})")
    # 2) CS validity, 3) acceptability (overall)
    print(f"[2] Human CS-valid: {rate([yn(r.get('human_cs_valid')) for r in rows])[0]}%")
    print(f"[3] Human overall-acceptable: {rate([yn(r.get('human_overall_acceptable')) for r in rows])[0]}%")

    # 4) AI judge vs human agreement (task correctness; + sentiment label kappa)
    jc = [judge_correct(r) for r in rows]
    hc = [yn(r.get("human_task_correct")) for r in rows]
    a, n = agree(jc, hc)
    print(f"[4] AI-judge vs human (task-correct): agreement {a}% (n={n}), kappa {kappa(jc, hc)}")
    sent = [r for r in rows if r["task"] == "sentiment"]
    sj = [(r.get('pipeline_task_correct_or_judge_label') or '').strip().lower() or None for r in sent]
    sh = [(r.get('human_sentiment_label') or '').strip().lower() or None for r in sent]
    print(f"    sentiment label (judge vs human): agreement {agree(sj, sh)[0]}% (n={agree(sj,sh)[1]}), kappa {kappa(sj, sh)}")

    # 5) TaskValidator vs human
    vv = [yn(r.get("task_validator_passed")) for r in rows]
    print(f"[5] TaskValidator vs human (task-correct): agreement {agree(vv, hc)[0]}% (n={agree(vv,hc)[1]}), kappa {kappa(vv, hc)}")

    # 6) masked vs non-masked human quality
    mk = [r for r in rows if (r.get("masked_case") or "").strip().lower() == "yes"]
    nm = [r for r in rows if (r.get("masked_case") or "").strip().lower() == "no"]
    print("[6] Masked vs non-masked (human):")
    print(f"    acceptable:  masked={rate([yn(r.get('human_overall_acceptable')) for r in mk])[0]}%  non={rate([yn(r.get('human_overall_acceptable')) for r in nm])[0]}%")
    print(f"    fluency:     masked={mean([num(r.get('human_fluency_1_5')) for r in mk])}  non={mean([num(r.get('human_fluency_1_5')) for r in nm])}"
          f"  (MW p={mann_whitney_p([num(r.get('human_fluency_1_5')) for r in mk],[num(r.get('human_fluency_1_5')) for r in nm])})")
    print(f"    naturalness: masked={mean([num(r.get('human_naturalness_1_5')) for r in mk])}  non={mean([num(r.get('human_naturalness_1_5')) for r in nm])}")

    # 7) neutral sentiment dispute resolution
    disp = [r for r in sent if "DISPUTED" in (r.get("notes_for_annotator") or "")]
    if disp:
        from collections import Counter
        hlab = Counter((r.get("human_sentiment_label") or "").strip().lower() for r in disp if r.get("human_sentiment_label"))
        agreed_target = sum(1 for r in disp if (r.get("human_sentiment_label") or "").strip().lower() == (r.get("target_label") or "").strip().lower())
        print(f"[7] Neutral disputes ({len(disp)}): human labels {dict(hlab)}; human agrees with TARGET (neutral) on {agreed_target}/{len(disp)}"
              f" -> {'generator/target right (judge wrong)' if agreed_target>len(disp)/2 else 'judge right (generator failed neutral)'}")
    else:
        print("[7] Neutral disputes: none in filled rows")

    # 8) NER human correctness + English-script compliance
    ner = [r for r in rows if r["task"] == "ner"]
    if ner:
        print(f"[8] NER human: ner_correct {rate([yn(r.get('human_ner_correct')) for r in ner])[0]}%, "
              f"required_types_present {rate([yn(r.get('required_entity_types_present')) for r in ner])[0]}%, "
              f"required_english_script {rate([yn(r.get('required_entities_english_script')) for r in ner])[0]}%")
    else:
        print("[8] NER: no rows")

    # 9) CS-ratio: human Arabic ratio vs deterministic
    errs = []
    for r in rows:
        ar, en, ot = num(r.get("human_arabic_token_count")), num(r.get("human_english_token_count")), num(r.get("human_other_token_count"))
        det = num(r.get("cs_ratio_deterministic"))
        if ar is not None and en is not None and det is not None:
            tot = ar + en + (ot or 0)
            if tot > 0:
                errs.append(abs(100 * ar / tot - det))
    if errs:
        print(f"[9] CS-ratio: human-vs-deterministic Arabic% MAE = {round(float(np.mean(errs)),2)} (n={len(errs)})")
    else:
        print("[9] CS-ratio: no token counts filled")


def main():
    sheets = [pathlib.Path(a) for a in sys.argv[1:]]
    sheets = [p if p.is_absolute() else (DIR / p) for p in sheets]
    if not sheets:
        sheets = [p for p in sorted(DIR.glob("*.csv")) if p.name != "analysis_summary.csv"]
    any_filled = False
    for p in sheets:
        if not p.exists():
            continue
        rows = load(p)
        if any(yn(r.get("human_overall_acceptable")) is not None or num(r.get("human_fluency_1_5")) is not None for r in rows):
            any_filled = True
        analyse(p.name, rows)
    if not any_filled:
        print(f"\nNo FILLED ratings yet. Annotators fill {SHEET}, save as annotator1.csv in\n  {DIR}\nthen re-run.")


if __name__ == "__main__":
    main()
