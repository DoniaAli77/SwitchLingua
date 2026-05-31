"""
step3_analyze_human_sheet.py — Analyze the filled per-dimension human sheet(s).
===============================================================================
For each dimension (fluency, naturalness, cultural) and a composite:
  - do humans rate MASKED sentences lower than neighbours?  (Mann-Whitney p)
  - does human rating agree with the AI's score?            (Spearman)
Plus monolingual-leak %, per-scenario sign test, and inter-annotator agreement.
No scipy (numpy + math only).

Usage:
  python step3_analyze_human_sheet.py                 # auto-detect filled sheets
  python step3_analyze_human_sheet.py annotator1.csv  # or name them
"""
import csv
import math
import pathlib
import sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DIR = HERE / "step3_human_check"
KEY = DIR / "answer_key.csv"

DIMS = ["fluency", "naturalness", "cultural"]          # human columns: <dim>_1to10
MACHINE = {"fluency": "machine_fluency", "naturalness": "machine_naturalness",
           "cultural": "machine_cultural"}


def _norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def mann_whitney_p(a, b):
    a, b = list(a), list(b)
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
    n = n1 + n2
    tie = (counts ** 3 - counts).sum()
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))) if n > 1 else 0
    if sigma == 0:
        return float("nan")
    z = (u1 - n1 * n2 / 2.0) / sigma
    return 2 * (1 - _norm_cdf(abs(z)))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def sign_test_p(wins, n):
    if n == 0:
        return float("nan")
    z = (wins - n / 2.0) / math.sqrt(n / 4.0)
    return 2 * (1 - _norm_cdf(abs(z)))


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


def _num(v):
    v = (v or "").strip()
    try:
        return float(v) if v != "" else None
    except ValueError:
        return None


def _yn(v):
    v = (v or "").strip().lower()
    return "yes" if v in {"yes", "y", "1", "true"} else ("no" if v in {"no", "n", "0", "false"} else None)


def load_key():
    return {r["id"]: r for r in csv.DictReader(KEY.open(encoding="utf-8-sig"))}


def load_filled(path):
    out = {}
    for r in csv.DictReader(path.open(encoding="utf-8-sig")):
        rid = (r.get("id") or "").strip()
        if not rid:
            continue
        out[rid] = {d: _num(r.get(f"{d}_1to10")) for d in DIMS}
        out[rid]["cs"] = _yn(r.get("is_real_codeswitch_yes_no"))
        vals = [out[rid][d] for d in DIMS if out[rid][d] is not None]
        out[rid]["composite"] = sum(vals) / len(vals) if vals else None
    return out


def find_filled():
    res = []
    for p in sorted(DIR.glob("*.csv")):
        if p.name in {"answer_key.csv", "analysis_summary.csv"}:
            continue
        f = load_filled(p)
        if any(v["composite"] is not None for v in f.values()):
            res.append(p)
    return res


def analyse(name, filled, key):
    print(f"--- {name} ---")
    rows = []
    # per dimension: MASKED vs neighbour, and human-vs-machine agreement
    for dim in DIMS:
        mask_v, neigh_v, hum, mach = [], [], [], []
        for rid, ans in filled.items():
            k = key.get(rid)
            if not k or ans[dim] is None:
                continue
            (mask_v if k["role"] == "MASKED" else neigh_v).append(ans[dim])
            mv = k.get(MACHINE[dim])
            if mv not in (None, "", "None"):
                hum.append(ans[dim]); mach.append(float(mv))
        p = mann_whitney_p(mask_v, neigh_v)
        rho = spearman(mach, hum)
        mm = round(float(np.mean(mask_v)), 2) if mask_v else None
        nm = round(float(np.mean(neigh_v)), 2) if neigh_v else None
        print(f"   {dim:12s}: MASKED={mm}  neighbour={nm}  gap={round(nm-mm,2) if mm and nm else None}"
              f"  MW_p={round(p,4) if p==p else None}  human-vs-AI rho={round(rho,3) if rho==rho else None}")
        rows.append({"annotator": name, "dimension": dim, "masked_mean": mm, "neighbour_mean": nm,
                     "gap": round(nm - mm, 2) if mm and nm else None,
                     "mann_whitney_p": round(p, 4) if p == p else None,
                     "human_vs_ai_spearman": round(rho, 3) if rho == rho else None})

    # composite per-scenario sign test
    per = {}
    for rid, ans in filled.items():
        k = key.get(rid)
        if not k or ans["composite"] is None:
            continue
        per.setdefault(k["scenario_idx"], {"m": [], "n": []})
        per[k["scenario_idx"]]["m" if k["role"] == "MASKED" else "n"].append(ans["composite"])
    wins = comp = 0
    for d in per.values():
        if d["m"] and d["n"]:
            comp += 1; wins += np.mean(d["m"]) < np.mean(d["n"])
    sp = sign_test_p(wins, comp)
    print(f"   composite: scenarios MASKED-lower = {wins}/{comp}  sign_p={round(sp,4) if sp==sp else None}")

    # monolingual leak
    mt = mn = nt = nn = 0
    for rid, ans in filled.items():
        k = key.get(rid)
        if not k or ans["cs"] is None:
            continue
        if k["role"] == "MASKED":
            mt += 1; mn += ans["cs"] == "no"
        else:
            nt += 1; nn += ans["cs"] == "no"
    print(f"   monolingual-leak: MASKED={round(100*mn/mt,1) if mt else None}%  neighbour={round(100*nn/nt,1) if nt else None}%\n")
    rows.append({"annotator": name, "dimension": "composite_scenario_sign",
                 "masked_mean": f"{wins}/{comp}", "neighbour_mean": "", "gap": "",
                 "mann_whitney_p": round(sp, 4) if sp == sp else None, "human_vs_ai_spearman": ""})
    return rows


def main():
    if not KEY.exists():
        print(f"No answer key: {KEY}. Run step3_build_human_sheet.py first.")
        return
    key = load_key()
    sheets = [pathlib.Path(a) for a in sys.argv[1:]] or find_filled()
    if not sheets:
        print("No FILLED sheet yet. Annotators fill human_check_sheet.csv,")
        print("save as annotator1.csv in step3_human_check/, then re-run.")
        return

    print(f"Analyzing: {[p.name for p in sheets]}\n")
    all_rows, comps = [], {}
    for p in sheets:
        f = load_filled(p)
        all_rows += analyse(p.name, f, key)
        comps[p.name] = {rid: v["composite"] for rid, v in f.items()}

    if len(sheets) >= 2:
        a, b = sheets[0].name, sheets[1].name
        ids = [i for i in comps[a] if comps[a].get(i) is not None and comps[b].get(i) is not None]
        rho = spearman([comps[a][i] for i in ids], [comps[b][i] for i in ids])
        print(f"--- agreement (composite, {a} vs {b}): Spearman = {round(rho,3) if rho==rho else None} ---\n")

    out = DIR / "analysis_summary.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        w.writeheader(); w.writerows(all_rows)
    print(f"wrote {out.name}")
    print("\nHOW TO READ: for each dimension, MASKED mean should be LOWER than neighbour")
    print("(gap>0, MW_p<0.05). Positive human-vs-AI rho means the AI's scores track humans.")


if __name__ == "__main__":
    main()
