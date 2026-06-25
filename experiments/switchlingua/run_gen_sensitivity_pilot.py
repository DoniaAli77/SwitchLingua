"""
run_gen_sensitivity_pilot.py — ISOLATED pilots for the generation-config SENSITIVITY experiment.
Each variant runs in its OWN dir + manifest; does NOT touch pilot_v1 / GEN daily_runs / GEN pool.
Cross-dedups every kept sample against GEN-960 and against other variant pilots.
No core/prompt change. Generation-config sensitivity study (NOT EESA-tailored).

Actions:
  gen     : generate + filter + cross-dedup + balance one variant pilot, write metrics.
  report  : load all variant pilots + GEN-960 baseline -> PILOT_REPORT.md (the 12 items).

Layout: multi-agent-bert/data/Sentiment/generated/gen_sensitivity/<variant>/
        {raw_states.jsonl, manifest.json, pilot.csv, pilot.jsonl, gen_stats.json, metrics.json}
"""
import argparse, asyncio, importlib, json, os, pathlib, random, statistics, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SW = ROOT / "experiments" / "switchlingua"
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
GENROOT = ROOT / "multi-agent-bert" / "data" / "Sentiment" / "generated"
SENS = GENROOT / "gen_sensitivity"
GEN960 = GENROOT / "merged" / "switchlingua_sentiment_train_960_320perlabel.jsonl"
MODEL = "gpt-4o-mini"
QTHRESH = 7.0

sys.path.insert(0, str(SW))
import manage_sentiment_data as M  # reuse extract_rows / assign_reasons / helpers


def _norm(t):
    return M._norm(t)


def cmi(r):  # code-mixing index from the deterministic ratios: 100 - dominant-language %
    return round(100 - max(r.get("cs_ar_ratio") or 0, r.get("cs_en_ratio") or 0), 2)


def toklen(t):
    return len(str(t).split())


def metrics(rows):
    if not rows:
        return {}
    ar = [r.get("cs_ar_ratio") or 0 for r in rows]
    cmis = [cmi(r) for r in rows]
    lens = [toklen(r["text"]) for r in rows]
    q = [float(r["quality_score"]) for r in rows if r.get("quality_score") not in (None, "")]
    bins = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for c in cmis:
        k = "80-100" if c >= 80 else "60-80" if c >= 60 else "40-60" if c >= 40 else "20-40" if c >= 20 else "0-20"
        bins[k] += 1
    return {
        "n": len(rows),
        "ar_pct_mean": round(statistics.mean(ar), 1), "en_pct_mean": round(100 - statistics.mean(ar), 1),
        "cmi_mean": round(statistics.mean(cmis), 1), "cmi_median": round(statistics.median(cmis), 1),
        "cmi_hist": bins,
        "len_mean": round(statistics.mean(lens), 1), "len_median": int(statistics.median(lens)),
        "len_p90": int(sorted(lens)[int(0.9 * (len(lens) - 1))]),
        "quality_min": round(min(q), 2) if q else None, "quality_max": round(max(q), 2) if q else None,
        "label_dist": dict(Counter(r["label"] for r in rows)),
    }


# ---------------------------------------------------------------- generation
async def _gen(config, base, mpath, mx, conc):
    import dotenv
    env = ROOT / "Modified_Version" / ".env"
    if env.exists():
        dotenv.load_dotenv(str(env), override=True)
    os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
    import ssl, httpx
    ssl._create_default_https_context = ssl._create_unverified_context
    _c = httpx.Client.__init__; httpx.Client.__init__ = lambda s, *a, **k: (k.setdefault("verify", False), _c(s, *a, **k))[-1]
    _a = httpx.AsyncClient.__init__; httpx.AsyncClient.__init__ = lambda s, *a, **k: (k.setdefault("verify", False), _a(s, *a, **k))[-1]
    if str(MODIFIED_CORE) not in sys.path:
        sys.path.insert(0, str(MODIFIED_CORE))
    for m in ("utils", "node_engine", "node_models", "prompt", "agents", "run_french"):
        sys.modules.pop(m, None)
    importlib.invalidate_caches()
    import utils as ut, node_engine as ne, run_french as rf
    ne.MODEL = MODEL
    (base / "_backup").mkdir(parents=True, exist_ok=True)
    ne.OUTPUT_DIR = str(base / "_backup")

    scn = [s for s in ut.generate_scenarios(ut.load_config(str(config))["pre_execute"]) if s.get("task") == "sentiment"]
    for s in scn:
        s["scenario_id"] = M._scenario_id(s)
    man = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"completed": {}}
    todo = [s for s in scn if s["scenario_id"] not in man["completed"]]
    random.seed(13); random.shuffle(todo); todo = todo[:mx]
    print(f"[{base.name}] scenarios total={len(scn)} done={len(man['completed'])} attempting={len(todo)}")
    sem = asyncio.Semaphore(conc); lock = asyncio.Lock(); prog = {"d": 0, "ok": 0, "f": 0}; states = []

    async def one(sc):
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(sc).run()
            except Exception as exc:
                st = None
                async with lock:
                    prog["f"] += 1
                    if ("429" in str(exc) or "rate_limit" in str(exc).lower()) and prog["f"] % 20 == 1:
                        print(f"[{base.name}] 429 rate-limited")
            async with lock:
                prog["d"] += 1
                if st:
                    st["scenario_id"] = sc["scenario_id"]; states.append(st)
                    man["completed"][sc["scenario_id"]] = 1; prog["ok"] += 1
                if prog["d"] % 20 == 0 or prog["d"] == len(todo):
                    print(f"[{base.name}] {prog['d']}/{len(todo)} ok={prog['ok']} fail={prog['f']}", flush=True)

    await asyncio.gather(*(one(s) for s in todo))
    with open(base / "raw_states.jsonl", "a", encoding="utf-8") as f:
        for st in states:
            f.write(json.dumps(st, ensure_ascii=False, default=str) + "\n")
    mpath.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    return len(todo), prog["ok"], prog["f"]


def _exclude_texts(exclude):
    """Texts to cross-dedup against: GEN-960 + the whole GEN pool (pilot_v1 + GEN daily_runs +
    240/480 snapshots) + any other variant pilots."""
    s = set()
    files = [GEN960,
             GENROOT / "pilot_v1" / "filtered_train.jsonl",
             GENROOT / "merged" / "switchlingua_sentiment_train_480_160perlabel.jsonl",
             GENROOT / "merged" / "switchlingua_sentiment_train_240_80perlabel.jsonl"]
    files += sorted((GENROOT / "daily_runs").glob("run_*_filtered.jsonl"))
    files += [SENS / d / "pilot.jsonl" for d in exclude]
    for p in files:
        if p.exists():
            for l in p.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    s.add(_norm(json.loads(l)["text"]))
    return s


def cmd_gen(a):
    variant = a.variant
    base = SENS / variant; base.mkdir(parents=True, exist_ok=True)
    mpath = base / "manifest.json"
    n_att, ok, fail = asyncio.run(_gen(pathlib.Path(a.config), base, mpath, a.max, a.concurrency))
    # filter
    states = [s for s in M._read_jsonl(base / "raw_states.jsonl") if s.get("task") == "sentiment"]
    rows = []
    for s in states:
        rows.extend(M.extract_rows(s, f"gen_{variant}"))
    M.assign_reasons(rows, QTHRESH)
    nonempty = [r for r in rows if (r["text"] or "").strip()]
    cs_valid_rate = round(100 * sum(1 for r in nonempty if r["cs_valid"]) / len(nonempty), 1) if nonempty else 0
    kept = [r for r in rows if r["filter_reason"] is None]
    # cross-dedup vs GEN-960 + other variants
    excl = _exclude_texts(a.exclude or [])
    cross = 0; kept2 = []
    for r in kept:
        if _norm(r["text"]) in excl:
            cross += 1
        else:
            kept2.append(r)
    # balance to target/label
    rng = random.Random(7); bal = []
    bylab = {}
    for r in kept2:
        bylab.setdefault(r["label"], []).append(r)
    for lab in ("positive", "negative", "neutral"):
        rows_l = bylab.get(lab, []); rng.shuffle(rows_l); bal.extend(rows_l[: a.target])
    rng.shuffle(bal)
    M._write_csv(base / "pilot.csv", bal); M._write_jsonl(base / "pilot.jsonl", bal)
    gs = {"variant": variant, "attempted_scenarios": n_att, "completed_scenarios": ok, "failed": fail,
          "raw_instances": len(rows), "validator_pass": sum(1 for r in nonempty if r["task_validator_passed"]),
          "cs_valid_rate_pct": cs_valid_rate, "kept_after_filters": len(kept),
          "cross_dup_removed": cross, "kept_after_crossdedup": len(kept2),
          "balanced": len(bal), "balanced_by_label": dict(Counter(r["label"] for r in bal)),
          "loss_by_reason": dict(Counter(r["filter_reason"] for r in rows if r["filter_reason"]))}
    (base / "gen_stats.json").write_text(json.dumps(gs, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "metrics.json").write_text(json.dumps(metrics(bal), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{variant}] raw={len(rows)} cs_valid={cs_valid_rate}% kept={len(kept)} crossdup={cross} balanced={dict(Counter(r['label'] for r in bal))}")
    print(f"[{variant}] metrics: {metrics(bal)}")


def cmd_report(a):
    base960 = [json.loads(l) for l in GEN960.read_text(encoding="utf-8").splitlines() if l.strip()] if GEN960.exists() else []
    m960 = metrics(base960)
    L = ["# GEN sensitivity pilots — report\n",
         "Generation-config **sensitivity** study (NOT EESA-tailored). Isolated pilots; cross-deduped vs GEN-960 and each other.\n",
         "## Baseline GEN-960", f"- {m960}\n"]
    variants = [d.name for d in sorted(SENS.iterdir()) if (d / "pilot.jsonl").exists()] if SENS.exists() else []
    for v in variants:
        gs = json.loads((SENS / v / "gen_stats.json").read_text(encoding="utf-8"))
        rows = [json.loads(l) for l in (SENS / v / "pilot.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        mt = metrics(rows)
        L += [f"## {v}",
              f"1. attempted scenarios: {gs['attempted_scenarios']} (completed {gs['completed_scenarios']}, failed {gs['failed']})",
              f"2. raw generated: {gs['raw_instances']}", f"3. valid kept (filters): {gs['kept_after_filters']} | after cross-dedup: {gs['kept_after_crossdedup']} | balanced pilot: {gs['balanced']}",
              f"4. label distribution: {mt.get('label_dist')}", f"5. CS-valid yield: {gs['cs_valid_rate_pct']}%",
              f"6. AR:EN ratio: {mt['ar_pct_mean']} : {mt['en_pct_mean']}  (GEN-960 {m960['ar_pct_mean']}:{m960['en_pct_mean']})",
              f"7. CMI mean {mt['cmi_mean']} (GEN-960 {m960['cmi_mean']}) | hist {mt['cmi_hist']}",
              f"8. length mean/median/p90: {mt['len_mean']}/{mt['len_median']}/{mt['len_p90']}  (GEN-960 {m960['len_mean']}/{m960['len_median']}/{m960['len_p90']})",
              f"9. quality range: {mt['quality_min']}–{mt['quality_max']}", f"10. cross-dup removed: {gs['cross_dup_removed']}",
              f"11. vs GEN-960: AR {mt['ar_pct_mean']-m960['ar_pct_mean']:+.1f}pp, CMI {mt['cmi_mean']-m960['cmi_mean']:+.1f}, len {mt['len_mean']-m960['len_mean']:+.1f}",
              "12. examples:"]
        for r in rows[:10]:
            L.append(f"   - [{r['label']}/{r.get('cs_ratio')}] ar%={r.get('cs_ar_ratio')} cmi={cmi(r)} len={toklen(r['text'])} | {r['text'][:90]}")
        L.append("")
    (SENS / "PILOT_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", SENS / "PILOT_REPORT.md")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--config", required=True); g.add_argument("--variant", required=True)
    g.add_argument("--max", type=int, default=80); g.add_argument("--target", type=int, default=20)
    g.add_argument("--concurrency", type=int, default=4); g.add_argument("--exclude", nargs="*", default=[])
    g.set_defaults(func=cmd_gen)
    r = sub.add_parser("report"); r.set_defaults(func=cmd_report)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
