"""
run_pilot_csfix.py — ISOLATED CS-validity-fix pilot (Experiment C, config-only).
Runs config_sentiment_expC_v2.yaml (cs_ratio 50/60/70 + Intrasentential-only) on a small scenario
cap and filters with the SAME criteria as the main workflow. Writes to a SEPARATE directory and does
NOT touch pilot_v1, the resume manifest, daily_runs, or the merged training set. No prompt/NER/pipeline change.

Output: multi-agent-bert/data/Sentiment/generated/pilot_v2_csfix/
        {raw_outputs.jsonl, filtered.csv, filtered.jsonl, PILOT_V2_REPORT.md}

Usage: python experiments/switchlingua/run_pilot_csfix.py --max 40 --concurrency 4
"""
import argparse, asyncio, importlib, json, os, pathlib, random, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODIFIED_CORE = ROOT / "Modified_Version" / "core"
SW = ROOT / "experiments" / "switchlingua"
CONFIG = SW / "config_sentiment_expC_v2.yaml"
OUT = ROOT / "multi-agent-bert" / "data" / "Sentiment" / "generated" / "pilot_v2_csfix"
MODEL = "gpt-4o-mini"

# baseline (from CS_VALIDITY_DIAGNOSIS.md) for the comparison table
BASE_CS_VALID_PCT = 30
BASE_FULLY_AR_OF_FAIL_PCT = 99.6

# env + SSL
import dotenv
_env = ROOT / "Modified_Version" / ".env"
if _env.exists():
    dotenv.load_dotenv(str(_env), override=True)
os.environ["API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["API_BASE"] = os.environ.get("OPENAI_BASE_URL", "")
import ssl, httpx
ssl._create_default_https_context = ssl._create_unverified_context
_c = httpx.Client.__init__
httpx.Client.__init__ = lambda self, *a, **k: (k.setdefault("verify", False), _c(self, *a, **k))[-1]
_a = httpx.AsyncClient.__init__
httpx.AsyncClient.__init__ = lambda self, *a, **k: (k.setdefault("verify", False), _a(self, *a, **k))[-1]

if str(SW) not in sys.path:
    sys.path.insert(0, str(SW))
if str(MODIFIED_CORE) not in sys.path:
    sys.path.insert(0, str(MODIFIED_CORE))


async def _generate(max_scn, concurrency):
    for m in ("utils", "node_engine", "node_models", "prompt", "agents", "run_french"):
        sys.modules.pop(m, None)
    importlib.invalidate_caches()
    import utils as ut, node_engine as ne, run_french as rf
    ne.MODEL = MODEL
    (OUT / "_backup").mkdir(parents=True, exist_ok=True)
    ne.OUTPUT_DIR = str(OUT / "_backup")

    cfg = ut.load_config(str(CONFIG))
    scn = [s for s in ut.generate_scenarios(cfg["pre_execute"]) if s.get("task") == "sentiment"]
    random.seed(13); random.shuffle(scn)
    scn = scn[:max_scn]
    print(f"[v2] {len(scn)} scenarios (cs_ratio 50/60/70, Intrasentential-only) | concurrency={concurrency}")

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    states, prog = [], {"done": 0, "ok": 0, "fail": 0}

    async def one(sc):
        async with sem:
            try:
                st = await rf.CodeSwitchingAgent(sc).run()
            except Exception as exc:
                st = None
                async with lock:
                    prog["fail"] += 1
                    if ("429" in str(exc) or "rate_limit" in str(exc).lower()) and prog["fail"] % 15 == 1:
                        print(f"[v2] rate-limited (429) — remaining will skip.")
            async with lock:
                prog["done"] += 1
                if st:
                    states.append(st); prog["ok"] += 1
                if prog["done"] % 10 == 0 or prog["done"] == len(scn):
                    print(f"[v2] {prog['done']}/{len(scn)} ok={prog['ok']} fail={prog['fail']}", flush=True)

    await asyncio.gather(*(one(s) for s in scn))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "raw_outputs.jsonl", "w", encoding="utf-8") as f:
        for st in states:
            f.write(json.dumps(st, ensure_ascii=False, default=str) + "\n")
    print(f"[v2] wrote {len(states)} raw states -> {OUT/'raw_outputs.jsonl'} (requested={len(scn)}, fail={prog['fail']})")
    return len(scn), prog["ok"], prog["fail"]


def _analyze(threshold):
    import manage_sentiment_data as M  # same extract/filter/funnel logic
    states = [json.loads(l) for l in (OUT / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for s in states:
        if s.get("task") == "sentiment":
            rows.extend(M.extract_rows(s, "pilot_v2_csfix"))
    M.assign_reasons(rows, threshold)
    stats, kept = M.funnel(rows)
    M._write_csv(OUT / "filtered.csv", kept)
    M._write_jsonl(OUT / "filtered.jsonl", kept)
    # CS-valid rate among non-empty (independent of quality filter)
    nonempty = [r for r in rows if (r["text"] or "").strip()]
    cs_valid = [r for r in nonempty if r["cs_valid"]]
    fails = [r for r in nonempty if not r["cs_valid"]]
    fully_ar = [r for r in fails if (r["cs_ar_ratio"] or 0) > 0 and (r["cs_en_ratio"] or 0) == 0]
    return stats, kept, rows, nonempty, cs_valid, fails, fully_ar


def _report(gen, stats, kept, nonempty, cs_valid, fails, fully_ar, threshold):
    req, ok, fail = gen
    cs_rate = (100 * len(cs_valid) / len(nonempty)) if nonempty else 0
    fa_of_fail = (100 * len(fully_ar) / len(fails)) if fails else 0
    by_label = Counter(r["label"] for r in kept)
    by_topic = Counter(r["topic"] for r in kept)
    by_ratio_csrate = {}
    from collections import defaultdict
    g = defaultdict(lambda: [0, 0])
    for r in nonempty:
        g[r["cs_ratio"]][1] += 1
        if r["cs_valid"]:
            g[r["cs_ratio"]][0] += 1
    for k, (v, n) in g.items():
        by_ratio_csrate[k] = f"{v}/{n} ({100*v/n:.0f}%)" if n else "—"

    L = ["# Pilot v2 (CS-validity fix) — Results\n",
         "Config-only change: `cs_ratio: [50%,60%,70%]`, `cs_type: [Intrasentential]`. No prompt/NER/pipeline "
         "change. Isolated run — **not merged** into the training set.\n",
         "## Run",
         f"- scenarios requested: **{req}** | completed: **{ok}** | failed (429/other): **{fail}**",
         f"- quality threshold: **{threshold}** (unchanged)\n",
         "## Yield vs baseline",
         "| metric | baseline (v1) | pilot v2 |", "|---|--:|--:|",
         f"| CS-valid rate (of non-empty) | {BASE_CS_VALID_PCT}% | **{cs_rate:.0f}%** ({len(cs_valid)}/{len(nonempty)}) |",
         f"| fully-Arabic share of failures | {BASE_FULLY_AR_OF_FAIL_PCT}% | **{fa_of_fail:.0f}%** ({len(fully_ar)}/{len(fails)}) |",
         "\n## CS-valid rate by cs_ratio target", "| cs_ratio | CS-valid |", "|---|---|"]
    for k in ("50%", "60%", "70%"):
        if k in by_ratio_csrate:
            L.append(f"| {k} | {by_ratio_csrate[k]} |")
    L += ["\n## Filter funnel (instances)", "| stage | count |", "|---|--:|",
          f"| raw generated | {stats['raw_instances']} |",
          f"| kept (validator+CS-valid+quality>={threshold}+dedup) | {stats['kept']} |",
          "\n**Filtering loss by reason:**", "| reason | count |", "|---|--:|"]
    for r in M_REASONS:
        L.append(f"| {r} | {stats['loss_by_reason'].get(r, 0)} |")
    L += [f"| duplicates removed | {stats['duplicates_removed']} |",
          "\n## Kept by label", "| label | kept |", "|---|--:|"]
    for l in ("positive", "negative", "neutral"):
        L.append(f"| {l} | {by_label.get(l, 0)} |")
    L += ["\n## Kept by topic", "| topic | kept |", "|---|--:|"]
    for t, c in sorted(by_topic.items(), key=lambda x: -x[1]):
        L.append(f"| {t} | {c} |")
    L += ["\n## Examples of newly-valid code-switched outputs"]
    for r in kept[:10]:
        L.append(f"- [{r['label']}/{r['cs_ratio']}/{r['intensity']}] ar%={r['cs_ar_ratio']} q={r['quality_score']} | {r['text'][:95]}")
    L += ["\n## Verdict",
          f"- CS-valid {BASE_CS_VALID_PCT}% → **{cs_rate:.0f}%** "
          f"({'IMPROVED' if cs_rate > BASE_CS_VALID_PCT else 'NOT improved'}).",
          "- **Not merged** into the main training set (per instruction; merge only if clearly better).",
          "- Multi-Agent BERT not trained from this."]
    (OUT / "PILOT_V2_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    return cs_rate, fa_of_fail, by_label, by_topic


M_REASONS = ["empty_text", "validator_failed", "not_cs_valid", "low_quality", "duplicate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--quality-threshold", type=float, default=7.0)
    ap.add_argument("--filter-only", action="store_true")
    args = ap.parse_args()

    gen = (0, 0, 0)
    if not args.filter_only:
        gen = asyncio.run(_generate(args.max, args.concurrency))
    elif (OUT / "raw_outputs.jsonl").exists():
        n = sum(1 for l in (OUT / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
        gen = (n, n, 0)

    stats, kept, rows, nonempty, cs_valid, fails, fully_ar = _analyze(args.quality_threshold)
    cs_rate, fa_of_fail, by_label, by_topic = _report(gen, stats, kept, nonempty, cs_valid, fails, fully_ar, args.quality_threshold)

    print(f"\n[v2] raw={stats['raw_instances']} CS-valid={len(cs_valid)}/{len(nonempty)} ({cs_rate:.0f}%, baseline 30%) "
          f"| fully-Arabic share of fails={fa_of_fail:.0f}% (baseline 99.6%)")
    print(f"[v2] kept={stats['kept']} by_label={dict(by_label)}")
    print(f"[v2] loss={stats['loss_by_reason']}")
    print(f"[v2] report -> {OUT/'PILOT_V2_REPORT.md'}  (NOT merged)")


if __name__ == "__main__":
    main()
