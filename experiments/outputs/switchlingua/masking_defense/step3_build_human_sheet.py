"""
step3_build_human_sheet.py — Build a BLIND per-dimension human-check sheet.
===========================================================================
Annotators rate each sentence on 3 dimensions (1-10) + a yes/no code-switch check:
  fluency_1to10, naturalness_1to10, cultural_1to10, is_real_codeswitch_yes_no

We take masking scenarios (average >= bar, but >=1 sentence below bar), keep WHOLE
scenarios (weak "MASKED" sentence + its good neighbours), select a balanced
subset of ~SUBSET_TARGET sentences, shuffle, and hide which is which.

The hidden answer key also stores the MACHINE's per-dimension scores, so the
analysis can validate the AI judge dimension-by-dimension.

Outputs to step3_human_check/:
  human_check_sheet.csv  — give to annotators (blind)
  answer_key.csv         — KEEP HIDDEN (roles + machine per-dimension scores)
  README.md              — instructions
"""
import csv
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "step1_raw_data" / "Arabic.jsonl"
OUT = HERE / "step3_human_check"
OUT.mkdir(parents=True, exist_ok=True)

BAR = 7.0
SEED = 42
SUBSET_TARGET = 50   # aim for ~50 sentences (whole scenarios kept together)


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _dim(arr, j, key):
    if isinstance(arr, list) and j < len(arr) and isinstance(arr[j], dict):
        try:
            return round(float(arr[j].get(key, 0) or 0), 3)
        except (TypeError, ValueError):
            return None
    return None


def main():
    recs = load(RAW)

    # group sentences by masking scenario
    scenarios = []  # list of (task, cs_type, [sentence_items])
    for idx, r in enumerate(recs):
        scores = r.get("sentence_scores", [])
        sents = r.get("data_generation_result", [])
        if not isinstance(scores, list) or len(scores) < 2:
            continue
        agg = float(r.get("score", 0) or 0)
        if not (agg >= BAR and any(s < BAR for s in scores)):
            continue
        flu = r.get("fluency_results_per_instances", [])
        nat = r.get("naturalness_results_per_instances", [])
        soc = r.get("social_cultural_results_per_instances", [])
        csr = r.get("cs_ratio_results_per_instances", [])
        items = []
        for j, (s, txt) in enumerate(zip(scores, sents)):
            items.append({
                "scenario_idx": idx, "task": r.get("task", "?"), "cs_type": r.get("cs_type", "?"),
                "sentence_pos": j, "sentence": txt,
                "role": "MASKED" if s < BAR else "neighbour",
                "machine_score": round(float(s), 3),
                "machine_fluency": _dim(flu, j, "fluency_score"),
                "machine_naturalness": _dim(nat, j, "naturalness_score"),
                "machine_cultural": _dim(soc, j, "socio_cultural_score"),
                "machine_cs_ratio": _dim(csr, j, "ratio_score"),
            })
        scenarios.append((r.get("task", "?"), items))

    # balanced subset: keep ALL topic scenarios (minority), then add sentiment
    # scenarios (seeded shuffle) until we reach ~SUBSET_TARGET sentences.
    topic = [s for s in scenarios if s[0] == "topic"]
    other = [s for s in scenarios if s[0] != "topic"]
    random.seed(SEED)
    random.shuffle(other)

    chosen, count = [], 0
    for _, items in topic:
        chosen.append(items); count += len(items)
    for _, items in other:
        if count >= SUBSET_TARGET:
            break
        chosen.append(items); count += len(items)

    items = [it for group in chosen for it in group]
    random.shuffle(items)  # blind: interleave masked & neighbours
    for i, it in enumerate(items, 1):
        it["id"] = f"S{i:03d}"

    n_masked = sum(1 for it in items if it["role"] == "MASKED")
    n_scen = len({it["scenario_idx"] for it in items})

    # blind sheet
    with open(OUT / "human_check_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "sentence", "fluency_1to10", "naturalness_1to10",
                    "cultural_1to10", "is_real_codeswitch_yes_no", "notes"])
        for it in items:
            w.writerow([it["id"], it["sentence"], "", "", "", "", ""])

    # hidden answer key (with machine per-dimension scores)
    with open(OUT / "answer_key.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "scenario_idx", "task", "cs_type", "sentence_pos", "role",
                    "machine_score", "machine_fluency", "machine_naturalness",
                    "machine_cultural", "machine_cs_ratio", "sentence"])
        for it in sorted(items, key=lambda x: x["id"]):
            w.writerow([it["id"], it["scenario_idx"], it["task"], it["cs_type"], it["sentence_pos"],
                        it["role"], it["machine_score"], it["machine_fluency"],
                        it["machine_naturalness"], it["machine_cultural"], it["machine_cs_ratio"],
                        it["sentence"]])

    (OUT / "README.md").write_text(f"""# Human Check — Masking Test (per-dimension)

## What annotators do
Open **human_check_sheet.csv** in Excel. For EACH sentence fill these columns:

1. **fluency_1to10** — is the grammar/wording correct & smooth? (1 = very bad … 10 = perfect)
2. **naturalness_1to10** — does it sound like a real Arabic-English bilingual speaker? (1…10)
3. **cultural_1to10** — is it culturally/socially appropriate & sensible? (1…10)
4. **is_real_codeswitch_yes_no** — does it genuinely MIX Arabic and English?
   - yes = mixes both  |  no = basically all Arabic or all English (monolingual)
5. **notes** — optional.

Do NOT change `id` or `sentence`. Rate each sentence on its own, in the order shown
(they are deliberately shuffled). If 2 people rate, each saves their own copy
(`annotator1.csv`, `annotator2.csv`) in this folder.

## Why
Each sentence is secretly either a machine-flagged weak ("MASKED") sentence or a
"good neighbour". The sheet hides which. Afterwards we check: do humans rate the
MASKED ones lower? If yes, masking is confirmed by people, not just the AI.
Rating the same 3 dimensions the AI uses also lets us check the AI's scoring is trustworthy.

## This sheet
- {len(items)} sentences from {n_scen} masking scenarios ({n_masked} MASKED, {len(items)-n_masked} neighbours)
- Bar = {BAR}, seed = {SEED}
""", encoding="utf-8")

    print(f"Built per-dimension sheet: {len(items)} sentences from {n_scen} scenarios")
    print(f"  MASKED: {n_masked} | neighbours: {len(items)-n_masked}")
    print("  columns: fluency_1to10, naturalness_1to10, cultural_1to10, is_real_codeswitch_yes_no")


if __name__ == "__main__":
    main()
