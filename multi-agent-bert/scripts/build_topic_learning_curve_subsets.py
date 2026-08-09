"""Build balanced, nested Topic-180 / Topic-360 subsets from the existing
Topic-540 SwitchLingua-generated dataset. No new data is generated: every row
is copied verbatim from switchlingua_topic_train_540_60perlabel.jsonl.

Nesting guarantee: for a given seed, the 60 examples of each label are shuffled
once with random.Random(seed); Topic-180 takes the first 20 per label,
Topic-360 the first 40, so
    Topic-180(seed)  subset-of  Topic-360(seed)  subset-of  Topic-540 (all 60).

Topic-540 is the ORIGINAL file, used unchanged, so the 540/seed-42 run is a
faithful reproduction attempt of the completed primary experiment.
"""
import json
import random
import collections
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "data/Topic/generated/merged/switchlingua_topic_train_540_60perlabel.jsonl"
OUT = ROOT / "data/Topic/generated/learning_curve"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
PER_LABEL = {180: 20, 360: 40}          # 540 uses the original file unchanged

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
by_label = collections.defaultdict(list)
for r in rows:
    by_label[r["label"]].append(r)
labels = sorted(by_label)
assert len(rows) == 540 and all(len(v) == 60 for v in by_label.values()), "unexpected source"
print(f"source: {SRC.name}  rows={len(rows)}  labels={len(labels)}  per-label=60")

manifest = {}
for seed in SEEDS:
    # one shuffle per label per seed -> prefixes are nested by construction
    order = {lbl: random.Random(seed).sample(by_label[lbl], 60) for lbl in labels}
    prev_ids = None
    for size in (180, 360):
        n = PER_LABEL[size]
        sel = [r for lbl in labels for r in order[lbl][:n]]
        assert len(sel) == size
        cnt = collections.Counter(r["label"] for r in sel)
        assert set(cnt.values()) == {n}, f"unbalanced: {cnt}"
        path = OUT / f"topic_{size}_seed{seed}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in sel:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        ids = {r["text"] for r in sel}
        if prev_ids is not None:
            assert prev_ids <= ids, f"NESTING VIOLATED: 180 not subset of 360 (seed {seed})"
        prev_ids = ids
        manifest[f"{size}_seed{seed}"] = {"path": str(path.relative_to(ROOT)),
                                           "rows": size, "per_label": n}
        print(f"  seed {seed}  size {size:>3}: {n}/label -> {path.name}")
    # verify both are subsets of the full 540
    all_txt = {r["text"] for r in rows}
    assert prev_ids <= all_txt, "360 not subset of 540"

manifest["540_all_seeds"] = {"path": str(SRC.relative_to(ROOT)), "rows": 540, "per_label": 60,
                              "note": "original file, used unchanged for all seeds"}
json.dump(manifest, open(OUT / "manifest.json", "w", encoding="utf-8"), indent=2)
print(f"\nNESTING VERIFIED for all seeds: 180 subset-of 360 subset-of 540")
print(f"wrote manifest -> {OUT/'manifest.json'}")
