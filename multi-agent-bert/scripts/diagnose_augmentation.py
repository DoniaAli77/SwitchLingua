"""Diagnostic: why didn't SwitchLingua-960 help as augmentation on real EESA?
Compares EESA train, EESA test, generated-960 across style / vocab / CS / labels.
Read-only analysis. Prints a structured profile used for the diagnosis report.
"""
from __future__ import annotations
import json, re, collections, statistics

AR = re.compile(r"[؀-ۿ]")
EN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"[؀-ۿA-Za-z']+")  # keep Arabic/Latin word tokens

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def toks(t): return WORD.findall(str(t))
def lang(tok):
    a, e = bool(AR.search(tok)), bool(EN.search(tok))
    return "ar" if a and not e else ("en" if e and not a else "mix" if a and e else "other")

def profile(rows):
    texts = [r["text"] for r in rows]
    n = len(texts)
    lens, ar_r, en_r, cmis, sw = [], [], [], [], []
    ar_words, en_words = collections.Counter(), collections.Counter()
    vocab = set()
    for t in texts:
        ws = toks(t); lens.append(len(ws))
        langs = [lang(w) for w in ws]
        a = sum(1 for L in langs if L == "ar"); e = sum(1 for L in langs if L == "en")
        ld = a + e
        ar_r.append(a/len(ws) if ws else 0); en_r.append(e/len(ws) if ws else 0)
        cmis.append(100*(1 - max(a, e)/ld) if ld else 0)
        seq = [L for L in langs if L in ("ar", "en")]
        sw.append(sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1]))
        for w, L in zip(ws, langs):
            lw = w.lower(); vocab.add(lw)
            if L == "ar": ar_words[w] += 1
            elif L == "en": en_words[lw] += 1
    return dict(n=n, avg_len=statistics.mean(lens) if lens else 0,
               ar_ratio=statistics.mean(ar_r), en_ratio=statistics.mean(en_r),
               cmi=statistics.mean(cmis), switches=statistics.mean(sw),
               ar_top=ar_words.most_common(15), en_top=en_words.most_common(15),
               vocab=vocab)

paths = {
 "EESA_train": "data/Sentiment/processed/eesa_sentiment_train.jsonl",
 "EESA_test":  "data/Sentiment/processed/eesa_sentiment_test.jsonl",
 "GEN_960":    "data/Sentiment/generated/merged/switchlingua_sentiment_train_960_320perlabel.jsonl",
}
data = {k: load(v) for k, v in paths.items()}
test_vocab = profile(data["EESA_test"])["vocab"]

print("="*70)
for name, rows in data.items():
    print(f"\n##### {name}  (n={len(rows)}) #####")
    by = collections.defaultdict(list)
    for r in rows: by[r["label"]].append(r)
    print("label dist:", {k: len(v) for k, v in sorted(by.items())})
    allp = profile(rows)
    voc_ov = len(allp["vocab"] & test_vocab)/len(test_vocab)
    print("ALL: avg_len=%.1f ar=%.2f en=%.2f CMI=%.1f switches=%.2f vocab=%d  test-vocab-coverage=%.3f"
          % (allp["avg_len"], allp["ar_ratio"], allp["en_ratio"], allp["cmi"], allp["switches"], len(allp["vocab"]), voc_ov))
    for lab in ["positive", "negative", "neutral"]:
        if lab not in by: continue
        p = profile(by[lab])
        ov = len(p["vocab"] & test_vocab)/len(test_vocab)
        print("  %-9s n=%4d len=%.1f ar=%.2f en=%.2f CMI=%.1f sw=%.2f testVocOv=%.3f"
              % (lab, p["n"], p["avg_len"], p["ar_ratio"], p["en_ratio"], p["cmi"], p["switches"], ov))
    print("  top AR:", [w for w, _ in allp["ar_top"][:12]])
    print("  top EN:", [w for w, _ in allp["en_top"][:12]])

# divergence examples
gen_vocab = profile(data["GEN_960"])["vocab"]
train_vocab = profile(data["EESA_train"])["vocab"]
ref_vocab = train_vocab | test_vocab
def oov_frac(t, ref):
    ws = [w.lower() for w in toks(t)]; return sum(1 for w in ws if w not in ref)/len(ws) if ws else 0
print("\n##### GENERATED samples most UNLIKE real EESA (high OOV vs EESA vocab) #####")
g = sorted(data["GEN_960"], key=lambda r: -oov_frac(r["text"], ref_vocab))[:6]
for r in g: print("  [%s oov=%.2f] %s" % (r["label"], oov_frac(r["text"], ref_vocab), r["text"][:90]))
print("\n##### EESA TEST samples NOT covered by generated style (high OOV vs GEN vocab) #####")
e = sorted(data["EESA_test"], key=lambda r: -oov_frac(r["text"], gen_vocab))[:6]
for r in e: print("  [%s oov=%.2f] %s" % (r["label"], oov_frac(r["text"], gen_vocab), r["text"][:90]))
# label-wise vocab overlap gen-vs-test
print("\n##### label-wise vocab overlap with EESA test (Jaccard) #####")
for src in ["EESA_train", "GEN_960"]:
    by = collections.defaultdict(list)
    for r in data[src]: by[r["label"]].append(r)
    tby = collections.defaultdict(list)
    for r in data["EESA_test"]: tby[r["label"]].append(r)
    row = []
    for lab in ["positive", "negative", "neutral"]:
        a = profile(by[lab])["vocab"]; b = profile(tby[lab])["vocab"]
        j = len(a & b)/len(a | b) if (a|b) else 0
        cov = len(a & b)/len(b) if b else 0
        row.append("%s J=%.3f cov=%.3f" % (lab, j, cov))
    print(" ", src, "|", "  ".join(row))
