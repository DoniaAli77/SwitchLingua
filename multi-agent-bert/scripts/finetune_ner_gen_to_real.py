"""scripts/finetune_ner_gen_to_real.py

Cross-domain generalization / data-augmentation study (mirrors the sentiment
methodology): TRAIN on the SwitchLingua-generated NER-240, TEST on the REAL
Sabty test set. Answers: does generated CS-NER data teach a model that transfers
to real data? Pure fine-tuning — no LLM, no credits.

Handles the label-scheme mismatch by canonicalizing BOTH datasets to one space:
BIO tags over 4 types {PER, LOC, ORG, MISC} (Sabty's IO 'I-PERS' -> 'B/I-PER').
Entity-level F1 (seqeval).

Configs (via --mode):
  gen        TRAIN generated-240        -> TEST real   (the core question)
  augment    TRAIN real + generated-240 -> TEST real   (does generated data help?)
  real       TRAIN real only            -> TEST real   (baseline, same pipeline)

    python scripts/finetune_ner_gen_to_real.py --mode gen --epochs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:  # Windows consoles default to cp1252 and cannot encode the arrows below.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ner_conll_loader import load_conll

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "NER" / "generated" / "expN" / "merged" / "switchlingua_ner_train_240_bio.jsonl"
REAL_TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
REAL_TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"

CANON = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
_TYPEMAP = {"PER": "PER", "PERS": "PER", "PERSON": "PER", "LOC": "LOC", "GPE": "LOC",
            "ORG": "ORG", "MISC": "MISC"}


def canon_bio(tags):
    """Map any BIO/IO tag list to canonical BIO over {PER,LOC,ORG,MISC}."""
    out, prev = [], None
    for t in tags:
        if t == "O" or "-" not in t:
            out.append("O"); prev = None; continue
        pre, ty = t.split("-", 1)
        cty = _TYPEMAP.get(ty.upper())
        if cty is None:
            out.append("O"); prev = None; continue
        out.append(f"B-{cty}" if (pre == "B" or cty != prev) else f"I-{cty}")
        prev = cty
    return out


def load_gen(path=None):
    p = Path(path) if path else GEN
    if not p.is_absolute():
        p = ROOT / p
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    return [{"tokens": r["tokens"], "tags": canon_bio(r["ner_tags"])} for r in rows]


def load_real(path):
    return [{"tokens": s["tokens"], "tags": canon_bio(s["tags"])} for s in load_conll(path)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["gen", "augment", "real"], default="gen")
    ap.add_argument("--gen_file", default=None,
                    help="generated BIO jsonl (default: the 240 set); use to run the size ladder")
    ap.add_argument("--base", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, Trainer, TrainingArguments)
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    gen, real_tr, test = load_gen(args.gen_file), load_real(REAL_TRAIN), load_real(REAL_TEST)
    train = {"gen": gen, "augment": real_tr + gen, "real": real_tr}[args.mode]
    print(f"MODE={args.mode}  train={len(train)} sents (gen={len(gen)}, real={len(real_tr)})  "
          f"test=REAL {len(test)} sents")
    print(f"CUDA: {torch.cuda.is_available()}")

    label2id = {l: i for i, l in enumerate(CANON)}
    id2label = {i: l for l, i in label2id.items()}
    tok = AutoTokenizer.from_pretrained(args.base)

    class DS(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            s = self.data[i]
            enc = tok(s["tokens"], is_split_into_words=True, truncation=True, max_length=args.maxlen)
            wid, lab, prev = enc.word_ids(), [], None
            for w in wid:
                lab.append(-100 if (w is None or w == prev) else label2id[s["tags"][w]])
                prev = w
            enc["labels"] = lab
            return enc

    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(CANON), id2label=id2label, label2id=label2id)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        tl, tp = [], []
        for pr, gl in zip(preds, p.label_ids):
            a, b = [], []
            for pi, gi in zip(pr, gl):
                if gi != -100:
                    a.append(id2label[gi]); b.append(id2label[int(pi)])
            tl.append(a); tp.append(b)
        return {"f1": round(f1_score(tl, tp), 4),
                "precision": round(precision_score(tl, tp), 4),
                "recall": round(recall_score(tl, tp), 4)}

    targs = TrainingArguments(
        output_dir=str(ROOT / "experiments" / "outputs" / "multi_agent_bert" / f"_ft_{args.mode}"),
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, fp16=torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="no", logging_steps=25, report_to=[], disable_tqdm=True)
    trainer = Trainer(model=model, args=targs, train_dataset=DS(train), eval_dataset=DS(test),
        data_collator=DataCollatorForTokenClassification(tok), compute_metrics=compute_metrics)
    trainer.train()
    m = trainer.evaluate()

    # Save before reporting: a failed print must never cost a trained model.
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir); tok.save_pretrained(args.save_dir)
        print(f"Saved model to: {args.save_dir}")

    print(f"\n=== {args.mode.upper()} → REAL Sabty test (entity-level) ===")
    print(f"  F1={m['eval_f1']}  P={m['eval_precision']}  R={m['eval_recall']}")

    # final per-type report
    lp = trainer.predict(DS(test))
    preds = np.argmax(lp.predictions, axis=2)
    tl, tp = [], []
    for pr, gl in zip(preds, lp.label_ids):
        a, b = [], []
        for pi, gi in zip(pr, gl):
            if gi != -100:
                a.append(id2label[gi]); b.append(id2label[int(pi)])
        tl.append(a); tp.append(b)
    print(classification_report(tl, tp, digits=3))
    print("Reference: real-train→real baseline = 0.816")


if __name__ == "__main__":
    main()
