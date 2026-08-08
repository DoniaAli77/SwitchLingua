"""scripts/finetune_ner_real_90split.py

Same recipe as finetune_ner_real.py, but trains on only 90% of real Sabty TRAIN
(the strong primary otherwise saw all 5306 sentences, so nothing was held out
for it). The remaining 10% becomes a leakage-free DEV set: per-type precision
computed there is safe to use for designing a competence router, since this
model never trained on those sentences and the real TEST file is never touched.

Split is a fixed-seed shuffle of TRAIN's sentence indices (seed=42); the dev
indices are saved alongside the model so later scripts reuse the EXACT same
held-out set.

    python scripts/finetune_ner_real_90split.py --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ner_conll_loader import load_conll

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
SAVE = ROOT / "models" / "xlmr_sabty_ner_90split"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dev_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--save_dir", default=str(SAVE))
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, Trainer, TrainingArguments,
    )

    all_train = load_conll(TRAIN)
    idx = list(range(len(all_train)))
    random.Random(args.seed).shuffle(idx)
    n_dev = int(len(idx) * args.dev_frac)
    dev_idx, train_idx = sorted(idx[:n_dev]), sorted(idx[n_dev:])
    train = [all_train[i] for i in train_idx]
    dev = [all_train[i] for i in dev_idx]
    if args.smoke:
        train, dev = train[:120], dev[:60]

    print(f"Full real train={len(all_train)} -> train_sub={len(train)} dev_held={len(dev)} "
          f"(seed={args.seed}, dev_frac={args.dev_frac})")

    labels = sorted({t for s in all_train for t in s["tags"]})
    labels = ["O"] + [l for l in labels if l != "O"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    entity_labels = [l for l in labels if l != "O"]
    print(f"labels={labels}  CUDA: {torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained(args.base)

    class DS(torch.utils.data.Dataset):
        def __init__(self, sents): self.sents = sents
        def __len__(self): return len(self.sents)
        def __getitem__(self, i):
            s = self.sents[i]
            enc = tok(s["tokens"], is_split_into_words=True,
                      truncation=True, max_length=args.maxlen)
            word_ids = enc.word_ids()
            lab, prev = [], None
            for w in word_ids:
                if w is None:
                    lab.append(-100)
                elif w != prev:
                    lab.append(label2id[s["tags"][w]])
                else:
                    lab.append(-100)
                prev = w
            enc["labels"] = lab
            return enc

    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(labels), id2label=id2label, label2id=label2id)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=2)
        yt, yp = [], []
        for pr, gl in zip(preds, p.label_ids):
            for pi, gi in zip(pr, gl):
                if gi != -100:
                    yt.append(id2label[gi]); yp.append(id2label[int(pi)])
        from src.evaluation.evaluator import _per_class_metrics
        per = {m.label: m for m in _per_class_metrics(yt, yp, labels)}
        macro = sum(per[l].f1 for l in entity_labels if l in per) / len(entity_labels)
        acc = sum(a == b for a, b in zip(yt, yp)) / max(1, len(yt))
        out = {"token_acc": round(acc, 4), "macro_f1": round(macro, 4)}
        for l in entity_labels:
            if l in per:
                out[f"f1_{l}"] = round(per[l].f1, 3)
        return out

    targs = TrainingArguments(
        output_dir=str(ROOT / "experiments" / "outputs" / "multi_agent_bert" / "_ft_ckpt_90split"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=(1 if args.smoke else args.epochs),
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=DS(train), eval_dataset=DS(dev),
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\n=== FINE-TUNED XLM-R (90% real train) on HELD-OUT 10% dev ===")
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"  {k[5:]:<12} {v}")

    if not args.smoke:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        with open(Path(args.save_dir) / "split_indices.json", "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "dev_frac": args.dev_frac,
                       "train_idx": train_idx, "dev_idx": dev_idx}, f)
        print(f"\nSaved model + split_indices.json to: {args.save_dir}")


if __name__ == "__main__":
    main()
