"""scripts/finetune_ner_real.py

Fine-tune XLM-R on the REAL Sabty Arabic-English CS NER train set and evaluate
on the real test set — the proper trained baseline (what Sabty did), as opposed
to the off-the-shelf Davlan model we ran zero-shot.

Trains ``xlm-roberta-base`` with a fresh token-classification head over the
corpus's own labels {O, I-PERS, I-LOC, I-ORG, I-MISC} (IO scheme). A fine-tuned
model learns THIS dataset's style AND the MISC category (the off-the-shelf model
had no MISC label -> scored 0 on it).

NER-only. Saves the model so it can later be plugged into the agentic pipeline
as the new primary.

Usage
-----
    python scripts/finetune_ner_real.py --smoke      # tiny/fast sanity run
    python scripts/finetune_ner_real.py --epochs 3   # full run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.ner_conll_loader import load_conll

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "NER" / "Train_AR-EN_NER.txt"
TEST = ROOT / "data" / "NER" / "Test_AR-EN_NER.txt"
SAVE = ROOT / "models" / "xlmr_sabty_ner"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--smoke", action="store_true", help="tiny fast run to catch errors")
    ap.add_argument("--save_dir", default=str(SAVE))
    args = ap.parse_args()

    import torch
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, Trainer, TrainingArguments,
    )

    train = load_conll(TRAIN)
    test = load_conll(TEST)
    if args.smoke:
        train, test = train[:120], test[:60]

    labels = sorted({t for s in train + test for t in s["tags"]})
    labels = ["O"] + [l for l in labels if l != "O"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    entity_labels = [l for l in labels if l != "O"]
    print(f"Train={len(train)} Test={len(test)}  labels={labels}")
    print(f"CUDA: {torch.cuda.is_available()}")

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
                    lab.append(-100)   # label only the first sub-word
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
        output_dir=str(ROOT / "experiments" / "outputs" / "multi_agent_bert" / "_ft_ckpt"),
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
        train_dataset=DS(train), eval_dataset=DS(test),
        data_collator=DataCollatorForTokenClassification(tok),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\n=== FINE-TUNED XLM-R on real Sabty test ===")
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"  {k[5:]:<12} {v}")

    if not args.smoke:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir)
        tok.save_pretrained(args.save_dir)
        print(f"\nSaved fine-tuned model to: {args.save_dir}")


if __name__ == "__main__":
    main()
