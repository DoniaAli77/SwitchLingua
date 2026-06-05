"""Fine-tune a pretrained multilingual transformer for sentiment classification.

This is **fine-tuning**, not training from scratch: it starts from a pretrained
Hugging Face checkpoint (default ``bert-base-multilingual-cased``) and adapts a
sequence-classification head to the {positive, negative, neutral} task. The
resulting checkpoint directory is loaded for inference by
``evaluate_pipeline.py`` via ``--primary_model transformer
--transformer_checkpoint <output_dir>`` — this script never imports or touches
the pipeline.

Design
------
The data/label/metric helpers (``load_examples``, ``build_label_maps``,
``compute_classification_metrics``, ``parse_args``) use only the standard
library, so they are unit-testable offline. ``torch`` / ``transformers`` are
imported lazily inside :func:`main`, so importing this module (e.g. in tests)
never requires the heavy ML stack and never downloads a model.

Inputs
------
- ``--train`` / ``--dev``: JSONL (``{"text","label"}``) or CSV. For CSV pass
  ``--text_col`` / ``--label_col`` (default ``text`` / ``label``).
- ``--labels``: the label set (default: positive negative neutral).
- ``--base_checkpoint``: pretrained model id / path to fine-tune.
- ``--output_dir``: where the fine-tuned checkpoint is written.

Outputs (written to ``--output_dir``)
-------------------------------------
- a standard HF checkpoint (``config.json`` with ``id2label``/``label2id``,
  ``model.safetensors``, tokenizer files)
- ``label_map.json``       — {"label2id": ..., "id2label": ...}
- ``training_metrics.json``— final training loss / steps / args summary
- ``dev_metrics.json``     — accuracy, macro/weighted F1, per-class P/R/F1

Example (EESA reference baseline)
---------------------------------
    python scripts/finetune_transformer_classifier.py \\
        --train data/Sentiment/processed/eesa_sentiment_train.jsonl \\
        --dev   data/Sentiment/processed/eesa_sentiment_dev.jsonl \\
        --base_checkpoint bert-base-multilingual-cased \\
        --output_dir experiments/checkpoints/eesa_mbert \\
        --epochs 4 --batch_size 16 --lr 2e-5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("finetune_transformer_classifier")

DEFAULT_LABELS = ("positive", "negative", "neutral")
DEFAULT_BASE_CHECKPOINT = "bert-base-multilingual-cased"


# ---------------------------------------------------------------------------
# Data loading (stdlib only — testable offline)
# ---------------------------------------------------------------------------

def load_examples(
    path: str,
    text_col: str = "text",
    label_col: str = "label",
    labels: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    """Load ``(text, label)`` pairs from a ``.jsonl`` or ``.csv`` file.

    JSONL lines must be objects with ``text``/``label`` keys (``text_col`` and
    ``label_col`` are honoured for JSONL too). CSV must have a header row.

    Rows with empty text are dropped. If ``labels`` is given, rows whose label
    is not in the set are dropped. Raises ``FileNotFoundError`` if missing and
    ``ValueError`` if no valid rows remain.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    label_set = set(labels) if labels else None
    rows: List[Tuple[str, str]] = []

    suffix = p.suffix.lower()
    if suffix == ".csv":
        with open(p, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for rec in reader:
                rows.append((rec.get(text_col, ""), rec.get(label_col, "")))
    else:  # treat everything else as JSONL
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append((obj.get(text_col, ""), obj.get(label_col, "")))

    cleaned: List[Tuple[str, str]] = []
    for text, label in rows:
        text = (text or "").strip()
        label = (label or "").strip()
        if not text:
            continue
        if label_set is not None and label not in label_set:
            continue
        cleaned.append((text, label))

    if not cleaned:
        raise ValueError(f"No valid (text, label) rows found in '{path}'.")
    return cleaned


def build_label_maps(labels: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return ``(label2id, id2label)`` in the given label order."""
    if not labels:
        raise ValueError("labels must not be empty.")
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels contains duplicates: {labels}")
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


# ---------------------------------------------------------------------------
# Metrics (stdlib only — testable offline; no sklearn dependency)
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[str],
) -> Dict:
    """Accuracy, macro F1, weighted F1, and per-class P/R/F1/support.

    ``y_true`` / ``y_pred`` are integer class ids aligned to ``labels``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")
    n = len(y_true)
    num_classes = len(labels)

    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    support = [0] * num_classes

    correct = 0
    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if t == p:
            tp[t] += 1
            correct += 1
        else:
            fp[p] += 1
            fn[t] += 1

    per_class = []
    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0
    for c in range(num_classes):
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append({
            "label": labels[c],
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": support[c],
        })
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support[c]

    return {
        "accuracy": round(correct / n, 6) if n else 0.0,
        "macro_f1": round(macro_f1_sum / num_classes, 6) if num_classes else 0.0,
        "weighted_f1": round(weighted_f1_sum / n, 6) if n else 0.0,
        "num_samples": n,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Argument parsing (stdlib only — testable offline)
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained transformer for sentiment classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", required=True, metavar="PATH",
                        help="Training data (.jsonl or .csv).")
    parser.add_argument("--dev", default=None, metavar="PATH",
                        help="Optional dev data (.jsonl or .csv) for evaluation.")
    parser.add_argument("--text_col", default="text",
                        help="Text column/key name.")
    parser.add_argument("--label_col", default="label",
                        help="Label column/key name.")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS),
                        help="Label set (order defines class ids).")
    parser.add_argument("--base_checkpoint", default=DEFAULT_BASE_CHECKPOINT,
                        help="Pretrained checkpoint to fine-tune (id or local path).")
    parser.add_argument("--output_dir", required=True, metavar="DIR",
                        help="Directory to write the fine-tuned checkpoint + metrics.")
    parser.add_argument("--epochs", type=float, default=4.0,
                        help="Number of fine-tuning epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Per-device train/eval batch size.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Tokenizer max sequence length.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Enable fp16 mixed precision (GPU only).")
    parser.add_argument("--grad_accum", type=int, default=1, metavar="N",
                        help="Gradient accumulation steps. Effective batch = "
                             "batch_size * grad_accum. Use to simulate a larger "
                             "batch on a small GPU.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Trade compute for VRAM (recompute activations in "
                             "backward). Helps fit large models on small GPUs.")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Fine-tuning (heavy deps imported lazily — NOT exercised by unit tests)
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    # Lazy heavy imports — keep module import (and tests) free of torch.
    try:
        import numpy as np
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        log.error(
            "Fine-tuning requires torch + transformers (and numpy). "
            "Install with: pip install torch transformers accelerate numpy. (%s)",
            exc,
        )
        return 1

    set_seed(args.seed)
    label2id, id2label = build_label_maps(args.labels)

    train_rows = load_examples(args.train, args.text_col, args.label_col, args.labels)
    log.info("Loaded %d training rows from %s", len(train_rows), args.train)
    dev_rows = (
        load_examples(args.dev, args.text_col, args.label_col, args.labels)
        if args.dev else []
    )
    if args.dev:
        log.info("Loaded %d dev rows from %s", len(dev_rows), args.dev)

    tokenizer = AutoTokenizer.from_pretrained(args.base_checkpoint)

    class _TextDataset(Dataset):
        def __init__(self, rows):
            self.texts = [t for t, _ in rows]
            self.labels = [label2id[l] for _, l in rows]
            self.enc = tokenizer(
                self.texts, truncation=True, max_length=args.max_length, padding=False,
            )

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds = _TextDataset(train_rows)
    dev_ds = _TextDataset(dev_rows) if dev_rows else None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_checkpoint,
        num_labels=len(args.labels),
        id2label=id2label,
        label2id=label2id,
    )

    def _compute_metrics(eval_pred):
        logits, gold = eval_pred
        preds = np.argmax(logits, axis=-1).tolist()
        return {
            k: v for k, v in compute_classification_metrics(
                list(gold), preds, args.labels
            ).items() if isinstance(v, (int, float))
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        fp16=args.fp16,
        eval_strategy="epoch" if dev_ds else "no",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )

    from transformers import DataCollatorWithPadding
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=_compute_metrics if dev_ds else None,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    log.info("Starting fine-tuning from '%s' …", args.base_checkpoint)
    train_result = trainer.train()

    # Save checkpoint + tokenizer (id2label/label2id baked into config.json).
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    (out_dir / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "training_metrics.json").write_text(
        json.dumps({
            "base_checkpoint": args.base_checkpoint,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "fp16": args.fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lr": args.lr,
            "train_samples": len(train_rows),
            "train_runtime_sec": round(train_result.metrics.get("train_runtime", 0.0), 2),
            "final_train_loss": round(train_result.metrics.get("train_loss", 0.0), 6),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if dev_ds:
        # Full per-class dev report (not just scalar Trainer metrics).
        pred_out = trainer.predict(dev_ds)
        preds = np.argmax(pred_out.predictions, axis=-1).tolist()
        gold = [label2id[l] for _, l in dev_rows]
        dev_metrics = compute_classification_metrics(gold, preds, args.labels)
        (out_dir / "dev_metrics.json").write_text(
            json.dumps(dev_metrics, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        log.info("Dev — acc=%.4f macroF1=%.4f weightedF1=%.4f",
                 dev_metrics["accuracy"], dev_metrics["macro_f1"], dev_metrics["weighted_f1"])

    log.info("Fine-tuned checkpoint written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
