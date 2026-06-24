#!/usr/bin/env bash
# Low-resource augmentation: EESA {10,25,50}% only vs +SwitchLingua-960.
# Matched recipe (fresh xlm-roberta-base, adafactor, seed 42). primary_only only.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
LR=data/Sentiment/processed/augmentation/lowresource
CKR=experiments/checkpoints/lowresource
OUTR=experiments/outputs/multi_agent_bert/experiment_LR_lowresource_augmentation
mkdir -p "$OUTR"

for TAG in eesa10_only eesa10_plus960 eesa25_only eesa25_plus960 eesa50_only eesa50_plus960; do
  CK="$CKR/$TAG"; OUT="$OUTR/$TAG"; mkdir -p "$OUT"
  echo "===== START $TAG $(date +%H:%M:%S) ====="
  rm -rf "$CK"
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$LR/$TAG.jsonl" --dev data/Sentiment/processed/eesa_sentiment_dev.jsonl \
    --base_checkpoint xlm-roberta-base --output_dir "$CK" \
    --max_steps 400 --load_best --eval_steps 50 \
    --batch_size 4 --grad_accum 4 --lr 2e-5 --max_length 256 \
    --seed 42 --fp16 --gradient_checkpointing --optim adafactor \
    > "$OUT/finetune.log" 2>&1
  echo "  finetune exit=$? $(date +%H:%M:%S)"; sleep 20
  "$PY" evaluate_pipeline.py \
    --dataset data/Sentiment/processed/eesa_sentiment_test.jsonl \
    --config src/config/default.yaml --active_task sentiment_classification \
    --pipeline_mode primary_only --mode full_pipeline \
    --primary_model transformer --transformer_checkpoint "$CK" \
    --transformer_device cuda --output_dir "$OUT/primary_only" --run_id "$TAG" \
    > "$OUT/eval.log" 2>&1
  echo "  eval exit=$? $(date +%H:%M:%S)"; sleep 20
done
echo "===== ALL LOW-RESOURCE RUNS DONE $(date +%H:%M:%S) ====="
