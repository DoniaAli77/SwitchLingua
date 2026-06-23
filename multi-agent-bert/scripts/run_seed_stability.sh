#!/usr/bin/env bash
# Seed-stability check for C2 (480) vs C3 (960) generated sentiment.
# 2 extra seeds each (123, 456); seed 42 already run as C2/C3. primary_only only.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CKROOT=experiments/checkpoints/seed_stability
OUTROOT=experiments/outputs/multi_agent_bert/experiment_seed_stability
mkdir -p "$OUTROOT"

declare -A TRAIN
TRAIN[480]=data/Sentiment/generated/merged/switchlingua_sentiment_train_480_160perlabel.jsonl
TRAIN[960]=data/Sentiment/generated/merged/switchlingua_sentiment_train_960_320perlabel.jsonl

for SIZE in 480 960; do
  for SEED in 123 456; do
    TAG="sz${SIZE}_seed${SEED}"
    CK="$CKROOT/$TAG"
    OUT="$OUTROOT/$TAG"
    mkdir -p "$OUT"
    echo "===== START $TAG $(date +%H:%M:%S) ====="
    rm -rf "$CK"
    "$PY" scripts/finetune_transformer_classifier.py \
      --train "${TRAIN[$SIZE]}" --dev data/Sentiment/processed/eesa_sentiment_dev.jsonl \
      --base_checkpoint xlm-roberta-base --output_dir "$CK" \
      --epochs 4 --batch_size 4 --grad_accum 4 --lr 2e-5 --max_length 256 \
      --seed "$SEED" --fp16 --gradient_checkpointing --optim adafactor \
      > "$OUT/finetune.log" 2>&1
    echo "  finetune exit=$? $(date +%H:%M:%S)"
    sleep 20   # let GPU memory reclaim before next load
    "$PY" evaluate_pipeline.py \
      --dataset data/Sentiment/processed/eesa_sentiment_test.jsonl \
      --config src/config/default.yaml --active_task sentiment_classification \
      --pipeline_mode primary_only --mode full_pipeline \
      --primary_model transformer --transformer_checkpoint "$CK" \
      --transformer_device cuda --output_dir "$OUT/primary_only" --run_id "$TAG" \
      > "$OUT/eval.log" 2>&1
    echo "  eval exit=$? $(date +%H:%M:%S)"
    sleep 20
  done
done
echo "===== ALL SEED RUNS DONE $(date +%H:%M:%S) ====="
