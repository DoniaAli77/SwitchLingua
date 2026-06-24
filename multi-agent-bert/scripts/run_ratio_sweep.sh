#!/usr/bin/env bash
# Augmentation ratio sweep: EESA {10,25,50}% + {0.25,0.5,1.0}x generated.
# Matched compute (max_steps 400 + best-dev), reuses LR real-only baselines.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
RS=data/Sentiment/processed/augmentation/ratiosweep
CKR=experiments/checkpoints/ratiosweep
OUTR=experiments/outputs/multi_agent_bert/experiment_E_ratio_sweep
mkdir -p "$OUTR"

for TAG in eesa10_gen025 eesa10_gen05 eesa10_gen10 eesa25_gen025 eesa25_gen05 eesa25_gen10 eesa50_gen025 eesa50_gen05 eesa50_gen10; do
  CK="$CKR/$TAG"; OUT="$OUTR/$TAG"; mkdir -p "$OUT"
  echo "===== START $TAG $(date +%H:%M:%S) ====="
  rm -rf "$CK"
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$RS/$TAG.jsonl" --dev data/Sentiment/processed/eesa_sentiment_dev.jsonl \
    --base_checkpoint xlm-roberta-base --output_dir "$CK" \
    --max_steps 400 --load_best --eval_steps 50 \
    --batch_size 4 --grad_accum 4 --lr 2e-5 --max_length 256 \
    --seed 42 --fp16 --gradient_checkpointing --optim adafactor \
    > "$OUT/finetune.log" 2>&1
  echo "  finetune exit=$? $(date +%H:%M:%S)"; sleep 15
  "$PY" evaluate_pipeline.py \
    --dataset data/Sentiment/processed/eesa_sentiment_test.jsonl \
    --config src/config/default.yaml --active_task sentiment_classification \
    --pipeline_mode primary_only --mode full_pipeline \
    --primary_model transformer --transformer_checkpoint "$CK" \
    --transformer_device cuda --output_dir "$OUT/primary_only" --run_id "$TAG" \
    > "$OUT/eval.log" 2>&1
  echo "  eval exit=$? $(date +%H:%M:%S)"; sleep 15
done
echo "===== ALL RATIO-SWEEP RUNS DONE $(date +%H:%M:%S) ====="
