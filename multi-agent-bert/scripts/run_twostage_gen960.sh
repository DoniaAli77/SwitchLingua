#!/usr/bin/env bash
# Two-stage augmentation test: gen-960-pretrained XLM-R -> fine-tune on real EESA{10,25,50}%.
# Tests whether generated data helps as PRETRAINING (real data gets the last word) where naive
# mixing failed. Same recipe as the LR baselines (max_steps 400, load_best, adafactor, seed 42),
# only --base_checkpoint changes from xlm-roberta-base to the gen-960 (C3) checkpoint.
# primary_only. Dev/test = EESA. New folders only. 4GB GPU -> long inter-run reclaim.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

GEN960=experiments/checkpoints/expC3_switchlingua_xlmr_960   # stage-1 init (gen-960 pretrained)
DEV=data/Sentiment/processed/eesa_sentiment_dev.jsonl
TEST=data/Sentiment/processed/eesa_sentiment_test.jsonl
LR=data/Sentiment/processed/augmentation/lowresource
CKR=experiments/checkpoints/expTwoStage_gen960
OUTR=experiments/outputs/multi_agent_bert/experiment_TwoStage_gen960_eesa
mkdir -p "$OUTR"

for R in 10 25 50; do
  TAG=eesa${R}_twostage; CK="$CKR/$TAG"; OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"
  echo "===== START $TAG $(date +%H:%M:%S) ====="
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$LR/eesa${R}_only.jsonl" --dev "$DEV" \
    --base_checkpoint "$GEN960" --output_dir "$CK" \
    --max_steps 400 --load_best --eval_steps 50 \
    --batch_size 4 --grad_accum 4 --lr 2e-5 --max_length 256 \
    --seed 42 --fp16 --gradient_checkpointing --optim adafactor \
    > "$OUT/finetune.log" 2>&1
  echo "  finetune exit=$? $(date +%H:%M:%S)"; sleep 40
  "$PY" evaluate_pipeline.py \
    --dataset "$TEST" --config src/config/default.yaml --active_task sentiment_classification \
    --pipeline_mode primary_only --mode full_pipeline \
    --primary_model transformer --transformer_checkpoint "$CK" \
    --transformer_device cuda --output_dir "$OUT/primary_only" --run_id "$TAG" \
    > "$OUT/eval.log" 2>&1
  echo "  eval exit=$? $(date +%H:%M:%S)"; sleep 40
done
echo "===== ALL TWO-STAGE RUNS DONE $(date +%H:%M:%S) ====="
