#!/usr/bin/env bash
# V1_lowerCS downstream tests (primary_only, XLM-R). Sensitivity/domain-compatibility.
#  C-V1: generated-only transfer, V1-480, 3 seeds (42,123,456). Mirror of C2/C3 recipe.
#  E-V1: low-resource augmentation, EESA{10,25,50}% + V1-gen (gen capped <=50%), seed 42.
# Exact recipe: xlm-roberta-base, adafactor, batch4 x grad_accum4 (eff16), fp16, grad-ckpt,
# lr2e-5, maxlen256; C-V1 4 epochs; E-V1 --max_steps 400 --load_best --eval_steps 50.
# Dev=EESA dev, test=EESA test (kept out of training). New folders only.
set -u
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

DEV=data/Sentiment/processed/eesa_sentiment_dev.jsonl
TEST=data/Sentiment/processed/eesa_sentiment_test.jsonl
V1=data/Sentiment/generated/variants/V1_lowerCS/switchlingua_sentiment_v1_lowerCS_480.jsonl
LRV1=data/Sentiment/processed/augmentation/lowresource_v1

CKROOT=experiments/checkpoints/expCV1_v1lowerCS_480
OUTC=experiments/outputs/multi_agent_bert/experiment_CV1_v1lowerCS
CKE=experiments/checkpoints/expEV1_v1lowerCS_aug
OUTE=experiments/outputs/multi_agent_bert/experiment_EV1_v1lowerCS_augmentation
mkdir -p "$OUTC" "$OUTE"

finetune_eval () {   # args: TRAIN CK OUT RUNID  <extra finetune args...>
  local TRAIN="$1" CK="$2" OUT="$3" RUNID="$4"; shift 4
  mkdir -p "$OUT"; rm -rf "$CK"
  echo "----- finetune $RUNID $(date +%H:%M:%S) -----"
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$TRAIN" --dev "$DEV" \
    --base_checkpoint xlm-roberta-base --output_dir "$CK" \
    --batch_size 4 --grad_accum 4 --lr 2e-5 --max_length 256 \
    --fp16 --gradient_checkpointing --optim adafactor "$@" \
    > "$OUT/finetune.log" 2>&1
  echo "  finetune exit=$? $(date +%H:%M:%S)"; sleep 20
  "$PY" evaluate_pipeline.py \
    --dataset "$TEST" --config src/config/default.yaml --active_task sentiment_classification \
    --pipeline_mode primary_only --mode full_pipeline \
    --primary_model transformer --transformer_checkpoint "$CK" \
    --transformer_device cuda --output_dir "$OUT/primary_only" --run_id "$RUNID" \
    > "$OUT/eval.log" 2>&1
  echo "  eval exit=$? $(date +%H:%M:%S)"; sleep 20
}

echo "===== C-V1 generated-only transfer (V1-480, 3 seeds) $(date +%H:%M:%S) ====="
for SEED in 42 123 456; do
  finetune_eval "$V1" "$CKROOT/seed$SEED" "$OUTC/seed$SEED" "CV1_seed$SEED" \
    --epochs 4 --seed "$SEED"
done

echo "===== E-V1 low-resource augmentation (EESA%+V1, seed 42) $(date +%H:%M:%S) ====="
for R in 10 25 50; do
  finetune_eval "$LRV1/eesa${R}_plusV1.jsonl" "$CKE/eesa${R}_plusV1" "$OUTE/eesa${R}_plusV1" "EV1_eesa${R}_plusV1" \
    --max_steps 400 --load_best --eval_steps 50 --seed 42
done

echo "===== ALL V1 DOWNSTREAM RUNS DONE $(date +%H:%M:%S) ====="
