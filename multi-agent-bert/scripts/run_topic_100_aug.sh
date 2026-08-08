#!/usr/bin/env bash
# Topic 100% row of the low-resource augmentation grid: real-only vs +540 mixed.
# BOTH cells are run here because the pre-existing 100% checkpoint
# (topic_arentcv2_xlmr, test macro F1 0.9947) was trained under an unknown recipe
# -- its trainer args were not preserved -- so it is not comparable to the
# max_steps-400 budget used across the 10/25/50% ladder. Running both cells under
# the matched recipe keeps the 100% row internally consistent.
# Checkpoints are written to D: (more headroom) and deleted after evaluation.
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

LR=data/Topic/processed/lowresource_arentc
DEV=data/Topic/processed/ARENTCV2/dev_sub1000.jsonl
TEST=data/Topic/processed/ARENTCV2/test.jsonl
CKR="D:/topic_lowresource_ckpt"
OUTR=experiments/outputs/multi_agent_bert/experiment_TopicLR_augmentation
mkdir -p "$CKR"

for ARM in only plus540; do
  TAG="arentc100_${ARM}"
  CK="$CKR/$TAG"; OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"
  echo "===== START $TAG  $(date +%H:%M:%S) ====="
  echo "  free C: $(df -h /c | tail -1 | awk '{print $4}')  D: $(df -h /d | tail -1 | awk '{print $4}')"
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$LR/${TAG}.jsonl" --dev "$DEV" \
    --labels business education finance health medical shopping social sports tech \
    --base_checkpoint xlm-roberta-base --output_dir "$CK" \
    --max_steps 400 --load_best --eval_steps 50 \
    --batch_size 4 --grad_accum 4 --lr 2e-5 --max_length 256 \
    --seed 42 --fp16 --gradient_checkpointing --optim adafactor \
    > "$OUT/finetune.log" 2>&1
  echo "  finetune exit=$? $(date +%H:%M:%S)"; sleep 30
  "$PY" evaluate_pipeline.py \
    --dataset "$TEST" --config src/config/default.yaml --active_task topic_classification \
    --pipeline_mode primary_only --mode full_pipeline \
    --primary_model transformer --transformer_checkpoint "$CK" \
    --transformer_device cuda --output_dir "$OUT/primary_only" --run_id "$TAG" \
    > "$OUT/eval.log" 2>&1
  echo "  eval exit=$? $(date +%H:%M:%S)"
  rm -rf "$CK"
  echo "  checkpoint removed; free C: $(df -h /c | tail -1 | awk '{print $4}')  D: $(df -h /d | tail -1 | awk '{print $4}')"
  sleep 30
done
echo "===== TOPIC 100% ROW DONE $(date +%H:%M:%S) ====="
