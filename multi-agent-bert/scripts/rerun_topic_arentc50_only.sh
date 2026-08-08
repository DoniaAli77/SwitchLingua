#!/usr/bin/env bash
# Re-run the ONE failed cell: arentc50_only (baseline for the 50% ratio).
# The original attempt died at step 350/400 with os error 112 (disk full) because
# C: had only ~13 GB free. This run writes its checkpoint to D: (15 GB free), so
# nothing has to be deleted. Recipe identical to the other five cells.
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

LR=data/Topic/processed/lowresource_arentc
DEV=data/Topic/processed/ARENTCV2/dev_sub1000.jsonl
TEST=data/Topic/processed/ARENTCV2/test.jsonl
CK="D:/topic_lowresource_ckpt/arentc50_only"      # <-- checkpoint on D:
OUT=experiments/outputs/multi_agent_bert/experiment_TopicLR_augmentation/arentc50_only
mkdir -p "$OUT"; rm -rf "$CK"; mkdir -p "D:/topic_lowresource_ckpt"

echo "===== START arentc50_only (retry, ckpt on D:) $(date +%H:%M:%S) ====="
echo "  free C: $(df -h /c | tail -1 | awk '{print $4}')   free D: $(df -h /d | tail -1 | awk '{print $4}')"
"$PY" scripts/finetune_transformer_classifier.py \
  --train "$LR/arentc50_only.jsonl" --dev "$DEV" \
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
  --transformer_device cuda --output_dir "$OUT/primary_only" --run_id arentc50_only \
  > "$OUT/eval.log" 2>&1
echo "  eval exit=$? $(date +%H:%M:%S)"

rm -rf "$CK"
echo "  checkpoint removed; free C: $(df -h /c | tail -1 | awk '{print $4}')  D: $(df -h /d | tail -1 | awk '{print $4}')"
echo "===== arentc50_only RETRY DONE $(date +%H:%M:%S) ====="
