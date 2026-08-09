#!/usr/bin/env bash
# Subset-variance check for the 180-sentence point at 136 optimizer steps.
#
# QUESTION: the matched-compute Original-180 result (0.6205 +/- 0.0081) rests on ONE
# subset (topic_180_seed42) with only the model seed varying. Is that subset
# representative, or a lucky draw?
#
# DESIGN: 3 subsets (topic_180_seed42/43/44) x 3 model seeds (42/43/44) = 9 cells.
# The three seed-42-subset cells already exist in experiment_topic_arabicdominant_mc136/
# (orig180_mseed{42,43,44}) and are NOT re-run; this script fills the remaining 6.
# Subset and model seed are then fully crossed, so their variance can be separated.
#
# Recipe is byte-identical to run_topic_arabicdominant_mc136.sh: 136 steps,
# xlm-roberta-base, batch 16, grad_accum 1, lr 2e-5, max_length 256, fp16,
# adamw_torch, no dev, load_best off, 9 labels frozen order, primary_only on Silver-1163.
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SILVER=experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl
LC=data/Topic/generated/learning_curve
OUTR=experiments/outputs/multi_agent_bert/experiment_topic180_subset_variance
CK=experiments/checkpoints/_sv_tmp
STEPS=136
mkdir -p "$OUTR"

for SUBSET in 43 44; do
  for MSEED in 42 43 44; do
    TAG="sub${SUBSET}_mseed${MSEED}"
    OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"
    TRAIN="$LC/topic_180_seed${SUBSET}.jsonl"

    echo "===== START $TAG  train=$(basename "$TRAIN")  steps=$STEPS  $(date +%H:%M:%S) ====="
    "$PY" scripts/finetune_transformer_classifier.py \
      --train "$TRAIN" \
      --labels business education health shopping medical sports tech finance social \
      --base_checkpoint xlm-roberta-base --output_dir "$CK" \
      --max_steps "$STEPS" --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
      --seed "$MSEED" --fp16 \
      > "$OUT/finetune.log" 2>&1
    echo "  finetune exit=$? $(date +%H:%M:%S)"

    "$PY" evaluate_pipeline.py \
      --dataset "$SILVER" --config src/config/default.yaml --active_task topic_classification \
      --pipeline_mode primary_only --mode full_pipeline \
      --primary_model transformer --transformer_checkpoint "$CK" \
      --transformer_device cuda --output_dir "$OUT" --run_id "$TAG" \
      > "$OUT/eval.log" 2>&1
    echo "  eval exit=$? $(date +%H:%M:%S)"

    rm -rf "$CK"
    echo "  checkpoint removed; free: $(df -h /c | tail -1 | awk '{print $4}')"
  done
done
echo "===== SUBSET VARIANCE DONE $(date +%H:%M:%S) ====="
