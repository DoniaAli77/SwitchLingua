#!/usr/bin/env bash
# FINAL matched-compute ArabicDominant ablation: exactly 136 optimizer steps.
#
# Two FIXED, balanced corpora (no new data generated):
#   Original-180      = data/Topic/generated/learning_curve/topic_180_seed42.jsonl  (fixed for ALL model seeds)
#   ArabicDominant-180= .../variants/ArabicDominant/merged/switchlingua_topic_arabicdominant_180_20perlabel.jsonl
# Each is trained with MODEL seeds 42/43/44 -> 6 runs. The corpus is the ONLY thing
# that differs between arms; the seed varies training randomness only, symmetrically
# for both arms (this removes the subset-sampling asymmetry of the 48-step diagnostic).
#
# --max_steps 136 overrides --epochs, exactly as in run_topic_matched_compute.sh.
# Identical everywhere else: xlm-roberta-base, batch 16, grad_accum 1, lr 2e-5,
# max_length 256, fp16, gradient_checkpointing OFF, optim adamw_torch (default),
# 9 labels in the frozen order, NO dev set, load_best OFF (final-step model = the
# checkpoint-selection procedure, identical for both arms).
#
# Evaluation: unchanged Silver-1163, primary_only (no router, no tau), identical code.
# Silver is FINAL EVALUATION ONLY - never training or selection.
#
# The 48-step results in experiment_topic_arabicdominant_ablation/ are an UNDERTRAINING
# DIAGNOSTIC and are deliberately written to a DIFFERENT directory so the two budgets
# are never pooled.
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SILVER=experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl
ORIG=data/Topic/generated/learning_curve/topic_180_seed42.jsonl
AD=data/Topic/generated/variants/ArabicDominant/merged/switchlingua_topic_arabicdominant_180_20perlabel.jsonl
OUTR=experiments/outputs/multi_agent_bert/experiment_topic_arabicdominant_mc136
CK=experiments/checkpoints/_mc136_tmp
STEPS=136
mkdir -p "$OUTR"

for SEED in 42 43 44; do
  for ARM in orig ad; do
    TAG="${ARM}180_mseed${SEED}"
    OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"

    if [ "$ARM" = "ad" ]; then TRAIN="$AD"; else TRAIN="$ORIG"; fi

    echo "===== START $TAG  train=$(basename "$TRAIN")  steps=$STEPS  $(date +%H:%M:%S) ====="
    "$PY" scripts/finetune_transformer_classifier.py \
      --train "$TRAIN" \
      --labels business education health shopping medical sports tech finance social \
      --base_checkpoint xlm-roberta-base --output_dir "$CK" \
      --max_steps "$STEPS" --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
      --seed "$SEED" --fp16 \
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
echo "===== MC136 ABLATION DONE $(date +%H:%M:%S) ====="
