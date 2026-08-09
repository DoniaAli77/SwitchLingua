#!/usr/bin/env bash
# EXPLORATORY configuration ablation: ArabicDominant-180 vs Original-180.
#
# Single factor under test: the generation config's cs_ratio, ["50%","60%"] (Original)
# vs ["85%","95%"] (ArabicDominant). Everything downstream is held identical.
#
# Recipe is byte-identical to scripts/run_topic_learning_curve.sh (which itself matches
# the completed Topic-540 primary experiment):
#   xlm-roberta-base, epochs 4, batch 16, grad_accum 1, lr 2e-5, max_length 256, fp16,
#   gradient_checkpointing OFF, optim adamw_torch (default), load_best OFF, no dev set,
#   9 labels in the frozen order.
# Both arms are exactly 180 rows, so "epochs 4" is an identical step budget for both
# (12 steps/epoch x 4 = 48 optimizer steps).
#
# Evaluation: unchanged Silver-1163, primary_only (no router, so no tau threshold),
# identical evaluation code. Silver is FINAL EVALUATION ONLY - never training/selection.
#
# ASYMMETRY, DOCUMENTED (do not silently average it away): the Original arm uses a
# DIFFERENT nested 180-subset per seed (topic_180_seed{42,43,44}, the existing learning-
# curve subsets), so its seed spread contains subset-sampling variance as well as
# training variance. The ArabicDominant arm has one fixed 180 corpus (only 205 sentences
# were accepted in total, so three near-independent subsets are not available), so its
# seed spread is training variance only.
#
# Checkpoints are deleted after each evaluation (4 GB GPU, ~16 GB free disk, 2.2 GB each).
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SILVER=experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl
LC=data/Topic/generated/learning_curve
AD=data/Topic/generated/variants/ArabicDominant/merged/switchlingua_topic_arabicdominant_180_20perlabel.jsonl
OUTR=experiments/outputs/multi_agent_bert/experiment_topic_arabicdominant_ablation
CK=experiments/checkpoints/_ad_tmp
mkdir -p "$OUTR"

for SEED in 42 43 44; do
  for ARM in orig ad; do
    TAG="${ARM}180_seed${SEED}"
    OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"

    if [ "$ARM" = "ad" ]; then TRAIN="$AD"; else TRAIN="$LC/topic_180_seed${SEED}.jsonl"; fi

    echo "===== START $TAG  train=$(basename "$TRAIN")  $(date +%H:%M:%S) ====="
    "$PY" scripts/finetune_transformer_classifier.py \
      --train "$TRAIN" \
      --labels business education health shopping medical sports tech finance social \
      --base_checkpoint xlm-roberta-base --output_dir "$CK" \
      --epochs 4 --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
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
echo "===== ABLATION DONE $(date +%H:%M:%S) ====="
