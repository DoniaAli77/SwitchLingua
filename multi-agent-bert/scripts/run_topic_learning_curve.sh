#!/usr/bin/env bash
# SwitchLingua topic learning curve: Topic-180 -> Topic-360 -> Topic-540, seeds 42/43/44.
#
# Recipe matched EXACTLY to the completed Topic-540 primary experiment
# (verified from experiments/checkpoints/topic_gen540_xlmr/training_metrics.json
#  and .../experiment_T_gen540_xlmr/finetune.log):
#   base xlm-roberta-base, epochs 4, batch 16, grad_accum 1, lr 2e-5,
#   max_length 256, fp16, gradient_checkpointing OFF, optim adamw_torch (default),
#   9 labels in the frozen order, load_best OFF (final-epoch model).
#
# Dev set: the original run passed ARENTCV2 dev, but trainer_state.json shows
# best_model_checkpoint=None / best_metric=None, i.e. --load_best was OFF, so the
# dev set NEVER influenced the weights (it was logged only). This study forbids
# ArEnTC, so no --dev is passed. The 540/seed-42 reproduction check validates
# that this has no material effect.
#
# Evaluation: unchanged Silver-1163, primary_only, identical evaluation code.
# Silver is used for FINAL EVALUATION ONLY - never for training or selection.
# Checkpoints are DELETED after each evaluation (only ~16 GB free, 2.2 GB each).
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SILVER=experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl
LC=data/Topic/generated/learning_curve
FULL540=data/Topic/generated/merged/switchlingua_topic_train_540_60perlabel.jsonl
OUTR=experiments/outputs/multi_agent_bert/experiment_topic_learning_curve
CK=experiments/checkpoints/_lc_tmp
mkdir -p "$OUTR"

for SEED in 42 43 44; do
  for SIZE in 180 360 540; do
    TAG="topic${SIZE}_seed${SEED}"
    OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"

    if [ "$SIZE" = "540" ]; then TRAIN="$FULL540"; else TRAIN="$LC/topic_${SIZE}_seed${SEED}.jsonl"; fi

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
echo "===== LEARNING CURVE DONE $(date +%H:%M:%S) ====="
