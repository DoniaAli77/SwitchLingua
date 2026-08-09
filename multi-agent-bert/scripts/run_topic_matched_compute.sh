#!/usr/bin/env bash
# Matched-compute diagnostic: Topic-360 vs Topic-540, BOTH trained for exactly
# 136 optimizer steps (--max_steps 136 overrides --epochs), seeds 42/43/44.
# 6 primary-only XLM-R runs. No ArEnTC, no agents, no LLM calls.
#
# Everything else identical to the learning-curve runs: same nested datasets,
# same preprocessing/label order, xlm-roberta-base, batch 16, grad_accum 1,
# lr 2e-5, max_length 256, fp16, gradient_checkpointing off, adamw_torch,
# load_best OFF, no dev set. Evaluated on the unchanged Silver-1163.
#
# Note: 540 at 136 steps == the learning-curve 540 run (34 steps/epoch x 4),
# so 540/seed-42 here is also a nondeterminism probe against 0.6148 / 0.5429.
# 360 at 136 steps ~= 5.9 epochs over 360 rows.
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SILVER=experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl
LC=data/Topic/generated/learning_curve
FULL540=data/Topic/generated/merged/switchlingua_topic_train_540_60perlabel.jsonl
OUTR=experiments/outputs/multi_agent_bert/experiment_topic_matched_compute
CK=experiments/checkpoints/_mc_tmp
STEPS=136
mkdir -p "$OUTR"

for SEED in 42 43 44; do
  for SIZE in 360 540; do
    TAG="mc${SIZE}_seed${SEED}"
    OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"

    if [ "$SIZE" = "540" ]; then TRAIN="$FULL540"; else TRAIN="$LC/topic_${SIZE}_seed${SEED}.jsonl"; fi

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
echo "===== MATCHED-COMPUTE DONE $(date +%H:%M:%S) ====="
