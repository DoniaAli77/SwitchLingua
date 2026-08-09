#!/usr/bin/env bash
# Topic-1080 at exactly 272 optimizer steps, model seeds 42/43/44.
#
# Paired against experiment_topic_matched_compute/mc540_seed{42,43,44}. THE TRAINING
# CORPUS IS THE ONLY DIFFERENCE. Every other setting is reused verbatim from that
# experiment: xlm-roberta-base, --max_steps 272, batch 16, grad_accum 1, lr 2e-5,
# max_length 256, fp16, gradient_checkpointing OFF, optim adamw_torch (default),
# 9 labels in the frozen order, NO dev set, load_best OFF (final-step model).
#
# WHY 272: 1080 rows / batch 16 = 68 steps per epoch, so 272 steps = EXACTLY 4.0 epochs,
# the same number of passes over the data that Topic-540 gets at 136 steps. This is the
# EPOCH-MATCHED comparison that the 136-step run could not provide (there 1080 got only
# 2.0 epochs). It also gives the step-matched comparison against Topic-540 at 272 steps,
# completing the corpus x budget 2x2.
#
# Evaluation: unchanged Silver-1163, primary_only (no router, no tau threshold).
# Silver is FINAL EVALUATION ONLY. No agents. No data generated. No other experiments.
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SILVER=experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl
TRAIN=data/Topic/generated/merged/switchlingua_topic_train_1080_120perlabel.jsonl
OUTR=experiments/outputs/multi_agent_bert/experiment_topic1080_272steps
CK=experiments/checkpoints/_1080x2_tmp
STEPS=272
mkdir -p "$OUTR"

for SEED in 42 43 44; do
  TAG="mc1080x2_seed${SEED}"
  OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"

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
echo "===== TOPIC-1080 272-STEP DONE $(date +%H:%M:%S) ====="
