#!/usr/bin/env bash
# Topic low-resource augmentation study — mirrors the sentiment design.
# Three arms per ratio on ARENTC {10,25,50}% real data:
#   _only      : real subset only            (fresh xlm-roberta-base)   <- baseline
#   _plus540   : real subset + generated 540 (fresh xlm-roberta-base)   <- DIRECT MIXING
#   _twostage  : gen-540 checkpoint -> real subset only                 <- TWO-STAGE
# Recipe matched to the sentiment low-resource/two-stage runs: max_steps 400,
# load_best, eval_steps 50, batch 4, grad_accum 4, lr 2e-5, max_length 256,
# seed 42, fp16, gradient_checkpointing, adafactor. primary_only evaluation.
# Dev/test = ARENTCV2. Checkpoints are DELETED after evaluation (disk is tight).
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

LR=data/Topic/processed/lowresource_arentc
# Model-selection dev set. The full ARENTCV2 dev is 10,562 rows (12.9x the sentiment
# dev of 818); with eval_steps 50 / max_steps 400 it would be re-scored 8 times per
# run, adding ~28 min/run of pure overhead. A stratified 999-row subsample (111 per
# label, seed 42, identical across all arms) selects the best checkpoint just as well
# and is never reported. Test evaluation still uses the FULL 21,134-row test set.
DEV=data/Topic/processed/ARENTCV2/dev_sub1000.jsonl
TEST=data/Topic/processed/ARENTCV2/test.jsonl
GEN540=experiments/checkpoints/topic_gen540_xlmr
CKR=experiments/checkpoints/topic_lowresource
OUTR=experiments/outputs/multi_agent_bert/experiment_TopicLR_augmentation
mkdir -p "$OUTR" "$CKR"

for R in 10 25 50; do
  for ARM in only plus540; do
    TAG="arentc${R}_${ARM}"
    CK="$CKR/$TAG"; OUT="$OUTR/$TAG"; mkdir -p "$OUT"; rm -rf "$CK"

    if [ "$ARM" = "twostage" ]; then
      TRAIN="$LR/arentc${R}_only.jsonl"; BASE="$GEN540"
    elif [ "$ARM" = "plus540" ]; then
      TRAIN="$LR/arentc${R}_plus540.jsonl"; BASE="xlm-roberta-base"
    else
      TRAIN="$LR/arentc${R}_only.jsonl"; BASE="xlm-roberta-base"
    fi

    echo "===== START $TAG  train=$(basename "$TRAIN")  base=$(basename "$BASE")  $(date +%H:%M:%S) ====="
    "$PY" scripts/finetune_transformer_classifier.py \
      --train "$TRAIN" --dev "$DEV" \
      --labels business education finance health medical shopping social sports tech \
      --base_checkpoint "$BASE" --output_dir "$CK" \
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

    # Free the checkpoint immediately - only ~14 GB free on C:.
    rm -rf "$CK"
    echo "  checkpoint removed; free: $(df -h /c | tail -1 | awk '{print $4}')"
    sleep 30
  done
done
echo "===== ALL TOPIC LOW-RESOURCE RUNS DONE $(date +%H:%M:%S) ====="
