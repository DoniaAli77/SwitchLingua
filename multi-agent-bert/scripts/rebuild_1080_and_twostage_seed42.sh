#!/usr/bin/env bash
# Rebuild the two deleted checkpoints, SEED 42 ONLY, saving to D: and DELETING NOTHING.
#
#  1) topic1080_272  : xlm-roberta-base, 1080 generated rows, max_steps 272
#                      -> recipe copied verbatim from scripts/run_topic1080_272steps.sh
#  2) twostage_gs    : stage 1 = gen-540 @136 steps (base xlm-roberta-base)
#                      stage 2 = fine-tune that on silver_hybrid_train, 4 epochs
#                      -> recipe copied verbatim from scripts/run_silver_adaptation.sh
#
# Training is deterministic on this setup (verified: matched-compute 540/seed-42
# reproduced the learning-curve run to delta 0.0000/0.0000), so these rebuilds
# reconstruct the original weights exactly rather than approximating them.
#
# Each model is re-evaluated primary_only on ITS OWN established test set:
#   topic1080_272 -> Silver-1163 (full)         [unchanged]
#   twostage_gs   -> silver_hybrid_test (300)   [unchanged]
# The reproduction check compares against the saved originals.
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

LABELS="business education health shopping medical sports tech finance social"
GEN540=data/Topic/generated/merged/switchlingua_topic_train_540_60perlabel.jsonl
GEN1080=data/Topic/generated/merged/switchlingua_topic_train_1080_120perlabel.jsonl
STRAIN=data/Topic/processed/silver_hybrid/silver_hybrid_train.jsonl
STEST=data/Topic/processed/silver_hybrid/silver_hybrid_test.jsonl
SILVER1163=experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl

CKROOT=/d/switchlingua_checkpoints          # persistent, NOT deleted
OUTR=experiments/outputs/multi_agent_bert/experiment_rebuild_seed42
mkdir -p "$CKROOT" "$OUTR"
SEED=42

evaluate () {  # $1=ckpt $2=dataset $3=outdir $4=run_id
  mkdir -p "$3"
  "$PY" evaluate_pipeline.py \
    --dataset "$2" --config src/config/default.yaml --active_task topic_classification \
    --pipeline_mode primary_only --mode full_pipeline \
    --primary_model transformer --transformer_checkpoint "$1" \
    --transformer_device cuda --output_dir "$3" --run_id "$4" \
    > "$3/eval.log" 2>&1
  echo "    eval $4 exit=$?"
}

echo "===== [1/3] topic1080_272 seed $SEED  $(date +%H:%M:%S) ====="
CK1080="$CKROOT/topic1080_272_seed${SEED}"
if [ -d "$CK1080" ] && [ -f "$CK1080/config.json" ]; then
  echo "  already present, skipping training"
else
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$GEN1080" --labels $LABELS \
    --base_checkpoint xlm-roberta-base --output_dir "$CK1080" \
    --max_steps 272 --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
    --seed "$SEED" --fp16 > "$OUTR/finetune_1080.log" 2>&1
  echo "  finetune exit=$? $(date +%H:%M:%S)"
fi
evaluate "$CK1080" "$SILVER1163" "$OUTR/topic1080_272_seed${SEED}" "topic1080_272_seed${SEED}"

echo "===== [2/3] two-stage: stage 1 gen540 @136  $(date +%H:%M:%S) ====="
CKG="$CKROOT/gen540_136_seed${SEED}"
if [ -d "$CKG" ] && [ -f "$CKG/config.json" ]; then
  echo "  already present, skipping training"
else
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$GEN540" --labels $LABELS \
    --base_checkpoint xlm-roberta-base --output_dir "$CKG" \
    --max_steps 136 --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
    --seed "$SEED" --fp16 > "$OUTR/finetune_gen540.log" 2>&1
  echo "  finetune exit=$? $(date +%H:%M:%S)"
fi

echo "===== [3/3] two-stage: stage 2 -> silver_hybrid_train, 4 epochs  $(date +%H:%M:%S) ====="
CKGS="$CKROOT/twostage_gs_seed${SEED}"
if [ -d "$CKGS" ] && [ -f "$CKGS/config.json" ]; then
  echo "  already present, skipping training"
else
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$STRAIN" --labels $LABELS \
    --base_checkpoint "$CKG" --output_dir "$CKGS" \
    --epochs 4 --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
    --seed "$SEED" --fp16 > "$OUTR/finetune_twostage.log" 2>&1
  echo "  finetune exit=$? $(date +%H:%M:%S)"
fi
evaluate "$CKGS" "$STEST" "$OUTR/twostage_gs_seed${SEED}" "twostage_gs_seed${SEED}"

echo "===== REBUILD DONE $(date +%H:%M:%S) ====="
echo "checkpoints kept at $CKROOT :"
du -sh "$CKROOT"/* 2>/dev/null
echo "C: free $(df -h /c | tail -1 | awk '{print $4}')   D: free $(df -h /d | tail -1 | awk '{print $4}')"
