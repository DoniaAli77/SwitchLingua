#!/usr/bin/env bash
# EXPLORATORY Silver topic-adaptation experiment: does pretraining XLM-R on the generated
# Topic-540 corpus improve adaptation to the limited Silver-labelled target domain?
#
# Three systems x model seeds 42/43/44, evaluated on the frozen 300-row Silver hybrid test:
#   A. G-only : Topic-540 136-step checkpoint, evaluated directly. Never sees Silver train.
#   B. S-only : fresh xlm-roberta-base, trained on the 860-row Silver train, 4 epochs.
#   C. G->S   : the Topic-540 checkpoint, fine-tuned on the same 860 rows, 4 epochs,
#               with a FRESH optimizer and scheduler (--base_checkpoint loads weights only;
#               no resume_from_checkpoint, so no optimizer state is restored).
#
# CHECKPOINT PROVENANCE: the per-seed Topic-540 136-step checkpoints from the matched-compute
# experiment were deleted by that script (rm -rf $CK after each eval). They are REGENERATED
# here with the identical recipe and the identical seed, and each is verified by re-evaluating
# it on Silver-1163 and comparing with the recorded accuracy (seed42 0.6148, seed43 0.6148,
# seed44 0.6139). Training is deterministic for a fixed seed, so these are the same models.
#
# LOCKED CONDITIONS (identical to the Topic-540 experiment): xlm-roberta-base tokenizer and
# preprocessing, 9 labels in the frozen order, max_length 256, adamw_torch, lr 2e-5,
# batch 16 x grad_accum 1 = effective 16, fp16, no dev split, no early stopping, no
# load_best_model_at_end, final checkpoint evaluated. 4 epochs over 860 rows = 216 steps
# (ceil(860/16)=54 per epoch); the observed count is asserted from each log by the summarizer.
# No rebalancing, upsampling, downsampling or class weights. No ArEnTC, no Topic-1080,
# no agents, no LLM calls. The full Silver corpus is used only for the reproduction check.
set -u
cd "$(dirname "$0")/.."
PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

LABELS="business education health shopping medical sports tech finance social"
GEN540=data/Topic/generated/merged/switchlingua_topic_train_540_60perlabel.jsonl
STRAIN=data/Topic/processed/silver_hybrid/silver_hybrid_train.jsonl
STEST=data/Topic/processed/silver_hybrid/silver_hybrid_test.jsonl
SILVER1163=experiments/outputs/multi_agent_bert/experiment_silver_topic540/silver_full1163_ordered.jsonl
OUTR=experiments/outputs/multi_agent_bert/experiment_silver_adaptation
CKG=experiments/checkpoints/_adapt_g_tmp
CKS=experiments/checkpoints/_adapt_s_tmp
CKGS=experiments/checkpoints/_adapt_gs_tmp
mkdir -p "$OUTR"

evaluate () {  # $1=checkpoint  $2=dataset  $3=outdir  $4=run_id
  mkdir -p "$3"
  "$PY" evaluate_pipeline.py \
    --dataset "$2" --config src/config/default.yaml --active_task topic_classification \
    --pipeline_mode primary_only --mode full_pipeline \
    --primary_model transformer --transformer_checkpoint "$1" \
    --transformer_device cuda --output_dir "$3" --run_id "$4" \
    > "$3/eval.log" 2>&1
  echo "    eval $4 exit=$?"
}

for SEED in 42 43 44; do
  echo "===================== SEED $SEED  $(date +%H:%M:%S) ====================="
  rm -rf "$CKG" "$CKS" "$CKGS"

  # ---- regenerate the Topic-540 136-step checkpoint for this seed ----
  echo "  [1/5] Topic-540 136 steps (regenerating checkpoint)  $(date +%H:%M:%S)"
  OUT="$OUTR/g540_seed${SEED}"; mkdir -p "$OUT"
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$GEN540" --labels $LABELS \
    --base_checkpoint xlm-roberta-base --output_dir "$CKG" \
    --max_steps 136 --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
    --seed "$SEED" --fp16 > "$OUT/finetune.log" 2>&1
  echo "    finetune exit=$? $(date +%H:%M:%S)"

  # reproduction check against the recorded matched-compute result
  echo "  [2/5] reproduction check on Silver-1163"
  evaluate "$CKG" "$SILVER1163" "$OUTR/repro1163_seed${SEED}" "repro1163_seed${SEED}"

  # ---- A. G-only on the 300-row Silver hybrid test ----
  echo "  [3/5] A. G-only on Silver hybrid test"
  evaluate "$CKG" "$STEST" "$OUTR/A_gonly_seed${SEED}" "A_gonly_seed${SEED}"

  # ---- C. G->S two-stage adaptation (fresh optimizer/scheduler) ----
  echo "  [4/5] C. G->S adaptation, 4 epochs on 860 rows  $(date +%H:%M:%S)"
  OUT="$OUTR/C_gs_seed${SEED}"; mkdir -p "$OUT"
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$STRAIN" --labels $LABELS \
    --base_checkpoint "$CKG" --output_dir "$CKGS" \
    --epochs 4 --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
    --seed "$SEED" --fp16 > "$OUT/finetune.log" 2>&1
  echo "    finetune exit=$? $(date +%H:%M:%S)"
  evaluate "$CKGS" "$STEST" "$OUT" "C_gs_seed${SEED}"

  # ---- B. S-only baseline ----
  echo "  [5/5] B. S-only, 4 epochs on 860 rows  $(date +%H:%M:%S)"
  OUT="$OUTR/B_sonly_seed${SEED}"; mkdir -p "$OUT"
  "$PY" scripts/finetune_transformer_classifier.py \
    --train "$STRAIN" --labels $LABELS \
    --base_checkpoint xlm-roberta-base --output_dir "$CKS" \
    --epochs 4 --batch_size 16 --grad_accum 1 --lr 2e-5 --max_length 256 \
    --seed "$SEED" --fp16 > "$OUT/finetune.log" 2>&1
  echo "    finetune exit=$? $(date +%H:%M:%S)"
  evaluate "$CKS" "$STEST" "$OUT" "B_sonly_seed${SEED}"

  rm -rf "$CKG" "$CKS" "$CKGS"
  echo "  seed $SEED done; free: $(df -h /c | tail -1 | awk '{print $4}')"
done
echo "===== SILVER ADAPTATION DONE $(date +%H:%M:%S) ====="
