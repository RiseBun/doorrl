#!/usr/bin/env bash
# Stage 1 pilot Y: isolate visibility weighting as the decoupled-vs-object
# collision-rate gap cause.
#
# Setup (see docs/stage1_pilot.md §5):
#   * Reuse v3 hparams verbatim: entropy_beta=0.003, action_clip=5, bs=128,
#     lr_scale=4, horizon=5, epochs=10, seed=7, 700 scenes.
#   * Only change: the representation. `wm_decoupled_no_vis` uses
#     `object_relation_decoupled` (same typed-budget top-k, NO visibility
#     multiplication on the dyn path).
#
# Expected outcomes:
#   (a) If visibility is the primary cause of the v3 decoupled gap:
#       collision drops sharply, stability (ego cos) drops, return persists.
#       => mechanism: "visibility × latent creates multiplicative noise in
#          imagination rollout, hurting policy learning".
#   (b) If no_vis does NOT close the gap:
#       the issue is decoupled's rel fusion / WM rollout error / actor's
#       use of the fused latent, and we need a different ablation.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
OUT_ROOT="${OUT_ROOT:-experiments/stage1_pilot_y}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR_SCALE="${LR_SCALE:-4}"
NUM_SCENES="${NUM_SCENES:-700}"
HORIZON="${HORIZON:-5}"
ENTROPY_BETA="${ENTROPY_BETA:-0.003}"
ACTION_CLIP="${ACTION_CLIP:-5.0}"

mkdir -p "${OUT_ROOT}/logs"

echo "=============================================================="
echo "Y ablation: wm_decoupled_no_vis  (entropy=${ENTROPY_BETA}, clip=${ACTION_CLIP})"
echo "=============================================================="
"${PY}" run_stage1_table4.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --condition wm_decoupled_no_vis \
    --seed "${SEED}" \
    --num-scenes "${NUM_SCENES}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr-scale "${LR_SCALE}" \
    --horizon "${HORIZON}" \
    --output-dir "${OUT_ROOT}" \
    --stage0-root experiments/table3_fair_fix2_seed7 \
    --entropy-beta "${ENTROPY_BETA}" \
    --action-sample-clip "${ACTION_CLIP}" \
    2>&1 | tee "${OUT_ROOT}/logs/y_seed${SEED}.log"

echo "Y done: ${OUT_ROOT}/seed${SEED}/wm_decoupled_no_vis/stage1_metrics.json"
