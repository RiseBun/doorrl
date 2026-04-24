#!/usr/bin/env bash
# Stage 1 pilot: 1 seed × 3 conditions (bc, wm_object, wm_decoupled).
# Goal: verify the imagination loop trains without numerical issues
# BEFORE committing to the full 3-seed × 6-condition sweep.
#
# Estimated wallclock on H20 (bs=128, BF16, 10 epochs, 560 train scenes):
#   bc            ~15 min
#   wm_object     ~35 min  (K=5 re-encode is the main cost)
#   wm_decoupled  ~35 min
#   total         ~1.5 h
#
# Each condition warm-starts from the matching Stage 0 seed-7 checkpoint
# (encoder / abstraction / WM), so we inherit the good representation and
# only need to learn the policy on top.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
OUT_ROOT="${OUT_ROOT:-experiments/stage1_pilot}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR_SCALE="${LR_SCALE:-4}"
NUM_SCENES="${NUM_SCENES:-700}"
HORIZON="${HORIZON:-5}"

mkdir -p "${OUT_ROOT}/logs"

"${PY}" run_stage1_table4.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --condition pilot \
    --seed "${SEED}" \
    --num-scenes "${NUM_SCENES}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr-scale "${LR_SCALE}" \
    --horizon "${HORIZON}" \
    --output-dir "${OUT_ROOT}" \
    --stage0-root experiments/table3_fair_fix2_seed7 \
    2>&1 | tee "${OUT_ROOT}/logs/pilot_seed${SEED}.log"

echo "pilot done: ${OUT_ROOT}/seed${SEED}/stage1_all.json"
