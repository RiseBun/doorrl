#!/usr/bin/env bash
# Stage 1 pilot v3: tame the wm_object actor.
#
# v2 finding (see docs/stage1_pilot.md): after Huber critic + reward clip
# stopped WM corruption, the `wm_object` actor saturated at |a|max ≈ 7.7
# with log_std at its clamp (+0.5) and collided 98.5 % of the time. v3
# applies two knobs expected to rein this in:
#   * --entropy-beta 0.003   (down from 0.01): less pressure on the
#                              policy to maintain high entropy.
#   * --action-sample-clip 5 (down from 8.0):  hard cap the reparam
#                              sample; still >3σ of the tanh-bounded
#                              distribution, so this is a safety net,
#                              not a hard constraint during training.
#
# We rerun BOTH wm_object (the problem child) and wm_decoupled (the
# control: the same knobs must not destroy its behaviour). bc is
# unchanged under these flags, so we skip it — its v2 numbers still
# stand.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
OUT_ROOT="${OUT_ROOT:-experiments/stage1_pilot_v3}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-7}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR_SCALE="${LR_SCALE:-4}"
NUM_SCENES="${NUM_SCENES:-700}"
HORIZON="${HORIZON:-5}"
ENTROPY_BETA="${ENTROPY_BETA:-0.003}"
ACTION_CLIP="${ACTION_CLIP:-5.0}"

mkdir -p "${OUT_ROOT}/logs"

for COND in wm_object wm_decoupled; do
    echo "=============================================================="
    echo "v3 ablation: condition=${COND}  entropy_beta=${ENTROPY_BETA}  action_clip=${ACTION_CLIP}"
    echo "=============================================================="
    "${PY}" run_stage1_table4.py \
        --config configs/debug_mvp.json \
        --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
        --condition "${COND}" \
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
        2>&1 | tee -a "${OUT_ROOT}/logs/v3_seed${SEED}.log"
done

echo "v3 done. Aggregates per-condition under ${OUT_ROOT}/seed${SEED}/"
