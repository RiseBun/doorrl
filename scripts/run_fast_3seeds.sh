#!/usr/bin/env bash
# Fast 3-seed pipeline for fair Stage 0 with decoupled variants.
#
# Optimisations vs scripts/run_decoupled_3seeds.sh:
#   - BF16 autocast (enabled by default in DoorRLTrainer on CUDA).
#   - Large batch size (default 128) + LR linearly scaled by --lr-scale.
#   - DataLoader with num_workers=2, persistent_workers, prefetch_factor=4.
#   - 3 seeds launched CONCURRENTLY on the same GPU (model is ~540 k params,
#     each process uses <2 GB VRAM; H20's 96 GB easily hosts 3).
#
# Usage:
#   bash scripts/run_fast_3seeds.sh                 # default bs=128, 3 seeds
#   BATCH_SIZE=256 LR_SCALE=8 bash scripts/run_fast_3seeds.sh
#
# Env knobs:
#   BATCH_SIZE   (default 128)
#   LR_SCALE     (default 4, matching bs 32 -> 128 linear scaling)
#   EPOCHS       (default 30)
#   NUM_SCENES   (default 700)
#   OUT_ROOT     (default experiments/table3_fast)
#   SEEDS        (default "7 42 2026")
#   VARIANT      (default all_with_decoupled)
#
# Exit code is 0 iff all seeds succeed.
set -uo pipefail

cd "$(dirname "$0")/.."

PY="/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR_SCALE="${LR_SCALE:-4}"
EPOCHS="${EPOCHS:-30}"
NUM_SCENES="${NUM_SCENES:-700}"
OUT_ROOT="${OUT_ROOT:-experiments/table3_fast}"
SEEDS="${SEEDS:-7 42 2026}"
VARIANT="${VARIANT:-all_with_decoupled}"

mkdir -p "${OUT_ROOT}/logs"

echo "[$(date)] launching seeds=${SEEDS} concurrently"
echo "  variant=${VARIANT}  bs=${BATCH_SIZE}  lr_scale=${LR_SCALE}  epochs=${EPOCHS}  num_scenes=${NUM_SCENES}"

pids=()
for seed in ${SEEDS}; do
    log="${OUT_ROOT}/logs/seed${seed}.log"
    echo "  -> seed=${seed}  log=${log}"
    "${PY}" run_stage0_table3.py \
        --config configs/debug_mvp.json \
        --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
        --variant "${VARIANT}" \
        --seed "${seed}" \
        --num-scenes "${NUM_SCENES}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr-scale "${LR_SCALE}" \
        --scene-val-ratio 0.2 \
        --output-dir "${OUT_ROOT}/seed${seed}" \
        > "${log}" 2>&1 &
    pids+=("$!")
    # Stagger launches by 30 s so the 3 processes don't hit the tokenisation
    # phase simultaneously (I/O contention) and their DataLoader worker
    # forks don't all race for shared memory at once.
    sleep 30
done

echo "[$(date)] all seeds running; waiting..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
        echo "[$(date)] pid=${pid} FAILED"
    fi
done

if [[ "${fail}" -ne 0 ]]; then
    echo "[$(date)] one or more seeds failed; check logs in ${OUT_ROOT}/logs/"
    exit 1
fi

echo "[$(date)] all seeds done; aggregating..."
run_jsons=()
for seed in ${SEEDS}; do
    run_jsons+=("${OUT_ROOT}/seed${seed}/table3_complete.json")
done
"${PY}" scripts/aggregate_fix2_seeds.py \
    --runs "${run_jsons[@]}" \
    --out "${OUT_ROOT}/aggregate.json" \
    > "${OUT_ROOT}/logs/aggregate.log" 2>&1 || {
    echo "aggregation failed; see ${OUT_ROOT}/logs/aggregate.log"
    exit 1
}

echo "[$(date)] done. results: ${OUT_ROOT}/aggregate.json"
