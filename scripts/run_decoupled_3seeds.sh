#!/usr/bin/env bash
# Run the new Decoupled (Route C) variants on the same 3 seeds as the
# fair Stage 0 fix2 pipeline. Reuses --evaluate-only behavior: if
# previous variants already have model.pt, they are loaded; only the new
# decoupled / decoupled+visibility variants will train.
#
# To get a clean combined table, point this at the SAME output dirs as
# table3_fair_fix2_seed{7,42,2026}/ — the orchestrator will skip already
# trained variants and just train + evaluate decoupled / decoupled+vis.
set -u

DOORRL_PY=/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python
ROOT=/mnt/volumes/cpfs/prediction/lipeinan/code
cd "$ROOT" || exit 1

SEEDS=(7 42 2026)
NUM_SCENES=700
EPOCHS=15
BATCH_SIZE=32

run_seed_decoupled () {
    local seed=$1
    local OUT="experiments/table3_fair_fix2_seed${seed}"
    local LOG="experiments/table3_fair_fix2_seed${seed}_decoupled.log"
    mkdir -p "$OUT"
    echo "================================================================"
    echo "[$(date '+%F %T')] Starting Route C decoupled run for seed=${seed}"
    echo "================================================================"
    # --variant all_with_decoupled enumerates base + decoupled + holistic.
    # --evaluate-only causes already-trained variants to be reloaded
    # from model.pt; missing ones (the two new decoupled variants) get
    # trained + evaluated. table3_complete.json gets overwritten with
    # the combined fair-Stage-0 + decoupled rows.
    "$DOORRL_PY" run_stage0_table3.py \
        --config configs/debug_mvp.json \
        --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
        --variant all_with_decoupled \
        --num-scenes "$NUM_SCENES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUT" \
        --seed "$seed" \
        --evaluate-only \
        > "$LOG" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] seed=${seed} decoupled finished with exit=${rc}"
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] seed=${seed} FAILED, stopping pipeline."
        exit $rc
    fi
}

for seed in "${SEEDS[@]}"; do
    run_seed_decoupled "$seed"
done

echo "================================================================"
echo "[$(date '+%F %T')] Decoupled runs done, re-aggregating..."
echo "================================================================"
"$DOORRL_PY" scripts/aggregate_fix2_seeds.py \
    --runs experiments/table3_fair_fix2_seed7/table3_complete.json \
           experiments/table3_fair_fix2_seed42/table3_complete.json \
           experiments/table3_fair_fix2_seed2026/table3_complete.json \
    --out experiments/table3_fair_fix2_aggregate.json
