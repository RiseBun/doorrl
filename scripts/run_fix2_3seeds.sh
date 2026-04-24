#!/usr/bin/env bash
# Run Fix #2 fair Stage 0 across 3 seeds sequentially, then aggregate.
# Seed 7: re-evaluates the 4 already-trained variants from saved
# checkpoints (with the patched evaluator that filters to dynamic-type
# slots) and trains the only missing variant (holistic).
# Seeds 42 & 2026: full train + eval from scratch.
# Output: experiments/table3_fair_fix2_seedXX/{table3_complete.json, *.log}
set -u

DOORRL_PY=/mnt/volumes/cpfs/prediction/lipeinan/environments/conda/envs/doorrl/bin/python
ROOT=/mnt/volumes/cpfs/prediction/lipeinan/code
cd "$ROOT" || exit 1

NUM_SCENES=700
EPOCHS=15
BATCH_SIZE=32  # conservative; flow_planner co-runs on the same H20

run_seed () {
    local seed=$1
    local extra_flag=${2:-}
    local OUT="experiments/table3_fair_fix2_seed${seed}"
    local LOG="experiments/table3_fair_fix2_seed${seed}.log"
    mkdir -p "$OUT"
    echo "================================================================"
    echo "[$(date '+%F %T')] Starting Fix#2 seed=${seed} (extra='${extra_flag}')"
    echo "================================================================"
    "$DOORRL_PY" run_stage0_table3.py \
        --config configs/debug_mvp.json \
        --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
        --variant all \
        --num-scenes "$NUM_SCENES" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUT" \
        --seed "$seed" \
        $extra_flag \
        > "$LOG" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] seed=${seed} finished with exit=${rc}"
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%F %T')] seed=${seed} FAILED, stopping pipeline."
        exit $rc
    fi
}

# Seed 7: 4/5 variants already trained -> re-evaluate from .pt with the
# patched evaluator; missing 'holistic' will be trained on the fly.
run_seed 7 "--evaluate-only"

# Seeds 42 and 2026: full train + eval.
run_seed 42 ""
run_seed 2026 ""

echo "================================================================"
echo "[$(date '+%F %T')] All 3 seeds finished, running aggregation..."
echo "================================================================"
"$DOORRL_PY" scripts/aggregate_fix2_seeds.py \
    --runs experiments/table3_fair_fix2_seed7/table3_complete.json \
           experiments/table3_fair_fix2_seed42/table3_complete.json \
           experiments/table3_fair_fix2_seed2026/table3_complete.json \
    --out experiments/table3_fair_fix2_aggregate.json
