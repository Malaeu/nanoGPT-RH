#!/bin/bash
# race_seeds.sh - Multi-seed racing for E5.5
#
# Launches N seeds in parallel, stops at 6k steps, shows results.
# Usage: ./scripts/race_seeds.sh [N_PARALLEL]
#
# Default: 4 seeds in parallel

set -e

N_PARALLEL=${1:-4}
MAX_STEPS=6000
SEEDS=(7 42 1337 2024)  # 4 seeds for conservative racing

echo "=== E5.5 Multi-Seed Racing ==="
echo "Seeds: ${SEEDS[*]}"
echo "Max steps: $MAX_STEPS"
echo "Parallel: $N_PARALLEL"
echo ""

# Create output directory
mkdir -p out/racing

# Launch seeds in parallel
for seed in "${SEEDS[@]}"; do
    echo "Launching seed $seed..."
    python src/train_mdn_postfix.py \
        --data-dir data/continuous_2M \
        --out-dir out/racing/s${seed} \
        --seed $seed \
        --slot-id-mode permute_per_batch \
        --use-aux-loss \
        --max-steps $MAX_STEPS \
        --perm-warmup-steps 2000 \
        --batch-size 512 --use-amp \
        > out/racing/s${seed}_stdout.log 2>&1 &
done

echo ""
echo "All seeds launched. Waiting for completion..."
echo "Monitor with: tail -f out/racing/s*/train.log"
echo ""

# Wait for all background jobs
wait

echo ""
echo "=== Racing Results ==="
echo ""

# Collect and sort results
for seed in "${SEEDS[@]}"; do
    if [ -f "out/racing/s${seed}/train.log" ]; then
        best=$(grep "best=" out/racing/s${seed}/train.log 2>/dev/null | tail -1 | grep -oE 'best=[0-9.]+' | cut -d= -f2)
        if [ -n "$best" ]; then
            echo "Seed $seed: best NLL = $best"
        else
            echo "Seed $seed: no results found"
        fi
    else
        echo "Seed $seed: log not found"
    fi
done | sort -t= -k2 -n

echo ""
echo "Top seeds can be continued with:"
echo "  python src/train_mdn_postfix.py --resume out/racing/s<SEED>/best.pt --out-dir out/E5_s<SEED>_full --early-stop --patience 10000"
