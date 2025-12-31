#!/bin/bash
# RunPod Setup Script for TASK_SPEC_2M

echo "=== RunPod Setup for TASK_SPEC_2M ==="

# 1. Install dependencies
pip install torch numpy scipy matplotlib rich

# 2. Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# 3. Run training (OPTIMIZED: FlashAttention + torch.compile + large batch)
echo "Starting SpacingMDN training (optimized)..."
python train_mdn.py \
    --data-dir data/continuous_2M \
    --out-dir out/mdn_baseline \
    --max-steps 20000 \
    --batch-size 2048 \
    --eval-interval 500 \
    --save-interval 5000 \
    --use-amp \
    --compile

echo "Baseline training complete!"

# 4. Run eval (with reduced samples to avoid OOM during Memory Bank training)
echo "Running eval suite..."
python eval_mdn.py \
    --ckpt out/mdn_baseline/best.pt \
    --data-dir data/continuous_2M \
    --output-dir reports/2M \
    --n-pit 1000 \
    --n-crps 1000 \
    --rollout 100 \
    --n-rollouts 16

# 5. Train Memory Bank (OPTIMIZED: AMP + torch.compile)
echo "Training Memory Bank (optimized)..."
python train_mdn_memory.py \
    --data-dir data/continuous_2M \
    --out-dir out/mdn_memory_v0 \
    --max-steps 10000 \
    --batch-size 1024 \
    --use-amp \
    --compile

# 6. Run memory diagnostics
echo "Running memory diagnostics..."
python diagnose_memory.py \
    --ckpt out/mdn_memory_v0/best.pt \
    --data-dir data/continuous_2M \
    --output-dir reports/2M

echo "=== All done! ==="
