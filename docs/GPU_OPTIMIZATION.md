# GPU Optimization Notes

## Current Hardware
- **GPU**: NVIDIA TITAN RTX 24GB VRAM
- **CPU**: AMD Ryzen Threadripper 2920X (24 cores)
- **RAM**: 64GB

## Current Usage (SpacingMDN training)
- **VRAM**: 8.2GB / 24GB (34%)
- **GPU Compute**: 99%
- **Batch size**: 128
- **Model**: 4.8M params (6L/8H/256E)

## Optimization Opportunities

| Parameter | Current | Recommended | Impact |
|-----------|---------|-------------|--------|
| Batch size | 128 | 256-512 | +VRAM, faster convergence |
| Model size | 4.8M | 10-20M | +VRAM, +capacity |
| Mixed precision | No | FP16/BF16 | 2x batch size |
| Gradient checkpointing | No | Yes | -VRAM, longer training |
| Compile (torch.compile) | No | Yes | +20-30% speed |

## Larger Model Configs

### Medium (10M params)
```bash
python train_mdn.py \
    --batch-size 256 \
    --n-layer 8 \
    --n-head 8 \
    --n-embd 384 \
    --n-components 8
```

### Large (20M params)
```bash
python train_mdn.py \
    --batch-size 256 \
    --n-layer 12 \
    --n-head 12 \
    --n-embd 512 \
    --n-components 16
```

## Mixed Precision Training

Add to train_mdn.py:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    result = model(x, targets=x)
    loss = result['loss']

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## torch.compile (PyTorch 2.0+)

```python
model = torch.compile(model)  # After creating model
```

## DataLoader Optimization

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,        # More workers for Threadripper
    pin_memory=True,      # Faster CPU->GPU transfer
    drop_last=True,
    persistent_workers=True,
)
```

## Multi-GPU (if available)

```python
model = torch.nn.DataParallel(model)
# or
model = torch.nn.parallel.DistributedDataParallel(model)
```

---
*Last updated: 2025-12-30*
