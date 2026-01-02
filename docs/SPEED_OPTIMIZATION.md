# Speed Optimization Guide for Ampere+ GPUs

> **Цель:** Максимальная скорость БЕЗ потери качества на A40/A100/L40S/H100/RTX 30xx+

---

## TL;DR — Копипасти это в training script

```python
import torch

# === CUDA OPTIMIZATIONS (Ampere+) ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')  # TF32 by default

# === DATA ON GPU (главная оптимизация!) ===
train_data = torch.load('train.pt').to(device)  # один раз в начале!

# В training loop:
idx = torch.randint(0, train_data.shape[0], (batch_size,), device=device)
x = train_data[idx]  # GPU→GPU = мгновенно

# === MIXED PRECISION (BF16 на Ampere+) ===
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(x)
    loss = criterion(output, target)

# === FUSED OPTIMIZER ===
optimizer = torch.optim.AdamW(params, lr=lr, fused=True)

# === OPTIONAL: torch.compile ===
model = torch.compile(model, mode='reduce-overhead')  # 10-30% speedup
```

---

## Полный Checklist

### 1. Data Loading (🔥 КРИТИЧНО — 10-50x speedup!)

| Метод | Скорость | Trade-off |
|-------|----------|-----------|
| DataLoader (CPU→GPU) | 🐌 | Работает с любым размером данных |
| **GPU-only indexing** | 🚀 | Данные должны влезать в VRAM |

```python
# ❌ МЕДЛЕННО — каждый батч копируется с CPU
for batch in DataLoader(dataset, batch_size=256):
    batch = batch.to(device)  # ~10-50ms на батч!

# ✅ БЫСТРО — данные уже на GPU
train_data = train_data.to(device)  # один раз
idx = torch.randint(0, N, (batch_size,), device=device)
x = train_data[idx]  # ~0.01ms
```

**Trade-off:** Данные должны влезать в GPU RAM.
- 687MB данных = OK для любой GPU
- 10GB данных = нужна A100 80GB или CPU fallback

---

### 2. Mixed Precision (3x speedup на matmul)

| Тип | Точность | Скорость | Совместимость |
|-----|----------|----------|---------------|
| FP32 | Высокая | 1x | Все GPU |
| FP16 + GradScaler | Высокая | 8-16x | Все GPU |
| **BF16** | Средняя | 8-16x | **Ampere+ only** |
| TF32 | Средняя | 2-3x | Ampere+ only |

```python
# ✅ Для Ampere+ (A40, A100, L40S, H100, RTX 30xx+)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    output = model(x)

# BF16 не требует GradScaler!
# Сразу backward() без scaling
```

**Почему BF16 лучше FP16:**
- BF16: 8-bit exponent = тот же range как FP32
- FP16: 5-bit exponent = overflow риск → нужен GradScaler
- На Ampere+ одинаковая скорость

---

### 3. TF32 (бесплатный 2-3x speedup)

```python
# ✅ Включить для всех matmul операций
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

TF32 = FP32 range (8-bit exp) + FP16 precision (10-bit mantissa)
- Автоматически для FP32 операций на Ampere+
- ~0.1% потеря точности — незаметно для ML

---

### 4. cuDNN Benchmark (5-20% speedup)

```python
torch.backends.cudnn.benchmark = True
```

**Что делает:** Ищет оптимальный алгоритм для conv/attention.

**Trade-off:**
- Первый forward МЕДЛЕННЫЙ (поиск алгоритма)
- Все последующие БЫСТРЫЕ
- Размеры input должны быть ФИКСИРОВАННЫЕ!

---

### 5. Fused Optimizer (10-20% speedup)

```python
# ✅ PyTorch 2.0+
optimizer = torch.optim.AdamW(params, lr=lr, fused=True)

# Или Apex (ещё быстрее, но нужна установка):
# from apex.optimizers import FusedAdam
# optimizer = FusedAdam(params, lr=lr)
```

**Что делает:** Объединяет 3 kernel launches в 1.

**Trade-off:** Никакого! Просто быстрее.

---

### 6. torch.compile (10-30% speedup)

```python
# Базовый режим (стабильный)
model = torch.compile(model)

# Агрессивный (быстрее, но дольше компиляция)
model = torch.compile(model, mode='reduce-overhead')

# Максимальный (самый быстрый, долгая компиляция)
model = torch.compile(model, mode='max-autotune')
```

**Trade-off:**
- Долгая первая компиляция (30-120 сек)
- Не все модели совместимы
- Debugging сложнее

**Трюк — региональная компиляция:**
```python
# Компилируем только один блок, reuse для всех
compiled_block = torch.compile(model.blocks[0])
for i in range(len(model.blocks)):
    model.blocks[i] = compiled_block
# Компиляция 7x быстрее!
```

---

### 7. CUDA Graphs (5x speedup для маленьких батчей!)

```python
# Для batch_size < 64 где CPU overhead доминирует
g = torch.cuda.CUDAGraph()

# Warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        output = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Capture
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay (очень быстро!)
g.replay()
```

**Trade-off:**
- Input/output должны быть статическими (те же тензоры)
- Сложнее debugging
- Максимальный эффект на маленьких батчах

---

### 8. Flash Attention (2-4x speedup, меньше памяти)

```python
# PyTorch 2.0+ — автоматически через SDPA
from torch.nn.functional import scaled_dot_product_attention

# Или явно:
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = scaled_dot_product_attention(q, k, v)
```

**Trade-off:** Только FP16/BF16, только Ampere+.

---

## Рекомендуемые настройки

### Быстрые эксперименты (скорость > всё)

```python
batch_size = 32
seq_len = 256
# GPU-only data
# BF16
# TF32
# Fused AdamW
# cudnn.benchmark = True
# НЕ torch.compile (долгая компиляция)
```

### Production training (баланс)

```python
batch_size = 256-512
seq_len = 256-512
# Всё выше +
# torch.compile(mode='reduce-overhead')
```

### Максимальная скорость (batch < 64)

```python
batch_size = 32
seq_len = 512
# Всё выше +
# CUDA Graphs
```

---

## Бенчмарки (наши данные, A40 GPU)

| Настройка | Steps/sec | Примечание |
|-----------|-----------|------------|
| E4 original (DataLoader) | ~0.5 | CPU→GPU bottleneck |
| + GPU-only data | ~5.0 | 10x speedup |
| + BF16 + TF32 | ~8.0 | 16x vs original |
| + Fused AdamW | ~9.0 | 18x vs original |
| + torch.compile | ~11.0 | 22x vs original |

---

## Sources

- [PyTorch Mixed Precision Training](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- [HuggingFace GPU Training Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one)
- [RunPod Mixed Precision Guide](https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training)
- [torch.compile Tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [PyTorch Performance Tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Graphs in PyTorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [Sebastian Raschka — Accelerating PyTorch](https://magazine.sebastianraschka.com/p/accelerating-pytorch-model-training)

---

*Last updated: 2026-01-02*
