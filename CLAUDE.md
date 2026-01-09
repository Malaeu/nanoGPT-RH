# CLAUDE.md — nanoGPT_RH Quick Reference

> **Читай это ПЕРВЫМ при входе в проект!**
> Здесь все пути, конвенции, workflow — чтобы не учить структуру заново.

---

## 🎯 QUICK REFERENCE (Jan 2026)

### Current Status
```
✅ E4 COMPLETE — ID-Detox works!
⭐ Best Model: checkpoints/E4_s7_best.pt (NLL=0.1942)
🔬 TESTING: Is π real or bias artifact?
   Model predictions: sₙ = 3.1084/(s₋₁+s₋₂+s₋₃) ← π!
   True values (--target true): sₙ = 2.83/(...) ← NOT π!

   Hypothesis: π = 2.92 × 1.062 (model bias creates π!)
📍 Run 5 IN PROGRESS: PySR with --target true (~30 min remaining)
```

### Key Paths (ЗАПОМНИ!)
```
INPUT:
  data/continuous_500M/train.pt  # ⭐ (1.76M, 256) 500M zeros!
  data/continuous_500M/val.pt    # (195K, 256) val_tail
  data/continuous_2M/            # (legacy, 7K windows)

CODE:
  src/train_mdn_postfix.py       # ⭐ Main training script
  src/data_loading.py            # 🚀 Streaming DataLoader (gpu-direct/mmap)
  src/eval_mdn.py                # Evaluation
  src/diagnose_memory_postfix.py # Diagnostics

OUTPUT:
  out/                           # Temporary (gitignored)
  checkpoints/                   # Final models (in git)
  results/                       # Diagnostics output (in git)

DOCS:
  docs/PROJECT_MAP.md            # ⭐ Main project map
  docs/E4_SPEC.md                # E4 specification
  docs/runpod_specs.md           # GPU comparison
  docs/SPEED_OPTIMIZATION.md     # 🚀 GPU speed tricks (Ampere+)
  docs/OPERATOR_EXTRACTION.md    # 🧬 Masters-inspired operator extraction

FLASH CODE (новое):
  src/flash/                     # Flash-оптимизированные модели
  src/flash/mdn_flash.py         # Base SpacingMDN + RoPE
  src/flash/memory_mdn_flash.py  # Q3 MemoryMDN
  src/flash/train_memory_q3_flash.py  # Fast training script

FLASH DATA (180M точек!):
  src/data/flash_residuals/      # Residuals = spacings - 1.0

LMFDB DATA (500M zeros!):
  data/lmfdb_raw/                # Raw .dat files from LMFDB
  data/continuous_500M/          # Processed train.pt, val.pt

LAW-GRADE SCRIPTS (новое):
  scripts/prepare_lmfdb_500M.py  # Download & process LMFDB zeros
  scripts/eval_law.py            # Coverage/Width/Rollout eval
  scripts/conformal_calibrate.py # Conformal interval calibration
  scripts/symbolic_distill_quantiles.py  # Q0.1/Q0.5/Q0.9 → formulas
  scripts/extract_operator.py    # 🧬 Operator extraction из attention
```

---

## 🚀 НОВЫЕ ФИЧИ (Jan 2026)

### 1. Streaming DataLoader (`src/data_loading.py`)
Три режима загрузки данных для масштабирования до 500M+ нулей:

```bash
# GPU-direct (данные на GPU, самый быстрый)
python src/train_mdn_postfix.py --data-mode gpu-direct ...

# MMap (ленивая загрузка с диска, экономит RAM)
python src/train_mdn_postfix.py --data-mode mmap ...

# Auto (автовыбор по размеру VRAM)
python src/train_mdn_postfix.py --data-mode auto ...
```

**Классы:** `GPUDirectBatcher`, `MMapBatcher`, `DataLoaderWrapper`

### 2. torch.compile() (20-30% speedup)
Компиляция модели для GPU SM≥8.0 (Ampere+):

```bash
python src/train_mdn_postfix.py --use-compile ...
```

### 3. W&B Tracking (эксперименты)
Логирование в Weights & Biases:

```bash
# Онлайн (нужен wandb login)
python src/train_mdn_postfix.py --use-wandb --wandb-project nanoGPT-RH ...

# Оффлайн
WANDB_MODE=offline python src/train_mdn_postfix.py --use-wandb ...
```

### 4. Operator Extraction (`scripts/extract_operator.py`)
Извлечение kernel K(s_i, s_j) из attention для сравнения с GUE:

```bash
python scripts/extract_operator.py \
  --checkpoint checkpoints/E4_s7_best.pt \
  --data-dir data/continuous_2M \
  --output-dir results/operator_extraction \
  --run-pysr  # опционально: символьная регрессия
```

**Выход:**
- `kernel_visualization.png` — три графика attention patterns
- `extraction_results.json` — корреляция с sine kernel, exp decay

### 5. Conformal Calibration (`scripts/conformal_calibrate.py`)
Калибровка confidence intervals для честных 90%:

```bash
python scripts/conformal_calibrate.py \
  --ckpt checkpoints/E4_s7_best.pt \
  --data-dir data/continuous_2M \
  --alpha 0.1 \
  --output results/calibrator.json
```

**Выход:** `adjustment_q` — поправка для расширения интервалов

---

## 📥 LMFDB 500M ZEROS DOWNLOAD

### Источник данных
- **URL:** https://beta.lmfdb.org/data/riemann-zeta-zeros/
- **Precision:** ±2^{-102} (David Platt, Turing method verified)
- **Format:** Binary delta-encoded (13 bytes per zero)
- **Total:** 103.8 billion zeros available

### Скачивание и препарация (одна команда!)
```bash
python scripts/prepare_lmfdb_500M.py --download --max-zeros 500

# Флаги:
--download          # Качает из LMFDB с cookie human=1
--download-dir      # Куда качать raw .dat (default: data/lmfdb_raw)
--max-zeros N       # В МИЛЛИОНАХ! (500=500M, 100=100M, 10=10M)
--output-dir        # Куда сохранять train/val.pt
```

### Примеры:
```bash
# Быстрый тест (2M zeros, 1 файл)
python scripts/prepare_lmfdb_500M.py --download --max-zeros 2 --output-dir data/test_2M

# Средний датасет (100M zeros)
python scripts/prepare_lmfdb_500M.py --download --max-zeros 100 --output-dir data/continuous_100M

# Полный 500M (239 файлов)
python scripts/prepare_lmfdb_500M.py --download --max-zeros 500

# Использовать уже скачанные файлы
python scripts/prepare_lmfdb_500M.py --input-dir data/lmfdb_raw --max-zeros 100
```

### Формат binary (delta encoding!)
```python
# Block header (32 bytes):
t0, t1, Nt0, Nt1 = struct.unpack('<ddQQ', header)
n_zeros = Nt1 - Nt0

# Zero records (13 bytes each, DELTA encoded):
Z = 0  # Accumulator
for _ in range(n_zeros):
    z1, z2, z3 = struct.unpack('<QIB', record)
    delta = z1 + (z2 << 64) + (z3 << 96)
    Z += delta  # ACCUMULATE!
    gamma = t0 + Z * 2**(-101)
```

### Валидация spacings
После unfolding (Variant B: u(γ) = γ/2π × log(γ/2πe)):
- Mean ≈ 1.0 ✓
- Std ≈ 0.41 ✓ (GUE)
- Autocorr(1) < 0 ✓ (level repulsion)

### Готовый датасет (Jan 2026)
```
data/continuous_500M/
├── train.pt   [1,757,812 × 256] float32  # 1.8M windows!
├── val.pt     [195,312 × 256] float32    # 195K windows (val_tail)
└── meta.pt    hash=02fc584870ed65ac

Statistics:
  500M zeros processed
  γ range: [14.13, 193,418,189]
  Mean=1.0000, Std=0.4142, Autocorr=-0.357
```

**ВАЖНО:** Файлы должны сортироваться ЧИСЛОВЫМ порядком!
- ❌ Алфавитный: zeros_101246000 < zeros_14 (WRONG!)
- ✅ Числовой: zeros_14 < zeros_5000 < zeros_26000 (CORRECT!)

---

## 🔄 EXPERIMENT WORKFLOW

### Запуск нового эксперимента (E5, E6, etc.)

**Шаг 1: Создай рабочую папку**
```bash
mkdir -p out/E5_experiment_name
```

**Шаг 2: Запусти тренировку**
```bash
python src/train_mdn_postfix.py \
  --data-dir data/continuous_2M \
  --out-dir out/E5_experiment_name \
  --seed 7 \
  [новые флаги] \
  --batch-size 512 --use-amp
```

**Шаг 3: После завершения — раскладываем результаты**
```bash
# Лучшая модель → checkpoints/
cp out/E5_experiment_name/best.pt checkpoints/E5_best.pt

# Диагностика → results/
mkdir -p results/E5
python src/diagnose_memory_postfix.py \
  --ckpt checkpoints/E5_best.pt \
  --data-dir data/continuous_2M \
  --output-dir results/E5

# Обнови docs/PROJECT_MAP.md с результатами!
```

**Шаг 4: Cleanup временных файлов**
```bash
# out/ в gitignore — можно удалить или оставить локально
rm -rf out/E5_experiment_name/ckpt_*.pt  # удалить промежуточные
```

---

## 📁 I/O CONVENTIONS

### Naming Rules
```
Checkpoints:  E{N}_{description}_best.pt
              E4_s7_best.pt, E5_ortho_best.pt

Results:      results/E{N}/
              results/E4_s7/postfix_diagnostics.jsonl

Logs:         out/E{N}_{name}/train.log (temporary)
```

### Input Paths (НИКОГДА не меняй!)
```python
DATA_DIR = "data/continuous_2M"
TRAIN_PATH = "data/continuous_2M/train.pt"  # (7035, 256)
VAL_PATH = "data/continuous_2M/val.pt"      # (781, 256)
```

### Output Paths
```python
# Temporary (during training):
OUT_DIR = f"out/{experiment_name}"

# Permanent (after training):
CHECKPOINT = f"checkpoints/{experiment_name}_best.pt"
RESULTS = f"results/{experiment_name}/"
```

---

## 📝 LOGGING RULES (КРИТИЧНО!)

### При тренировке ОБЯЗАТЕЛЬНО логировать:

**В начале:**
- GPU name, VRAM
- batch_size
- Experiment config (flags)

**Каждый eval:**
```
Step 1000/20000 | val_nll=0.358 (best=0.358) | 5.8 steps/s | ETA: 55m | elapsed: 2.9m
```

**В конце:**
- Total time, steps/sec
- Best NLL achieved
- Cost estimate ($/hr × time)

---

## 📋 ПОСЛЕ КАЖДОГО ЭКСПЕРИМЕНТА

### Checklist:
- [ ] Скопировать `best.pt` в `checkpoints/`
- [ ] Запустить диагностику в `results/`
- [ ] Обновить `docs/PROJECT_MAP.md`:
  - Experiments Timeline
  - E{N} Results table
  - Key Insights Log
- [ ] Commit & push

---

## 🏗️ Repository Structure

```
nanoGpt_RH/
│
├── 📁 src/                      # MAIN CODE
│   ├── train_mdn.py             # Base SpacingMDN (MDNConfig)
│   ├── train_mdn_postfix.py     # ⭐ E4 training (ID-Detox, aux-loss)
│   ├── train_mdn_memory.py      # PREFIX memory (deprecated)
│   ├── eval_mdn.py              # Evaluation (NLL, CRPS, PIT)
│   ├── diagnose_memory.py       # PREFIX diagnostics
│   └── diagnose_memory_postfix.py # ⭐ POSTFIX diagnostics (A-K)
│
├── 📁 scripts/                  # UTILITIES
│   ├── prepare_continuous_2M.py # Unfolding zeros → spacings
│   ├── prepare_zeros.py         # Raw zeros processing
│   ├── prepare_primes.py        # Prime gaps dataset
│   └── runpod_setup.sh          # RunPod setup script
│
├── 📁 checkpoints/              # TRAINED MODELS (Git LFS)
│   ├── E0_baseline_best.pt      # SpacingMDN no memory
│   ├── E1_prefix_best.pt        # PREFIX (decorative)
│   ├── E2_prefix_best.pt        # PREFIX (seed variance)
│   ├── E3_postfix_s1337_best.pt # POSTFIX (ID-crutch)
│   ├── E4_s7_best.pt            # ⭐ BEST! NLL=0.1942
│   └── E4_s1337_best.pt         # E4 (stuck seed)
│
├── 📁 data/                     # DATASET
│   └── continuous_2M/           # Main dataset
│       ├── train.pt             # (7035, 256)
│       ├── val.pt               # (781, 256)
│       └── meta.pt
│
├── 📁 docs/                     # DOCUMENTATION
│   ├── PROJECT_MAP.md           # ⭐ MAIN PROJECT MAP
│   ├── E4_SPEC.md               # E4 specification
│   └── runpod_specs.md          # GPU comparison
│
├── 📁 results/                  # DIAGNOSTICS OUTPUT
│   └── E4_s7/
│       ├── postfix_diagnostics.jsonl
│       └── postfix_diagnostics.png
│
├── 📁 out/                      # TEMPORARY (gitignored)
│
└── 📁 archive/                  # OLD CODE (gitignored)
```

---

## 🖥️ RunPod Quick Start

### ⚠️ SSH Setup (ВАЖНО! Один раз навсегда)

**Проблема:** RunPod НЕ читает SSH ключи с твоего компа автоматически!
Ключи нужно добавить в RunPod Account Settings ЗАРАНЕЕ.

**Решение (один раз):**
1. Скопируй публичный ключ: `cat ~/.ssh/id_ed25519.pub`
2. Иди в [RunPod Settings → SSH Keys](https://www.runpod.io/console/user/settings)
3. Добавь ключ
4. Все НОВЫЕ поды будут работать без пароля

**Если под уже создан без ключа** — добавь через Web Terminal:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDmeQP05UiH0tXgAhL+Nx6nJZTgon9G63shnpUY9qL+2 emalam@example.com" \
>> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
```

**Подключение по SSH:**
```bash
# Вариант 1: через прокси (без SCP/SFTP)
ssh vjex2o62haxaew-644114f5@ssh.runpod.io -i ~/.ssh/id_ed25519

# Вариант 2: прямой TCP (с SCP/SFTP)
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```

Источник: [RunPod SSH Docs](https://docs.runpod.io/pods/configuration/use-ssh)

---

### 🚀 SCP File Transfer (ИСПОЛЬЗУЕМ!)

**Нативный SCP** — быстро, надёжно, не тормозит Mac.
SSHFS/macFUSE удалены — они грузили систему.

**Upload на RunPod:**
```bash
# Один файл
scp -P <PORT> -i ~/.ssh/id_ed25519 local_file.py root@<IP>:/workspace/pair-correlation/

# Несколько файлов
scp -P <PORT> -i ~/.ssh/id_ed25519 scripts/*.py root@<IP>:/workspace/pair-correlation/scripts/

# Целая папка
scp -rP <PORT> -i ~/.ssh/id_ed25519 src/ root@<IP>:/workspace/pair-correlation/src/
```

**Download с RunPod:**
```bash
# Один файл
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/pair-correlation/results/file.json ./

# Чекпоинт
scp -P <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/pair-correlation/out/best.pt checkpoints/

# Целая папка
scp -rP <PORT> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/pair-correlation/results/ ./results/
```

**Пример с текущим подом:**
```bash
# Upload скрипта
scp -P 22066 scripts/symbolic_distillation.py root@69.30.85.23:/workspace/pair-correlation/scripts/

# Download результатов
scp -P 22066 root@69.30.85.23:/workspace/pair-correlation/results/*.json ./results/
```

---

### 📦 Package & Send (альтернатива)
```bash
tar czf runpod_package.tar.gz \
  src/train_mdn.py src/train_mdn_postfix.py \
  src/eval_mdn.py src/diagnose_memory_postfix.py \
  scripts/runpod_setup.sh data/continuous_2M

runpodctl send runpod_package.tar.gz
```

### On Pod
```bash
runpodctl receive <CODE> && tar xzf runpod_package.tar.gz

# Training
python src/train_mdn_postfix.py \
  --data-dir data/continuous_2M \
  --out-dir out/E5_experiment \
  --seed 7 \
  --slot-id-mode permute_per_batch \
  --use-aux-loss \
  --early-stop --patience 800 \
  --batch-size 512 --use-amp
```

### Download Results
```bash
# On pod:
tar czf results.tar.gz out/E5_experiment/ && runpodctl send results.tar.gz

# On Mac:
runpodctl receive <CODE>
tar xzf results.tar.gz
```

### GPU Selection
```
BEST:    RTX 6000 Ada @ $0.77/hr (48GB, дешевле и быстрее L40S!)
         ↳ Low availability, но стоит подождать
DEFAULT: L40S @ $0.86/hr (48GB, ML-optimized, high availability)
BUDGET:  A40 @ $0.40/hr (48GB, best $/perf)
FAST:    H100 @ $2.69/hr (80GB, 2.5x speed)
```

**Benchmark (3 parallel, batch 512):**
- RTX 6000 Ada: ~2.0 steps/sec each
- L40S: ~1.6 steps/sec each
- RTX 6000 Ada wins: дешевле ($0.77 vs $0.86) И быстрее!

---

## ⚙️ E4 Training Flags Reference

```bash
python src/train_mdn_postfix.py \
  --data-dir data/continuous_2M \      # INPUT: always this!
  --out-dir out/experiment_name \      # OUTPUT: temporary
  --seed 7 \                           # Seed (7 worked best)
  --slot-id-mode permute_per_batch \   # ID-detox (E4)
  --content-mode normal \              # or zeroed
  --use-aux-loss \                     # Q3-proxy supervision
  --early-stop --patience 800 \        # Early stopping
  --batch-size 512 \                   # 512 for 48GB GPU
  --use-amp                            # Mixed precision
```

---

## 🔬 Diagnostics Metrics

### A-K метрики в diagnose_memory_postfix.py:
```
A) Ablation Δ      — важность слота (цель: >0.02)
B) Slot Similarity — косинусное сходство (цель: <0.5)
C) Grad Correlation— корреляция градиентов (цель: <0.7)
D) Readout Weights — веса чтения (цель: uniform)
E) Effect Entropy  — энтропия эффекта (цель: >1.5)
F) Slot Norms      — нормы слотов
G) Rollout Drift   — рост ошибки (цель: slope<0.5)
H) Cross-Block CV  — distribution shift (цель: <0.3)
I) Attention CoM   — center of mass (цель: std>10)
J) Permutation Inc — ID-reliance (цель: <10%)
K) Gradient Rank   — effective rank (цель: >50%)
```

---

## 📊 Experiments History

| ID | Architecture | Best NLL | Key Result |
|----|--------------|----------|------------|
| E0 | Baseline MDN | -0.25 | No memory |
| E1 | PREFIX Memory | -0.38 | Memory decorative |
| E2 | PREFIX Memory | -0.38 | Seed variance |
| E3 | POSTFIX Memory | 0.304 | ID-crutch detected |
| **E4** | **POSTFIX+ID-Detox** | **0.1942** | **ID-Detox works!** |

---

## ❌ COMMON MISTAKES (не делай!)

1. **Не запускай из root без указания путей!**
   ```bash
   # WRONG:
   python train_mdn_postfix.py

   # RIGHT:
   python src/train_mdn_postfix.py --data-dir data/continuous_2M
   ```

2. **Не забывай --out-dir!**
   - Без него output пойдет в случайное место

3. **Не коммить out/ !**
   - Только checkpoints/ и results/

4. **После эксперимента — обнови docs/PROJECT_MAP.md!**

---

## 🎓 Project Overview

**nanoGPT_RH** — Neural telescope for Riemann Hypothesis spectral analysis.

**Goal:** Train transformer on 2M unfolded zeta zeros to:
1. Learn GUE-like spacing distribution
2. Extract operator/kernel via attention
3. Compare with Q3 formal structures

**Architecture:** SpacingMDN + POSTFIX Memory Bank
- Memory slots AFTER data (bottleneck readout)
- ID-Detox prevents slot-ID cheating
- Q3-proxy aux loss for supervision

**Data:** Unfolded spacings with mean ≈ 1
```
s_n = Δ_n * log(γ_n) / (2π)
```

---

*Last updated: 2026-01-02*
