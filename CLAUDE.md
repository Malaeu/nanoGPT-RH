# CLAUDE.md — nanoGPT_RH Quick Reference

> **Читай это ПЕРВЫМ при входе в проект!**
> Здесь все пути, конвенции, workflow — чтобы не учить структуру заново.

---

## 🎯 QUICK REFERENCE (Jan 2026)

### Current Status
```
✅ E4 COMPLETE — ID-Detox works!
⭐ Best Model: checkpoints/E4_s7_best.pt (NLL=0.1942)
📍 Next: E5 (slot specialization) OR symbolic extraction
```

### Key Paths (ЗАПОМНИ!)
```
INPUT:
  data/continuous_2M/train.pt    # (7035, 256) training data
  data/continuous_2M/val.pt      # (781, 256) validation data

CODE:
  src/train_mdn_postfix.py       # ⭐ Main training script
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
```

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
├── 📁 runpod_workspace/         # ⚡ SSHFS mount to RunPod /workspace
│   └── pair-correlation/        # Current experiment on pod
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

### 🚀 SSHFS Live Mount (РЕКОМЕНДУЕТСЯ!)

**Лучший способ работы с RunPod** — примонтировать `/workspace` локально через SSHFS.
Никакого `runpodctl send/receive`! Все файлы синхронизируются автоматически.

**Установка macFUSE + SSHFS (один раз):**
```bash
brew install macfuse
brew install gromgit/fuse/sshfs-mac
```

**Монтирование RunPod workspace:**
```bash
# Создай точку монтирования (один раз)
mkdir -p runpod_workspace

# Примонтируй (при каждом запуске пода)
sshfs root@<POD_IP>:/workspace runpod_workspace/ -p <PORT> \
  -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3

# Пример:
sshfs root@69.30.85.23:/workspace runpod_workspace/ -p 22022
```

**Проверка:**
```bash
mount | grep runpod  # должен показать macfuse mount
ls runpod_workspace/  # видишь файлы с пода!
```

**Workflow с SSHFS:**
```bash
# Мониторинг тренировки в реальном времени
tail -f runpod_workspace/pair-correlation/train_flash.log

# Копирование результатов — просто cp!
cp runpod_workspace/out/experiment/best.pt checkpoints/

# Редактирование кода на поде — прямо из VS Code!
code runpod_workspace/
```

**Отмонтирование:**
```bash
umount runpod_workspace/
# или если завис:
diskutil unmount force runpod_workspace/
```

**Текущий mount:**
```
runpod_workspace/ → root@69.30.85.23:/workspace (pair-correlation)
```

---

### 📦 Package & Send (старый способ)
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
