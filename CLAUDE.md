# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## TRAINING LOGGING RULES (КРИТИЧНО!)

При запуске тренировки ВСЕГДА показывать в логах:

1. **В начале тренировки:**
   - GPU name (nvidia-smi)
   - VRAM total / used
   - batch_size
   - Estimated steps/sec (после первых 100 steps)

2. **Каждые N steps (eval):**
   - Current step / total steps
   - val_nll (и best)
   - Elapsed time
   - **steps/sec** (текущая скорость!)
   - **ETA** (сколько осталось!)
   - patience (если early-stop)

3. **В конце тренировки:**
   - Total time
   - Total steps
   - Final steps/sec
   - samples/sec
   - Cost estimate ($/hr × time)

**Пример формата лога:**
```
Step 1000/20000 | val_nll=0.358 (best=0.358) | 5.8 steps/s | ETA: 55m | elapsed: 2.9m
```

**ЗАЧЕМ:** Чтобы сравнивать GPU и выбирать оптимальный под!
См. `docs/runpod_specs.md` для бенчмарков.

---

## DOCUMENTATION RULES (ОБЯЗАТЕЛЬНО!)

### После успешных изменений обновляй:

1. **docs/PROJECT_MAP.md** — главная карта проекта
   - Experiments Timeline (новые эксперименты)
   - File Map (новые/измененные скрипты)
   - Implementation Checklist (что сделано)
   - Key Insights Log (важные открытия)

2. **e*_summary.md** — саммари эксперимента (e3_summary.md, e4_summary.md и т.д.)
   - Создавать при запуске нового эксперимента
   - Обновлять при получении результатов

### Что документировать:
- Новые training скрипты
- Изменения архитектуры
- Результаты экспериментов (NLL, метрики)
- Баги и их решения
- Инсайты и выводы

### Формат Key Insights:
```markdown
### YYYY-MM-DD: Краткое название
- **Problem/Observation:** что обнаружили
- **Evidence:** данные/метрики
- **Conclusion/Action:** что делать дальше
```

---

## Project Overview

**nanoGPT_RH** — Neural telescope for Riemann Hypothesis spectral analysis. We train a small transformer (nanoGPT) on 2M unfolded zeta zeros to:
1. Learn stationary statistics (GUE-like spacing distribution, spectral rigidity)
2. Extract hidden state geometry ("helmet" manifold)
3. Distill operator/kernel approximation via attention logits + PySR symbolic regression

This is NOT a "prove RH with neural nets" project. It's a controlled lab to study spectral invariants and compare with Q3 kernel/operator structures (Toeplitz-RKHS bridge, prime cap, uniform floor c*=11/10).

## Architecture Philosophy

- **nanoGPT over large LLMs** — we need observability, not chat power
- **Continuous or binned spacings** — NOT text tokenization (avoid "14.1347" as chunks)
- **Tabula rasa** — if small model learns spectral invariants from scratch, that's stronger scientific signal
- **Attention ≈ Kernel** — attention logits A_ij as function of distance d=|u_i-u_j|

## Data Pipeline

### Unfolding (critical preprocessing)

Raw zeros → unfolded spacings with mean ≈ 1:

**Variant A (local density):**
```
Δ_n = γ_{n+1} - γ_n
s_n = Δ_n * log(γ_n) / (2π)
```

**Variant B (unfolded coordinates):**
```
u(γ) = (γ/2π) * log(γ/(2πe))
s_n = u(γ_{n+1}) - u(γ_n)
```

Quality check: mean(s) ≈ 1 on large blocks.

### Sequence formatting
- Sequence length L=256 (configurable)
- Train/val split BY BLOCKS (no shuffling — preserves structure)

## Commands

```bash
# Setup environment (uv + Python 3.13)
uv venv && source .venv/bin/activate
uv pip install torch numpy scipy matplotlib rich

# Data prep
python scripts/prepare_continuous_2M.py --input zeros_2M.txt --output data/continuous_2M

# E4 Training (POSTFIX + ID-Detox)
python src/train_mdn_postfix.py \
  --data-dir data/continuous_2M \
  --out-dir out/mdn_postfix_E4_s7 \
  --seed 7 \
  --slot-id-mode permute_per_batch \
  --use-aux-loss \
  --early-stop --patience 800 \
  --batch-size 512 --use-amp

# Evaluation
python src/eval_mdn.py --ckpt checkpoints/E4_s7_best.pt --data-dir data/continuous_2M

# Diagnostics
python src/diagnose_memory_postfix.py \
  --ckpt checkpoints/E4_s7_best.pt \
  --data-dir data/continuous_2M \
  --output-dir results/E4_s7
```

## Key Experiments

### Baselines (mandatory sanity checks)
1. **Shuffled spacings** — destroy correlations, keep marginals
2. **i.i.d. resample** — sample from empirical distribution
3. **Positional encoding only** — no learned weights

### Metrics
- Spacing histogram vs GUE Wigner surmise: P(s) = (πs/2)exp(-πs²/4)
- Spectral form factor: ramp → plateau transition
- Hidden state manifold stability across seeds

### Kernel extraction
1. Collect attention logits A_ij before softmax
2. Build dataset (d_k, y_k) where d = |u_i - u_j|
3. Run PySR: look for sine-kernel-like forms on Pareto front

## Model Choice

**Phase 1:** nanoGPT / minGPT / tiny Transformer (fast iterations on Mac)
**Phase 2:** Larger models when metrics and signal are clear

## Loss Design

- **Primary:** next-spacing prediction (MSE for regression, CE for bins)
- **Diagnostics only (not in loss):** Mehta/GUE distribution — use as external validator, not built-in (avoids "you forced the network" criticism)
- **Optional soft regularizers:** penalty for too many tiny spacings (level repulsion)

## Repository Structure (Jan 2026)

```
nanoGpt_RH/
│
├── 📁 src/                      # MAIN CODE
│   ├── train_mdn.py             # Base SpacingMDN (MDNConfig, transformer)
│   ├── train_mdn_postfix.py     # E4 POSTFIX training (ID-Detox, aux-loss)
│   ├── train_mdn_memory.py      # PREFIX memory (deprecated)
│   ├── eval_mdn.py              # Evaluation (NLL, CRPS, PIT)
│   ├── diagnose_memory.py       # PREFIX diagnostics
│   └── diagnose_memory_postfix.py # POSTFIX diagnostics (A-K metrics)
│
├── 📁 scripts/                  # UTILITIES
│   ├── prepare_continuous_2M.py # Unfolding zeros → spacings
│   ├── prepare_zeros.py         # Raw zeros processing
│   ├── prepare_primes.py        # Prime gaps dataset
│   └── runpod_setup.sh          # RunPod setup script
│
├── 📁 checkpoints/              # TRAINED MODELS (Git LFS)
│   ├── E0_baseline_best.pt      # SpacingMDN no memory
│   ├── E1_prefix_best.pt        # PREFIX memory (decorative)
│   ├── E2_prefix_best.pt        # PREFIX memory (seed variance)
│   ├── E3_postfix_s1337_best.pt # POSTFIX (ID-crutch)
│   ├── E4_s7_best.pt            # ⭐ BEST! NLL=0.1942, ID-Detox works!
│   └── E4_s1337_best.pt         # POSTFIX + ID-Detox (stuck seed)
│
├── 📁 data/                     # DATASET
│   ├── continuous_2M/           # Main dataset
│   │   ├── train.pt             # (7035, 256) training spacings
│   │   ├── val.pt               # (781, 256) validation spacings
│   │   └── meta.pt              # Dataset metadata
│   ├── train.pt, val.pt         # Copies in root
│   └── *_primes.pt              # Prime gaps dataset
│
├── 📁 docs/                     # DOCUMENTATION
│   ├── PROJECT_MAP.md           # ⭐ MAIN PROJECT MAP (experiments, results)
│   ├── E4_SPEC.md               # E4 specification (ID-Detox)
│   ├── runpod_specs.md          # GPU comparison & benchmarks
│   ├── PROJECT_GUIDE.md         # Detailed project guide
│   └── *.md                     # Session summaries, drafts
│
├── 📁 results/                  # DIAGNOSTICS OUTPUT
│   └── E4_s7/
│       ├── postfix_diagnostics.jsonl  # Metrics JSON
│       └── postfix_diagnostics.png    # Visualization
│
├── 📁 archive/                  # OLD CODE (gitignored)
│   ├── old_training/            # Deprecated train_*.py
│   ├── analysis/                # Old analysis scripts
│   └── images/                  # Old PNG files
│
├── CLAUDE.md                    # THIS FILE
├── README.md                    # Project readme
└── .gitignore                   # Git ignore rules
```

### Current Status: E4 COMPLETE ✅
- **Best Model:** `checkpoints/E4_s7_best.pt`
- **NLL:** 0.1942 (+36% vs E3!)
- **ID-Detox:** Works! Perm Inc = 1.0%
- **Next:** E5 (slot specialization) or symbolic extraction

## Q3 Integration Points

Cross-reference with Q3 formal structures:
- Attention logits → compare with sine kernel / Toeplitz symbol
- Operator norm cap → check if learned interactions respect bounds
- Prime block structure → compare attention patterns with ρ(t) formulation

## Notes

- 256 bins for spacing classification gives stable perplexity metric
- RoPE / sinusoidal positional encoding justified for sequence + phase structure
- Save hidden states during training for manifold analysis

## RunPod GPU Training Guide

### ОДНА КОМАНДА для полного pipeline

**НА МАКЕ (подготовка):**
```bash
cd /Users/emalam/Documents/GitHub/nanoGpt_RH

# Создать пакет
tar czf runpod_package.tar.gz \
  src/train_mdn.py src/train_mdn_postfix.py \
  src/eval_mdn.py src/diagnose_memory_postfix.py \
  scripts/runpod_setup.sh data/continuous_2M

# Отправить (получишь код типа: 2406-final-rufus-fashion-5)
runpodctl send runpod_package.tar.gz
```

**НА ПОДЕ (одна команда!):**
```bash
cd /workspace && runpodctl receive <КОД> && tar xzf runpod_package.tar.gz && chmod +x runpod_setup.sh && ./runpod_setup.sh
```

**СКАЧАТЬ РЕЗУЛЬТАТЫ:**
```bash
# На поде:
tar czf results.tar.gz out/ reports/ && runpodctl send results.tar.gz

# На маке:
runpodctl receive <КОД>
tar xzf results.tar.gz
```

### Проблемы с Web Terminal / Jupyter

**ИЗВЕСТНЫЙ БАГ RUNPOD!** Веб-терминал и Jupyter часто не работают:
- После рестарта пода веб-терминал может не запуститься
- Jupyter показывает 502 Bad Gateway
- Терминал в Jupyter пустой/не печатает

**РЕШЕНИЯ:**
1. **Используй SSH вместо веб-терминала** (надежнее):
   ```bash
   # Proxy SSH (работает всегда, но без SCP):
   ssh espzoobm5yxuif-64410f63@ssh.runpod.io -i ~/.ssh/id_ed25519

   # Или Direct TCP (если настроен):
   ssh -p <PORT> root@<IP> -i ~/.ssh/id_ed25519
   ```

2. **Перезапусти под** - иногда веб-терминал работает только при первом запуске

3. **Проверь шаблон пода** - только официальные шаблоны RunPod поддерживают Jupyter

4. **Подожди 2-3 минуты** после старта пода - иногда нужно время

### Установка runpodctl (один раз)
```bash
brew install runpod/runpodctl/runpodctl
```

### SSH Access (опционально)
Для Direct TCP SSH, добавь ключ НА ПОДЕ:
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "ssh-ed25519 AAAA... твой_ключ" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### GPU Selection (Jan 2026)

**DEFAULT CHOICE: L40S @ $0.86/hr**
- 48GB VRAM, Ada architecture, ML-optimized
- High availability, datacenter drivers
- Best balance of price/performance/availability

**BUDGET: A40 @ $0.40/hr**
- 48GB VRAM, Ampere (older but cheap)
- Best $/performance for single jobs

**PARALLEL SEEDS: RTX 6000 Ada @ $0.77/hr**
- 48GB VRAM, can run 3 seeds @ batch=512

See `docs/runpod_specs.md` for full GPU comparison.

### Performance Tips
- **L40S 48GB**: batch_size 512-1024, --use-amp
- **H100 80GB**: batch_size 2048, --use-amp --compile
- **A40 48GB**: batch_size 512, --use-amp
- FlashAttention включен по умолчанию в train_mdn.py

### Что делает runpod_setup.sh
1. pip install torch numpy scipy matplotlib rich
2. Проверка GPU
3. train_mdn.py (20k steps, ~50 мин на H100)
4. eval_mdn.py (метрики)
5. train_mdn_memory.py (10k steps, ~30 мин)
6. diagnose_memory.py (диагностика)

## GPU Benchmark (2024-12-31)

### Speed Comparison (SpacingMDN+Memory, 4.8M params)

| GPU | $/hr | steps/s | batch | Δ per 500 steps | samples/s |
|-----|------|---------|-------|-----------------|-----------|
| Mac M4 Max | FREE | 1.3 | 128 | ~3.2m | 166 |
| A40 48GB | $0.40 | 5.8 | 512 | ~1.4m | 2,969 |
| H100 80GB | $3.69 | ~15* | 512 | ~33s* | ~7,680* |

*H100 estimated based on typical 2.5x speedup over A40

### Cost Analysis (20k steps)

| GPU | Time | Cost | Cost/10k steps |
|-----|------|------|----------------|
| Mac M4 Max | ~4.3 hr | $0 | $0 |
| A40 | ~57 min | ~$0.38 | $0.19 |
| H100 | ~22 min | ~$1.35 | $0.68 |

### Recommendation

**A40 = лучший выбор по цена/производительность для наших моделей!**
- 18x быстрее Mac по samples/sec
- 3.5x дешевле H100 при 2.5x меньшей скорости
- 48GB VRAM достаточно для batch_size=512

Для маленьких моделей (<10M params) H100 избыточна.
