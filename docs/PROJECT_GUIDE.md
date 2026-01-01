# nanoGPT_RH: SpacingMDN + Memory Bank

## Что это за проект?

**Цель:** Обучить нейросеть предсказывать расстояния между нулями дзета-функции Римана.

**Почему это интересно:**
- Нули дзета-функции имеют статистику как у GUE (Gaussian Unitary Ensemble)
- Это связано с гипотезой Римана и квантовым хаосом
- Если модель выучит паттерны — можем извлечь символические формулы через PySR

---

## Архитектура модели

### Base: SpacingMDN (Mixture Density Network)

```
Input: [s1, s2, ..., sT]  — последовательность spacing'ов (T=255)
                          (spacing = расстояние между соседними нулями)

┌─────────────────────────────────────────────────────────┐
│  Transformer Encoder (6 layers, 8 heads, 256 embed)    │
│  - Causal attention (каждая позиция видит только прошлое)│
│  - RoPE positional encoding                             │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  MDN Head (Mixture Density Network)                     │
│  - K=8 компонент гауссовой смеси                        │
│  - Output: π (weights), μ (means), σ (stds)             │
│  - P(s_next) = Σ π_k * N(μ_k, σ_k²)                    │
└─────────────────────────────────────────────────────────┘

Параметры: ~4.5M
```

### Memory Bank (надстройка)

**Архитектура: PREFIX TOKENS (не отдельный модуль!)**

```
┌─────────────────────────────────────────────────────────┐
│  Memory Bank: M=8 learnable slots                       │
│  - Каждый slot = вектор размера 256                     │
│  - Slot-ID embeddings (уникальная идентичность)         │
│  - PREPEND к input sequence (prefix tokens)             │
└─────────────────────────────────────────────────────────┘
```

**Как это работает в forward():**

```python
# train_mdn_memory.py, строки 298-316:

memory = self.memory_bank(B, device)     # (B, M, D) = (B, 8, 256)
x = self.input_proj(x)                   # (B, T, D) = (B, 255, 256)

# КЛЮЧЕВОЙ МОМЕНТ: конкатенация!
x = torch.cat([memory, x], dim=1)        # (B, M+T, D) = (B, 263, 256)
#              ^^^^^^   ^
#              prefix   data

# Positional embeddings для ВСЕЙ последовательности
pos = torch.arange(0, M + T, device=device)  # [0, 1, 2, ..., 262]
x = x + self.wpe(pos)

# Transformer видит ВСЮ последовательность
for block in self.blocks:
    x = block(x)

# После: разделяем обратно
memory_outputs = x[:, :M, :]   # (B, 8, 256)  → для aux_loss
data_outputs   = x[:, M:, :]   # (B, 255, 256) → для MDN
```

**Визуально:**

```
Input to Transformer (263 позиции):
┌──────────────────────────────────────────────────────────────────┐
│ pos:  0    1    2    3    4    5    6    7    8    9   ...  262 │
│      [M0] [M1] [M2] [M3] [M4] [M5] [M6] [M7] [s1] [s2] ... [sT] │
│       └───────────── memory prefix ──────────┘ └──── data ────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  Transformer (6 layers, causal attention)
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Causal mask:                                                    │
│    - M0 видит только себя                                        │
│    - M7 видит M0..M7                                             │
│    - s1 видит M0..M7, s1  ← MEMORY ДОСТУПНА!                    │
│    - sT видит M0..M7, s1..sT                                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Output split:                                                   │
│    memory_outputs = x[:, :8, :]   → aux_loss (Q3 invariants)    │
│    data_outputs   = x[:, 8:, :]   → MDN head (π, μ, σ)          │
└──────────────────────────────────────────────────────────────────┘
```

**Почему PREFIX, а не отдельный модуль?**

| Подход | Плюсы | Минусы |
|--------|-------|--------|
| **PREFIX tokens** ✅ | Простота, memory участвует в attention | Memory "съедает" capacity |
| Cross-attention | Чистое разделение | Сложнее, доп. параметры |
| Side module | Независимость | Нет прямого gradient flow |

**Наш выбор: PREFIX** — проще, attention flow естественный, легче диагностировать.

**Зачем Memory Bank?**
- Transformer с causal attention видит только прошлое
- Memory даёт доступ к "глобальному состоянию"
- Гипотеза: memory может хранить Q3 инварианты (M0-M7)

---

## Параметры модели

### Архитектурные параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `n_layer` | 6 | Количество transformer блоков |
| `n_head` | 8 | Количество attention heads |
| `n_embd` | 256 | Размерность embeddings |
| `block_size` | 256 | Максимальная длина последовательности |
| `n_components` | 8 | Количество компонент MDN (K) |
| `n_memory_slots` | 8 | Количество memory slots (M) |
| `use_slot_id` | True | Добавлять уникальные ID к слотам |

### Training параметры

| Параметр | Значение | Описание |
|----------|----------|----------|
| `batch_size` | 512-1024 | Размер batch (зависит от GPU) |
| `lr` | 3e-4 | Learning rate для основной модели |
| `memory_lr` | 9e-5 | Learning rate для memory (0.3x base) |
| `max_steps` | 20000 | Количество шагов обучения |
| `warmup_steps` | 500 | Warmup для learning rate |
| `weight_decay` | 0.1 | L2 regularization |

### Drift Aux параметры (E2+)

| Параметр | E1 | E2 | Описание |
|----------|-----|-----|----------|
| `drift_aux_weight` | 0.01 | 0.1 | Вес drift loss в total loss |
| `drift_aux_every` | 20 | 10 | Как часто считать drift loss |
| `drift_aux_horizon` | 10 | 20 | Длина rollout для drift |
| `drift_aux_batch` | 16 | 32 | Batch size для drift loss |

---

## Метрики и их интерпретация

### 1. NLL (Negative Log-Likelihood)

```
NLL = -log P(y_true | model)

Меньше = лучше (модель точнее предсказывает)

Типичные значения:
  Начало обучения: +0.5 ... +1.0
  После 20k steps:  -0.3 ... -0.6

Baseline (без memory): -0.57
Memory v1:             -0.44 (хуже!)
```

**Интерпретация:** NLL показывает качество one-step prediction. Memory модели часто хуже по NLL из-за дополнительных параметров.

### 2. mem_entropy (Entropy of attention to memory)

```
mem_entropy = -Σ p_i * log(p_i)

где p_i = доля внимания на slot i

Максимум: ln(8) ≈ 2.08 (равномерное внимание)
Минимум: 0 (всё внимание на один slot)

Типичные значения:
  mem_entropy ≈ 2.08 → равномерное (не значит бесполезно!)
  mem_entropy < 1.5  → есть фавориты
```

**Интерпретация:** mem_entropy=2.08 НЕ означает что memory бесполезна! Внимание может быть равномерным, но содержимое слотов разным.

### 3. eff% (Effect / MAE)

```
eff% = max_slot_effect / baseline_MAE × 100

где:
  max_slot_effect = max изменение предсказания при отключении слота
  baseline_MAE = средняя ошибка модели

Интерпретация:
  0-1%    → BALLAST (слоты бесполезны)
  1-3%    → MARGINAL (слабое влияние)
  3-10%   → USEFUL (норм влияние)
  10-30%  → STRONG (сильное влияние) ← ИДЕАЛ
  30%+    → DOMINANT (переобучение на слоты)
```

**Интерпретация:** Показывает насколько один slot влияет на one-step prediction относительно общей ошибки.

### 4. Err@H (Rollout Error at horizon H)

```
Err@10 = MAE после 10 шагов авторегрессии

Процесс:
  1. Даём модели seed [s1..s10]
  2. Предсказываем s11'
  3. Кормим [s1..s10, s11'] → предсказываем s12'
  4. Повторяем 10 раз
  5. Сравниваем [s11'..s20'] с ground truth

Типичные значения:
  Err@10 ≈ 0.25-0.35
```

**Интерпретация:** Ошибки накапливаются при rollout. Это ключевая метрика для реального использования модели.

### 5. rollout_effect (Δ Err@H при отключении слота)

```
rollout_effect = Err@10(slot_off) - Err@10(base)

Положительный: отключение слота УХУДШАЕТ rollout → слот ПОМОГАЕТ
Отрицательный: отключение слота УЛУЧШАЕТ rollout → слот МЕШАЕТ

Интерпретация:
  Δ < 0%   → слот = BALLAST (даже вредит!)
  Δ = 0-2% → слот = MARGINAL
  Δ > 5%   → слот = HOLDS FOCUS (работает!)
```

**КЛЮЧЕВАЯ МЕТРИКА!** Именно rollout_effect показывает работает ли memory как "global lens".

---

## Концепции

### Teacher Forcing vs Rollout

```
TEACHER FORCING (training):
  Модель всегда видит ground truth
  Input:  [GT1, GT2, GT3, GT4]
  Target: [GT2, GT3, GT4, GT5]

  Проблема: модель не учится исправлять свои ошибки

ROLLOUT (inference):
  Модель кормится своими предсказаниями
  Step 1: [GT1, GT2, GT3] → pred4
  Step 2: [GT1, GT2, GT3, pred4] → pred5
  Step 3: [GT1, GT2, GT3, pred4, pred5] → pred6

  Проблема: ошибки НАКАПЛИВАЮТСЯ (exposure bias)
```

### Local Conditioner vs Global Lens

```
LOCAL CONDITIONER:
  Memory влияет на текущий шаг, но не помогает rollout
  eff% высокий, rollout_effect низкий

  Пример: v1 имел eff%=3.45%, rollout Δ=0.16%

GLOBAL LENS:
  Memory держит "фокус" на протяжении всего rollout
  eff% средний, rollout_effect высокий

  Цель: достичь rollout Δ > 5%
```

### Drift Aux Loss

```
drift_loss = |cumsum(predictions) - cumsum(ground_truth)|

Идея: штрафуем за накопление ошибки на rollout
      Это учит memory "держать фокус"

Total Loss = NLL + drift_aux_weight × drift_loss
```

---

## Memory Map Q3: Замкнутая цепь инвариантов

### Философия: Memory = Learnable Lens

Каждый memory slot — это **настраиваемый элемент линзы/решета**.
Q3 условия — это **замкнутая цепь**, все звенья должны быть ВСЕГДА активны.

```
┌─────────────────────────────────────────────────────────────────┐
│  PRIMES (источники света)                                       │
│     ↓                                                            │
│  SIEVE (фильтр/решето) = MEMORY BANK                            │
│     ↓                                                            │
│  1/2 LINE (фокус) = предсказание нулей                          │
└─────────────────────────────────────────────────────────────────┘
```

### ⚠️ ВАЖНО: Модель НЕ ЗНАЕТ наших ярлыков!

**Факт:** для модели M0-M7 — это просто 8 обучаемых векторов.
Она кладёт туда что угодно. Названия — это НАША интерпретация.

**Текущий aux_loss (4 слота):**
- Slot 0 → mean
- Slot 1 → std
- Slot 2 → autocorr lag-1
- Slot 3 → autocorr lag-5
- Slots 4-7 → **СВОБОДНЫЕ**

**Проблема:** это базовые статистики, НЕ Q3-инварианты!

### M0–M7: Детальная карта (Q3 → Линза → ML-метрика)

#### M0 — Нормировка T0

| Аспект | Значение |
|--------|----------|
| **Линза** | "Шкала экрана / яркость фона правильная" |
| **Цель слота** | `Δmean = mean(x) - 1.0` (отклонение от идеала) |
| **Почему Q3** | Фиксирует базовую "калибровку света" |
| **Тех. термин** | **aux loss** — доп. лосс для семантики слота |

#### M1 — Покрытие режимов A1'

| Аспект | Значение |
|--------|----------|
| **Линза** | "Линза умеет разные апертуры, не залипает" |
| **Цель слота** | **Энтропия гистограммы** (8-16 бинов) или `q90 - q10` |
| **Почему Q3** | A1' = "достаточный базис", форма распределения лучше чем std |
| **Тех. термин** | **entropy** — мера разнообразия распределения |

#### M2 — Непрерывность / Lipschitz A2

| Аспект | Значение |
|--------|----------|
| **Линза** | "Линза не дрожит от микро-шевеления" |
| **Цель слота** | `max|Δx|` и/или `RMS|Δx|` (скачки между соседними) |
| **Почему Q3** | Прямой датчик "модуля непрерывности" |
| **Тех. термин** | **Lipschitz** — макс. прыжок выхода при малом изменении входа |

#### M3 — Floor A3 (Level Repulsion)

| Аспект | Значение |
|--------|----------|
| **Линза** | "Нет провалов/дырок в подсветке" |
| **Цель слота** | **Низкий квантиль** `q1%` (стабильнее чем min!) |
| **Почему Q3** | Floor = нижний барьер, квантиль устойчивее |
| **Тех. термин** | **quantile** — значение, ниже которого лежит X% данных |

#### M4 — Smoothness ω (гладкость PSF)

| Аспект | Значение |
|--------|----------|
| **Линза** | "Пятно фокуса (PSF) без ряби" |
| **Цель слота** | **Энергия 2-й разности**: `|x[t+1] - 2*x[t] + x[t-1]|` |
| **Почему Q3** | ω = гладкость символа, отсутствие "кривизны" |
| **Тех. термин** | **PSF** — форма пятна фокуса (как линза размазывает точку) |

#### M5 — Toeplitz (устойчивость к дискретизации)

| Аспект | Значение |
|--------|----------|
| **Линза** | "Линза не ломается на пикселях; смена сетки не рушит фокус" |
| **Цель слота** | **Расхождение статистик** 1-й vs 2-й половины окна |
| **Почему Q3** | Toeplitz = зависит только от расстояния, не от индекса |
| **Тех. термин** | **Toeplitz** — матрица с трансляционной инвариантностью |

#### M6 — RKHS Cap ("простые не кричат")

| Аспект | Значение |
|--------|----------|
| **Линза** | "Ни один источник не выжигает экран; вклад ограничен" |
| **Цель слота** | **High-freq energy**: FFT power на верхней 1/4 частот для Δx |
| **Почему Q3** | RKHS = контроль нормы/гладкости, высокие частоты = "рваность" |
| **Тех. термин** | **RKHS** — пространство с бюджетом на рваность |

#### M7 — Spectral Gap / Rigidity

| Аспект | Значение |
|--------|----------|
| **Линза** | "Фокус устойчив на дальних масштабах" |
| **Цель слота** | **Локальная rigidity**: отклонение cumsum от линейки |
| **Почему Q3** | Spectral gap = паразитные режимы подавлены |
| **Тех. термин** | **spectral gap** — "разрыв спектра", долгий фокус |

### Ловушка "Студент-ботан" и решение

**Проблема:** модель может:
- ✅ Выполнять aux-задачи через память
- ❌ Но НЕ использовать память для предсказаний!

Память становится "студентом, который решает контрольные, но не участвует в игре".

**Решение — связать память с предсказанием:**

| Метод | Как работает |
|-------|--------------|
| **FiLM** | Память задаёт scale/shift для активаций (как ручки линзы) |
| **Gating** | Без памяти сигнал не проходит полноценно |
| **MDN readout** | MDN head получает summary(memory) напрямую |

### Как модель "узнаёт" Q3 инварианты?

**Путь 1: Semantic Supervision (aux_loss)** — слоты имеют смысл

```python
def compute_q3_invariants(x):  # x: [B, T] spacings
    dx = x[:, 1:] - x[:, :-1]  # первые разности
    d2x = dx[:, 1:] - dx[:, :-1]  # вторые разности

    m0 = x.mean(dim=1) - 1.0                      # T0: отклонение от 1
    m1 = hist_entropy(x, bins=16)                 # A1': энтропия гистограммы
    m2 = dx.abs().max(dim=1)[0]                   # A2: max скачок
    m3 = torch.quantile(x, 0.01, dim=1)           # A3: floor (q1%)
    m4 = d2x.abs().mean(dim=1)                    # A3: smoothness
    m5 = half_window_divergence(x)                # Toeplitz: стабильность
    m6 = high_freq_energy(dx)                     # RKHS: high-freq cap
    m7 = local_rigidity(x)                        # Spectral: Δ3 proxy

    return torch.stack([m0, m1, m2, m3, m4, m5, m6, m7], dim=1)
```

**Путь 2: Post-hoc Interpretation (ablation)** — проверить реальное использование

- Какой слот влияет на mean? → ablate, measure Δmean
- Какой слот влияет на tail? → ablate, measure Δ(q99)
- Какой слот стабилизирует rollout? → ablate, measure Δ(Err@100)

**Правильный ответ: ГИБРИД обоих путей!**

### Регуляризаторы: все включены, но с ramp

```
Все регуляризаторы ВКЛЮЧЕНЫ с первого шага, но с плавным нарастанием:

┌────────────────────┬────────────┬────────────┬──────────────────────┐
│ Регуляризатор      │ start      │ target     │ ramp                 │
├────────────────────┼────────────┼────────────┼──────────────────────┤
│ Orthogonality      │ 1e-4       │ 1e-2       │ linear 0→5k steps    │
│ Attention entropy  │ 0.0        │ 0.005      │ linear 0→2k steps    │
│ Mass cap (30%)     │ 0.0        │ 0.01       │ step at 1k           │
│ Memory norm cap    │ always on  │ max=5.0    │ projection           │
│ MDN entropy        │ 0.005      │ 0.005      │ constant             │
└────────────────────┴────────────┴────────────┴──────────────────────┘

НИКАКОГО slot dropout — все слоты ВСЕГДА присутствуют!
```

### Диагностика: 6 феноменов (не "проблем")

| Феномен | Что измеряем | Скрипт |
|---------|--------------|--------|
| A) Shortcut | Куда модель запихивает? | `diagnose_memory.py` |
| B) Symmetry | Все слоты одинаковые? | `diagnose_memory.py` |
| C) Co-adaptation | Слоты "договариваются"? | `diagnose_memory.py` |
| D) Gradient correlation | Учатся вместе? | `diagnose_memory.py` |
| E) Train vs Rollout | Стабильность при переносе? | `diagnose_memory.py` |
| F) Slot effect norm | Реальный импакт? | `diagnose_memory.py` |

**Ключевой принцип:** Это ФИЧИ для измерения, не баги для исправления!

---

## История экспериментов

### Baseline (без memory)
```
Результаты:
  val_nll:  -0.5721 (лучший NLL!)
  CRPS:     0.0835
  Err@100:  0.2772

Вывод: Сильный baseline. Memory должна улучшить rollout.
```

### Memory v0 (первый прогон)
```
Конфигурация:
  n_memory_slots: 8
  use_slot_id: False
  aux_loss: None

Результаты:
  mem_entropy: 2.08 (равномерное)
  Все слоты одинаковые (permutation invariance)

Вывод: Без slot-ID слоты не дифференцируются.
```

### Memory v1 (+ slot-ID + window-stats aux)
```
Конфигурация:
  use_slot_id: True
  aux_loss_weight: 0.1 (window statistics)

Результаты:
  val_nll:  -0.4413 (хуже baseline)
  eff%:     3.45% (USEFUL)
  rollout Δ: +0.16% (BALLAST!)

Вывод: Memory работает как LOCAL CONDITIONER, не GLOBAL LENS.
       Window-stats aux не помогает rollout.
```

### E1 (drift aux, слабые параметры)
```
Конфигурация:
  drift_aux_weight: 0.01 (слабый)
  drift_aux_every: 20
  drift_aux_horizon: 10

Результаты:
  val_nll:  -0.3793 (ещё хуже)
  eff%:     2.39% (MARGINAL)
  rollout Δ: -0.16% (отрицательный! слоты мешают)

Вывод: weight=0.01 слишком мал. Drift aux не успевает повлиять.
```

### E2 (drift aux, агрессивные параметры) ← ТЕКУЩИЙ
```
Конфигурация:
  drift_aux_weight: 0.1 (10x от E1)
  drift_aux_every: 10 (2x чаще)
  drift_aux_horizon: 20 (2x длиннее)
  drift_aux_batch: 32 (2x больше)

Промежуточные результаты (step 3000):
  val_nll:  0.1936 (ещё падает)
  eff%:     4.9% (РАСТЁТ! было 3.7% на step 2000)
  Err@10:   0.3266

Статус: IN PROGRESS
ETA: ~90 минут
```

---

## Следующие шаги

### Если E2 покажет хороший rollout_effect (> 5%):
1. Увеличить horizon (H=50, H=100)
2. Попробовать разные weight schedules (linear, exponential)
3. Извлечь формулы через PySR

### Если E2 не поможет:
1. **E3**: Ещё агрессивнее (weight=0.2, horizon=30)
2. **Альтернатива**: Slot dropout (случайно отключать слоты при обучении)
3. **Альтернатива**: Orthogonality loss (заставить слоты быть разными)

### Долгосрочные цели:
1. Достичь rollout Δ > 10%
2. Извлечь символическую формулу для attention kernel
3. Сравнить с теоретическим sine kernel из RMT

---

## Команды

### Запуск обучения
```bash
python train_mdn_memory.py \
  --data-dir data/continuous_2M \
  --out-dir out/mdn_memory_E2 \
  --use-slot-id \
  --drift-aux-weight 0.1 \
  --drift-aux-every 10 \
  --drift-aux-horizon 20 \
  --max-steps 20000 \
  --batch-size 512 \
  --use-amp
```

### Проверка статуса
```bash
# На RunPod:
tail -f /workspace/out/mdn_memory_E2/train.log

# Checkpoint info:
python3 -c "import torch; e=torch.load('best.pt',map_location='cpu',weights_only=False); print('step=',e['step'],'val_nll=',round(e['val_nll'],4))"
```

### Тест slot_effect
```bash
python test_slot_effect.py
```

---

## Файловая структура

```
nanoGpt_RH/
├── train_mdn.py           # Baseline SpacingMDN
├── train_mdn_memory.py    # SpacingMDN + Memory Bank
├── eval_mdn.py            # Evaluation metrics
├── test_slot_effect.py    # Slot importance testing
├── data/
│   └── continuous_2M/     # Preprocessed zeta zeros
│       ├── train.pt
│       └── val.pt
├── out/
│   ├── mdn_baseline/      # Baseline results
│   ├── mdn_memory_v1/     # v1 results
│   ├── mdn_memory_E1/     # E1 results
│   └── mdn_memory_E2/     # E2 results (current)
└── docs/
    └── PROJECT_GUIDE.md   # This file
```

---

*Last updated: 2026-01-01*
*Current experiment: E2 (in progress)*
*Memory Map Q3: 8 slots mapped to Q3 invariants*
