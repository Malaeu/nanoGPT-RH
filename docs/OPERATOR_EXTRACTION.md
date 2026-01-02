# Operator Extraction via Progressive Distillation

> **Цель:** Извлечь символическую формулу оператора из натренированного трансформера
> **Метод:** Masters-inspired progressive masking + symbolic distillation
> **Источник:** arXiv:2512.22238 "Masking Teacher and Reinforcing Student"

---

## TL;DR

```
Натренированный Transformer ──[mask 80%]──> Minimal Core ──[distill]──> Symbolic Formula
       E4 (NLL=0.19)                         10% весов                    s_n = f(...)
```

---

## Ключевые Идеи из Masters

### 1. Progressive Masking

Маскируем веса учителя по magnitude, постепенно восстанавливая:

```
Masking Schedule: 0.80 → 0.60 → 0.40 → 0.20 → 0.00
                   ↓       ↓       ↓       ↓       ↓
Student learns: simple → medium → complex → full → refinement
```

**Почему это работает:**
- Маленькие веса = шум, не критичны для предсказания
- Большие веса = ядро знаний = структура оператора
- Студент учится сначала простым паттернам, потом сложным

### 2. Dual Rewards

| Reward | Формула | Что измеряет |
|--------|---------|--------------|
| **Accuracy** | `R_acc = NLL(student, data)` | Качество предсказаний |
| **Distillation** | `R_distill = KL(teacher ‖ student)` | Насколько student повторяет teacher |

### 3. Symbolic Distillation

Вместо маленького трансформера — используем символический регрессор:
- Teacher: наш E4 (или masked E4)
- Student: формула с 10-100 параметрами

---

## План Извлечения Оператора

### Phase 1: Progressive Masking Analysis

**Цель:** Найти минимальный % весов, сохраняющий качество

```python
# scripts/masking_analysis.py

import torch
from pathlib import Path

def mask_by_magnitude(model, ratio: float):
    """Mask weights with smallest magnitudes."""
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.abs().flatten())

    all_weights = torch.cat(all_weights)
    threshold = torch.quantile(all_weights, ratio)

    # Create masked model
    masked_model = copy.deepcopy(model)
    for name, param in masked_model.named_parameters():
        if 'weight' in name:
            mask = param.abs() >= threshold
            param.data *= mask.float()

    return masked_model

# Experiment
results = []
for ratio in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
    masked = mask_by_magnitude(E4_model, ratio)
    nll = evaluate(masked, val_data)
    results.append((ratio, nll))
    print(f"Mask {ratio*100:.0f}%: NLL = {nll:.4f}")

# Expected output:
# Mask 0%:  NLL = 0.1942 (baseline)
# Mask 20%: NLL = 0.1950 (almost same!)
# Mask 40%: NLL = 0.2100 (slight degradation)
# Mask 60%: NLL = 0.2500 (noticeable)
# Mask 80%: NLL = 0.3500 (significant)
# Mask 90%: NLL = 0.5000 (broken)

# → Найти точку перегиба = минимальное ядро
```

### Phase 2: Attention Head Ablation

**Цель:** Какие attention heads критичны?

```python
# scripts/attention_ablation.py

def ablate_head(model, layer_idx, head_idx):
    """Zero out specific attention head."""
    model.blocks[layer_idx].attn.heads[head_idx].weight.data.zero_()
    return model

baseline_nll = evaluate(E4_model, val_data)

critical_heads = []
for layer in range(n_layers):
    for head in range(n_heads):
        ablated = ablate_head(copy.deepcopy(model), layer, head)
        delta = evaluate(ablated, val_data) - baseline_nll
        if delta > 0.05:  # >5% degradation
            critical_heads.append((layer, head, delta))
            print(f"CRITICAL: Layer {layer}, Head {head}: Δ={delta:.4f}")

# → Критичные heads содержат структуру оператора!
```

### Phase 3: Symbolic Distillation

**Цель:** Дистиллировать в символическую формулу

```python
# scripts/symbolic_distillation.py

from pysr import PySRRegressor

# Подготовка данных
X = []  # input features (previous spacings)
Y = []  # teacher predictions (MDN mean)

with torch.no_grad():
    for batch in val_loader:
        result = E4_model(batch)
        pred_mean = (result['pi'] * result['mu']).sum(-1)

        # Используем последние K spacings как features
        for i in range(batch.shape[1] - 1):
            features = batch[:, max(0,i-K):i+1]  # context
            target = pred_mean[:, i]
            X.append(features.numpy())
            Y.append(target.numpy())

X = np.concatenate(X)
Y = np.concatenate(Y)

# Symbolic Regression
model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sin", "cos", "exp", "log"],
    complexity_of_operators={"sin": 2, "cos": 2},
    maxsize=30,
    populations=20,
)

model.fit(X, Y)

# Извлекаем формулу!
print("Best formula:", model.sympy())
# → Например: s_n = 1.0 + 0.23*sin(2*pi*n/100) + ...
```

### Phase 4: Validate Extracted Operator

**Цель:** Проверить что формула работает

```python
# scripts/validate_operator.py

from sympy import sympify, lambdify

# Конвертируем формулу в callable
formula = "1.0 + 0.23*sin(2*pi*n/100)"  # пример
f = lambdify('n', sympify(formula))

# Сравниваем с ground truth
predicted = [f(n) for n in range(len(spacings))]
actual = spacings.numpy()

correlation = np.corrcoef(predicted, actual)[0, 1]
mse = np.mean((predicted - actual)**2)

print(f"Correlation: {correlation:.4f}")
print(f"MSE: {mse:.6f}")

# Если correlation > 0.9 → Мы нашли оператор!
```

---

## Архитектура Distillation

```
┌─────────────────────────────────────────────────────────────────┐
│                         TEACHER                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  E4 Transformer (NLL=0.19)                               │   │
│   │  - 6 layers, 8 heads, 256 dim                           │   │
│   │  - 8 memory slots (Q3 structure)                        │   │
│   │  - ~2M parameters                                        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│                    Progressive Masking                           │
│                    (80% → 60% → 40% → 20% → 0%)                 │
│                              ↓                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Masked Teacher (10-20% active weights)                  │   │
│   │  - Still NLL < 0.25                                      │   │
│   │  - Minimal core of knowledge                             │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Knowledge Distillation
                    (KL divergence on logits)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         STUDENT                                  │
│   Option A: Symbolic Regressor (PySR)                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  s_n = 1.0 + Σ a_k * sin(2πn/λ_k + φ_k)                 │   │
│   │  ~10-30 parameters                                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Option B: Tiny MLP                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Linear(K, 32) → ReLU → Linear(32, 1)                   │   │
│   │  ~100 parameters                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Option C: Toeplitz Kernel                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  K(i,j) = f(|i-j|)  — pair correlation function         │   │
│   │  ~50 parameters (first row of Toeplitz matrix)          │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dual Rewards для Нашей Задачи

### Accuracy Reward

```python
def accuracy_reward(student, data):
    """How well student predicts spacings."""
    pred = student(data[:, :-1])
    target = data[:, 1:]
    nll = -torch.distributions.Normal(pred, 0.1).log_prob(target).mean()
    return -nll  # Higher is better
```

### Distillation Reward

```python
def distillation_reward(teacher, student, data):
    """How well student matches teacher's distribution."""
    with torch.no_grad():
        teacher_result = teacher(data)
        teacher_dist = MixtureDensity(
            teacher_result['pi'],
            teacher_result['mu'],
            teacher_result['sigma']
        )

    student_pred = student(data[:, :-1])
    student_dist = Normal(student_pred, 0.1)

    # KL divergence (approximate via sampling)
    samples = teacher_dist.sample((100,))
    kl = teacher_dist.log_prob(samples) - student_dist.log_prob(samples)

    return -kl.mean()  # Lower KL = higher reward
```

### Total Reward

```python
R_total = α * R_accuracy + (1-α) * R_distillation
# α = 0.5 балансирует обе цели
```

---

## Ожидаемые Результаты

### Masking Analysis

| Mask Ratio | Active Params | NLL | Status |
|------------|---------------|-----|--------|
| 0% | 100% | 0.194 | Baseline |
| 20% | 80% | ~0.195 | OK |
| 40% | 60% | ~0.210 | Slight degradation |
| 60% | 40% | ~0.250 | Noticeable |
| **80%** | **20%** | ~0.300 | **Critical point** |
| 90% | 10% | ~0.500 | Broken |

**Инсайт:** ~20% весов содержат 80% знаний (Pareto principle!)

### Symbolic Extraction

Возможные формулы (гипотезы):

```python
# Option 1: Fourier-like
s_n = 1.0 + Σ A_k * sin(2π * n / λ_k)

# Option 2: Pair correlation
s_n = 1.0 + K(s_{n-1} - 1) + K(s_{n-2} - 1) + ...

# Option 3: GUE kernel
K(r) = (sin(πr) / πr)² — sinc² kernel

# Option 4: Memory-based
s_n = 1.0 + Σ w_m * M_m(n)  — weighted memory readout
```

---

## Связь с Q3 Теорией

Если извлечённый оператор имеет форму:

```
K(i,j) = f(|i-j|)  — Toeplitz structure
```

Это подтверждает:
1. **Pair correlations** — главный источник структуры
2. **Translation invariance** — статистика не зависит от позиции
3. **GUE universality** — асимптотика к sinc² kernel

Если находим периодические компоненты:

```
s_n ~ 1 + A*sin(2πn/λ)
```

Это может указывать на:
1. **Дискретные симметрии** в нулях
2. **Периодические орбиты** (связь с динамическими системами)
3. **Modular structure** (связь с L-функциями)

---

## Следующие Шаги

1. [ ] Реализовать `scripts/masking_analysis.py`
2. [ ] Запустить на E4_s7_best.pt
3. [ ] Найти критическую точку masking
4. [ ] Реализовать attention ablation
5. [ ] Установить PySR для символической регрессии
6. [ ] Дистиллировать в формулу
7. [ ] Валидировать на held-out данных
8. [ ] Сравнить с известными GUE формулами

---

## References

- arXiv:2512.22238 — Masters: Masking Teacher and Reinforcing Student
- PySR — https://github.com/MilesCranmer/PySR
- GUE correlations — Montgomery, Odlyzko (1973-1987)
- Symbolic regression — Cranmer et al. (2020)

---

*Last updated: 2026-01-02*
