# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
uv pip install torch numpy scipy matplotlib pysr tqdm rich

# Data prep (when scripts exist)
python data/prepare_zeros.py --input zeros_2M.txt --output data/unfolded.pt

# Training
python train.py --config config/spacing_gpt.py

# Evaluation
python eval.py --checkpoint out/model.pt --metrics all

# PySR extraction (attention → symbolic formula)
python extract_kernel.py --checkpoint out/model.pt --output formulas/
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

## File Structure (planned)

```
nanoGpt_RH/
├── data/
│   ├── prepare_zeros.py    # unfolding pipeline
│   └── unfolded.pt         # preprocessed tensors
├── model/
│   ├── gpt.py              # transformer architecture
│   └── config.py           # model configs
├── train.py                # training loop
├── eval.py                 # metrics and diagnostics
├── extract_kernel.py       # PySR symbolic extraction
├── notebooks/
│   └── analysis.ipynb      # visualization
└── config/
    └── spacing_gpt.py      # experiment config
```

## Q3 Integration Points

Cross-reference with Q3 formal structures:
- Attention logits → compare with sine kernel / Toeplitz symbol
- Operator norm cap → check if learned interactions respect bounds
- Prime block structure → compare attention patterns with ρ(t) formulation

## Notes

- 256 bins for spacing classification gives stable perplexity metric
- RoPE / sinusoidal positional encoding justified for sequence + phase structure
- Save hidden states during training for manifold analysis
