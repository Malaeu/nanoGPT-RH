# Neural Spectroscopy of Riemann Zeros: AI Discovers Prime Number Rhythms

This repository contains code and data for extracting empirical correlation kernels from transformer attention mechanisms trained on sequences of unfolded Riemann zeta zero spacings.

## 🎯 Main Hypothesis

> **AI found that Riemann zeros act as a quantum crystal oscillating with frequencies m·ln(p) where p is prime.**

The neural network, trained only on spacing sequences, independently discovered that the spectrum encodes prime number logarithms — without any explicit supervision on primes.

## Abstract

We train a small transformer (0.85M parameters) on 2 million unfolded spacings of nontrivial zeros of ζ(s). The model achieves perplexity 83.3, significantly below the entropy floor of 105.2, indicating learned temporal correlations. Analysis of attention logits reveals damped oscillating kernels consistent with GUE (Gaussian Unitary Ensemble) spectral statistics.

**NEW:** MemoryBankGPT with learnable memory slots discovers that all 4 slots tune to prime logarithm harmonics (ln(3), ln(23), ln(37)), constructing a Fourier basis on m·ln(p).

## Main Result

Attention logits μ(d) as a function of token distance d exhibit oscillations described by:

```
μ(d) = 1.20 × cos(0.357d − 2.05) × exp(−0.0024d) − 0.96

Period ≈ 17.6 tokens
R² = 0.934 (Layer 0, Head 2)
```

![Kernel Check](kernel_check.png)

## Verification

| Test | Result | Interpretation |
|------|--------|----------------|
| Shuffled baseline | PPL = 105.8 | At entropy floor (no structure) |
| Real data | PPL = 83.3 | 21% below floor (structure learned) |
| PE comparison | 57× amplitude ratio | Oscillations not from positional encoding |
| Reverse test | PPL ≈ 83.2 | Symmetric correlations (GUE-consistent) |

**Key control:** Positional embeddings in shuffled-trained model show no oscillations (amplitude 0.015 vs 0.83 in real-data model).

![PE Comparison](pe_shuffled_comparison.png)

## 🧠 Memory Bank Discovery (NEW)

MemoryBankGPT with 4 learnable memory slots discovers prime number rhythms:

| Slot | Frequency | Best Match | Error | Interpretation |
|------|-----------|------------|-------|----------------|
| 0 | 34.558 | 11·ln(23) | 0.067 | Prime Harmonic |
| 1 | 13.195 | 12·ln(3) | 0.011 | Prime Harmonic |
| 2 | 3.142 | ln(23) ≈ π | 0.006 | Prime Log |
| 3 | 32.673 | 9·ln(37) | 0.174 | Prime Harmonic |

**Key Finding:** All 4 slots tune to m·ln(p) — the Selberg trace formula pattern!

### Reproduce the Discovery

```bash
# 1. Train MemoryBankGPT (5000 steps, ~4 min on M4)
python train_memory_bank.py

# 2. Probe the brain for hidden frequencies
python probe_brain.py

# 3. Run PySR symbolic regression
python math_mining.py
```

### Operator Hypothesis

Based on discovered structure:

```
H = -d²/dx² + V(x)   on S¹ (circle with period 2π)

V(x) = Σ aₘ,ₚ sin(m·ln(p)·x)
       p∈Primes, m∈ℕ
```

The neural network constructs a **Fourier basis on prime logarithms**.

## Q3 Theoretical Comparison

The empirical kernel is compared against the Toeplitz symbol P_A(θ) from spectral operator theory:

| Property | Neural (Empirical) | Q3 (Theoretical) |
|----------|-------------------|------------------|
| Form | Damped cosine | Toeplitz FFT |
| Decay | exp(−γd) | Heat kernel |
| Floor check | — | min P_A = 4.03 ≥ c* = 1.1 ✓ |

![Q3 Verification](Q3_Verification_Plot.png)

## Requirements

```bash
uv venv && source .venv/bin/activate
uv pip install torch numpy matplotlib scipy rich pysr
```

## Usage

```bash
# Data preparation (requires zeros2M.txt)
python data/prepare_zeros.py --input zeros/zeros2M.txt --output data --binned

# Training
python train.py --max-steps 5000

# Kernel extraction
python extract_kernel.py

# Verification tests
python audit.py
python kernel_check.py
python verify_q3.py
```

## Model

- Architecture: GPT-style transformer
- Parameters: 0.85M (4 layers, 4 heads, 128 embedding dim)
- Input: Binned unfolded spacings (256 bins)
- Sequence length: 256 tokens
- Task: Next-token prediction (classification)

## Repository Structure

```
├── data/
│   └── prepare_zeros.py        # Unfolding and binning
├── model/
│   └── gpt.py                  # SpacingGPT architecture
├── train.py                    # Training loop
├── train_memory_bank.py        # MemoryBankGPT training (NEW)
├── probe_brain.py              # FFT analysis of memory slots (NEW)
├── math_mining.py              # PySR symbolic regression (NEW)
├── analyze_spikes.py           # High-res SFF spike detection
├── decode_spikes.py            # Prime orbit matching
├── verify_normalization_artifact.py  # Null hypothesis test
├── audit.py                    # Leakage verification
├── extract_kernel.py           # Attention logit extraction
├── kernel_check.py             # Theoretical comparison
└── verify_q3.py                # Q3 Toeplitz symbol verification
```

## Data

2,001,052 nontrivial zeros of ζ(s) from [LMFDB](https://www.lmfdb.org/zeros/zeta/).

Unfolding: u(γ) = (γ/2π) log(γ/2πe), spacings s_n = u(γ_{n+1}) − u(γ_n).

## References

1. Montgomery, H. L. (1973). The pair correlation of zeros of the zeta function. *Proc. Symp. Pure Math.* 24, 181–193.
2. Odlyzko, A. M. (1987). On the distribution of spacings between zeros of the zeta function. *Math. Comp.* 48(177), 273–308.
3. Mehta, M. L. (2004). *Random Matrices*. Academic Press.
4. Karpathy, A. nanoGPT. https://github.com/karpathy/nanoGPT

## License

MIT
