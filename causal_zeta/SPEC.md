# CAUSAL ZETA: Multi-Agent Causal Discovery System

## Спецификация v1.0

---

## 1. ОБЗОР СИСТЕМЫ

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CAUSAL ZETA SYSTEM                               │
│                                                                         │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│   │ SpacingGPT  │───▶│   Latent    │───▶│   Causal    │                │
│   │   Model     │    │  Extractor  │    │   States    │                │
│   └─────────────┘    └─────────────┘    └──────┬──────┘                │
│                                                 │                       │
│                                                 ▼                       │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    MULTI-AGENT ORCHESTRATOR                      │  │
│   │                                                                  │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │  │
│   │  │Hypothesis│ │  Tester  │ │Intervent.│ │Validator │           │  │
│   │  │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │           │  │
│   │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │  │
│   │       │            │            │            │                   │  │
│   │       └────────────┴─────┬──────┴────────────┘                   │  │
│   │                          ▼                                       │  │
│   │                   ┌──────────┐                                   │  │
│   │                   │Synthesis │                                   │  │
│   │                   │  Agent   │                                   │  │
│   │                   └────┬─────┘                                   │  │
│   └────────────────────────┼────────────────────────────────────────┘  │
│                            ▼                                           │
│                     ┌─────────────┐                                    │
│                     │   Causal    │                                    │
│                     │    Graph    │                                    │
│                     └─────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. КАУЗАЛЬНЫЙ ГРАФ (v0.1)

### 2.1 Узлы (Variables)

```
┌─────────────────────────────────────────────────────────────────┐
│                      CAUSAL VARIABLES                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐                                                  │
│  │  S_{t-1}  │  Previous spacing (observable)                   │
│  │           │  Source: data/model output                       │
│  │  [scalar] │  Range: [0, 3] after unfolding                   │
│  └───────────┘                                                  │
│                                                                 │
│  ┌───────────┐                                                  │
│  │    Z_t    │  Latent mode (2D phase vector)                   │
│  │           │  Source: PCA(hidden_states[-1])                  │
│  │  [dim=2]  │  Z_t[0] = phase, Z_t[1] = amplitude              │
│  └───────────┘                                                  │
│                                                                 │
│  ┌───────────┐                                                  │
│  │    R_t    │  Rigidity proxy (local variance)                 │
│  │           │  Source: Var(S[t-10:t]) / 0.178                  │
│  │  [scalar] │  R_t ~ 1.0 = GUE, < 1 = rigid, > 1 = loose       │
│  └───────────┘                                                  │
│                                                                 │
│  ┌───────────┐                                                  │
│  │    Y_t    │  Target spacing (next token)                     │
│  │           │  Source: model prediction / data                 │
│  │  [scalar] │  What we want to explain causally                │
│  └───────────┘                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 DAG Structure

```
                    CAUSAL DAG v0.1
    ═══════════════════════════════════════════

              ┌─────────┐
              │ S_{t-1} │ (Previous Spacing)
              └────┬────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 │
     ┌─────────┐            │
     │   Z_t   │            │ (Direct Repulsion)
     │ (Phase) │            │
     └────┬────┘            │
          │                 │
     ┌────┴────┐            │
     │         │            │
     ▼         │            │
┌─────────┐    │            │
│   R_t   │    │            │
│(Rigid.) │    │            │
└────┬────┘    │            │
     │         │            │
     │         ▼            ▼
     │    ┌─────────────────────┐
     └───▶│        Y_t          │
          │  (Target Spacing)   │
          └─────────────────────┘


    EDGES:
    ───────
    S_{t-1} ──▶ Z_t     : Spacing encodes to latent phase
    S_{t-1} ──▶ Y_t     : Direct repulsion (short-range GUE)
    Z_t ──────▶ R_t     : Phase modulates global rigidity
    Z_t ──────▶ Y_t     : Phase mediates long-range correlations
    R_t ──────▶ Y_t     : Rigidity constrains feasible outputs
```

### 2.3 D-Separation Implications

```
┌──────────────────────────────────────────────────────────────┐
│              IMPLIED INDEPENDENCIES                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  If graph is correct:                                        │
│                                                              │
│  1. Y_t ⊥ S_{t-1} | Z_t                                     │
│     "Phase screens off local effect"                         │
│     → Direct repulsion mediated through phase                │
│                                                              │
│  2. R_t ⊥ S_{t-1} | Z_t                                     │
│     "Rigidity is global, not local"                          │
│     → R_t determined by phase, not recent spacing            │
│                                                              │
│  These are TESTABLE with CI tests!                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. MULTI-AGENT ARCHITECTURE

### 3.1 Agent Definitions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           5 AGENTS                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  HYPOTHESIS AGENT                                                │   │
│  │  ════════════════                                                │   │
│  │  Role: Generate causal hypotheses to test                        │   │
│  │  Input: Current graph, past results                              │   │
│  │  Output: List of (X causes Y) statements + test specs            │   │
│  │  Runs: Solo (Round 1)                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  TESTER AGENT                                                    │   │
│  │  ════════════                                                    │   │
│  │  Role: Run HSIC conditional independence tests                   │   │
│  │  Input: Test specifications, causal states data                  │   │
│  │  Output: HSIC values, p-values, interpretation                   │   │
│  │  Runs: Parallel with Intervention Agent                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  INTERVENTION AGENT                                              │   │
│  │  ══════════════════                                              │   │
│  │  Role: Design and specify do-operations                          │   │
│  │  Input: Claims to test, baseline metrics                         │   │
│  │  Output: Intervention specs + expected outcomes                  │   │
│  │  Runs: Parallel with Tester Agent                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  VALIDATOR AGENT                                                 │   │
│  │  ═══════════════                                                 │   │
│  │  Role: Check Q3 constraints on trajectories                      │   │
│  │  Input: Generated trajectories, intervention context             │   │
│  │  Output: Pass/fail for each constraint + interpretation          │   │
│  │  Runs: After interventions                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SYNTHESIS AGENT                                                 │   │
│  │  ═══════════════                                                 │   │
│  │  Role: Combine findings into updated causal model                │   │
│  │  Input: All results from other agents                            │   │
│  │  Output: Graph updates, insights, next experiments               │   │
│  │  Runs: Solo (final step)                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Orchestration Flow

```
                    ORCHESTRATION WORKFLOW
    ════════════════════════════════════════════════════

    ROUND 1: EXPLORE
    ─────────────────
                    ┌──────────────┐
                    │  Hypothesis  │
                    │    Agent     │
                    └──────┬───────┘
                           │
                           ▼
                    [Hypotheses List]


    ROUND 2: TEST (PARALLEL)
    ────────────────────────
            ┌──────────────┐     ┌──────────────┐
            │    Tester    │     │ Intervention │
            │    Agent     │     │    Agent     │
            └──────┬───────┘     └──────┬───────┘
                   │                    │
                   └────────┬───────────┘
                            │
                            ▼
                   [Test Results + Designs]


    ROUND 3: VALIDATE
    ─────────────────
                    ┌──────────────┐
                    │  Validator   │
                    │    Agent     │
                    └──────┬───────┘
                           │
                           ▼
                   [Q3 Validation Results]


    ROUND 4: SYNTHESIZE
    ───────────────────
                    ┌──────────────┐
                    │  Synthesis   │
                    │    Agent     │
                    └──────┬───────┘
                           │
                           ▼
                   [Updated Graph + Insights]


    ROUND 5+: ITERATE (FULL PARALLEL)
    ──────────────────────────────────
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │Hypothesis│ │  Tester  │ │Intervent.│ │Validator │
    │  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         │           │            │            │
         └───────────┴─────┬──────┴────────────┘
                           │
                           ▼
                    ┌──────────┐
                    │Synthesis │
                    └────┬─────┘
                         │
                         ▼
                  [Check Convergence]
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
         [CONVERGED]           [NOT CONVERGED]
              │                     │
              ▼                     │
        Final Report          ◄─────┘
                              (Loop back)
```

---

## 4. DATA FLOW

### 4.1 From Model to Agents

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Odlyzko Zeros (2M)                                                    │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────┐                                                   │
│   │   Unfolding     │  s_n = Δ_n * log(γ_n) / 2π                       │
│   │   (normalize)   │  mean(s) ≈ 1.0                                   │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐                                                   │
│   │   Binning       │  256 bins, range [0, 3]                          │
│   │   (discretize)  │  bin_idx = int(s / 3 * 256)                      │
│   └────────┬────────┘                                                   │
│            │                                                            │
│            ▼                                                            │
│   ┌─────────────────┐         ┌─────────────────┐                      │
│   │   SpacingGPT    │────────▶│  Hidden States  │                      │
│   │   (forward)     │         │  [B, T, 128]    │                      │
│   └────────┬────────┘         └────────┬────────┘                      │
│            │                           │                                │
│            │                           ▼                                │
│            │                  ┌─────────────────┐                      │
│            │                  │   PCA (n=2)     │                      │
│            │                  │   (fit on 10k)  │                      │
│            │                  └────────┬────────┘                      │
│            │                           │                                │
│            ▼                           ▼                                │
│   ┌─────────────────┐         ┌─────────────────┐                      │
│   │    S_t, Y_t     │         │      Z_t        │                      │
│   │   (spacings)    │         │   (latent 2D)   │                      │
│   └────────┬────────┘         └────────┬────────┘                      │
│            │                           │                                │
│            │      ┌────────────────────┘                               │
│            │      │                                                     │
│            ▼      ▼                                                     │
│   ┌─────────────────────────────────────────────┐                      │
│   │              CausalState                     │                      │
│   │  ─────────────────────────────────────────  │                      │
│   │  t: int          (time index)               │                      │
│   │  S_t: float      (current spacing)          │                      │
│   │  Z_t: [2,]       (latent mode)              │                      │
│   │  R_t: float      (rigidity)                 │                      │
│   │  Y_t: float      (target spacing)           │                      │
│   └─────────────────────────────────────────────┘                      │
│                          │                                              │
│                          │ (2000+ samples)                             │
│                          ▼                                              │
│              ┌───────────────────────┐                                 │
│              │   Agent Consumption   │                                 │
│              └───────────────────────┘                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Z_t Computation Detail

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Z_t EXTRACTION PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: idx [B, T] (token indices)                                     │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  SpacingGPT Forward Pass                                         │  │
│   │  ════════════════════════                                        │  │
│   │                                                                  │  │
│   │  idx ──▶ wte ──▶ +wpe ──▶ Block[0] ──▶ Block[1] ──▶ ...         │  │
│   │                                         │                        │  │
│   │                              ──▶ Block[3] ──▶ ln_f ──▶ lm_head   │  │
│   │                                    │                             │  │
│   │                                    ▼                             │  │
│   │                              hidden_states[-1]                   │  │
│   │                              [B, T, 128]                         │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    │ Extract last token                │
│                                    ▼                                    │
│                              h_t [B, 128]                               │
│                                    │                                    │
│                                    │ PCA transform                      │
│                                    ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                      Z_t [B, 2]                                  │  │
│   │  ═════════════════════════════                                   │  │
│   │                                                                  │  │
│   │  Z_t[0] = PC1 projection                                        │  │
│   │           (explains ~40% variance)                               │  │
│   │           Interpretation: "Phase position in oscillation"        │  │
│   │                                                                  │  │
│   │  Z_t[1] = PC2 projection                                        │  │
│   │           (explains ~15% variance)                               │  │
│   │           Interpretation: "Energy/amplitude level"               │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 R_t Computation Detail

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    R_t COMPUTATION                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: spacings [T,]                                                  │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Sliding Window (L=10)                                           │  │
│   │  ══════════════════════                                          │  │
│   │                                                                  │  │
│   │  spacings: [s_0, s_1, s_2, ..., s_t, ...]                       │  │
│   │                    └─────────────┘                               │  │
│   │                     window [t-10:t]                              │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Variance Calculation                                            │  │
│   │  ════════════════════                                            │  │
│   │                                                                  │  │
│   │  var_t = Var(spacings[t-10:t])                                  │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │  Normalization                                                   │  │
│   │  ═════════════                                                   │  │
│   │                                                                  │  │
│   │  R_t = var_t / GUE_VARIANCE                                     │  │
│   │                                                                  │  │
│   │  where GUE_VARIANCE = 0.178 (theoretical for mean=1 GUE)        │  │
│   │                                                                  │  │
│   │  Interpretation:                                                 │  │
│   │  ───────────────                                                 │  │
│   │  R_t < 1.0  →  Super-rigid (crystal-like)                       │  │
│   │  R_t ≈ 1.0  →  Normal GUE behavior                              │  │
│   │  R_t > 1.0  →  Sub-rigid (Poisson-like)                         │  │
│   │                                                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. TESTS & INTERVENTIONS

### 5.1 CI Tests (HSIC)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONDITIONAL INDEPENDENCE TESTS                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TEST 1: Y_t ⊥ S_{t-1} | Z_t                                           │
│  ════════════════════════════                                           │
│  Question: Does Z_t screen off the effect of S_{t-1} on Y_t?           │
│                                                                         │
│  Method:                                                                │
│  1. Residualize Y_t on Z_t: Y_res = Y - KRR(Y|Z)                       │
│  2. Residualize S on Z_t:   S_res = S - KRR(S|Z)                       │
│  3. Compute HSIC(Y_res, S_res)                                         │
│  4. Permutation test for p-value                                        │
│                                                                         │
│  If PASS (p > 0.05): Z_t mediates, consider removing S→Y edge          │
│  If FAIL (p < 0.05): Direct repulsion is real, keep edge               │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  TEST 2: R_t ⊥ S_{t-1} | Z_t                                           │
│  ════════════════════════════                                           │
│  Question: Is rigidity global (via phase) or local?                     │
│                                                                         │
│  If PASS: Rigidity determined by global state, not local spacing       │
│  If FAIL: Need edge S_{t-1} → R_t                                      │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  TEST 3: Y_t ~ Z_t (unconditional)                                     │
│  ═════════════════════════════════                                      │
│  Question: Is Z_t → Y_t edge real?                                      │
│                                                                         │
│  Expected: FAIL (dependent) → edge is valid                            │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  TEST 4: Y_t ~ R_t (unconditional)                                     │
│  ═════════════════════════════════                                      │
│  Question: Is R_t → Y_t edge real?                                      │
│                                                                         │
│  Expected: FAIL (dependent) → edge is valid                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Interventions (do-operations)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTERVENTIONS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INTERVENTION 1: do(S_t := S_t + δ)                                    │
│  ══════════════════════════════════                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Context:  [s_0, s_1, ..., s_9, s_10, s_11, ...]               │   │
│  │                              ▲                                   │   │
│  │                              │                                   │   │
│  │                         INTERVENE                                │   │
│  │                         s_10 += δ                                │   │
│  │                                                                  │   │
│  │  Then generate: [s_10+δ] ──▶ [y_11, y_12, ..., y_60]            │   │
│  │                                                                  │   │
│  │  Measure: |y_t - baseline_t| over time (healing curve)          │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Expected if "rigidity" is real:                                        │
│  • Healing time < 5 steps (system corrects quickly)                    │
│  • Deviation decays exponentially                                       │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  INTERVENTION 2: do(Z_t := Z_t + δ_vec)                                │
│  ══════════════════════════════════════                                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Method: Hook into transformer layer                             │   │
│  │                                                                  │   │
│  │  def perturb_hook(module, input, output):                       │   │
│  │      # Convert δ_vec to hidden space                            │   │
│  │      δ_hidden = PCA.inverse_transform(δ_vec)                    │   │
│  │      output[:, -1, :] += δ_hidden                               │   │
│  │      return output                                               │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Tests: Does perturbing phase affect R_t and Y_t as predicted?         │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  INTERVENTION 3: do(R_t := const)                                      │
│  ════════════════════════════════                                       │
│                                                                         │
│  Method: Constrain output logits during generation                      │
│  • If current_var > target: penalize extreme tokens                    │
│  • If current_var < target: boost extreme tokens                       │
│                                                                         │
│  Tests: Does fixing rigidity collapse Y_t distribution?                │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  INTERVENTION 4: do(H_t := H_t + noise)                                │
│  ══════════════════════════════════════                                 │
│                                                                         │
│  Method: Add Gaussian noise to hidden state                             │
│  • noise_std = 0.01 (small) or 0.05 (large)                            │
│                                                                         │
│  Tests: Is hidden state the "root cause"? How fast does it heal?       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Q3 VALIDATORS

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Q3 CONSTRAINTS                                    │
│                                                                         │
│  These are VALIDATORS not TEACHERS.                                     │
│  We check if generated trajectories are "physically plausible".         │
│  We do NOT train the model to satisfy them.                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  C1: RIGIDITY                                                           │
│  ════════════                                                           │
│  Var(spacings) / 0.178 < 2.0                                           │
│                                                                         │
│  Interpretation: Spectral variance bounded (GUE universality)          │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  C2: REPULSION                                                          │
│  ═══════════                                                            │
│  min(spacing) > 0.01                                                    │
│  fraction(spacing < 0.01) < 1%                                          │
│                                                                         │
│  Interpretation: Level repulsion (no exact degeneracies)               │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  C3: MEAN CONSTRAINT                                                    │
│  ══════════════════                                                     │
│  |mean(spacing) - 1.0| < 0.1                                           │
│                                                                         │
│  Interpretation: Proper unfolding preserved                             │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  C4: GUE DISTRIBUTION                                                   │
│  ════════════════════                                                   │
│  K-S test vs Wigner surmise: p > 0.01                                  │
│                                                                         │
│  Wigner surmise: P(s) = (πs/2) exp(-πs²/4)                             │
│                                                                         │
│  ───────────────────────────────────────────────────────────────────── │
│                                                                         │
│  C5: SPECTRAL FORM FACTOR                                               │
│  ════════════════════════                                               │
│  SFF K(τ) shows:                                                        │
│  • Linear ramp for τ < 1                                               │
│  • Plateau at 1 for τ ≥ 1                                              │
│                                                                         │
│  Interpretation: Long-range spectral correlations correct              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. FILE STRUCTURE

```
causal_zeta/
├── __init__.py          # Package exports
├── SPEC.md              # This specification ← YOU ARE HERE
│
├── variables.py         # Z_t, R_t extraction
│   ├── LatentExtractor  # PCA on hidden states
│   ├── RigidityCalculator
│   └── collect_causal_states()
│
├── graph.py             # Causal DAG
│   ├── CausalGraph      # DAG structure
│   ├── CausalEdge       # Edge with hypothesis
│   └── implied_independencies()
│
├── ci_tests.py          # Independence tests
│   ├── HSICTest         # HSIC computation
│   ├── ConditionalHSIC  # X ⊥ Y | Z tests
│   └── run_ci_tests()
│
├── interventions.py     # do-operations
│   ├── SpacingIntervention
│   ├── LatentIntervention
│   ├── RigidityIntervention
│   ├── HiddenNoiseIntervention
│   └── InterventionSuite
│
├── validators.py        # Q3 constraints
│   ├── Q3Validator      # All constraints
│   └── SFFValidator     # Spectral form factor
│
├── llm_agents.py        # Agent definitions
│   ├── HYPOTHESIS_AGENT
│   ├── TESTER_AGENT
│   ├── INTERVENTION_AGENT
│   ├── VALIDATOR_AGENT
│   ├── SYNTHESIS_AGENT
│   └── format_agent_prompt()
│
├── orchestrator.py      # Multi-agent coordination
│   ├── CausalZetaOrchestrator
│   ├── RoundPlan
│   └── generate_orchestrator_instructions()
│
├── run_mvp.py           # Standalone pipeline
│   └── main()           # Run without agents
│
└── outputs/             # Results directory
    ├── graph.dot        # Final DAG (Graphviz)
    ├── summary.json     # All results
    └── orchestrator_state.json
```

---

## 8. QUICK START

### 8.1 Standalone (no agents)

```bash
cd /Users/emalam/Documents/GitHub/nanoGpt_RH
python -m causal_zeta.run_mvp --checkpoint out/best.pt
```

### 8.2 Multi-Agent (via Claude)

```python
from causal_zeta import CausalZetaOrchestrator

# Initialize
orch = CausalZetaOrchestrator()

# Get prompts for Round 1
plan = orch.plan_round(1)
prompts = orch.get_agent_prompts(plan)
# prompts = [("hypothesis", "...full prompt...")]

# Claude launches agent via Task tool:
# Task(subagent_type='general-purpose', prompt=prompts[0][1])

# After agent returns, integrate:
# orch.integrate_results([result])
# orch.save_state()
```

---

## 9. CONVERGENCE CRITERIA

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONVERGENCE CHECK                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  The system is CONVERGED when:                                          │
│                                                                         │
│  1. ✓ At least 3 rounds completed                                       │
│                                                                         │
│  2. ✓ No graph changes in last synthesis                               │
│       (graph_updates = [])                                              │
│                                                                         │
│  3. ✓ All Q3 validations pass                                          │
│       (overall_valid = true)                                            │
│                                                                         │
│  4. ✓ CI tests stable                                                   │
│       (same results as previous round)                                  │
│                                                                         │
│  When converged:                                                        │
│  → Generate final report                                                │
│  → Export graph as DOT/PNG                                              │
│  → Summarize causal mechanism                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 10. EXPECTED OUTCOMES

### 10.1 If Z_t Mediation Hypothesis is TRUE

```
Expected Results:
─────────────────
• CI Test 1 PASSES: Y_t ⊥ S_{t-1} | Z_t
• Intervention on S_t heals quickly (<5 steps)
• Intervention on Z_t causes permanent shift
• Final graph: S_{t-1} → Z_t → Y_t (no direct S→Y edge)

Interpretation:
───────────────
"The transformer has learned a latent phase representation
that captures the sine-kernel-like structure of GUE spacings.
Local repulsion is MEDIATED through this phase, not direct."
```

### 10.2 If Direct Repulsion Hypothesis is TRUE

```
Expected Results:
─────────────────
• CI Test 1 FAILS: Y_t still depends on S_{t-1} given Z_t
• Both S and Z interventions cause similar effects
• Final graph: S_{t-1} → Y_t (direct edge remains)

Interpretation:
───────────────
"Local repulsion is a DIRECT effect that cannot be fully
captured by the latent phase. The model implements both
short-range (direct) and long-range (phase-mediated) correlations."
```

---

## 11. Q3 ABLATION STUDY: РЕЖИМЫ R0-R3

### 11.1 Философия: Q3 как эталон-валидатор

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Q3 INTEGRATION PHILOSOPHY                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  КЛЮЧЕВАЯ ИДЕЯ:                                                         │
│  ══════════════                                                         │
│  Q3 работает как ЭТАЛОН-ВАЛИДАТОР в нейро-телескопе,                   │
│  а НЕ как "мы доказали RH нейросетью".                                 │
│                                                                         │
│  Q3 в текущей формулировке — это Tier-2 ⟹ RH условно:                 │
│  • Цель в базе: Q(Φ) ≥ 0                                               │
│  • Без подмен нормировок/констант                                       │
│  • Без "RH proven"                                                      │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ДВА РЕЖИМА РАБОТЫ:                                                     │
│  ══════════════════                                                     │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  РЕЖИМ A (без Q3):                                               │   │
│  │  ═════════════════                                               │   │
│  │  • Чистая нейромодель + данные                                   │   │
│  │  • Модель учится сама, без внешних ограничений                   │   │
│  │  • Baseline: "что она выучит без подсказок?"                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  РЕЖИМ B (с Q3):                                                 │   │
│  │  ═══════════════                                                 │   │
│  │  • Та же нейромодель                                             │   │
│  │  • Q3 работает как эталон/аксиома-валидатор                      │   │
│  │  • Сравниваем: меняется ли поведение/качество/контрфакты?        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  АУДИТ-ВОПРОСЫ:                                                         │
│  ═══════════════                                                        │
│                                                                         │
│  1. Совместимы ли данные Odlyzko с "Q3-сценой"?                        │
│     (нет ли статистических аномалий, которые принуждают                │
│      нарушать ключевые ограничения)                                    │
│                                                                         │
│  2. Помогает ли Q3 как валидатор делать генерации                      │
│     устойчивее/физичнее?                                               │
│     (меньше патологий, лучше long-range метрики)                       │
│                                                                         │
│  3. Если Q3 включить как prior, меняется ли извлечённое                │
│     нейро-ядро/каузальная модель предсказуемо?                         │
│     (а не "натренили подтверждение")                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.2 Экспериментальная матрица: 4 режима

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    R0-R3 EXPERIMENT MATRIX                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  R0 — NEURAL ONLY (BASELINE)                                           │
│  ════════════════════════════                                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SpacingGPT                                                      │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  Обучается на unfolded spacings                                  │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  Генерация/контрфакты БЕЗ фильтра                               │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  [Результаты R0]                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  R1 — NEURAL + Q3 REJECTOR                                             │
│  ═════════════════════════════                                          │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SpacingGPT                                                      │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  Генерит K кандидатов (beam/sampling)                           │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  ┌───────────────────────────────────┐                          │   │
│  │  │  Q3-ВАЛИДАТОР (REJECTOR)          │                          │   │
│  │  │  ─────────────────────────────    │                          │   │
│  │  │  • Отбрасывает кандидатов,        │                          │   │
│  │  │    которые ломают инварианты      │                          │   │
│  │  │  • Честные метрики, не "s<0.1"    │                          │   │
│  │  └───────────────────────────────────┘                          │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  [Принятые кандидаты] → [Результаты R1]                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ⚠️  Валидатор должен быть честным:                                    │
│      НЕ "spacing<0.1", а метрики, оправданные базой и данными         │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  R2 — NEURAL + МЯГКИЙ Q3 REGULARIZER                                   │
│  ════════════════════════════════════                                   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SpacingGPT                                                      │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  Loss = L_CE + λ * L_Q3                                          │   │
│  │                  ▲                                               │   │
│  │                  │                                               │   │
│  │  Q3 добавляется В LOSS маленьким весом (λ ~ 0.01)               │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  [Результаты R2]                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ⚠️  САМЫЙ ОПАСНЫЙ РЕЖИМ с точки зрения самообмана!                   │
│      Идёт ТРЕТЬИМ, после того как R1 показал пользу                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  R3 — Q3-SYNTHETIC PRETRAIN → FINE-TUNE                                │
│  ═══════════════════════════════════════                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Phase 1: PRETRAIN                                               │   │
│  │  ─────────────────                                               │   │
│  │  • Генерируем синтетику, которая удовлетворяет валидаторам      │   │
│  │  • Обучаем SpacingGPT на синтетике                              │   │
│  │                                                                  │   │
│  │  Phase 2: FINE-TUNE                                              │   │
│  │  ──────────────────                                              │   │
│  │  • Fine-tune на реальных данных Odlyzko                         │   │
│  │  • Тест: "насколько мало нужно переучиваться?"                  │   │
│  │                                                                  │   │
│  │      │                                                           │   │
│  │      ▼                                                           │   │
│  │  [Результаты R3] + [Transfer Gap]                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.3 Допустимые vs недопустимые валидаторы

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    VALIDATOR POLICY                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ❌ НЕЛЬЗЯ (пока):                                                      │
│  ═════════════════                                                      │
│                                                                         │
│  • Переводить min(P_A) в "минимальный spacing"                         │
│    → Это неверная логика!                                              │
│                                                                         │
│  • Жёстко запрещать малые spacing'и                                    │
│    → В GUE они РЕДКИ, но НЕ нулевые                                    │
│    → P(s→0) ~ s^β где β=2 для GUE (level repulsion)                   │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ✅ МОЖНО (paper-grade):                                                │
│  ════════════════════════                                               │
│                                                                         │
│  1. ИНВАРИАНТЫ БАЗЫ Q3 (чек-лист корректности реализации):             │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  • Знак: Q = Q_arch - Q_prime                                │    │
│     │  • T0-нормировка                                             │    │
│     │  • Period-1 тор                                              │    │
│     │  • Floor: c* = 11/10                                         │    │
│     │  • Prime-cap: t_rkhs ≥ 1 ⟹ ρ(1) < 1/25                      │    │
│     │  (и другие из PROSHKA v3)                                    │    │
│     └─────────────────────────────────────────────────────────────┘    │
│     → Это "валидатор корректности пайплайна", не данных               │
│                                                                         │
│  2. RMT-МЕТРИКИ как наблюдаемая "сцена":                               │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  • Spacing distribution (Wigner surmise)                     │    │
│     │  • Number variance / rigidity proxy                          │    │
│     │  • SFF ramp→plateau (ЛУЧШИЙ long-range тест!)               │    │
│     └─────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  3. КАЛИБРОВКА ПО ODLYZKO:                                             │
│     ┌─────────────────────────────────────────────────────────────┐    │
│     │  Валидатор задаёт НЕ абсолютный порог,                       │    │
│     │  а "не хуже, чем real-data baseline по метрике M"            │    │
│     └─────────────────────────────────────────────────────────────┘    │
│     → Это честно: Q3 как "ожидаемая физика",                          │
│       но границы проверяются на данных                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.4 Метрики сравнения режимов

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    COMPARISON METRICS                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  METRIC           │  R0    │  R1    │  R2    │  R3    │  TARGET   │ │
│  ├───────────────────┼────────┼────────┼────────┼────────┼───────────┤ │
│  │  PPL (val)        │  ?.?   │  ?.?   │  ?.?   │  ?.?   │  min      │ │
│  │  SFF ramp slope   │  ?.?   │  ?.?   │  ?.?   │  ?.?   │  ≈ 1.0    │ │
│  │  SFF plateau      │  ?.?   │  ?.?   │  ?.?   │  ?.?   │  ≈ 1.0    │ │
│  │  Rigidity drift   │  ?.?   │  ?.?   │  ?.?   │  ?.?   │  min      │ │
│  │  Healing time     │  ?.?   │  ?.?   │  ?.?   │  ?.?   │  < 5      │ │
│  │  Rejection rate*  │  N/A   │  ?.?   │  N/A   │  N/A   │  < 10%    │ │
│  │  Transfer gap**   │  N/A   │  N/A   │  N/A   │  ?.?   │  min      │ │
│  └───────────────────┴────────┴────────┴────────┴────────┴───────────┘ │
│                                                                         │
│  *  Rejection rate: % кандидатов отвергнутых Q3-валидатором (только R1)│
│  ** Transfer gap: разница PPL до/после fine-tune (только R3)          │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ:                                               │
│  ═══════════════════════                                                │
│                                                                         │
│  • SFF score = |slope - 1| + |plateau - 1| (чем меньше, тем лучше)    │
│  • K-S statistic vs Wigner surmise                                     │
│  • Counterfactual healing curve (δ → recovery)                         │
│  • CI test stability (те же результаты что в прошлом раунде?)         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.5 Негативный тест (убийственно убедительно)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NEGATIVE TEST                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ИДЕЯ: Намеренно сломать один инвариант Q3 и показать,                 │
│        что валидатор и метрики НЕМЕДЛЕННО деградируют.                 │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ПРИМЕР: Инвертировать знак Q                                          │
│  ═══════════════════════════                                            │
│                                                                         │
│  Правильно:  Q = Q_arch - Q_prime                                      │
│  Сломано:    Q = Q_prime - Q_arch  (знак инвертирован!)               │
│                                                                         │
│  Ожидаемый результат:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  • SFF score ↑ (ухудшается)                                      │   │
│  │  • Rejection rate ↑↑ (много патологий)                          │   │
│  │  • Healing time ↑ (медленнее восстанавливается)                 │   │
│  │  • K-S statistic ↑ (распределение отклоняется от Wigner)        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Если это происходит → Q3 действительно ловит физику,                  │
│                       а не просто "любой регуляризатор помогает"       │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ДРУГИЕ НЕГАТИВНЫЕ ТЕСТЫ:                                              │
│  ════════════════════════                                               │
│                                                                         │
│  • Убрать floor c* = 11/10 → должны появиться "пустоты"               │
│  • Убрать prime-cap ρ(1) < 1/25 → должны быть кластеры               │
│  • Рандомный регуляризатор вместо Q3 → не должен помогать так же      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.6 Diff-отчёт по базе Q3 (PROSHKA v3)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Q3 INVARIANTS CHECKLIST                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Для каждого раунда R0-R3 проверяем:                                   │
│  "Мы НЕ сломали инварианты 1-8 из PROSHKA v3"                          │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │  #  │  INVARIANT              │  R0  │  R1  │  R2  │  R3  │ NOTE │ │
│  ├─────┼─────────────────────────┼──────┼──────┼──────┼──────┼──────┤ │
│  │  1  │  Sign: Q = Q_a - Q_p    │  ✓   │  ✓   │  ✓   │  ✓   │      │ │
│  │  2  │  T0-normalization       │  ✓   │  ✓   │  ✓   │  ✓   │      │ │
│  │  3  │  Period-1 torus         │  ✓   │  ✓   │  ✓   │  ✓   │      │ │
│  │  4  │  Floor c* = 11/10       │  ✓   │  ✓   │  ✓   │  ✓   │      │ │
│  │  5  │  Prime-cap ρ(1) < 1/25  │  ✓   │  ✓   │  ✓   │  ✓   │      │ │
│  │  6  │  t_rkhs ≥ 1 condition   │  ✓   │  ✓   │  ✓   │  ✓   │      │ │
│  │  7  │  Q(Φ) ≥ 0 goal          │  ?   │  ?   │  ?   │  ?   │ MAIN │ │
│  │  8  │  No RH claim            │  ✓   │  ✓   │  ✓   │  ✓   │ META │ │
│  └─────┴─────────────────────────┴──────┴──────┴──────┴──────┴──────┘ │
│                                                                         │
│  Этот чеклист включается в каждый отчёт ablation study.                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 11.7 Команда запуска ablation study

```bash
# Запуск всех 4 режимов с одинаковым seed:
python -m causal_zeta.ablation \
  --checkpoint out/best.pt \
  --data-dir data \
  --report reports/q3_ablation_round_001.md \
  --modes R0 R1 R2 R3 \
  --seed 42

# Только R0 vs R1 (минимальный эксперимент):
python -m causal_zeta.ablation \
  --checkpoint out/best.pt \
  --modes R0 R1 \
  --report reports/r0_vs_r1.md
```

---

## 12. PROSHKA v3 REFERENCE

**📄 Полная спецификация Q3:**
`/Users/emalam/Documents/GitHub/chen_q3/full/q3.lean.aristotle/PROSHKA_REQUEST_3.md`

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Q3 THEORY CONTEXT                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PROSHKA v3 — формальная база Q3 теории.                               │
│                                                                         │
│  КЛЮЧЕВЫЕ ОБЪЕКТЫ:                                                      │
│  ═════════════════                                                      │
│                                                                         │
│  • Q(Φ) = Q_arch(Φ) - Q_prime(Φ)                                       │
│    где Φ — тестовая функция на торе                                    │
│                                                                         │
│  • Q_arch — "архитектурная" часть (геометрия)                          │
│  • Q_prime — "простая" часть (вклад простых чисел)                     │
│                                                                         │
│  ЦЕЛЬ (Tier-2 ⟹ RH условно):                                           │
│  ════════════════════════════                                           │
│                                                                         │
│  Показать Q(Φ) ≥ 0 для всех допустимых Φ                               │
│  → Это ЭКВИВАЛЕНТНО RH (условно, требует аудита)                       │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  В КОНТЕКСТЕ НЕЙРО-ТЕЛЕСКОПА:                                           │
│  ════════════════════════════                                           │
│                                                                         │
│  • SpacingGPT учится на данных (не на Q3)                              │
│  • Q3 служит ВНЕШНИМ валидатором                                       │
│  • Мы НЕ "доказываем RH нейросетью"                                    │
│  • Мы ИЗУЧАЕМ: "согласуются ли данные с Q3-сценой?"                    │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  СВЯЗЬ С RMT:                                                           │
│  ════════════                                                           │
│                                                                         │
│  • GUE (Gaussian Unitary Ensemble) — базовая модель                    │
│  • Sine kernel K(x,y) = sin(π(x-y))/(π(x-y))                           │
│  • SFF ramp→plateau — ключевой тест универсальности                    │
│  • Rigidity = подавление variance (log vs linear growth)               │
│                                                                         │
│  Q3 теория объясняет ПОЧЕМУ зета-нули ведут себя как GUE.              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 12.1 Секции PROSHKA_REQUEST_3.md

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROSHKA v3 SECTIONS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  §0. Статус и цель                                                      │
│      → Цель: Q(Φ) ≥ 0 для всех Φ ∈ W                                   │
│      → Статус: Tier-2 ⟹ RH (условно)                                  │
│                                                                         │
│  §1. Нормировка / знак / тор                                           │
│      → Знак: Q = Q_arch - Q_prime (МИНУС!)                             │
│      → T0-нормировка: a*(ξ) = 2π a(ξ)                                  │
│      → Period-1 тор: T = [-1/2, 1/2]                                   │
│                                                                         │
│  §2. Tier-1 факты                                                       │
│      → Weil criterion: Q ≥ 0 на W ⟺ RH                                │
│      → Szegő-Böttcher: λ_min bounds                                    │
│                                                                         │
│  §3. Tier-2 модули                                                      │
│      → A1': Плотность Fejér×heat-конуса                                │
│      → A2: Lipschitz-контроль                                          │
│      → A3 floor: P_A(θ) ≥ c* = 11/10                                   │
│      → RKHS-cap: ||T_P|| ≤ ρ(1) < 1/25                                 │
│      → Rayleigh-мост                                                    │
│                                                                         │
│  §4. Дискретизация                                                      │
│      → M₀^unif = ⌈C_SB · L* / c*⌉                                      │
│      → M ≥ M₀ ⟹ λ_min(T_M[P_A]) ≥ c*/2                                │
│                                                                         │
│  §5. Позитивность на генераторах                                        │
│      → A3 + Discretisation + RKHS-cap                                  │
│      → λ_min(T_M[P_A] - T_P) ≥ c*/4 > 0                                │
│      → Q(Φ) ≥ 0 ✓                                                      │
│                                                                         │
│  §6-§8. Детали (density, approximation, full proof)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**END OF SPECIFICATION**

*Version: 1.1*
*Date: 2025-12-27*
*Author: Causal Zeta Team*
*Update: Added Q3 Ablation Study (R0-R3) and PROSHKA v3 reference*
