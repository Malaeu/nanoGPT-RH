#!/usr/bin/env python3
"""Test slot_effect_norm with proper normalization + rollout_effect."""
import torch
import sys
sys.path.insert(0, '/workspace')
from train_mdn_memory import SpacingMDNMemory, slot_effect_norm, format_slot_effects, MDNConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load checkpoint
ckpt_path = "/workspace/out/mdn_memory_v1/best.pt"
print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# Recreate model
config = MDNConfig(**ckpt["config"])
model = SpacingMDNMemory(
    config,
    n_memory_slots=ckpt["n_memory_slots"],
    use_slot_id=ckpt.get("use_slot_id", True),
    use_aux_loss=ckpt.get("use_aux_loss", False)
).to(device)

# Load weights
state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
model.load_state_dict(state_dict)
model.eval()

# Load val data
val_data = torch.load("/workspace/data/continuous_2M/val.pt", weights_only=False)
print(f"Val data: {val_data.shape}")

xb = val_data[:16].to(device)
x_input = xb[:, :-1]
y_target = xb[:, 1:]

# ============================================================================
# TEST A: Normalized effect (effect / baseline_MAE)
# ============================================================================
print("\n" + "="*60)
print("TEST A: Normalized slot_effect_norm")
print("="*60)

# Get baseline MAE (model prediction vs ground truth)
with torch.no_grad():
    pi, mu, sigma, _ = model(x_input)
    pred_mean = (pi * mu).sum(dim=-1)  # (B, T)
    baseline_mae = (pred_mean - y_target).abs().mean().item()

print(f"Baseline MAE (pred vs GT): {baseline_mae:.5f}")

# Get slot effects
eff_l1, eff_l2 = slot_effect_norm(model, x_input, n_slots=8)
max_effect = max(eff_l1)
effect_pct = (max_effect / baseline_mae) * 100

print(f"Max slot effect (L1): {max_effect:.5f}")
print(f"Effect / MAE: {effect_pct:.2f}%")
print(f"\nCriterion: <1% = ballast, >=3% = useful")
print(f"Result: {'BALLAST' if effect_pct < 1 else 'MARGINAL' if effect_pct < 3 else 'USEFUL'}")

# ============================================================================
# TEST B: Rollout effect @10 for top-2 slots
# ============================================================================
print("\n" + "="*60)
print("TEST B: Rollout effect @h=10 (top-2 slots)")
print("="*60)

def rollout_err(model, x, h=10, slot_off=None):
    """Autoregressive rollout for h steps, return mean error."""
    model.eval()
    B, T = x.shape

    # Use middle of sequence as seed
    seed_len = T - h - 10
    current = x[:, :seed_len].clone()
    gt = x[:, seed_len:seed_len+h]

    preds = []
    with torch.no_grad():
        for step in range(h):
            pi, mu, sigma, _ = model(current, slot_off=slot_off)
            # Sample from mixture (use mean for deterministic)
            pred_mean = (pi * mu).sum(dim=-1)[:, -1:]  # last position
            preds.append(pred_mean)
            current = torch.cat([current, pred_mean], dim=1)

    preds = torch.cat(preds, dim=1)  # (B, h)
    err = (preds - gt).abs().mean().item()
    return err

# Base rollout
base_err = rollout_err(model, xb, h=10, slot_off=None)
print(f"Base Err@10: {base_err:.5f}")

# Top-2 slots by effect
sorted_slots = sorted(range(8), key=lambda i: eff_l1[i], reverse=True)
top2 = sorted_slots[:2]
print(f"Top-2 slots: {top2} (effects: {eff_l1[top2[0]]:.5f}, {eff_l1[top2[1]]:.5f})")

for slot in top2:
    err_off = rollout_err(model, xb, h=10, slot_off=slot)
    delta = err_off - base_err
    delta_pct = (delta / base_err) * 100
    print(f"  Slot {slot} OFF: Err@10={err_off:.5f}, Δ={delta:+.5f} ({delta_pct:+.2f}%)")

print(f"\nCriterion: ΔErr@10 <2% = ballast, >=5% = holds focus")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Effect/MAE: {effect_pct:.2f}%")
print(f"Base Err@10: {base_err:.5f}")

# Bar chart
top, chart = format_slot_effects(eff_l1)
print(f"\nSlot effects (L1):")
print(chart)
