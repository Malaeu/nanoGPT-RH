#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heatmap_loss_landscape.py — 2D Loss Landscape Visualization for POSTFIX-MDN

Builds a heatmap of val_NLL around a checkpoint to diagnose:
- Is the "lucky minimum" (NLL~0.19) a narrow sharp basin or wide flat valley?
- Where does training trajectory go relative to the landscape?

Usage:
  # Basic 2D heatmap (41x41 grid)
  python src/heatmap_loss_landscape.py \
    --data-dir data/continuous_2M \
    --ckpt checkpoints/E4_s7_best.pt \
    --out-dir out/heatmap_E4_s7 \
    --grid 41 --eps 0.5

  # With trajectory overlay (multiple checkpoints)
  python src/heatmap_loss_landscape.py \
    --data-dir data/continuous_2M \
    --ckpt checkpoints/E4_s7_best.pt \
    --out-dir out/heatmap_E4_s7 \
    --traj "out/E4_s7/ckpt_5000.pt,out/E4_s7/ckpt_7500.pt,out/E4_s7/ckpt_10000.pt"

  # Quick 1D sharpness scan (21 points)
  python src/heatmap_loss_landscape.py \
    --data-dir data/continuous_2M \
    --ckpt checkpoints/E4_s7_best.pt \
    --out-dir out/heatmap_E4_s7 \
    --mode 1d --grid 21

Output:
  - heatmap.csv: raw NLL values on grid
  - heatmap.png: visualization with optional trajectory
  - sharpness.json: metrics about basin width/depth
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for RunPod
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from train_mdn import MDNConfig, mdn_loss
from train_mdn_postfix import SpacingMDNPostfix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Loss landscape heatmap for POSTFIX-MDN")

    # Required
    p.add_argument('--data-dir', type=str, required=True, help='Path to data dir with val.pt')
    p.add_argument('--ckpt', type=str, required=True, help='Checkpoint to analyze')
    p.add_argument('--out-dir', type=str, required=True, help='Output directory')

    # Grid settings
    p.add_argument('--mode', type=str, default='2d', choices=['1d', '2d'],
                   help='1d=sharpness scan, 2d=full heatmap')
    p.add_argument('--grid', type=int, default=41, help='Grid size (41 -> 41x41 for 2d)')
    p.add_argument('--eps', type=float, default=0.5,
                   help='Radius in normalized direction units')

    # Eval settings
    p.add_argument('--val-subset', type=int, default=4096,
                   help='Number of val samples to use (fixed subset for stability)')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--no-amp', action='store_true', help='Disable AMP for stable eval')

    # Trajectory overlay
    p.add_argument('--traj', type=str, default='',
                   help='Comma-separated checkpoint paths for trajectory overlay')

    # Misc
    p.add_argument('--seed', type=int, default=42, help='Seed for directions and subset')

    return p.parse_args()


def load_val_data(data_dir: str, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load fixed validation subset."""
    val_path = Path(data_dir) / 'val.pt'
    data = torch.load(val_path, map_location='cpu')

    if isinstance(data, dict) and 'val' in data:
        data = data['val']

    # Random subset for stability
    g = torch.Generator().manual_seed(seed)
    n = min(n_samples, data.shape[0])
    idx = torch.randperm(data.shape[0], generator=g)[:n]
    subset = data[idx]

    # x = all but last, y = last token
    x = subset[:, :-1].float()
    y = subset[:, -1].float()

    return x, y


def load_model_from_ckpt(ckpt_path: str) -> Tuple[SpacingMDNPostfix, Dict]:
    """Load model from checkpoint with all config."""
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Extract config and params
    config = ckpt['config']
    n_memory_slots = ckpt.get('n_memory_slots', 8)
    slot_id_mode = ckpt.get('slot_id_mode', 'fixed')
    content_mode = ckpt.get('content_mode', 'normal')
    use_aux_loss = ckpt.get('use_aux_loss', False)

    # Create model
    model = SpacingMDNPostfix(
        config,
        n_memory_slots=n_memory_slots,
        slot_id_mode=slot_id_mode,
        content_mode=content_mode,
        use_aux_loss=use_aux_loss
    )

    # Load weights
    model.load_state_dict(ckpt['model'], strict=True)

    # FIX: Force fixed eval mode to avoid permutation noise in heatmap!
    # Without this, each grid point gets different random permutation = noise
    model.memory_bank.eval_slot_id_mode = 'fixed'
    model.memory_bank.set_eval_perm_seed(42)  # Reproducible

    return model, ckpt


def state_dict_to_vec(sd: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size]]]:
    """Flatten state dict to single vector, return metadata for reconstruction."""
    parts = []
    meta = []
    for k, v in sd.items():
        if torch.is_floating_point(v):
            parts.append(v.flatten())
            meta.append((k, v.shape))
    return torch.cat(parts), meta


def vec_to_state_dict(vec: torch.Tensor, meta: List[Tuple[str, torch.Size]],
                      ref_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Reconstruct state dict from vector using metadata."""
    out = {}
    offset = 0
    for k, shape in meta:
        n = int(np.prod(shape))
        out[k] = vec[offset:offset+n].view(shape)
        offset += n

    # Keep non-float tensors unchanged
    for k, v in ref_sd.items():
        if k not in out:
            out[k] = v.clone()

    return out


def make_random_direction(sd: Dict[str, torch.Tensor], meta: List[Tuple[str, torch.Size]],
                          seed: int) -> torch.Tensor:
    """Create layer-wise normalized random direction (always on CPU)."""
    g = torch.Generator().manual_seed(seed)
    parts = []

    for k, shape in meta:
        v = sd[k].cpu()  # Ensure CPU
        # Random direction
        r = torch.randn(shape, generator=g)
        # Layer-wise normalization: scale to match param norm
        v_norm = torch.norm(v).clamp(min=1e-12)
        r_norm = torch.norm(r).clamp(min=1e-12)
        r = r * (v_norm / r_norm)
        parts.append(r.flatten())

    dir_vec = torch.cat(parts)
    # Global normalization to unit length
    return dir_vec / torch.norm(dir_vec).clamp(min=1e-12)


@torch.no_grad()
def eval_nll(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
             batch_size: int, device: str, use_amp: bool) -> float:
    """Evaluate NLL on data."""
    model.eval()
    x, y = x.to(device), y.to(device)

    total_nll = 0.0
    n_samples = 0

    for i in range(0, len(x), batch_size):
        xb = x[i:i+batch_size]
        yb = y[i:i+batch_size]

        with torch.amp.autocast('cuda', enabled=use_amp):
            # Model returns (pi, mu, sigma) tuple
            pi, mu, sigma = model(xb)
            # mdn_loss returns (loss, nll, entropy) - we want nll
            _, nll, _ = mdn_loss(pi, mu, sigma, yb)

        total_nll += nll.item() * len(xb)
        n_samples += len(xb)

    return total_nll / max(n_samples, 1)


def compute_heatmap_2d(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                       vec0: torch.Tensor, meta: List, ref_sd: Dict,
                       u: torch.Tensor, v: torch.Tensor,
                       grid: int, eps: float, batch_size: int,
                       device: str, use_amp: bool) -> np.ndarray:
    """Compute 2D loss landscape heatmap."""
    alphas = np.linspace(-eps, eps, grid, dtype=np.float32)
    betas = np.linspace(-eps, eps, grid, dtype=np.float32)

    Z = np.zeros((grid, grid), dtype=np.float32)
    total = grid * grid

    with Progress(
        TextColumn("[cyan]Heatmap"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Computing", total=total)

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                # Perturb weights
                vec = vec0 + float(a) * u + float(b) * v
                sd = vec_to_state_dict(vec, meta, ref_sd)
                model.load_state_dict(sd, strict=True)
                model.to(device)

                # Evaluate
                nll = eval_nll(model, x, y, batch_size, device, use_amp)
                Z[i, j] = nll

                progress.advance(task)

    return Z


def compute_sharpness_1d(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                         vec0: torch.Tensor, meta: List, ref_sd: Dict,
                         u: torch.Tensor, grid: int, eps: float,
                         batch_size: int, device: str, use_amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1D sharpness scan along direction u."""
    alphas = np.linspace(-eps, eps, grid, dtype=np.float32)
    nlls = np.zeros(grid, dtype=np.float32)

    with Progress(
        TextColumn("[cyan]1D Scan"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("Computing", total=grid)

        for i, a in enumerate(alphas):
            vec = vec0 + float(a) * u
            sd = vec_to_state_dict(vec, meta, ref_sd)
            model.load_state_dict(sd, strict=True)
            model.to(device)

            nlls[i] = eval_nll(model, x, y, batch_size, device, use_amp)
            progress.advance(task)

    return alphas, nlls


def compute_trajectory_coords(traj_paths: List[str], vec0: torch.Tensor,
                               u: torch.Tensor, v: torch.Tensor) -> List[Tuple[float, float, float]]:
    """Project trajectory checkpoints onto (u, v) plane. Returns (beta, alpha, nll) tuples."""
    pts = []
    for path in traj_paths:
        if not Path(path).exists():
            console.print(f"[yellow]Warning: {path} not found, skipping")
            continue

        ckpt = torch.load(path, map_location='cpu')
        sd = ckpt['model']
        vec, _ = state_dict_to_vec(sd)

        # Delta from base
        d = vec - vec0

        # Project onto directions
        alpha = torch.dot(d, u).item()
        beta = torch.dot(d, v).item()
        nll = ckpt.get('val_nll', 0.0)

        pts.append((beta, alpha, nll))

    return pts


def save_results(out_dir: Path, Z: Optional[np.ndarray], alphas: np.ndarray,
                 betas: Optional[np.ndarray], base_nll: float,
                 traj_pts: List[Tuple[float, float, float]], mode: str):
    """Save CSV, PNG, and sharpness metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == '2d':
        # Save CSV
        csv_path = out_dir / 'heatmap.csv'
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['alpha\\beta'] + [f'{b:.4f}' for b in betas])
            for i, a in enumerate(alphas):
                w.writerow([f'{a:.4f}'] + [f'{Z[i,j]:.6f}' for j in range(len(betas))])

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Use log scale if range is large
        vmin, vmax = Z.min(), Z.max()
        if vmax / max(vmin, 0.01) > 10:
            im = ax.imshow(Z, origin='lower', extent=[betas[0], betas[-1], alphas[0], alphas[-1]],
                          aspect='auto', cmap='viridis', norm=LogNorm(vmin=max(vmin, 0.01), vmax=vmax))
        else:
            im = ax.imshow(Z, origin='lower', extent=[betas[0], betas[-1], alphas[0], alphas[-1]],
                          aspect='auto', cmap='viridis')

        plt.colorbar(im, ax=ax, label='val_NLL')

        # Mark center (base checkpoint)
        ax.scatter([0], [0], c='red', s=100, marker='*', label=f'Base (NLL={base_nll:.4f})')

        # Plot trajectory if available
        if traj_pts:
            xs, ys, _ = zip(*traj_pts)
            ax.plot(xs, ys, 'w-o', markersize=5, linewidth=2, label='Trajectory')

        ax.set_xlabel('beta (direction 2)')
        ax.set_ylabel('alpha (direction 1)')
        ax.set_title(f'Loss Landscape (base NLL={base_nll:.4f})')
        ax.legend()

        plt.tight_layout()
        plt.savefig(out_dir / 'heatmap.png', dpi=180)
        plt.close()

        # Sharpness metrics
        center_idx = len(alphas) // 2
        eps_small = abs(alphas[1] - alphas[0])  # One grid step

        # Local sharpness: max NLL increase within 1 grid step
        local_patch = Z[max(0, center_idx-1):center_idx+2, max(0, center_idx-1):center_idx+2]
        local_sharpness = float(local_patch.max() - base_nll)

        # Basin width: how far can we go before NLL doubles?
        nll_threshold = base_nll * 2
        basin_width = 0.0
        for r in range(1, center_idx + 1):
            ring = []
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    if abs(di) == r or abs(dj) == r:
                        i, j = center_idx + di, center_idx + dj
                        if 0 <= i < len(alphas) and 0 <= j < len(betas):
                            ring.append(Z[i, j])
            if ring and max(ring) > nll_threshold:
                basin_width = float(r * eps_small)
                break
        else:
            basin_width = float(alphas[-1])  # Didn't exceed threshold

        metrics = {
            'base_nll': base_nll,
            'min_nll': float(Z.min()),
            'max_nll': float(Z.max()),
            'local_sharpness': local_sharpness,
            'basin_width': basin_width,
            'grid_size': len(alphas),
            'eps': float(alphas[-1]),
        }

    else:  # 1d mode
        # For 1D, alphas is x, Z is y (NLL values stored in "betas" for reuse)
        nlls = betas  # Hack: we stored 1D results in betas slot

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(alphas, nlls, 'b-o', markersize=4)
        ax.axhline(base_nll, color='r', linestyle='--', label=f'Base NLL={base_nll:.4f}')
        ax.axvline(0, color='gray', linestyle=':')

        ax.set_xlabel('alpha (perturbation)')
        ax.set_ylabel('val_NLL')
        ax.set_title('1D Sharpness Scan')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'sharpness_1d.png', dpi=180)
        plt.close()

        # Save CSV
        with open(out_dir / 'sharpness_1d.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['alpha', 'nll'])
            for a, nll in zip(alphas, nlls):
                w.writerow([f'{a:.6f}', f'{nll:.6f}'])

        # Metrics
        center_idx = len(alphas) // 2
        local_sharpness = float(max(nlls[center_idx-1:center_idx+2]) - base_nll)

        metrics = {
            'base_nll': base_nll,
            'min_nll': float(nlls.min()),
            'max_nll': float(nlls.max()),
            'local_sharpness': local_sharpness,
            'eps': float(alphas[-1]),
        }

    # Save metrics
    with open(out_dir / 'sharpness.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = not args.no_amp and device == 'cuda'

    console.print(f"[bold cyan]Loss Landscape Analysis[/]")
    console.print(f"  Checkpoint: {args.ckpt}")
    console.print(f"  Mode: {args.mode}, Grid: {args.grid}, Eps: {args.eps}")
    console.print(f"  Device: {device}, AMP: {use_amp}")

    # Load data
    console.print("\n[dim]Loading validation data...[/]")
    x, y = load_val_data(args.data_dir, args.val_subset, args.seed)
    console.print(f"  Val subset: {len(x)} samples")

    # Load model
    console.print("[dim]Loading checkpoint...[/]")
    model, ckpt = load_model_from_ckpt(args.ckpt)
    model.to(device)

    # Base NLL
    base_nll = eval_nll(model, x, y, args.batch_size, device, use_amp)
    console.print(f"  Base NLL: {base_nll:.6f}")
    console.print(f"  Checkpoint NLL: {ckpt.get('val_nll', 'N/A')}")

    # Vectorize params (keep on CPU for direction math)
    ref_sd = {k: v.cpu() for k, v in model.state_dict().items()}
    vec0, meta = state_dict_to_vec(ref_sd)
    console.print(f"  Parameters: {len(vec0):,}")

    # Create random directions (on CPU)
    u = make_random_direction(ref_sd, meta, args.seed + 11)
    v = make_random_direction(ref_sd, meta, args.seed + 23)

    # Compute landscape
    out_dir = Path(args.out_dir)

    if args.mode == '2d':
        console.print(f"\n[bold]Computing 2D heatmap ({args.grid}x{args.grid})...[/]")
        Z = compute_heatmap_2d(model, x, y, vec0, meta, ref_sd, u, v,
                               args.grid, args.eps, args.batch_size, device, use_amp)
        alphas = np.linspace(-args.eps, args.eps, args.grid)
        betas = np.linspace(-args.eps, args.eps, args.grid)

        # Trajectory overlay
        traj_pts = []
        if args.traj.strip():
            traj_paths = [p.strip() for p in args.traj.split(',') if p.strip()]
            console.print(f"[dim]Computing trajectory ({len(traj_paths)} points)...[/]")
            traj_pts = compute_trajectory_coords(traj_paths, vec0, u, v)

        metrics = save_results(out_dir, Z, alphas, betas, base_nll, traj_pts, '2d')

    else:  # 1d
        console.print(f"\n[bold]Computing 1D sharpness scan ({args.grid} points)...[/]")
        alphas, nlls = compute_sharpness_1d(model, x, y, vec0, meta, ref_sd, u,
                                            args.grid, args.eps, args.batch_size, device, use_amp)

        metrics = save_results(out_dir, None, alphas, nlls, base_nll, [], '1d')

    # Report
    console.print(f"\n[bold green]Done![/]")
    console.print(f"  Output: {out_dir}")
    console.print(f"  Base NLL: {base_nll:.4f}")
    console.print(f"  Local sharpness: {metrics['local_sharpness']:.4f}")
    if 'basin_width' in metrics:
        console.print(f"  Basin width: {metrics['basin_width']:.4f}")

    # Restore original weights
    model.load_state_dict(ref_sd, strict=True)


if __name__ == '__main__':
    main()
