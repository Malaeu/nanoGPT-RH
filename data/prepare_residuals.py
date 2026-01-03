#!/usr/bin/env python3
"""
Prepare Residual Data for training.
Instead of predicting spacing s_n, we predict r_n = s_n - 1.0.
This centers the data around zero and focuses the model on fluctuations (GUE chaos).
"""

import torch
from pathlib import Path
from rich.console import Console

console = Console()

def transform_to_residuals(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Transforming data from {input_path} to residuals...[/]")

    for split in ["train.pt", "val.pt"]:
        if not (input_path / split).exists():
            continue
            
        data = torch.load(input_path / split, weights_only=False)
        # Residuals: r = s - 1.0
        # This makes the "Physicist's Anchor" implicitly 1.0
        residuals = data - 1.0
        
        torch.save(residuals, output_path / split)
        console.print(f"[green]Saved {split}: {residuals.shape} (mean={residuals.mean():.6f}, std={residuals.std():.6f})[/]")

    # Copy meta if exists
    if (input_path / "meta.pt").exists():
        meta = torch.load(input_path / "meta.pt", weights_only=False)
        meta['is_residual'] = True
        torch.save(meta, output_path / "meta.pt")

if __name__ == "__main__":
    transform_to_residuals("data/continuous_clean", "data/continuous_residuals")
