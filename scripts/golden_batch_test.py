#!/usr/bin/env python3
"""
Golden Batch Test - Compare NLL on identical input between checkpoints.

Usage:
    python scripts/golden_batch_test.py --ckpt checkpoints/E4_s7_best.pt
    python scripts/golden_batch_test.py --ckpt checkpoints/E5_s7_best.pt
"""

import argparse
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, 'src')
from train_mdn_postfix import SpacingMDNPostfix, mdn_loss_1step, SpacingNextDataset
from train_mdn import MDNConfig


def load_model(ckpt_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = ckpt.get('config', None)
    if config is None:
        config = MDNConfig()

    # Infer n_memory_slots from state dict
    state_dict = ckpt['model']
    n_slots = 8
    for key in state_dict.keys():
        if 'memory_bank.memory' in key:
            n_slots = state_dict[key].shape[0]
            break

    # Get slot_id_mode and content_mode from checkpoint metadata
    slot_id_mode = ckpt.get('slot_id_mode', 'permute_per_batch')
    content_mode = ckpt.get('content_mode', 'normal')
    experiment = ckpt.get('experiment', 'unknown')

    print(f"Loading {ckpt_path}")
    print(f"  experiment: {experiment}")
    print(f"  n_slots: {n_slots}")
    print(f"  slot_id_mode: {slot_id_mode}")
    print(f"  content_mode: {content_mode}")

    # Build model
    model = SpacingMDNPostfix(
        config=config,
        n_memory_slots=n_slots,
        slot_id_mode=slot_id_mode,
        content_mode=content_mode,
        use_aux_loss=True
    )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, ckpt


def compute_nll_on_loader(model, loader, device, eval_mode='fixed', max_batches=10):
    """Compute mean NLL over data loader."""
    model.eval()

    # Set eval mode
    if hasattr(model, 'memory_bank'):
        model.memory_bank.eval_slot_id_mode = eval_mode

    total_nll = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            pi, mu, sigma = model(x)
            nll = mdn_loss_1step(pi, mu, sigma, y, entropy_reg=0.0)
            total_nll += nll.item()
            n_batches += 1

    return total_nll / n_batches if n_batches > 0 else float('inf')


def main():
    parser = argparse.ArgumentParser(description='Golden Batch Test')
    parser.add_argument('--ckpt', required=True, help='Checkpoint path')
    parser.add_argument('--data-dir', default='data/continuous_2M', help='Data directory')
    parser.add_argument('--seq-len', type=int, default=257, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-batches', type=int, default=10, help='Max batches to evaluate')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"=== Golden Batch Test ===")
    print(f"Device: {args.device}")

    # Load validation data (same way as training)
    val_data = torch.load(f'{args.data_dir}/val.pt')
    print(f"Val data shape: {val_data.shape}")

    # Create dataset and loader (deterministic)
    val_dataset = SpacingNextDataset(val_data, seq_len=args.seq_len)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Val dataset: {len(val_dataset)} samples")

    # Load model
    model, ckpt = load_model(args.ckpt, args.device)

    # Compute NLL with fixed slot IDs (deterministic)
    nll_fixed = compute_nll_on_loader(model, val_loader, args.device,
                                       eval_mode='fixed', max_batches=args.max_batches)

    # Compute NLL with permuted slot IDs
    nll_permuted = compute_nll_on_loader(model, val_loader, args.device,
                                          eval_mode='permute_per_batch', max_batches=args.max_batches)

    print(f"\n{'='*50}")
    print(f"NLL (fixed IDs):    {nll_fixed:.6f}")
    print(f"NLL (permuted IDs): {nll_permuted:.6f}")
    diff = nll_permuted - nll_fixed
    pct = diff / abs(nll_fixed) * 100 if nll_fixed != 0 else 0
    print(f"Difference:         {diff:.6f} ({pct:.2f}%)")
    print(f"{'='*50}")

    return nll_fixed, nll_permuted


if __name__ == '__main__':
    main()
