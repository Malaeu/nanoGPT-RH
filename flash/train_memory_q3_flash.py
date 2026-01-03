import argparse
import math
import time
import sys
from pathlib import Path
import torch
from memory_mdn_flash import MemoryMDN, MemoryMDNConfig

# Defaults
DEFAULT_DATA_DIR = Path("data/continuous_residuals")
DEFAULT_OUTPUT_DIR = Path("out/mdn_memory_q3_flash")

def load_data_to_gpu(data_dir: Path, seq_len: int, device: torch.device):
    """Load and reshape data directly to GPU memory."""
    print(f"[*] Loading data to GPU ({device})...")
    train_data = torch.load(data_dir / "train.pt", weights_only=False).to(device)
    val_data = torch.load(data_dir / "val.pt", weights_only=False).to(device)
    
    # Flatten and reshape to target seq_len
    # Assuming data is (N, L_old), we flatten to (N*L_old) then view as (N_new, seq_len)
    def reshape(data, L):
        flat = data.view(-1)
        n_elements = flat.numel()
        n_seq = n_elements // L
        # Crop excess
        flat = flat[:n_seq * L]
        return flat.view(n_seq, L)

    train_data = reshape(train_data, seq_len)
    val_data = reshape(val_data, seq_len)
    
    print(f"[+] GPU Data Ready. Train: {train_data.shape}, Val: {val_data.shape}")
    return train_data, val_data

@torch.no_grad()
def evaluate(model, data, batch_size, device):
    model.eval()
    total_nll = 0.0
    n_batches = 0
    for i in range(0, data.shape[0], batch_size):
        x = data[i:i+batch_size]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = model(x, targets=x)
        total_nll += result['nll'].item()
        n_batches += 1
    model.train()
    return total_nll / n_batches if n_batches > 0 else 0

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def train(args):
    print("\n" + "="*60)
    print(" FLASH ULTRA-TURBO: GPU-ONLY TRAINING ")
    print("="*60)

    device = torch.device("cuda")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 🚀 DATA DIRECTLY ON GPU
    train_data, val_data = load_data_to_gpu(Path(args.data_dir), args.seq_len, device)

    config = MemoryMDNConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        seq_len=args.seq_len, dropout=args.dropout, n_components=args.n_components,
        n_memory=args.n_memory, memory_dropout=args.memory_dropout,
        memory_cap=args.memory_cap, diversity_weight=args.diversity_weight,
        memory_lr_mult=args.memory_lr_mult,
    )

    model = MemoryMDN(config).to(device)
    
    # 🚀 PURE EAGER MODE (No Compile) - Fastest and most stable for this setup
    compiled_model = model 
    print("[+] Model Ready (Eager Mode).")

    # 🚀 CUDA OPTIMIZATION
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    param_groups = model.get_param_groups(base_lr=args.lr)
    # 🚀 OPTIMIZATION: Fused AdamW
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay, fused=True)

    start_step = 0
    best_val_nll = float('inf')
    if args.resume and Path(args.resume).exists():
        print(f"[*] Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        best_val_nll = ckpt.get('val_nll', float('inf'))

    print(f"[*] Ultra Start! Context: {args.seq_len} | Batch Size: {args.batch_size}")
    print("-" * 110)
    
    t_start = time.time()
    t_window_start = t_start
    window_size = 10
    dt_prev_avg = 0.0
    step = start_step

    model.train()
    while step < args.max_steps:
        # 🚀 GPU-GPU BATCHING (No DataLoader!)
        idx = torch.randint(0, train_data.shape[0], (args.batch_size,), device=device)
        x = train_data[idx]

        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr * config.memory_lr_mult if pg.get('is_memory') else lr

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = compiled_model(x, targets=x)
            loss = result['loss']
            nll = result['nll']
            pi, mu = result['pi'], result['mu']
            pred_mean = torch.sum(pi * mu, dim=-1)
            mae = torch.abs(pred_mean[:, :-1] - x[:, 1:]).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1
        
        if step % window_size == 0:
            t_now = time.time()
            dt_avg = ((t_now - t_window_start) / window_size) * 1000
            t_window_start = t_now
            diff_str = f"{dt_avg - dt_prev_avg:+.1f}ms" if dt_prev_avg > 0 else "0.0ms"
            dt_prev_avg = dt_avg
            sps = 1000.0 / dt_avg
            eta = (args.max_steps - step) / sps if sps > 0 else 0
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
            
            status = (
                f"\rStep [{step:5d}/{args.max_steps}] | "
                f"L: {loss.item():.3f} | NLL: {nll.item():.3f} | MAE: {mae.item():.3f} | "
                f"{int(sps)} st/s | dt: {int(dt_avg)}ms ({diff_str}) | "
                f"ETA: {eta_str} "
            )
            sys.stdout.write(status)
            sys.stdout.flush()

        if step % args.eval_interval == 0:
            print(f"\n" + "-"*40)
            val_nll = evaluate(compiled_model, val_data, args.batch_size, device)
            marker = " [NEW BEST]" if val_nll < best_val_nll else ""
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                raw_model = model.module if hasattr(model, 'module') else model
                torch.save({'model': raw_model.state_dict(), 'config': config, 'step': step, 'val_nll': val_nll}, output_dir / 'best.pt')
            print(f"| EVAL | Step {step} | Val NLL: {val_nll:.4f}{marker}")
            print("-" * 40)

        if step % args.save_interval == 0:
            raw_model = model.module if hasattr(model, 'module') else model
            torch.save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'step': step}, output_dir / f'ckpt_{step}.pt')

    print(f"\n[+] Training Complete. Best Val NLL: {best_val_nll:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256) 
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--seq-len", type=int, default=256) # Stable Baseline
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-memory", type=int, default=8)
    parser.add_argument("--memory-dropout", type=float, default=0.1)
    parser.add_argument("--memory-cap", type=float, default=2.0)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--memory-lr-mult", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--n-memory", type=int, default=8)
    parser.add_argument("--memory-dropout", type=float, default=0.1)
    parser.add_argument("--memory-cap", type=float, default=2.0)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--memory-lr-mult", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()