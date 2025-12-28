#!/usr/bin/env python3
"""
HERMITICITY TEST: Does the neural network learn Hermitian structure?

Hilbert-Pólya Conjecture: Riemann zeros are eigenvalues of a Hermitian operator.
If the network learns to model this data, its weight matrices might become symmetric.

Test:
1. Extract W_q, W_k from memory bank attention
2. Compute H_eff = W_q^T @ W_k (effective Hamiltonian)
3. Check symmetry: ||H - H^T|| / ||H + H^T||
4. Check eigenvalues: are they real?
5. Check memory slot orthogonality
"""

import torch
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sys

console = Console()

sys.path.insert(0, '.')
from train_memory_bank import MemoryBankGPT, MemoryBankConfig


def check_hermiticity():
    console.print(Panel.fit(
        "[bold magenta]⚛️ HERMITICITY TEST[/]\n"
        "Testing if neural network learned Hermitian structure",
        title="HILBERT-PÓLYA CHECK"
    ))

    # 1. Load Model
    try:
        ckpt = torch.load('out/memory_bank_best.pt', map_location='cpu', weights_only=False)
        config = ckpt['config']
        model = MemoryBankGPT(config)
        model.load_state_dict(ckpt['model'])
        console.print("[green]Model loaded.[/]")
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/]")
        return

    # 2. Extract Interaction Matrices
    W_q = model.memory_bank.query_proj.weight.detach()  # [dim, dim]
    W_k = model.memory_bank.key_proj.weight.detach()    # [dim, dim]

    console.print(f"[dim]W_q shape: {W_q.shape}, W_k shape: {W_k.shape}[/]")

    # Construct Effective Hamiltonian: H = W_q^T @ W_k
    H_eff = torch.matmul(W_q.T, W_k)
    console.print(f"[bold]Effective Hamiltonian H_eff: {H_eff.shape}[/]")

    # 3. Symmetry Check (Hermiticity)
    H_T = H_eff.T
    diff_norm = torch.norm(H_eff - H_T).item()
    sum_norm = torch.norm(H_eff + H_T).item()
    symmetry_score = diff_norm / sum_norm if sum_norm > 0 else 1.0

    console.print(f"\n[bold yellow]🔍 SYMMETRY SCORE[/] (0=Hermitian, 1=Random)")
    console.print(f"||H - H^T|| = {diff_norm:.4f}")
    console.print(f"||H + H^T|| = {sum_norm:.4f}")
    console.print(f"Score = {symmetry_score:.4f}")

    if symmetry_score < 0.1:
        sym_verdict = "[bold green]HERMITIAN![/]"
    elif symmetry_score < 0.3:
        sym_verdict = "[yellow]Quasi-Hermitian[/]"
    else:
        sym_verdict = "[red]Non-Hermitian[/]"

    console.print(f"Verdict: {sym_verdict}")

    # 4. Spectral Check (Eigenvalues)
    eigvals = torch.linalg.eigvals(H_eff)
    real_parts = eigvals.real
    imag_parts = eigvals.imag

    max_imag = torch.max(torch.abs(imag_parts)).item()
    avg_real = torch.mean(torch.abs(real_parts)).item()
    ratio = max_imag / avg_real if avg_real > 0 else float('inf')

    console.print(f"\n[bold cyan]📉 EIGENVALUE ANALYSIS[/]")
    console.print(f"Max |Im(λ)|: {max_imag:.6f}")
    console.print(f"Avg |Re(λ)|: {avg_real:.6f}")
    console.print(f"Im/Re Ratio: {ratio:.4f}")

    if ratio < 0.1:
        eig_verdict = "[bold green]REAL SPECTRUM![/]"
    elif ratio < 0.5:
        eig_verdict = "[yellow]Mostly Real[/]"
    else:
        eig_verdict = "[red]Complex Spectrum[/]"

    console.print(f"Verdict: {eig_verdict}")

    # 5. Memory Slot Orthogonality
    mem_vectors = model.memory_bank.memory.detach()  # [4, 128]
    mem_norm = torch.nn.functional.normalize(mem_vectors, p=2, dim=1)
    similarity = torch.matmul(mem_norm, mem_norm.T)

    console.print(f"\n[bold]Memory Slot Similarity Matrix:[/]")
    for i in range(4):
        row = " ".join([f"{similarity[i,j].item():+.3f}" for j in range(4)])
        console.print(f"  [{row}]")

    off_diag = similarity - torch.eye(4)
    max_corr = torch.max(torch.abs(off_diag)).item()

    console.print(f"\nMax off-diagonal: {max_corr:.4f}")
    if max_corr < 0.2:
        orth_verdict = "[green]Orthogonal[/]"
    else:
        orth_verdict = "[yellow]Correlated[/]"

    console.print(f"Verdict: {orth_verdict}")

    # 6. NULL HYPOTHESIS: Random matrices
    console.print(f"\n[bold]Null Hypothesis (Random Init):[/]")

    random_scores = []
    for _ in range(100):
        W_q_rand = torch.randn_like(W_q) * 0.02
        W_k_rand = torch.randn_like(W_k) * 0.02
        H_rand = torch.matmul(W_q_rand.T, W_k_rand)
        H_rand_T = H_rand.T
        diff = torch.norm(H_rand - H_rand_T).item()
        summ = torch.norm(H_rand + H_rand_T).item()
        random_scores.append(diff / summ if summ > 0 else 1.0)

    random_mean = np.mean(random_scores)
    random_std = np.std(random_scores)
    z_score = (symmetry_score - random_mean) / random_std

    console.print(f"Random symmetry score: {random_mean:.4f} ± {random_std:.4f}")
    console.print(f"Trained symmetry score: {symmetry_score:.4f}")
    console.print(f"Z-score: {z_score:.2f}")

    if z_score < -2:
        null_verdict = "[bold green]SIGNIFICANTLY MORE SYMMETRIC than random![/]"
    elif z_score < -1:
        null_verdict = "[yellow]Somewhat more symmetric[/]"
    else:
        null_verdict = "[red]Not more symmetric than random[/]"

    console.print(f"Verdict: {null_verdict}")

    # Final Summary
    console.print("\n")
    table = Table(title="⚛️ HERMITICITY TEST SUMMARY")
    table.add_column("Test", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Verdict")

    table.add_row("Symmetry", f"{symmetry_score:.4f}", sym_verdict)
    table.add_row("Eigenvalues", f"Im/Re={ratio:.4f}", eig_verdict)
    table.add_row("Orthogonality", f"{max_corr:.4f}", orth_verdict)
    table.add_row("vs Random", f"z={z_score:.2f}", null_verdict)

    console.print(table)


if __name__ == "__main__":
    check_hermiticity()
