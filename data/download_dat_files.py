#!/usr/bin/env python3
"""
Download Platt .dat files from LMFDB for a given number of zeros.
"""

import argparse
import sqlite3
import subprocess
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn

console = Console()

BASE_URL = "https://beta.lmfdb.org/riemann-zeta-zeros/data"


def get_required_files(db_path: Path, num_zeros: int) -> list[str]:
    """Get list of files needed for first N zeros."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('SELECT DISTINCT filename FROM zero_index WHERE N <= ? ORDER BY filename', (num_zeros,))
    files = [r[0] for r in c.fetchall()]

    conn.close()
    return files


def download_file(filename: str, output_dir: Path) -> bool:
    """Download a single .dat file using curl."""
    url = f"{BASE_URL}/{filename}"
    output_path = output_dir / filename

    if output_path.exists():
        console.print(f"  [dim]Skipping {filename} (already exists)[/]")
        return True

    # Use curl with cookie
    cmd = [
        "curl", "-sL",
        "-b", "human=1",
        "-o", str(output_path),
        "--max-time", "600",
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=660)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
        else:
            console.print(f"  [red]Failed to download {filename}[/]")
            if output_path.exists():
                output_path.unlink()
            return False
    except subprocess.TimeoutExpired:
        console.print(f"  [red]Timeout downloading {filename}[/]")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Platt .dat files from LMFDB")
    parser.add_argument("--index", type=Path, default=Path("data/raw/index.db"), help="Path to index.db")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Output directory")
    parser.add_argument("--num-zeros", type=int, default=100_000_000, help="Number of zeros needed")
    args = parser.parse_args()

    console.print("[bold magenta]LMFDB .dat File Downloader[/]")
    console.print()

    if not args.index.exists():
        console.print(f"[red]Index file not found: {args.index}[/]")
        return

    # Get list of required files
    files = get_required_files(args.index, args.num_zeros)
    console.print(f"[cyan]Need {len(files)} files for {args.num_zeros:,} zeros[/]")

    # Check which files already exist
    existing = [f for f in files if (args.output_dir / f).exists()]
    to_download = [f for f in files if f not in existing]

    console.print(f"[green]Already have: {len(existing)} files[/]")
    console.print(f"[yellow]Need to download: {len(to_download)} files[/]")

    if not to_download:
        console.print("[green]All files already downloaded![/]")
        return

    # Download missing files
    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print()
    success = 0
    failed = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Downloading...", total=len(to_download))

        for filename in to_download:
            progress.update(task, description=f"[cyan]{filename}")

            if download_file(filename, args.output_dir):
                success += 1
                size = (args.output_dir / filename).stat().st_size / 1024 / 1024
                console.print(f"  [green]Downloaded {filename} ({size:.1f} MB)[/]")
            else:
                failed.append(filename)

            progress.advance(task)

    console.print()
    console.print(f"[green]Successfully downloaded: {success} files[/]")
    if failed:
        console.print(f"[red]Failed: {len(failed)} files[/]")
        for f in failed:
            console.print(f"  [red]{f}[/]")


if __name__ == "__main__":
    main()
