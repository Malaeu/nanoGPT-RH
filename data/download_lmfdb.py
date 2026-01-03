#!/usr/bin/env python3
"""
LMFDB Zeta Zeros Bulk Downloader.

Downloads zeros from https://www.lmfdb.org/zeros/zeta/list in parallel.
Uses the plain text API endpoint for efficient downloading.

Uses asyncio + aiohttp for efficient parallel downloading.
"""

import argparse
import asyncio
import time
from pathlib import Path

import aiohttp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

# LMFDB plain text API endpoint
LIST_URL = "https://www.lmfdb.org/zeros/zeta/list"
ZEROS_PER_REQUEST = 10000  # Can request up to 10000 at once


async def fetch_zeros(session: aiohttp.ClientSession, start_n: int, limit: int, semaphore: asyncio.Semaphore, retries: int = 3) -> list[tuple[int, str]]:
    """Fetch zeros starting at index start_n."""
    async with semaphore:
        url = f"{LIST_URL}?N={start_n}&limit={limit}"
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        console.print(f"[red]Error {response.status} at N={start_n}[/]")
                        await asyncio.sleep(1)
                        continue

                    text = await response.text()

                    # Parse plain text format: "N imaginary_part" per line
                    zeros = []
                    for line in text.strip().split('\n'):
                        if line:
                            parts = line.split()
                            if len(parts) >= 2:
                                n = int(parts[0])
                                t = parts[1]
                                zeros.append((n, t))

                    return zeros

            except asyncio.TimeoutError:
                console.print(f"[yellow]Timeout at N={start_n}, attempt {attempt+1}/{retries}[/]")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                console.print(f"[red]Error at N={start_n}: {e}[/]")
                await asyncio.sleep(1)

        return []


async def download_range(start: int, end: int, output_file: Path, max_concurrent: int = 10, delay: float = 0.2, batch_size: int = 10000):
    """Download zeros from start to end."""
    total_zeros = end - start
    total_requests = (total_zeros + batch_size - 1) // batch_size

    console.print(f"[cyan]Downloading zeros {start:,} to {end:,}[/]")
    console.print(f"[cyan]Total: {total_zeros:,} zeros in {total_requests:,} requests ({batch_size} per request)[/]")
    console.print(f"[cyan]Concurrent requests: {max_concurrent}, delay: {delay}s[/]")

    semaphore = asyncio.Semaphore(max_concurrent)

    # Prepare request indices
    request_starts = list(range(start, end, batch_size))

    all_zeros = []

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Downloading...", total=len(request_starts))

            # Process in batches to avoid overwhelming the server
            concurrent_batch = max_concurrent * 2
            for batch_idx in range(0, len(request_starts), concurrent_batch):
                batch = request_starts[batch_idx:batch_idx + concurrent_batch]

                tasks = [fetch_zeros(session, n, batch_size, semaphore) for n in batch]
                results = await asyncio.gather(*tasks)

                for zeros in results:
                    all_zeros.extend(zeros)
                    progress.advance(task)

                # Rate limiting
                await asyncio.sleep(delay * len(batch))

                # Periodic status
                if len(all_zeros) % 1000000 == 0 and len(all_zeros) > 0:
                    console.print(f"[green]Progress: {len(all_zeros):,} zeros downloaded[/]")

    # Sort by N (should already be sorted, but just in case)
    all_zeros.sort(key=lambda x: x[0])

    # Save to file
    console.print(f"[cyan]Saving {len(all_zeros):,} zeros to {output_file}...[/]")
    with open(output_file, 'w') as f:
        for n, t in all_zeros:
            f.write(f"{t}\n")

    console.print(f"[green]Done! Saved {len(all_zeros):,} zeros[/]")
    return len(all_zeros)


def main():
    parser = argparse.ArgumentParser(description="Download zeta zeros from LMFDB")
    parser.add_argument("--start", type=int, default=1, help="Starting zero index")
    parser.add_argument("--end", type=int, default=100_000_000, help="Ending zero index")
    parser.add_argument("--output", type=Path, default=Path("data/raw/zeros_100M.txt"), help="Output file")
    parser.add_argument("--concurrent", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between batches (seconds)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Zeros per request (max ~5000, safe=1000)")
    args = parser.parse_args()

    console.print("[bold magenta]LMFDB Zeta Zeros Downloader[/]")
    console.print()

    # Estimate time
    total_requests = (args.end - args.start) // args.batch_size
    estimated_time = total_requests * args.delay / args.concurrent
    console.print(f"[yellow]Total requests: {total_requests:,}[/]")
    console.print(f"[yellow]Estimated time: {estimated_time/60:.1f} minutes ({estimated_time/3600:.2f} hours)[/]")
    console.print()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    asyncio.run(download_range(args.start, args.end, args.output, args.concurrent, args.delay, args.batch_size))
    elapsed = time.time() - start_time

    console.print(f"\n[bold green]Completed in {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)[/]")


if __name__ == "__main__":
    main()
