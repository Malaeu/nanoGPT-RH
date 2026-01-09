"""
data_loading.py - Streaming DataLoader for nanoGPT_RH

Three data loading strategies:
  - gpu-direct: All data on GPU, randint batching (fastest, needs VRAM)
  - mmap: Memory-mapped numpy, lazy loading (low RAM usage)
  - dataloader: Legacy PyTorch DataLoader (backward compat)

Usage:
    from data_loading import load_data

    train_batcher, val_batcher, info = load_data(
        data_dir='data/continuous_500M',
        batch_size=512,
        device=torch.device('cuda'),
        mode='auto'  # or 'gpu-direct', 'mmap', 'dataloader'
    )

    # Training loop
    x, y = train_batcher.get_batch()
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GPUDirectBatcher:
    """GPU-resident data with randint batching.

    Fastest option when data fits in GPU memory.
    No CPU-GPU transfer during training.
    """

    def __init__(self, data: torch.Tensor, batch_size: int, device: torch.device):
        self.data = data.to(device)
        self.batch_size = batch_size
        self.device = device
        self.n_samples = len(data)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get random batch directly from GPU memory."""
        idx = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        windows = self.data[idx]  # [B, seq_len]
        x = windows[:, :-1]  # [B, seq_len-1] context
        y = windows[:, -1]   # [B] target
        return x, y

    def get_batch_full(self) -> torch.Tensor:
        """Get full windows without x/y split (for eval)."""
        idx = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        return self.data[idx]

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self):
        """Iterate over all data in batches (for eval compatibility)."""
        for i in range(0, self.n_samples, self.batch_size):
            end = min(i + self.batch_size, self.n_samples)
            windows = self.data[i:end]
            x = windows[:, :-1]
            y = windows[:, -1]
            yield x, y


class MMapBatcher:
    """Memory-mapped numpy with lazy loading.

    Uses minimal RAM, loads data from disk on demand.
    Slower than GPU-direct due to disk I/O.
    """

    def __init__(self, npy_path: Path, batch_size: int, device: torch.device):
        self.data = np.load(npy_path, mmap_mode='r')
        self.batch_size = batch_size
        self.device = device
        self.n_samples = len(self.data)
        self._rng = np.random.default_rng()

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lazy load batch from disk."""
        idx = self._rng.integers(0, self.n_samples, size=self.batch_size)
        # .copy() required for mmap slicing
        windows = torch.from_numpy(self.data[idx].copy()).float().to(self.device)
        x = windows[:, :-1]
        y = windows[:, -1]
        return x, y

    def get_batch_full(self) -> torch.Tensor:
        """Get full windows without x/y split."""
        idx = self._rng.integers(0, self.n_samples, size=self.batch_size)
        return torch.from_numpy(self.data[idx].copy()).float().to(self.device)

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self):
        """Iterate over all data in batches (for eval compatibility)."""
        for i in range(0, self.n_samples, self.batch_size):
            end = min(i + self.batch_size, self.n_samples)
            windows = torch.from_numpy(self.data[i:end].copy()).float().to(self.device)
            x = windows[:, :-1]
            y = windows[:, -1]
            yield x, y


class SpacingNextDataset(Dataset):
    """Legacy Dataset for DataLoader mode.

    Copied from train_mdn_postfix.py for backward compatibility.
    """

    def __init__(self, spacings: torch.Tensor, seq_len: int = 257):
        # Handle both windowed [N, L] and flat [N] formats
        if spacings.dim() == 2:
            self.spacings = spacings.float()
            self.n_samples = len(spacings)
            self.seq_len = spacings.shape[1]
            self.windowed = True
        else:
            self.spacings = spacings.float()
            self.seq_len = seq_len
            self.n_samples = len(spacings) - seq_len
            self.windowed = False

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.windowed:
            window = self.spacings[idx]
        else:
            window = self.spacings[idx:idx + self.seq_len]
        x = window[:-1]
        y = window[-1]
        return x, y


class DataLoaderWrapper:
    """Wrap DataLoader to match Batcher interface."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self._iter = iter(loader)
        self.n_samples = len(loader.dataset)

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch, restart iterator if exhausted."""
        try:
            x, y = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            x, y = next(self._iter)
        return x.to(self.device), y.to(self.device)

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self):
        """Iterate over all batches (for eval compatibility)."""
        for x, y in self.loader:
            yield x.to(self.device), y.to(self.device)


def convert_pt_to_npy(pt_path: Path, npy_path: Path) -> Path:
    """Convert .pt to .npy for mmap support."""
    if npy_path.exists():
        return npy_path

    print(f"[*] Converting {pt_path.name} to numpy for mmap...")
    data = torch.load(pt_path, weights_only=False)

    # Handle dict format
    if isinstance(data, dict):
        data = data.get('data', data.get('spacings'))

    np.save(npy_path, data.numpy())
    print(f"[+] Created {npy_path.name}")
    return npy_path


def auto_detect_mode(data_path: Path, device: torch.device) -> str:
    """Auto-detect best data loading mode based on file size and VRAM."""
    data_size = data_path.stat().st_size

    if device.type == 'cuda':
        try:
            props = torch.cuda.get_device_properties(0)
            # Use 40% of VRAM as safety margin
            vram_available = props.total_memory * 0.4

            if data_size < vram_available:
                return 'gpu-direct'
            else:
                return 'mmap'
        except Exception:
            pass

    # CPU or detection failed
    if data_size < 500_000_000:  # 500MB
        return 'dataloader'
    else:
        return 'mmap'


def load_data(
    data_dir: str | Path,
    batch_size: int,
    device: torch.device,
    mode: str = 'auto',
    seq_len: int = 257,
    train_fraction: float = 1.0,
    num_workers: int = 4
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load train/val data with appropriate strategy.

    Args:
        data_dir: Directory containing train.pt, val.pt
        batch_size: Batch size for training
        device: torch.device (cuda or cpu)
        mode: 'auto', 'gpu-direct', 'mmap', or 'dataloader'
        seq_len: Sequence length (for dataloader mode)
        train_fraction: Fraction of training data to use
        num_workers: Number of DataLoader workers (for dataloader mode)

    Returns:
        train_batcher: Object with .get_batch() method
        val_batcher: Object with .get_batch() method
        info: Dict with metadata (mode, samples, memory usage)
    """
    data_dir = Path(data_dir)
    train_pt = data_dir / 'train.pt'
    val_pt = data_dir / 'val.pt'

    # Validate paths
    if not train_pt.exists():
        raise FileNotFoundError(f"Train data not found: {train_pt}")
    if not val_pt.exists():
        raise FileNotFoundError(f"Val data not found: {val_pt}")

    # Auto-detect mode
    if mode == 'auto':
        mode = auto_detect_mode(train_pt, device)

    info: Dict[str, Any] = {'mode': mode, 'device': str(device)}

    if mode == 'gpu-direct':
        # Load directly to GPU
        print(f"[*] Loading data to GPU ({device})...")
        train_data = torch.load(train_pt, weights_only=False)
        val_data = torch.load(val_pt, weights_only=False)

        # Handle dict format
        if isinstance(train_data, dict):
            train_data = train_data.get('data', train_data.get('spacings'))
        if isinstance(val_data, dict):
            val_data = val_data.get('data', val_data.get('spacings'))

        # Apply train_fraction (grokking mode)
        if train_fraction < 1.0:
            n_orig = len(train_data)
            n_keep = int(n_orig * train_fraction)
            train_data = train_data[:n_keep]
            info['train_fraction_applied'] = True
            info['train_original'] = n_orig

        train_batcher = GPUDirectBatcher(train_data, batch_size, device)
        val_batcher = GPUDirectBatcher(val_data, batch_size, device)

        info['train_samples'] = len(train_data)
        info['val_samples'] = len(val_data)
        info['gpu_memory_bytes'] = train_data.numel() * 4 + val_data.numel() * 4

    elif mode == 'mmap':
        # Convert to numpy if needed and use mmap
        train_npy = data_dir / 'train.npy'
        val_npy = data_dir / 'val.npy'

        convert_pt_to_npy(train_pt, train_npy)
        convert_pt_to_npy(val_pt, val_npy)

        train_batcher = MMapBatcher(train_npy, batch_size, device)
        val_batcher = MMapBatcher(val_npy, batch_size, device)

        info['train_samples'] = train_batcher.n_samples
        info['val_samples'] = val_batcher.n_samples
        info['using_npy'] = True

    else:  # dataloader (legacy)
        print("[*] Using legacy DataLoader mode...")
        train_data = torch.load(train_pt, weights_only=False)
        val_data = torch.load(val_pt, weights_only=False)

        # Handle dict format
        if isinstance(train_data, dict):
            train_data = train_data.get('data', train_data.get('spacings'))
        if isinstance(val_data, dict):
            val_data = val_data.get('data', val_data.get('spacings'))

        # Apply train_fraction
        if train_fraction < 1.0:
            n_orig = len(train_data)
            n_keep = int(n_orig * train_fraction)
            train_data = train_data[:n_keep]

        train_dataset = SpacingNextDataset(train_data, seq_len=seq_len)
        val_dataset = SpacingNextDataset(val_data, seq_len=seq_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        train_batcher = DataLoaderWrapper(train_loader, device)
        val_batcher = DataLoaderWrapper(val_loader, device)

        info['train_samples'] = len(train_dataset)
        info['val_samples'] = len(val_dataset)
        info['num_workers'] = num_workers

    print(f"[+] Data loaded: mode={mode}, train={info['train_samples']:,}, val={info['val_samples']:,}")

    return train_batcher, val_batcher, info
