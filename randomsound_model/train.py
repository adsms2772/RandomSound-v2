"""
RandomSound: train a small neural audio model on your own WAV library.

Expected dataset:
  dataset/
    sound_001.wav
    sound_002.wav
    ...

This starter model learns short raw-audio chunks with a causal 1-D convolutional
network. It is an educational baseline, not a state-of-the-art audio generator.
For useful quality, begin with clean, mono WAV clips at 16 kHz and lots of data.
"""

import argparse
import glob
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
def prepare_audio_files(folder, sample_rate=16000):
    """Find WAV and MP3 files, converting MP3 files with FFmpeg."""
    folder = Path(folder).resolve()

    if not folder.is_dir():
        raise RuntimeError(f"Dataset folder does not exist: {folder}")

    cache_dir = folder / "_converted_mp3"
    cache_dir.mkdir(parents=True, exist_ok=True)

    mp3_files = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".mp3"
        and cache_dir not in p.parents
    ]

    print(f"Found {len(mp3_files)} MP3 files.")

    for mp3_path in tqdm(mp3_files, desc="Converting MP3"):
        relative_path = mp3_path.relative_to(folder)
        output_path = cache_dir / relative_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if (
            output_path.exists()
            and output_path.stat().st_mtime >= mp3_path.stat().st_mtime
        ):
            continue

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(mp3_path),
            "-vn", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le", str(output_path),
        ]

        try:
            subprocess.run(
                command, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as exc:
            print(f"Failed to convert {mp3_path}: {exc.stderr.strip()}")
            if output_path.exists():
                output_path.unlink()

    original_wavs = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() == ".wav"
        and cache_dir not in p.parents
    ]

    converted_wavs = [
        p for p in cache_dir.rglob("*.wav") if p.is_file()
    ]

    all_wavs = sorted(set(original_wavs + converted_wavs))

    if not all_wavs:
        raise RuntimeError(f"No WAV or MP3 files found in: {folder}")

    print(f"Ready: {len(original_wavs)} original WAV files, "
          f"{len(converted_wavs)} converted WAV files.")

    return all_wavs

class AudioChunkDataset(Dataset):
    def __init__(self, folder, sample_rate=16000, chunk_seconds=1.0):
        self.paths = []
        for ext in ("*.wav", "*.WAV"):
         self.paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        self.sample_rate = sample_rate
        self.chunk_len = int(sample_rate * chunk_seconds)
        self.audio = []

        if not self.paths:
            raise RuntimeError(f"No WAV files found in: {folder}")

        print(f"Loading {len(self.paths)} WAV files...")
        for path in tqdm(self.paths):
            try:
                x, sr = sf.read(path, dtype="float32", always_2d=True)
                x = x.mean(axis=1)  # stereo -> mono
                if sr != sample_rate:
                    # Lightweight linear resampling to avoid another dependency.
                    old_t = np.arange(len(x), dtype=np.float64) / sr
                    new_len = max(1, round(len(x) * sample_rate / sr))
                    new_t = np.arange(new_len, dtype=np.float64) / sample_rate
                    x = np.interp(new_t, old_t, x).astype(np.float32)
                x = np.clip(x, -1.0, 1.0)
                if len(x) >= self.chunk_len + 1:
                    self.audio.append(torch.from_numpy(x))
            except Exception as exc:
                print(f"Skipping {path}: {exc}")

        if not self.audio:
            raise RuntimeError("No usable WAV clips long enough for the chosen chunk size.")

        self.index = []
        for audio_i, audio in enumerate(self.audio):
            count = max(1, (len(audio) - self.chunk_len - 1) // max(1, self.chunk_len // 2))
            for _ in range(count):
                self.index.append(audio_i)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        audio = self.audio[self.index[idx]]
        start = random.randint(0, len(audio) - self.chunk_len - 1)
        chunk = audio[start:start + self.chunk_len + 1]
        # Predict the next sample at each time step.
        return chunk[:-1], chunk[1:]


class CausalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=0, dilation=dilation
        )
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return F.gelu(self.norm(self.conv(x)))


class TinyAudioModel(nn.Module):
    def __init__(self, channels=64, layers=6, kernel_size=3):
        super().__init__()
        self.input = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList([
            CausalBlock(channels, kernel_size, 2 ** (i % 6))
            for i in range(layers)
        ])
        self.output = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.Conv1d(channels, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # x shape: [batch, time]
        h = self.input(x.unsqueeze(1))
        for block in self.blocks:
            h = h + block(h)
        return self.output(h).squeeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset", help="Folder containing WAV files")
    parser.add_argument("--out", default="checkpoints/randomsound.pt")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    ds = AudioChunkDataset(args.data, args.sample_rate, args.chunk_seconds)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True)
    model = TinyAudioModel(args.channels, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # Baseline next-sample regression objective.
            loss = F.l1_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg = float(np.mean(losses)) if losses else float("nan")
        print(f"Epoch {epoch}: mean L1 loss = {avg:.5f}")
        torch.save({
            "model": model.state_dict(),
            "sample_rate": args.sample_rate,
            "chunk_seconds": args.chunk_seconds,
            "channels": args.channels,
            "layers": args.layers,
            "epoch": epoch,
            "loss": avg,
        }, args.out)
        print(f"Saved checkpoint: {args.out}")

    print("Training finished.")


if __name__ == "__main__":
    main()
