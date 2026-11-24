"""Mel framing and datasets.

Manifest: TSV with at least the columns `path` (wav path relative to the wav root) and `split` (train | val | test).
F0: <f0_root>/<path minus .wav>.npy, float32 [T], written by prep_f0.py.
"""
import os, csv
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def f0_path(f0_root, rel):
    return os.path.join(f0_root, os.path.splitext(rel)[0] + '.npy')


def read_rows(manifest, split):
    rows = [r for r in csv.DictReader(open(manifest, encoding='utf-8'), delimiter='\t') if r['split'] == split]
    return sorted(rows, key=lambda r: r['path'])


class DSMel(nn.Module):
    """DiffSinger framing: reflect pad ((n_fft-hop)//2, (n_fft-hop+1)//2), center=False, periodic Hann, magnitude,
    slaney mel, ln(clamp(., 1e-5)). Frames = len // hop."""

    def __init__(self, sr=44100, n_fft=2048, win_size=2048, hop=512, n_mels=128, fmin=40, fmax=16000):
        super().__init__()
        from librosa.filters import mel as librosa_mel_fn
        self.n_fft, self.win_size, self.hop = n_fft, win_size, hop
        mb = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.register_buffer('mel_basis', torch.from_numpy(mb).float(), persistent=False)
        self.register_buffer('window', torch.hann_window(win_size), persistent=False)

    @classmethod
    def from_h(cls, h):
        return cls(h.sampling_rate, h.n_fft, h.win_size, h.hop_size, h.num_mels, h.fmin, h.fmax)

    def forward(self, y):
        """y [B, L] -> [B, n_mels, L // hop]"""
        with torch.autocast(device_type=y.device.type, enabled=False):
            y = y.float()
            pl, pr = (self.n_fft - self.hop) // 2, (self.n_fft - self.hop + 1) // 2
            y = F.pad(y.unsqueeze(1), (pl, pr), mode='reflect').squeeze(1)
            spec = torch.stft(y, self.n_fft, hop_length=self.hop, win_length=self.win_size, window=self.window,
                              center=False, normalized=False, onesided=True, return_complex=True).abs()
            return torch.log(torch.clamp(torch.matmul(self.mel_basis, spec), min=1e-5))


class CropSet(Dataset):
    """Random hop-aligned crops: audio [1, segment], f0 [segment // hop]."""

    def __init__(self, rows, wav_root, f0_root, segment=16384, hop=512):
        assert segment % hop == 0
        self.rels = [r['path'] for r in rows]
        self.wav_root, self.f0_root = wav_root, f0_root
        self.segment, self.hop, self.frames = segment, hop, segment // hop

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, i):
        rel = self.rels[i]
        f0 = np.load(f0_path(self.f0_root, rel))
        T = len(f0)
        s = np.random.randint(0, T - self.frames + 1) if T > self.frames else 0
        x, sr = sf.read(os.path.join(self.wav_root, rel), start=s * self.hop, frames=self.segment, dtype='float32',
                        always_2d=True)
        x = x.mean(1)
        if len(x) < self.segment:
            x = np.pad(x, (0, self.segment - len(x)))
        f = f0[s:s + self.frames]
        if len(f) < self.frames:
            f = np.pad(f, (0, self.frames - len(f)), mode='edge')
        return {'audio': torch.from_numpy(x)[None], 'f0': torch.from_numpy(f.astype(np.float32))}


class FullSet(Dataset):
    """Whole clips trimmed to T*hop: audio [1, T*hop], f0 [T]."""

    def __init__(self, rows, wav_root, f0_root, hop=512):
        self.rels = [r['path'] for r in rows]
        self.wav_root, self.f0_root, self.hop = wav_root, f0_root, hop

    def __len__(self):
        return len(self.rels)

    def __getitem__(self, i):
        rel = self.rels[i]
        f0 = np.load(f0_path(self.f0_root, rel)).astype(np.float32)
        x, sr = sf.read(os.path.join(self.wav_root, rel), dtype='float32', always_2d=True)
        x = x.mean(1)
        T = min(len(f0), len(x) // self.hop)
        return {'audio': torch.from_numpy(x[:T * self.hop])[None], 'f0': torch.from_numpy(f0[:T])}


def val_subset(manifest, n=200, seed=0):
    rows = read_rows(manifest, 'val')
    rng = np.random.RandomState(seed)
    pick = sorted(rng.choice(len(rows), min(n, len(rows)), replace=False).tolist())
    return [rows[i] for i in pick]
