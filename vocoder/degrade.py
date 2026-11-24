"""Degraded copies of clean audio, for restoration training and for evaluation.

Training (config key "degrade"): with probability `prob` the generator's input mel is computed from a degraded copy of
the clean clip. The target and every loss keep using the clean clip, so the model learns to give clean audio for a
degraded input and to leave a clean input alone.

All functions take and return float32 tensors [B, L] and draw from torch's global generator (seed it for a fixed
result). Levels are absolute (dBFS RMS), not relative to the clip, so a silent clip gets noise too.
"""
import math

import torch

DEFAULT = {
    'prob': 0.3,
    'noise': {'p': 0.7, 'level_db': [-75.0, -45.0], 'alpha': [-1.0, 2.0]},   # power ~ f^-alpha: -1 blue, 0 white, 1 pink, 2 brown
    'hum': {'p': 0.15, 'level_db': [-70.0, -45.0]},                          # 50 or 60 Hz with harmonics
    'clip': {'p': 0.15, 'ratio': [0.5, 0.9]},                                # threshold = ratio * peak, hard or soft
    'lowpass': {'p': 0.15, 'hz': [11000.0, 16000.0]},
    'click': {'p': 0.10, 'rate': [1.0, 3.0], 'amp_db': [-30.0, -10.0]},      # impulses per second, peak level
}


def _amp(db):
    return (10.0 ** (db / 20.0)).view(-1, 1)


def colored_noise(batch, length, sr, alpha, device):
    """unit-RMS noise [B, L] with power spectrum ~ f^-alpha (alpha [B])"""
    spec = torch.fft.rfft(torch.randn(batch, length, device=device))
    f = torch.fft.rfftfreq(length, 1.0 / sr).to(device)
    f[0] = f[1]
    n = torch.fft.irfft(spec * (f.view(1, -1) / 1000.0) ** (-alpha.view(-1, 1) / 2.0), n=length)
    return n / n.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-9)


def add_noise(y, sr, level_db, alpha):
    return y + _amp(level_db) * colored_noise(y.shape[0], y.shape[1], sr, alpha, y.device)


def add_hum(y, sr, level_db, mains_hz):
    t = torch.arange(y.shape[1], device=y.device).view(1, -1) / sr
    h = 0
    for k, a in ((1, 1.0), (2, 0.5), (3, 0.35), (5, 0.2)):
        h = h + a * torch.sin(2 * math.pi * k * mains_hz.view(-1, 1) * t + 2 * math.pi * torch.rand(y.shape[0], 1, device=y.device))
    return y + _amp(level_db) * h / h.pow(2).mean(1, keepdim=True).sqrt()


def clip(y, ratio, soft):
    c = (ratio.view(-1, 1) * y.abs().amax(1, keepdim=True)).clamp(min=1e-6)
    return torch.where(soft.view(-1, 1), c * torch.tanh(y / c), torch.maximum(torch.minimum(y, c), -c))


def lowpass(y, sr, cutoff_hz):
    """raised-cosine roll-off over 1 kHz above the cutoff"""
    f = torch.fft.rfftfreq(y.shape[1], 1.0 / sr).to(y.device).view(1, -1)
    g = 0.5 * (1 + torch.cos(math.pi * ((f - cutoff_hz.view(-1, 1)) / 1000.0).clamp(0, 1)))
    return torch.fft.irfft(torch.fft.rfft(y) * g, n=y.shape[1])


def add_clicks(y, sr, rate_per_s, amp_db):
    hit = (torch.rand_like(y) < rate_per_s.view(-1, 1) / sr).to(y.dtype)
    imp = hit * torch.where(torch.rand_like(y) < 0.5, -1.0, 1.0) * _amp(amp_db)
    return y + imp + 0.6 * torch.roll(imp, 1, 1)


class Degrader:
    def __init__(self, cfg, sr):
        self.sr = sr
        self.c = {k: (dict(v, **cfg.get(k, {})) if isinstance(v, dict) else cfg.get(k, v)) for k, v in DEFAULT.items()}

    @torch.no_grad()
    def __call__(self, y):
        """y [B, L] clean -> [B, L]; about `prob` of the items are degraded, each with at least one kind"""
        B, dev, c = y.shape[0], y.device, self.c
        u = lambda r: r[0] + (r[1] - r[0]) * torch.rand(B, device=dev)
        pick = torch.rand(B, device=dev) < c['prob']
        if not bool(pick.any()):
            return y
        on = {k: pick & (torch.rand(B, device=dev) < c[k]['p']) for k in ('clip', 'lowpass', 'click', 'hum', 'noise')}
        on['noise'] = on['noise'] | (pick & ~torch.stack(list(on.values())).any(0))
        out = y
        sel = lambda k, z: torch.where(on[k].view(-1, 1), z, out)
        if bool(on['clip'].any()):
            out = sel('clip', clip(out, u(c['clip']['ratio']), torch.rand(B, device=dev) < 0.5))
        if bool(on['lowpass'].any()):
            out = sel('lowpass', lowpass(out, self.sr, u(c['lowpass']['hz'])))
        if bool(on['click'].any()):
            out = sel('click', add_clicks(out, self.sr, u(c['click']['rate']), u(c['click']['amp_db'])))
        if bool(on['hum'].any()):
            out = sel('hum', add_hum(out, self.sr, u(c['hum']['level_db']), torch.where(torch.rand(B, device=dev) < 0.5, 50.0, 60.0)))
        if bool(on['noise'].any()):
            out = sel('noise', add_noise(out, self.sr, u(c['noise']['level_db']), u(c['noise']['alpha'])))
        return out
