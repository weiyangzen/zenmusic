#!/usr/bin/env python3
"""Score the F0 analyzers in f0.py against signals whose F0 is known exactly.

Singing-like harmonic signals (moving formants, shimmer, noise floor at -38 dB) are synthesised from a per-sample F0
curve: fast runs, very fast runs, vibrato and glides at seven base pitches from 98 to 1100 Hz. Each analyzer is
compared with the true F0 at the frame centres, grouped by how fast the pitch is moving.

Usage: f0_bench.py [--hop 512] [--procs 8] [--seeds 2]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, argparse, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f0 import extract_f0, ANALYZERS, SR

BASES = [98, 147, 220, 330, 494, 740, 1100]
KINDS = ['run', 'run_fast', 'vib', 'glide']
VOWELS = [(800, 1200, 2600, 3500), (300, 2300, 3000, 3600), (350, 800, 2400, 3400), (500, 1900, 2600, 3500),
          (500, 900, 2500, 3400)]
BW = (80, 100, 140, 180)


def smooth_steps(vals, durs, kernel_s, n):
    """piecewise-constant values -> per-sample curve, transitions shaped by a Hann kernel of kernel_s seconds"""
    st = np.zeros(n); pos = 0
    for v, d in zip(vals, durs):
        e = min(n, pos + int(d * SR)); st[pos:e] = v; pos = e
        if pos >= n:
            break
    if pos < n:
        st[pos:] = vals[-1]
    L = max(3, int(kernel_s * SR)) | 1
    k = np.hanning(L); k /= k.sum()
    return np.convolve(np.pad(st, (L // 2, L // 2), mode='edge'), k, mode='valid')[:n]


def trajectory(kind, base, dur, rng):
    n = int(dur * SR); t = np.arange(n) / SR
    vals, durs, cur, tot = [], [], 0.0, 0.0
    if kind in ('run', 'run_fast'):                 # notes of 70-140 ms, steps of 1-4 semitones
        while tot < dur:
            step = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            cur = cur + step if abs(cur + step) <= 7 else cur - step
            d = rng.uniform(0.07, 0.14); vals.append(cur); durs.append(d); tot += d
        st = smooth_steps(vals, durs, rng.uniform(0.04, 0.07) if kind == 'run' else rng.uniform(0.02, 0.03), n)
    elif kind == 'vib':                             # held notes with 5.5-7 Hz vibrato, 0.5-1 semitone deep
        while tot < dur:
            step = rng.choice([-5, -3, -2, 2, 3, 5])
            cur = cur + step if abs(cur + step) <= 7 else cur - step
            d = rng.uniform(0.4, 0.8); vals.append(cur); durs.append(d); tot += d
        st = smooth_steps(vals, durs, rng.uniform(0.06, 0.09), n)
        st = st + rng.uniform(0.5, 1.0) * np.sin(2 * np.pi * rng.uniform(5.5, 7.0) * t + rng.uniform(0, 6.28))
    else:                                           # glides of 5-12 semitones
        sgn = 1
        while tot < dur:
            cur = sgn * rng.uniform(2.5, 6.0); sgn = -sgn
            d = rng.uniform(0.25, 0.45); vals.append(cur); durs.append(d); tot += d
        st = smooth_steps(vals, durs, rng.uniform(0.12, 0.25), n)
    w = np.hanning(2001); jit = np.convolve(rng.randn(n + 2000), w / w.sum(), mode='valid')[:n]
    st = st + 0.03 * jit / (jit.std() + 1e-9)       # slow drift of about 3 cent
    return base * 2 ** (st / 12)


def synth(f0, rng):
    n = len(f0)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    vals, durs, tot = [], [], 0.0
    while tot < n / SR:
        d = rng.uniform(0.3, 0.6); vals.append(rng.randint(len(VOWELS))); durs.append(d); tot += d
    F = [smooth_steps([VOWELS[v][i] for v in vals], durs, 0.08, n) for i in range(4)]
    y = np.zeros(n)
    for k in range(1, int(16000 / f0.min()) + 1):
        f = k * f0
        a = 1.0 / k
        for Fi, Bi in zip(F, BW):
            a = a / np.sqrt((1 - (f / Fi) ** 2) ** 2 + (f * Bi / Fi ** 2) ** 2)
        y += a * (f < 16000) * np.sin(k * phase + rng.uniform(0, 6.28))
    y /= np.abs(y).max() + 1e-9
    w = np.hanning(4411)
    y = y * (1 + 0.05 * np.convolve(rng.randn(n + 4410), w / w.sum() * 60, mode='valid')[:n])
    y = y + 10 ** (-38 / 20) * rng.randn(n)
    fade = int(0.03 * SR); y[:fade] *= np.linspace(0, 1, fade); y[-fade:] *= np.linspace(1, 0, fade)
    pad = int(0.15 * SR)
    return (np.concatenate([np.zeros(pad), y, np.zeros(pad)]) * 0.5).astype(np.float32), \
        np.concatenate([np.zeros(pad), f0, np.zeros(pad)])


def make(key, hop):
    bi, ki, s = key
    rng = np.random.RandomState(1000 * bi + 10 * ki + s)
    x, f0 = synth(trajectory(KINDS[ki], BASES[bi], 4.0, rng), rng)
    c = np.arange(len(x) // hop) * hop + hop // 2
    stc = 12 * np.log2(np.maximum(f0, 1e-3)); d = int(0.001 * SR)
    speed = np.abs(stc[np.minimum(c + d, len(f0) - 1)] - stc[np.maximum(c - d, 0)]) / (2 * d / SR)
    ok = (f0[c] > 0) & (f0[np.maximum(c - 2205, 0)] > 0) & (f0[np.minimum(c + 2205, len(f0) - 1)] > 0)
    return x, f0[c], speed, ok


def work(job):
    key, hop = job
    x, truth, speed, ok = make(key, hop)
    return key, truth, speed, ok, {a: extract_f0(x, hop, analyzer=a)[:len(truth)] for a in ANALYZERS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hop', type=int, default=512); ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args()
    jobs = [((bi, ki, s), a.hop) for bi in range(len(BASES)) for ki in range(len(KINDS)) for s in range(a.seeds)]
    with Pool(a.procs) as pool:
        res = pool.map(work, jobs, chunksize=1)
    buckets = [('all', lambda k, s: s >= 0), ('< 15 st/s', lambda k, s: s < 15), ('15-50', lambda k, s: (s >= 15) & (s < 50)),
               ('50-100', lambda k, s: (s >= 50) & (s < 100)), ('> 100 st/s', lambda k, s: s >= 100),
               ('<= 147 Hz', lambda k, s: np.full(len(s), BASES[k[0]] <= 147)),
               ('220-494 Hz', lambda k, s: np.full(len(s), 220 <= BASES[k[0]] <= 494)),
               ('>= 740 Hz', lambda k, s: np.full(len(s), BASES[k[0]] >= 740))]
    print('%d clips, hop %d. Cells: RMSE cent (errors under 600 cent) / share of frames off by more than 50 cent' % (len(jobs), a.hop))
    print('| analyzer | ' + ' | '.join(b[0] for b in buckets) + ' |')
    print('|---|' + '---|' * len(buckets))
    for an in ANALYZERS:
        cells = []
        for _, sel in buckets:
            e = []
            for key, truth, speed, ok, f in res:
                v = ok & sel(key, speed) & (f[an] > 0)
                e.append(1200 * np.log2(f[an][v] / truth[v]))
            e = np.concatenate(e)
            cells.append('%.1f / %.1f %%' % (np.sqrt(np.mean(e[np.abs(e) < 600] ** 2)), 100 * np.mean(np.abs(e) > 50)))
        print('| %s | %s |' % (an, ' | '.join(cells)))


if __name__ == '__main__':
    main()
