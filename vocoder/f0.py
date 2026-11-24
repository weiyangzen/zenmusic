"""Frame-level F0 for training and inference.

pm_ac        Praat autocorrelation (parselmouth to_pitch_ac, 40-2000 Hz, voicing threshold 0.6) sampled at frame
             centres t*hop + hop/2 (DiffSinger framing, T = len(x) // hop). Its analysis window is 3 / floor = 75 ms
             whatever the hop, so fast pitch movement is smeared.
pm_ac_stone  the same voicing and octave decisions; every voiced frame is then refined with WORLD StoneMask
             (instantaneous frequency of the lowest harmonics over three local pitch periods). A frame that moves
             more than 300 cent away from the coarse value keeps the coarse value.

Both return f0 [T] in Hz with 0 on unvoiced frames. interp_uv fills unvoiced frames by linear interpolation.
"""
import numpy as np

SR = 44100
ANALYZERS = ('pm_ac', 'pm_ac_stone')


def resample_track(xs, f0, tt):
    """Praat frames (times xs, 0 = unvoiced) -> values at times tt. Linear interpolation between two voiced frames;
    next to an unvoiced frame the nearest frame is taken as is, so a voiced value is never averaged with 0."""
    if len(xs) < 2:
        return np.full(len(tt), f0[0] if len(xs) else 0.0)
    i = np.clip(np.searchsorted(xs, tt) - 1, 0, len(xs) - 2)
    w = np.clip((tt - xs[i]) / (xs[i + 1] - xs[i]), 0.0, 1.0)
    l, r = f0[i], f0[i + 1]
    out = np.where((l > 0) & (r > 0), l + w * (r - l), np.where(w < 0.5, l, r))
    out[(tt < xs[0]) | (tt > xs[-1])] = 0
    return out


def f0_pm_ac(x, hop, sr=SR, f0_min=40.0, f0_max=2000.0):
    import parselmouth
    T = len(x) // hop
    snd = parselmouth.Sound(x.astype(np.float64), sr)
    p = snd.to_pitch_ac(time_step=hop / sr, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0 = p.selected_array['frequency']
    tt = (np.arange(T) * hop + hop / 2) / sr
    return resample_track(p.xs(), f0, tt)


def stonemask_refine(x, f0, hop, sr=SR, max_cent=300.0):
    import pyworld as pw
    f0 = np.ascontiguousarray(f0, dtype=np.float64)
    tt = (np.arange(len(f0)) * hop + hop / 2) / sr
    r = pw.stonemask(np.ascontiguousarray(x, dtype=np.float64), f0, tt, sr)
    ok = (f0 > 0) & (r > 0)
    ok[ok] = np.abs(1200 * np.log2(r[ok] / f0[ok])) <= max_cent
    out = f0.copy()
    out[ok] = r[ok]
    return out


def extract_f0(x, hop, sr=SR, analyzer='pm_ac'):
    """x [L] float -> f0 [L // hop] in Hz, 0 on unvoiced frames"""
    f0 = f0_pm_ac(x, hop, sr)
    if analyzer == 'pm_ac':
        return f0
    if analyzer == 'pm_ac_stone':
        return stonemask_refine(x, f0, hop, sr)
    raise ValueError(analyzer)


def interp_uv(f0):
    """fill unvoiced frames by linear interpolation -> (f0, uv mask)"""
    uv = f0 == 0
    if uv.all():
        return f0 + 0, uv
    f = f0.copy()
    idx = np.arange(len(f))
    f[uv] = np.interp(idx[uv], idx[~uv], f[~uv])
    return f, uv
