"""Frame-level F0 used for training and inference.

Praat autocorrelation (parselmouth to_pitch_ac, 40-2000 Hz, voicing threshold 0.6) sampled at frame centres
t*hop + hop/2 (DiffSinger framing, T = len(x) // hop). Unvoiced frames are filled by linear interpolation.
"""
import numpy as np

SR = 44100


def extract_f0(x, hop, sr=SR, f0_min=40.0, f0_max=2000.0):
    """x [L] float -> f0 [L // hop] in Hz, 0 on unvoiced frames"""
    import parselmouth
    T = len(x) // hop
    snd = parselmouth.Sound(x.astype(np.float64), sr)
    p = snd.to_pitch_ac(time_step=hop / sr, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0 = p.selected_array['frequency']
    tt = (np.arange(T) * hop + hop / 2) / sr
    xs = p.xs()
    if not len(xs):
        return np.zeros(T)
    out = np.interp(tt, xs, f0, left=0, right=0)
    nearest = np.clip(np.rint((tt - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1) if len(xs) > 1 \
        else np.zeros(T, int)
    out[f0[nearest] == 0] = 0
    return out


def interp_uv(f0):
    """fill unvoiced frames by linear interpolation -> (f0, uv mask)"""
    uv = f0 == 0
    if uv.all():
        return f0 + 0, uv
    f = f0.copy()
    idx = np.arange(len(f))
    f[uv] = np.interp(idx[uv], idx[~uv], f[~uv])
    return f, uv
