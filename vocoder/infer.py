#!/usr/bin/env python3
"""Copy synthesis: wav -> (mel, f0) -> wav.

Usage: infer.py --ckpt model.ckpt --wav in.wav --out out.wav [--device cuda]
"""
import os, sys, argparse
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import load_generator
from data import DSMel
from f0 import extract_f0, interp_uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--wav', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    g, h = load_generator(a.ckpt, a.device)
    x, sr = sf.read(a.wav, dtype='float32', always_2d=True)
    assert sr == h.sampling_rate, 'expected %d Hz, got %d' % (h.sampling_rate, sr)
    x = x.mean(1)
    x = x[:len(x) // h.hop_size * h.hop_size]
    mel = DSMel.from_h(h).to(a.device)(torch.from_numpy(x)[None].to(a.device))       # [1, 128, T]
    f0, _ = interp_uv(extract_f0(x, h.hop_size, sr))
    with torch.no_grad():
        y = g(mel, torch.from_numpy(f0).float()[None].to(a.device))                    # [1, 1, T * hop]
    sf.write(a.out, y.squeeze().cpu().numpy(), sr, subtype='PCM_24')


if __name__ == '__main__':
    main()
