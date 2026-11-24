#!/usr/bin/env python3
"""Extract frame-level F0 for every clip in the manifest.

Usage: prep_f0.py --manifest M.tsv --wav-root DIR --f0-root DIR [--hop 512] [--procs 16]
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, csv, time, argparse
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f0 import extract_f0, interp_uv, SR
from data import f0_path

A = None


def work(rel):
    import soundfile as sf
    out = f0_path(A.f0_root, rel)
    try:
        if os.path.exists(out):
            return rel, 'skip'
        x, sr = sf.read(os.path.join(A.wav_root, rel), dtype='float32', always_2d=True)
        assert sr == SR, sr
        f, _ = interp_uv(extract_f0(x.mean(1), A.hop))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out + '.tmp.npy', f.astype(np.float32))
        os.replace(out + '.tmp.npy', out)
        return rel, 'ok'
    except Exception as e:
        return rel, 'err:%r' % (e,)


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True); ap.add_argument('--wav-root', required=True)
    ap.add_argument('--f0-root', required=True); ap.add_argument('--hop', type=int, default=512)
    ap.add_argument('--procs', type=int, default=16)
    A = ap.parse_args()
    rels = [r['path'] for r in csv.DictReader(open(A.manifest, encoding='utf-8'), delimiter='\t')]
    t0 = time.time(); n_err = 0
    with Pool(A.procs) as pool:
        for i, (rel, st) in enumerate(pool.imap_unordered(work, rels, chunksize=16)):
            if st.startswith('err'):
                n_err += 1; print(rel, st, flush=True)
            if i % 5000 == 0:
                print(i, len(rels), '%.0fs' % (time.time() - t0), flush=True)
    print('done, errors=%d, %.0fs' % (n_err, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
