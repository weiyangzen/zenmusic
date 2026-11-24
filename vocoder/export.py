#!/usr/bin/env python3
"""Export the generator from a training checkpoint.

Writes <run_dir>/export/<stem>/model.ckpt ({'generator': state_dict}, fp32, weight norm kept) + config.json.
Usage: export.py --ckpt <run_dir>/ckpt/step_0400000.ckpt
"""
import os, json, argparse
import torch


def export_generator(ckpt_path):
    ckpt_path = os.path.abspath(ckpt_path)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = {k[len('generator.'):]: v.float().clone() for k, v in ck['state_dict'].items() if k.startswith('generator.')}
    h = dict(ck['hyper_parameters']['h'])
    h['train_step'] = int(ck.get('global_step', 0)) // 2
    out = os.path.join(run_dir, 'export', stem)
    os.makedirs(out, exist_ok=True)
    torch.save({'generator': sd}, os.path.join(out, 'model.ckpt'))
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(h, f, indent=2)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    print(export_generator(ap.parse_args().ckpt))
