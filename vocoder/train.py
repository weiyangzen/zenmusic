#!/usr/bin/env python3
"""NSF-HiFiGAN GAN training (Lightning, manual optimization, DDP).

Losses
  discriminators: LSGAN, summed over the sub-discriminators of MPD and MSD
  generator     : adv + feature matching (MPD + MSD)
                  + loss_aux_weight * (0.5 * L_stft + L_mel + L_mel_extra + loss_input_mel_weight * L_mel_in)
    L_stft      : mean over loss_stft_resolutions of L1 on the complex STFT
    L_mel       : mean over loss_mel_resolutions of L1 on the full-band log-mel (num_mels bands)
    L_mel_extra : sum of weight * L1 log-mel for each [n_fft, hop, win, n_mels, weight] in loss_mel_extra (optional);
                  added on top of L_mel so an extra scale has its own band count and does not dilute the mean
    L_mel_in    : L1 between the mel the generator was given and the same mel of its output (the input definition:
                  DiffSinger framing, fmin..fmax, hop_size); used when loss_input_mel_weight > 0
Discriminator width follows discriminator_channel_mult (1.0 = HiFi-GAN sizes).
Restoration training (config key "degrade", see degrade.py): with probability degrade.prob the input mel of an item is
computed from a degraded copy of the clip (noise, hum, clipping, low-pass, clicks); targets and losses stay clean.
A step = one discriminator update + one generator update. A step whose loss is not finite is skipped and counted.
Optional config keys: grad_clip_norm (clip the gradient norm of each update), lr_warmup_steps (linear warm-up; the
per-epoch decay lr * lr_decay^epoch stays the same).

Every --log-every steps each sub-discriminator is checked: mean output on real and generated audio, output spread,
largest feature magnitude, its share of the feature-matching loss. A sub-discriminator whose output no longer moves,
or whose features run away, is reported on the console as DISC-WARN (TensorBoard: disc/*).

Usage:
  train.py --config configs/small_n3_h17.json --manifest M.tsv --wav-root DIR --f0-root DIR --run-name NAME
           --batch 12 --devices 3 [--precision 32-true] [--tf32] [--resume auto|PATH] [--max-steps N] [--probe-steps N]
Checkpoints: <runs-dir>/<run-name>/ckpt/step_XXXXXXX.ckpt every --ckpt-every steps (newest 3 kept), each one exported
to <runs-dir>/<run-name>/export/step_XXXXXXX/{model.ckpt, config.json}.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, glob, time, argparse, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models import (AttrDict, Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss,
                    feature_loss, generator_loss)
from data import DSMel, CropSet, FullSet, read_rows, val_subset
from export import export_generator
from degrade import Degrader



def loss_mel_transform(sr, n_fft, hop, win, n_mels):
    from torchaudio.transforms import MelSpectrogram
    return MelSpectrogram(sample_rate=sr, n_fft=n_fft, win_length=win, hop_length=hop, f_min=0, f_max=sr // 2,
                          n_mels=n_mels, center=True, power=1.0, pad_mode='reflect', norm='slaney', mel_scale='slaney')


class GAN(pl.LightningModule):
    def __init__(self, h):
        super().__init__()
        self.save_hyperparameters({'h': dict(h)})
        self.h = AttrDict(h)
        self.generator = Generator(self.h)
        d_mult = float(self.h.get('discriminator_channel_mult', 1.0))
        self.mpd = MultiPeriodDiscriminator(self.h.discriminator_periods, d_mult)
        self.msd = MultiScaleDiscriminator(d_mult)
        self.mel = DSMel.from_h(self.h)
        # plain lists: not registered, not in state_dict
        self._loss_mels = [loss_mel_transform(self.h.sampling_rate, n, hp, w, self.h.num_mels)
                           for n, hp, w in self.h.loss_mel_resolutions]
        self._loss_mels_x = [(loss_mel_transform(self.h.sampling_rate, int(n), int(hp), int(w), int(nm)), float(wt))
                             for n, hp, w, nm, wt in self.h.get('loss_mel_extra', [])]
        self._loss_mels_dev = None
        self.automatic_optimization = False
        self.nonfinite = 0          # steps skipped because a loss was not finite
        self.log_every = 50
        self.disc_names = ['mpd%d' % p for p in self.h.discriminator_periods] + ['msd%d' % i for i in range(3)]
        self.disc_warn = ''
        self.clip = float(self.h.get('grad_clip_norm', 0) or 0)
        self.warmup = int(self.h.get('lr_warmup_steps', 0) or 0)
        dg = self.h.get('degrade')
        self.degrader = Degrader(dg, self.h.sampling_rate) if dg and float(dg.get('prob', 0)) > 0 else None

    def configure_optimizers(self):
        h = self.h
        optim_g = torch.optim.AdamW(self.generator.parameters(), lr=h.learning_rate, betas=(h.adam_b1, h.adam_b2))
        optim_d = torch.optim.AdamW(itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                                    lr=h.learning_rate, betas=(h.adam_b1, h.adam_b2))
        return [optim_g, optim_d], [ExponentialLR(optim_g, h.lr_decay), ExponentialLR(optim_d, h.lr_decay)]

    @property
    def step(self):
        return self.global_step // 2

    def loss_mels(self, device):
        if self._loss_mels_dev is None:
            self._loss_mels_dev = [t.to(device) for t in self._loss_mels]
            self._loss_mels_x = [(t.to(device), wt) for t, wt in self._loss_mels_x]
        return self._loss_mels_dev

    def aux_losses(self, y, y_hat):
        """y, y_hat [B, L] float32"""
        with torch.autocast(device_type=y.device.type, enabled=False):
            loss_stft = 0
            for n_fft, hop, win in self.h.loss_stft_resolutions:
                s = torch.view_as_real(torch.stft(y, n_fft, hop, win, return_complex=True))
                s_hat = torch.view_as_real(torch.stft(y_hat, n_fft, hop, win, return_complex=True))
                loss_stft = loss_stft + F.l1_loss(s_hat, s)
            loss_stft = loss_stft / len(self.h.loss_stft_resolutions)
            loss_mel = 0
            for t in self.loss_mels(y.device):
                m = torch.log(torch.clamp(t(y), min=1e-5))
                m_hat = torch.log(torch.clamp(t(y_hat), min=1e-5))
                loss_mel = loss_mel + F.l1_loss(m_hat, m)
            loss_mel = loss_mel / len(self.h.loss_mel_resolutions)
            loss_mel_x = y.new_zeros(())
            for t, wt in self._loss_mels_x:
                loss_mel_x = loss_mel_x + wt * F.l1_loss(torch.log(torch.clamp(t(y_hat), min=1e-5)),
                                                         torch.log(torch.clamp(t(y), min=1e-5)))
        return loss_stft, loss_mel, loss_mel_x

    @torch.no_grad()
    def disc_health(self, d_real, d_fake, fm_real, fm_fake):
        """per sub-discriminator: mean output on real / generated, output spread, largest feature, feature-matching share"""
        vals, warn = {}, []
        fms = [float(2 * sum(torch.mean(torch.abs(a.float() - b.float())) for a, b in zip(fr, fg)))
               for fr, fg in zip(fm_real, fm_fake)]
        med = float(np.median(fms)) + 1e-9
        for name, r, g, fr, fm in zip(self.disc_names, d_real, d_fake, fm_real, fms):
            r, g = r.float(), g.float()
            spread = float(torch.cat([r.flatten(), g.flatten()]).std())
            feat = max(float(a.float().abs().max()) for a in fr[:-1])
            vals.update({'disc/%s_real' % name: float(r.mean()), 'disc/%s_fake' % name: float(g.mean()),
                         'disc/%s_spread' % name: spread, 'disc/%s_featmax' % name: feat, 'disc/%s_fm' % name: fm})
            if spread < 1e-3: warn.append('%s output constant (spread %.1e)' % (name, spread))
            if feat > 1e4: warn.append('%s features %.1e' % (name, feat))
            if fm > 10 * med and fm > 5: warn.append('%s fm %.1f vs median %.1f' % (name, fm, med))
        return vals, '; '.join(warn)

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()
        if self.warmup > 0:
            lr = self.h.learning_rate * self.h.lr_decay ** self.current_epoch * min(1.0, (self.step + 1) / self.warmup)
            for o in (optim_g, optim_d):
                for group in o.param_groups:
                    group['lr'] = lr
        y, f0 = batch['audio'].float(), batch['f0'].float()
        with torch.no_grad():
            mel_clean = self.mel(y.squeeze(1))
            mel = mel_clean if self.degrader is None else self.mel(self.degrader(y.squeeze(1)))
        y_hat = self.generator(mel, f0)

        # discriminators
        optim_d.zero_grad(set_to_none=True)
        y_df_r, y_df_g, _, _ = self.mpd(y, y_hat.detach())
        y_ds_r, y_ds_g, _, _ = self.msd(y, y_hat.detach())
        loss_d_f = discriminator_loss(y_df_r, y_df_g)
        loss_d_s = discriminator_loss(y_ds_r, y_ds_g)
        loss_d = loss_d_f + loss_d_s
        ok_d = bool(torch.isfinite(loss_d))
        if ok_d:
            self.manual_backward(loss_d)
            if self.clip > 0:
                self.clip_gradients(optim_d, gradient_clip_val=self.clip, gradient_clip_algorithm='norm')
            optim_d.step()

        # generator
        optim_g.zero_grad(set_to_none=True)
        loss_stft, loss_mel, loss_mel_x = self.aux_losses(y.squeeze(1), y_hat.float().squeeze(1))
        loss_aux = 0.5 * loss_stft + loss_mel + loss_mel_x
        w_in = float(self.h.get('loss_input_mel_weight', 0.0))
        loss_mel_in = y.new_zeros(())
        if w_in > 0:
            loss_mel_in = F.l1_loss(self.mel(y_hat.float().squeeze(1))[..., :mel_clean.shape[-1]], mel_clean)
            loss_aux = loss_aux + w_in * loss_mel_in
        y_df_r, y_df_g, fm_f_r, fm_f_g = self.mpd(y, y_hat)
        y_ds_r, y_ds_g, fm_s_r, fm_s_g = self.msd(y, y_hat)
        loss_fm_f, loss_fm_s = feature_loss(fm_f_r, fm_f_g), feature_loss(fm_s_r, fm_s_g)
        loss_gen_f, loss_gen_s = generator_loss(y_df_g), generator_loss(y_ds_g)
        loss_g = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_aux * self.h.loss_aux_weight
        ok_g = bool(torch.isfinite(loss_g))
        if ok_g:
            self.manual_backward(loss_g)
            if self.clip > 0:
                self.clip_gradients(optim_g, gradient_clip_val=self.clip, gradient_clip_algorithm='norm')
            optim_g.step()
        self.nonfinite += int(not (ok_d and ok_g))

        if self.trainer.is_last_batch and self.warmup == 0:
            for s in self.lr_schedulers():
                s.step()

        if batch_idx % self.log_every == 0:
            vals = {'train/loss_d': loss_d, 'train/loss_d_mpd': loss_d_f, 'train/loss_d_msd': loss_d_s,
                    'train/loss_g': loss_g, 'train/adv_mpd': loss_gen_f, 'train/adv_msd': loss_gen_s,
                    'train/fm_mpd': loss_fm_f, 'train/fm_msd': loss_fm_s,
                    'train/stft': loss_stft, 'train/mel': loss_mel, 'train/mel_x': loss_mel_x,
                    'train/mel_in': loss_mel_in}
            vals = {k: float(v.detach()) for k, v in vals.items()}
            vals['train/lr'] = optim_g.param_groups[0]['lr']
            vals['train/nonfinite_logged'] = self.nonfinite
            health, self.disc_warn = self.disc_health(y_df_r + y_ds_r, y_df_g + y_ds_g, fm_f_r + fm_s_r, fm_f_g + fm_s_g)
            vals.update(health)
            scaler = getattr(self.trainer.precision_plugin, 'scaler', None)
            if scaler is not None:
                vals['train/amp_scale'] = float(scaler.get_scale())
            self.last_vals = vals
            if self.logger is not None and self.global_rank == 0:
                self.logger.log_metrics(vals, step=self.step)

    def on_validation_start(self):
        self._bench = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False

    def on_validation_end(self):
        torch.backends.cudnn.benchmark = self._bench

    def validation_step(self, batch, batch_idx):
        y, f0 = batch['audio'].float(), batch['f0'].float()
        mel = self.mel(y.squeeze(1))
        y_hat = self.generator(mel, f0).float()
        mel_hat = self.mel(y_hat.squeeze(1))[..., :mel.shape[-1]]
        l1 = (mel - mel_hat).abs().mean()
        self.log('val/mel_l1', l1, on_step=False, on_epoch=True, sync_dist=True, batch_size=1)
        if self.global_rank == 0 and batch_idx < 4 and self.logger is not None:
            tb = self.logger.experiment
            sr = self.h.sampling_rate
            tb.add_audio('val_%d/pred' % batch_idx, y_hat[0, 0].cpu().numpy(), self.step, sample_rate=sr)
            if self.step <= self.trainer.val_check_interval:
                tb.add_audio('val_%d/gt' % batch_idx, y[0, 0].cpu().numpy(), self.step, sample_rate=sr)


class StepTasks(Callback):
    """Console log, periodic checkpoint + generator export."""

    def __init__(self, run_dir, ckpt_every, keep=3, print_every=100, max_steps=0, probe_steps=0):
        self.run_dir, self.ckpt_every, self.keep = run_dir, ckpt_every, keep
        self.print_every, self.max_steps, self.probe_steps = print_every, max_steps, probe_steps
        self.last_ckpt = -1
        self.t_last, self.s_last = None, None
        self.t_warm = None

    def on_train_start(self, trainer, pl_module):
        self.last_ckpt = pl_module.step
        self.t_last, self.s_last = time.time(), pl_module.step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = pl_module.step
        if self.probe_steps:                      # timing / memory probe: no checkpoints, stop after probe_steps
            if step == 50:
                torch.cuda.synchronize(); self.t_warm = time.time()
            if step >= self.probe_steps:
                torch.cuda.synchronize()
                res = {'rank': trainer.global_rank, 'steps': step, 'it_s': (step - 50) / (time.time() - self.t_warm),
                       'max_alloc_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                       'last': getattr(pl_module, 'last_vals', {})}
                with open(os.path.join(self.run_dir, 'probe_rank%d.json' % trainer.global_rank), 'w') as f:
                    json.dump(res, f, indent=1)
                trainer.should_stop = True
            return
        if trainer.is_global_zero and step % self.print_every == 0 and step != self.s_last:
            t = time.time()
            ips = (step - self.s_last) / max(t - self.t_last, 1e-6)
            self.t_last, self.s_last = t, step
            v = getattr(pl_module, 'last_vals', {})
            reals = [v[k] for k in v if k.startswith('disc/') and k.endswith('_real')] or [np.nan]
            feats = [v[k] for k in v if k.startswith('disc/') and k.endswith('_featmax')] or [np.nan]
            print('[step %d] ep %d it/s %.3f mem %.1fG | d %.3f g %.3f mel %.4f stft %.4f fm %.3f/%.3f adv %.3f/%.3f '
                  'lr %.3g scale %s nonfinite %d | D(real) %.2f..%.2f featmax %.3g' % (
                      step, trainer.current_epoch, ips, torch.cuda.max_memory_allocated() / 2 ** 30,
                      v.get('train/loss_d', np.nan), v.get('train/loss_g', np.nan), v.get('train/mel', np.nan),
                      v.get('train/stft', np.nan), v.get('train/fm_mpd', np.nan), v.get('train/fm_msd', np.nan),
                      v.get('train/adv_mpd', np.nan), v.get('train/adv_msd', np.nan), v.get('train/lr', np.nan),
                      v.get('train/amp_scale', '-'), v.get('train/nonfinite_logged', 0), min(reals), max(reals),
                      max(feats)), flush=True)
            if pl_module.disc_warn:
                print('[step %d] DISC-WARN %s' % (step, pl_module.disc_warn), flush=True)
        if self.ckpt_every and step % self.ckpt_every == 0 and step != self.last_ckpt:
            self.last_ckpt = step
            self.save(trainer, step)
        if self.max_steps and step >= self.max_steps:
            trainer.should_stop = True

    def save(self, trainer, step):
        cdir = os.path.join(self.run_dir, 'ckpt')
        path = os.path.join(cdir, 'step_%07d.ckpt' % step)
        trainer.save_checkpoint(path)
        if not trainer.is_global_zero:
            return
        for p in sorted(glob.glob(os.path.join(cdir, 'step_*.ckpt')))[:-self.keep]:
            os.remove(p)
        print('[ckpt] %s -> %s' % (path, export_generator(path)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--wav-root', required=True)
    ap.add_argument('--f0-root', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--batch', type=int, required=True, help='per GPU')
    ap.add_argument('--devices', type=int, default=1)
    ap.add_argument('--precision', default='32-true', help='32-true | 16-mixed')
    ap.add_argument('--tf32', action='store_true', help='let convolutions / matmul use TF32 (faster, 10-bit mantissa)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--resume', default=None, help='auto | path')
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--probe-steps', type=int, default=0, help='time N steps and stop; nothing is saved')
    ap.add_argument('--ckpt-every', type=int, default=5000)
    ap.add_argument('--val-every', type=int, default=5000)
    ap.add_argument('--val-n', type=int, default=200)
    ap.add_argument('--runs-dir', default='runs')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--log-every', type=int, default=50)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    torch.backends.cudnn.allow_tf32 = a.tf32
    h = json.load(open(a.config))
    run_dir = os.path.join(a.runs_dir, a.run_name)
    os.makedirs(os.path.join(run_dir, 'ckpt'), exist_ok=True)
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if rank == 0 and not a.probe_steps:
        json.dump(h, open(os.path.join(run_dir, 'config.json'), 'w'), indent=2)
        with open(os.path.join(run_dir, 'cmdline.txt'), 'a') as f:
            f.write(time.strftime('%F %T ') + ' '.join(sys.argv) + '\n')
    pl.seed_everything(a.seed, workers=True)

    assert int(np.prod(h['upsample_rates'])) == h['hop_size'], (h['upsample_rates'], h['hop_size'])
    train_set = CropSet(read_rows(a.manifest, 'train'), a.wav_root, a.f0_root, h['segment_size'], h['hop_size'])
    train_loader = DataLoader(train_set, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = None
    if not a.probe_steps:
        val_loader = DataLoader(FullSet(val_subset(a.manifest, a.val_n), a.wav_root, a.f0_root, h['hop_size']),
                                batch_size=1, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    model = GAN(h)
    model.log_every = a.log_every
    logger = False if a.probe_steps else TensorBoardLogger(a.runs_dir, name=a.run_name, version='tb',
                                                           default_hp_metric=False)
    cb = StepTasks(run_dir, 0 if a.probe_steps else a.ckpt_every, max_steps=a.max_steps, probe_steps=a.probe_steps)
    trainer = pl.Trainer(
        accelerator='gpu', devices=a.devices, precision=a.precision,
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend='nccl') if a.devices > 1 else 'auto',
        max_epochs=-1, max_steps=-1, logger=logger, callbacks=[cb], enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=(rank == 0), benchmark=True,
        val_check_interval=a.val_every, check_val_every_n_epoch=None,
        limit_val_batches=0 if a.probe_steps else None, num_sanity_val_steps=0, log_every_n_steps=50,
        default_root_dir=run_dir,
    )
    ckpt = None
    if a.resume == 'auto':
        cs = sorted(glob.glob(os.path.join(run_dir, 'ckpt', 'step_*.ckpt')))
        ckpt = cs[-1] if cs else None
    elif a.resume:
        ckpt = a.resume
    if rank == 0:
        n_g = sum(p.numel() for p in model.generator.parameters())
        print('generator params %d, train clips %d, batch/gpu %d x %d gpus, precision %s, resume %s' % (
            n_g, len(train_set), a.batch, a.devices, a.precision, ckpt), flush=True)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt)


if __name__ == '__main__':
    main()
