"""NSF-HiFiGAN generator / discriminators.

Adapted from DiffSinger (openvpi) modules/nsf_hifigan/models.py (Apache-2.0), which builds on HiFi-GAN (MIT,
Copyright (c) 2020 Jungil Kong). See NOTICE.


Generator parameter names follow DiffSinger's modules/nsf_hifigan/models.py (conv_pre, ups.i, resblocks.k,
noise_convs.i, m_source.l_linear, conv_post; weight_norm weight_g / weight_v) so exported state dicts load there.

Differences from the upstream generator:
- per-stage channel widths come from h.upsample_channels ([conv_pre_out, stage_1, ..., stage_n]) when present;
- the harmonic count of the source module comes from h.harmonic_num (overtones above f0);
- harmonics whose frequency k*f0 reaches sr/2 are zeroed per sample before the merge layer (h.nyquist_mask).
f0 input is frame level [B, T] in Hz; it is held over each hop and integrated with a continuous phase.

Discriminators: widths follow h.discriminator_channel_mult (1.0 = HiFi-GAN sizes). Activations are not passed through
nan_to_num: a non-finite value has to surface in the loss instead of being replaced by a finite one.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def stage_channels(h):
    """[conv_pre_out, stage_1_out, ..., stage_n_out]"""
    chs = h.get("upsample_channels")
    if chs:
        assert len(chs) == len(h.upsample_rates) + 1, chs
        return [int(c) for c in chs]
    c0 = h.upsample_initial_channel
    return [c0 // (2 ** i) for i in range(len(h.upsample_rates) + 1)]


class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation[:3]
        ])
        self.convs1.apply(init_weights)
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in range(3)
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation[:2]
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class SineGen(nn.Module):
    """Harmonic sine source from frame-level f0.

    f0 [B, T, 1] Hz per frame -> sines [B, T*upp, dim] (dim = harmonic_num + 1).
    Each frame value is held for upp samples; phase is accumulated continuously across frames.
    """

    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0,
                 nyquist_mask=True):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.nyquist_mask = nyquist_mask

    def _f02sine(self, f0, upp):
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], -1, 1)
        rad = rad * torch.arange(1, self.dim + 1, device=f0.device).reshape(1, 1, -1)
        rand_ini = torch.rand(f0.shape[0], 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        return torch.sin(2 * np.pi * rad)

    def harmonic_mask(self, f0, upp):
        """f0 [B, T, 1] -> [B, T*upp, dim], 1 where k*f0 < sr/2."""
        k = torch.arange(1, self.dim + 1, device=f0.device, dtype=f0.dtype).reshape(1, 1, -1)
        m = (f0 * k < self.sampling_rate / 2).to(f0.dtype)
        return m.repeat_interleave(upp, dim=1)

    @torch.no_grad()
    def forward(self, f0, upp):
        f0 = f0.float().unsqueeze(-1)
        sine_waves = self._f02sine(f0, upp) * self.sine_amp
        if self.nyquist_mask:
            sine_waves = sine_waves * self.harmonic_mask(f0, upp)
        uv = (f0 > self.voiced_threshold).float()
        uv = uv.repeat_interleave(upp, dim=1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)
        return sine_waves * uv + noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0,
                 nyquist_mask=True):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold,
                                 nyquist_mask=nyquist_mask)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, f0, upp):
        sine_wavs = self.l_sin_gen(f0, upp)
        return self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.dtype)))


class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.upp = int(np.prod(h.upsample_rates))
        chs = stage_channels(h)
        self.m_source = SourceModuleHnNSF(sampling_rate=h.sampling_rate, harmonic_num=int(h.get("harmonic_num", 8)),
                                          nyquist_mask=bool(h.get("nyquist_mask", True)))
        self.noise_convs = nn.ModuleList()
        self.conv_pre = weight_norm(Conv1d(h.num_mels, chs[0], 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            c_in, c_out = chs[i], chs[i + 1]
            self.ups.append(weight_norm(ConvTranspose1d(c_in, c_out, k, u, padding=(k - u) // 2)))
            for kk, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(h, c_out, kk, d))
            if i + 1 < len(h.upsample_rates):
                stride_f0 = int(np.prod(h.upsample_rates[i + 1:]))
                self.noise_convs.append(Conv1d(1, c_out, kernel_size=stride_f0 * 2, stride=stride_f0,
                                               padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_out, kernel_size=1))
        self.conv_post = weight_norm(Conv1d(chs[-1], 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x, f0):
        """x: mel [B, num_mels, T]; f0: [B, T] Hz -> wav [B, 1, T*hop]"""
        har_source = self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x = x + self.noise_convs[i](har_source)
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def scaled(channels, mult):
    """discriminator widths scaled by h.discriminator_channel_mult, kept multiples of 16 (MSD group convs)"""
    return [max(16, int(round(c * mult / 16)) * 16) if c >= 64 else max(4, int(round(c * mult))) for c in channels]


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, ch_mult=1.0):
        super().__init__()
        self.period = period
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        pad = (get_padding(5, 1), 0)
        c = scaled([32, 128, 512, 1024, 1024], ch_mult)
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, c[0], (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(Conv2d(c[0], c[1], (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(Conv2d(c[1], c[2], (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(Conv2d(c[2], c[3], (kernel_size, 1), (stride, 1), padding=pad)),
            norm_f(Conv2d(c[3], c[4], (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(c[4], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=None, ch_mult=1.0):
        super().__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([DiscriminatorP(p, ch_mult=ch_mult) for p in self.periods])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False, ch_mult=1.0):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        c = scaled([128, 128, 256, 512, 1024, 1024, 1024], ch_mult)
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, c[0], 15, 1, padding=7)),
            norm_f(Conv1d(c[0], c[1], 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(c[1], c[2], 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(c[2], c[3], 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(c[3], c[4], 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(c[4], c[5], 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(c[5], c[6], 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(c[6], 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE, inplace=True)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return torch.flatten(x, 1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, ch_mult=1.0):
        super().__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=True, ch_mult=ch_mult),
                                             DiscriminatorS(ch_mult=ch_mult), DiscriminatorS(ch_mult=ch_mult)])
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r); fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g); fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.float() - gl.float()))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        loss += torch.mean((1 - dr.float()) ** 2) + torch.mean(dg.float() ** 2)
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg.float()) ** 2)
    return loss


def load_generator(ckpt_path, device='cpu'):
    """Load an exported generator: <dir>/model.ckpt ({'generator': state_dict}) + <dir>/config.json."""
    import os, json
    h = AttrDict(json.load(open(os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), 'config.json'))))
    g = Generator(h)
    g.load_state_dict(torch.load(ckpt_path, map_location='cpu')['generator'])
    g.remove_weight_norm()
    return g.to(device).eval(), h
