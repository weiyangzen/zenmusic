# zenmusic

Open singing-voice models.

| | |
|---|---|
| [`vocoder/`](vocoder/) | zen-vocoder-small: NSF-HiFiGAN, 44.1 kHz, hop 512, 4.92 M parameters |

## What's New

### 2026-11-24

`zen-vocoder-small`, step 150,000: NSF-HiFiGAN singing-voice vocoder, 44.1 kHz, hop 512, 4.92 M parameters.
[Weights](https://github.com/weiyangzen/zenmusic/releases/tag/vocoder-small-150k) ·
[Model card](vocoder/README.md)

#### Comparison with stable-v1 and pc-nsf-hifigan 2025.02

Copy synthesis on 300 held-out clips. Lower is better unless the row says otherwise; 👑 marks the best value of each row.

| | stable-v1 | pc-nsf-hifigan 2025.02 | zen-vocoder-small 150k |
|---|---|---|---|
| mel L1 | 0.313 | 👑 **0.258** | 0.334 |
| HF L1 | 1.121 | 1.149 | 👑 **1.002** |
| MR-STFT | 0.805 | 0.785 | 👑 **0.782** |
| F0 RMSE (cent) | 17.3 | 19.0 | 👑 **15.5** |
| F0 gross error (%) | 1.45 | 2.23 | 👑 **1.42** |
| Clips with F0 RMSE > 100 cent | 7 | 6 | 👑 **2** |
| V/UV error (%) | 2.3 | 👑 **1.2** | 1.5 |
| Level in quiet frames (dBFS) | −64.9 | −66.0 | 👑 **−68.0** |
| HNR deviation (dB) | 1.91 | 👑 **1.50** | 1.68 |
| HNR bias (dB, closest to 0 is best) | −0.36 | +0.48 | 👑 **+0.17** |

| Model | Checkpoint | Generator parameters | Compute per second of audio | Hop |
|---|---|---|---|---|
| stable-v1 | fish-diffusion `nsf_hifigan-stable-v1` | 14.20 M | 28.1 GMACs | 512 |
| pc-nsf-hifigan 2025.02 | openvpi `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` | 14.13 M | 27.9 GMACs | 512 |
| zen-vocoder-small 150k | this repository, release `vocoder-small-150k` | 4.91 M | 18.3 GMACs | 512 |

#### Reconstruction loss of each model

The mel L1 row measures one specific quantity: the L1 on the 128-bin input mel (40–16000 Hz, n_fft 2048, hop 512,
DiffSinger framing). How much weight each model's training puts on that quantity differs:

| Model | Reconstruction loss used in training |
|---|---|
| pc-nsf-hifigan 2025.02 | openvpi SingingVocoders, `modules/loss/HiFiloss.py` with `configs/ft_hifigan.yaml`: 45 × L1 on the mel with this framing (n_fft 2048, hop 512, 128 bins); the STFT loss is switched off (`use_stftloss: false`). |
| stable-v1 | fish-diffusion recipe: 45 × (0.5 × complex-STFT L1 at three resolutions + log-mel L1 at three scales). |
| zen-vocoder-small 150k | 45 × (0.5 × complex-STFT L1 at three resolutions + log-mel L1 at three scales over 0–22.05 kHz), see [Loss](vocoder/README.md#loss). None of the three mel scales uses the framing of the mel L1 row. |

The weighting between the mel term and the other terms will be revised in later checkpoints, with listening quality
as the target.

#### How the numbers are computed

Test set: the first 300 clips of the test split (176 female, 124 male; singers not in the training set), 44.1 kHz mono.

For each clip and each model:

1. `x` = the recording, cut to a multiple of 512 samples; `T = len(x) / 512` frames.
2. `mel` = the mel of `x` as defined in the [model card](vocoder/README.md) (`T` frames).
3. `f0` = `pm_ac` F0 of `x` at hop 512 (frame centres `512·t + 256`), unvoiced frames filled by linear interpolation.
4. `y = G(mel, f0)`, cut or zero-padded to `len(x)`. Each checkpoint is loaded with its own configuration; parameters are counted after weight-norm removal.
5. The metrics below are computed on `(x, y)`. A table entry is the mean over the 300 clips, except the clip count.

| Metric | Definition |
|---|---|
| mel L1 | `mean |M(x) − M(y)|` over all bins and frames. `M` = the input mel: 128 bins, 40–16000 Hz, n_fft 2048, hop 512, `ln(clamp(·, 1e-5))`. |
| HF L1 | `S(·) = ln(clamp(|STFT(·)|, 1e-5))`, n_fft 2048, hop 512, Hann window 2048, centred. `mean |S(x) − S(y)|` over the bins at or above 16 kHz and all frames. |
| MR-STFT | For n in 512, 1024, 2048: `S_n` as above with n_fft n, hop n/4, Hann window n; `mean |S_n(x) − S_n(y)|` over all bins and frames. The three values are averaged. |
| F0 RMSE (cent) | `f_x`, `f_y` = `pm_ac` F0 of `x` and of `y` at hop 512, 0 = unvoiced, no interpolation. On the frames voiced in both: `c = 1200·|log2(f_y / f_x)|`; clip value `sqrt(mean(c²))`. No outliers are removed. |
| F0 gross error (%) | Share of the frames voiced in both with `c > 50`. |
| Clips with F0 RMSE > 100 cent | Number of clips, out of 300, whose F0 RMSE exceeds 100 cent. |
| V/UV error (%) | Share of all frames where exactly one of `f_x`, `f_y` is voiced. |
| Level in quiet frames (dBFS) | Frames are consecutive blocks of 512 samples. A frame is quiet when the RMS of `x` in it is below −55 dBFS. Clip value: mean over the quiet frames of `20·log10(RMS of y in the frame)`. Mean over the 256 clips that have at least 3 quiet frames (8,711 frames). The recordings themselves are at −66.2 dBFS in these frames. Lower means the output is quieter where the recording is quiet. |
| HNR deviation (dB) | HNR = harmonics-to-noise ratio per frame, Praat harmonicity (cross-correlation), time step 10 ms, minimum pitch 75 Hz, computed on `x` and on `y`. Over the frames with `HNR(x) > 0 dB` in which `y` is voiced: `mean |HNR(y) − HNR(x)|`. |
| HNR bias (dB) | Same frames: `mean (HNR(y) − HNR(x))`. Positive: the output is more periodic than the recording; negative: noisier. |

`pm_ac`: Praat autocorrelation (parselmouth `to_pitch_ac`), time step 512 / 44100 s, 40–2000 Hz, voicing threshold 0.6.
The generators add random noise to the harmonic source, so repeated runs differ in the third decimal.

## License

Code: [Apache-2.0](LICENSE). Model weights: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
