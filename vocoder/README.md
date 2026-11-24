# zen-vocoder-small

NSF-HiFiGAN singing-voice vocoder, trained from scratch. Checkpoint: step 150,000.

| | |
|---|---|
| Input | 128-bin log-mel + frame-level F0 (Hz) |
| Output | 44.1 kHz mono waveform |
| Mel | n_fft 2048, win 2048, hop 512, 40–16000 Hz, slaney, magnitude, `ln(clamp(x, 1e-5))`, DiffSinger framing (`center=False`, reflect pad `(n_fft-hop)/2`) |
| F0 | one value per frame, held over the hop with continuous phase; unvoiced frames interpolated |
| Generator | 4.92 M parameters (4.91 M after weight-norm removal), 18.9 MB (fp32) |
| Compute | 18.3 GMACs per second of audio |
| Speed | 4× real time on 8 CPU threads, 230× on one RTX 3080 (fp32, PyTorch, 10 s input) |

## Comparison

Copy synthesis on 300 held-out clips. Lower is better in every row; the best value of each row is in bold.

| | stable-v1 | pc-nsf-hifigan 2025.02 | zen-vocoder-small 150k |
|---|---|---|---|
| mel L1 | 0.313 | **0.258** | 0.334 |
| HF L1 | 1.121 | 1.149 | **1.002** |
| MR-STFT | 0.805 | 0.785 | **0.782** |
| F0 RMSE (cent) | 17.3 | 19.0 | **15.5** |
| F0 gross error (%) | 1.45 | 2.23 | **1.42** |
| Clips with F0 RMSE > 100 cent | 7 | 6 | **2** |
| V/UV error (%) | 2.3 | **1.2** | 1.5 |

| Model | Checkpoint | Generator parameters | Compute per second of audio | Hop |
|---|---|---|---|---|
| stable-v1 | fish-diffusion `nsf_hifigan-stable-v1` | 14.20 M | 28.1 GMACs | 512 |
| pc-nsf-hifigan 2025.02 | openvpi `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02` | 14.13 M | 27.9 GMACs | 512 |
| zen-vocoder-small 150k | this repository, release `vocoder-small-150k` | 4.91 M | 18.3 GMACs | 512 |

### Reconstruction loss of each model

The mel L1 row measures one specific quantity: the L1 on the 128-bin input mel (40–16000 Hz, n_fft 2048, hop 512,
DiffSinger framing). How much weight each model's training puts on that quantity differs:

| Model | Reconstruction loss used in training |
|---|---|
| pc-nsf-hifigan 2025.02 | openvpi SingingVocoders, `modules/loss/HiFiloss.py` with `configs/ft_hifigan.yaml`: 45 × L1 on the mel with this framing (n_fft 2048, hop 512, 128 bins); the STFT loss is switched off (`use_stftloss: false`). The training objective is the mel L1 row. |
| stable-v1 | fish-diffusion recipe: 45 × (0.5 × complex-STFT L1 at three resolutions + log-mel L1 at three scales). |
| zen-vocoder-small 150k | 45 × (0.5 × complex-STFT L1 at three resolutions + log-mel L1 at three scales over 0–22.05 kHz), see [Loss](#loss). None of the three mel scales uses the framing of the mel L1 row. |

The weighting between the mel term and the other terms will be revised in later checkpoints, with listening quality
as the target.

### How the numbers are computed

Test set: the first 300 clips of the test split (176 female, 124 male; singers not in the training set), 44.1 kHz mono.

For each clip and each model:

1. `x` = the recording, cut to a multiple of 512 samples; `T = len(x) / 512` frames.
2. `mel` = the mel of `x` as defined in the table at the top (`T` frames).
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

`pm_ac`: Praat autocorrelation (parselmouth `to_pitch_ac`), time step 512 / 44100 s, 40–2000 Hz, voicing threshold 0.6.
The generators add random noise to the harmonic source, so repeated runs differ in the third decimal.

## Model size

| | |
|---|---|
| Upsampling | 8 · 8 · 2 · 2 · 2 (= 512), transposed-conv kernels 16 · 16 · 4 · 4 · 4 |
| Channels | 256 → 128 → 96 → 64 → 32 → 16 |
| Residual blocks | ResBlock1, kernels 3 / 7 / 11, dilations 1 / 3 / 5, three per stage |
| Harmonic source | 17 sines (F0 + 16 overtones), amplitude 0.1; overtones at or above Nyquist are zeroed; merged by linear + tanh; injected at every stage |
| Discriminators (training only) | MPD, periods 3 / 5 / 7 / 11 / 17 / 23 / 37, and 3-scale MSD; 87 M parameters |

Full configuration: [`configs/small_n3_h17.json`](configs/small_n3_h17.json).

## Recipe

| | |
|---|---|
| Steps | 150,000 (one step = one discriminator update + one generator update) |
| Batch | 36 (3 GPUs × 12), random crops of 16,384 samples aligned to the hop |
| Optimizer | AdamW, lr 2e-4, betas (0.8, 0.99), for both generator and discriminators |
| Schedule | lr × 0.999 per epoch (2,214 steps) |
| Precision | 16-bit mixed; losses computed in fp32 |
| Augmentation | none |
| F0 | `pm_ac`, resampled to frame centres with plain linear interpolation (voiced values averaged with 0 at the edges of voiced segments) |
| Data | 80 h of voiced dry singing (24 h male, 56 h female), 44.1 kHz mono; singer-disjoint train / val / test. Not released. |

## Loss

Discriminators (LSGAN, summed over all sub-discriminators of MPD and MSD):

```
L_D = Σ mean((1 − D(y))²) + mean(D(ŷ)²)
```

Generator:

```
L_G = L_adv + L_fm + 45 · (0.5 · L_stft + L_mel)

L_adv  = Σ mean((1 − D(ŷ))²)
L_fm   = 2 · Σ mean|feat(y) − feat(ŷ)|                 every layer of every sub-discriminator
L_stft = mean over 3 resolutions of L1 on the complex STFT (real and imaginary parts)
         (n_fft, hop, win) = (512, 50, 240), (1024, 120, 600), (2048, 240, 1200)
L_mel  = mean over 3 scales of L1 on the log-mel, 128 bins over 0–22.05 kHz
         (n_fft, hop, win) = (2048, 512, 2048), (2048, 270, 1080), (4096, 540, 2160)
```

The discriminator gradients are cleared before every discriminator step.

Not used for this checkpoint: `discriminator_channel_mult` (discriminator width), `loss_mel_extra` (log-mel terms
`[n_fft, hop, win, n_mels, weight]` added to `L_mel`), `loss_input_mel_weight` (L1 on the input mel definition,
added to `L_mel`).

## F0

`f0.py` has two analyzers.

- `pm_ac`: Praat autocorrelation, 40–2000 Hz, voicing threshold 0.6; analysis window 75 ms at any hop.
- `pm_ac_stone`: voicing and octave decisions of `pm_ac`; each voiced frame refined with WORLD StoneMask; a frame
  that moves more than 300 cent from the `pm_ac` value keeps the `pm_ac` value.

Synthetic singing-like signals with exactly known F0 (`python f0_bench.py --hop 512`, 56 clips, 98–1100 Hz).
Cells: RMSE in cent / share of frames off by more than 50 cent.

| True pitch speed | `pm_ac` | `pm_ac_stone` |
|---|---|---|
| All frames | 20.4 / 3.3 % | 9.8 / 0.6 % |
| < 15 semitones/s | 11.4 / 0.8 % | 9.8 / 0.7 % |
| 15–50 | 17.6 / 1.9 % | 9.5 / 0.4 % |
| 50–100 | 33.1 / 10.0 % | 8.9 / 0.5 % |
| > 100 | 60.6 / 32.5 % | 11.9 / 0.8 % |

## Use

Weights: [`zen-vocoder-small-150k.zip`](https://github.com/weiyangzen/zenmusic/releases/download/vocoder-small-150k/zen-vocoder-small-150k.zip)
(`model.ckpt` + `config.json`).

```
pip install -r requirements.txt
python infer.py --ckpt model.ckpt --wav in.wav --out out.wav          # config.json next to model.ckpt
```

`model.ckpt` holds `{'generator': state_dict}` with DiffSinger's NSF-HiFiGAN parameter names. `config.json` carries
three keys the stock DiffSinger loader does not read (`upsample_channels`, `harmonic_num`, `nyquist_mask`); load the
checkpoint with `models.load_generator`.

## Train

```
python prep_f0.py --manifest data.tsv --wav-root wavs --f0-root f0            # --analyzer pm_ac_stone for the refined F0
python train.py --config configs/small_n3_h17.json --manifest data.tsv --wav-root wavs --f0-root f0 \
    --run-name small_n3_h17 --batch 12 --devices 3 --precision 16-mixed
```

Every 50 steps each sub-discriminator is checked (mean output on real and generated audio, output spread, largest
feature, feature-matching share); `DISC-WARN` is printed when one fails the check. Default precision is `32-true`.

`data.tsv` is tab-separated with the columns `path` (wav path relative to `--wav-root`, 44.1 kHz) and `split`
(`train` / `val` / `test`).

| File | |
|---|---|
| `models.py` | generator, discriminators, GAN losses |
| `train.py` | training loop, auxiliary losses, discriminator checks, checkpointing |
| `data.py` | mel framing, datasets |
| `f0.py`, `prep_f0.py` | F0 analyzers and extraction |
| `f0_bench.py` | F0 analyzers scored on synthetic signals with known F0 |
| `export.py` | training checkpoint → `model.ckpt` + `config.json` |
| `infer.py` | copy synthesis |

## License

Code: Apache-2.0. Weights: CC BY 4.0.

## Credits

`models.py` is adapted from the NSF-HiFiGAN modules of [DiffSinger (openvpi)](https://github.com/openvpi/DiffSinger)
(Apache-2.0), which build on [HiFi-GAN](https://github.com/jik876/hifi-gan) (MIT). The training recipe follows
[fish-diffusion](https://github.com/fishaudio/fish-diffusion) (MIT). See [`NOTICE`](../NOTICE).
