# zen-vocoder-small

NSF-HiFiGAN singing-voice vocoder, trained from scratch. Checkpoint: step 400,000.

| | |
|---|---|
| Input | 128-bin log-mel + frame-level F0 (Hz) |
| Output | 44.1 kHz mono waveform |
| Mel | n_fft 2048, win 2048, hop 512, 40–16000 Hz, slaney, magnitude, `ln(clamp(x, 1e-5))`, DiffSinger framing (`center=False`, reflect pad `(n_fft-hop)/2`) |
| F0 | one value per frame, held over the hop with continuous phase; unvoiced frames interpolated |
| Generator | 4.92 M parameters, 18.9 MB (fp32) |
| Compute | 18.3 GMACs per second of audio |
| Speed | 4× real time on 8 CPU threads, 230× on one RTX 3080 (fp32, PyTorch, 10 s input) |

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
| Steps | 400,000 (one step = one discriminator update + one generator update) |
| Batch | 36 (3 GPUs × 12), random crops of 16,384 samples aligned to the hop |
| Optimizer | AdamW, lr 2e-4, betas (0.8, 0.99), for both generator and discriminators |
| Schedule | lr × 0.999 per epoch (2,214 steps) |
| Precision | 16-bit mixed; losses computed in fp32 |
| Augmentation | none |
| F0 | Praat autocorrelation (parselmouth), 40–2000 Hz, voicing threshold 0.6 |
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

## Results

Copy synthesis on 300 held-out clips from singers not seen in training. Mel and F0 are extracted from the recording and
fed back to the model. Lower is better.

| | n | mel L1 | HF L1 | MR-STFT | F0 RMSE (cent) | F0 gross (%) | V/UV error |
|---|---|---|---|---|---|---|---|
| All | 300 | 0.384 | 1.030 | 0.805 | 21.2 | 2.26 | 0.015 |
| Female | 176 | 0.383 | 1.086 | 0.824 | 21.5 | 1.65 | 0.016 |
| Male | 124 | 0.385 | 0.950 | 0.778 | 20.9 | 3.12 | 0.015 |
| Median F0 C3–C4 | 160 | 0.380 | 0.968 | 0.783 | 23.6 | 2.83 | 0.019 |
| Median F0 C4–C5 | 126 | 0.384 | 1.104 | 0.827 | 18.5 | 1.53 | 0.011 |
| Median F0 C5–C6 | 12 | 0.439 | 1.108 | 0.876 | 18.4 | 2.06 | 0.013 |

- mel L1: L1 between input and output log-mel (the 128 input bins).
- HF L1: L1 on the log-magnitude spectrum from 16 to 22.05 kHz, the band the model generates without input.
- MR-STFT: L1 on the log-magnitude STFT, mean of n_fft 512 / 1024 / 2048.
- F0 RMSE and gross (> 50 cent): on frames voiced in both recording and output. V/UV error: share of frames whose voicing differs.

## Limitations

- Fast pitch runs can sound buzzy. F0 is tracked with a 75 ms window and applied in 11.6 ms steps.
- F0 accuracy is lowest on low male voices (26 cent RMSE, 4.2 % gross on the lowest register group).
- Content above 16 kHz is generated, not reconstructed.
- Trained on clean dry vocals only. Noisy, reverberant or mixed input is out of scope.

## Use

Weights: [`zen-vocoder-small-400k.zip`](https://github.com/weiyangzen/zenmusic/releases/download/vocoder-small-400k/zen-vocoder-small-400k.zip)
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
python prep_f0.py --manifest data.tsv --wav-root wavs --f0-root f0
python train.py --config configs/small_n3_h17.json --manifest data.tsv --wav-root wavs --f0-root f0 \
    --run-name small_n3_h17 --batch 12 --devices 3
```

`data.tsv` is tab-separated with the columns `path` (wav path relative to `--wav-root`, 44.1 kHz) and `split`
(`train` / `val` / `test`).

| File | |
|---|---|
| `models.py` | generator, discriminators, GAN losses |
| `train.py` | training loop, auxiliary losses, checkpointing |
| `data.py` | mel framing, datasets |
| `f0.py`, `prep_f0.py` | F0 extraction |
| `export.py` | training checkpoint → `model.ckpt` + `config.json` |
| `infer.py` | copy synthesis |

## License

Code: Apache-2.0. Weights: CC BY 4.0.

## Credits

`models.py` is adapted from the NSF-HiFiGAN modules of [DiffSinger (openvpi)](https://github.com/openvpi/DiffSinger)
(Apache-2.0), which build on [HiFi-GAN](https://github.com/jik876/hifi-gan) (MIT). The training recipe follows
[fish-diffusion](https://github.com/fishaudio/fish-diffusion) (MIT). See [`NOTICE`](../NOTICE).
