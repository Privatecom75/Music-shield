# Metrics: how much change survives a lossy re-encode

Everything on this page measures **change remaining**, not protection
efficacy. A number here tells you how different the protected file still is
from the original after an MP3/AAC round-trip. It does **not** tell you
whether any model copies, imitates, or trains on the track any worse. We have
not measured that and do not claim it.

Reproduce with (needs `ffmpeg` with `libmp3lame`; AAC uses ffmpeg's native
encoder):

```sh
python -m music_shield codec-metrics                       # busy clip, all presets, MP3 192k + 128k
python -m music_shield codec-metrics --clip demo --codecs mp3,aac
python -m music_shield codec-metrics --clip my-track.wav --presets light --json
```

## Method

For one clip, preset, codec and bitrate:

```
original ─── protect ───▶ protected
    │                        │
    ▼ ffmpeg encode/decode   ▼ ffmpeg encode/decode
codec(original)         codec(protected)
```

Decoded audio is checked for sample alignment against the input by
cross-correlation (ffmpeg's MP3 and AAC paths are gapless-aware; measured lag
is 0 samples) so a codec delay can never be mistaken for perturbation.

Four comparisons are reported:

| | compares | meaning |
| --- | --- | --- |
| **A** | protected vs original | the change we added |
| **B** | codec(original) vs original | codec-only control: what the codec does to an unprotected file |
| **C** | codec(protected) vs original | what a listener/tool sees relative to the master — our change *and* the codec's error together |
| **D** | codec(protected) vs codec(original) | change that is still there after the codec |

Per comparison, two distances:

- **residual dB** = `10·log10( Σ(candidate − reference)² / Σ reference² )`, i.e.
  minus the SNR. Closer to 0 means a bigger difference. `-30 dB` means the
  difference has 1/1000 of the reference energy.
- **log-mel dB** = mean `|Δ dB|` over a 64-band mel log-spectrogram restricted to
  0–15 kHz (so encoder low-pass behaviour does not dominate), only over cells
  within 60 dB of the frame's loudest band. This is the closest proxy here
  to the features many audio models consume. Also split into low (<500 Hz),
  mid (500 Hz–4 kHz) and high (4–15 kHz).

And two survival numbers derived from A and D:

- **retained** = `<D, A> / <A, A>`: the least-squares fraction of the *specific*
  change we added that is still present after the codec. 100 % means the codec
  passed it through; 0 % means it removed it. Because codec noise is
  uncorrelated with A this is not inflated by the codec re-randomising its own
  quantisation error between the two encodes (a plain `RMS(D)/RMS(A)` is —
  it reads 190–270 % for AAC on the demo clip, which is meaningless). Values a
  few percent above 100 % are estimator noise when the codec error is much
  larger than A.
- **corr** = correlation of D with A: how much of the difference between the
  two decoded files is actually our change, as opposed to codec noise that
  happened to land differently. Low corr with high retained means "still
  there, but buried".

Clips are synthetic and free of third-party material (`music_shield/synth.py`):

- `busy` — 8 s, 44.1 kHz stereo: saw-like chords with harmonics to 16 kHz, gated
  bass, kick/snare/hat at 120 BPM. Broadband and dense, so libmp3lame is
  bit-constrained at 128 kbps and its quantisation noise sits near the masking
  threshold, as on real mixes. **This is the honest stress case.**
- `demo` — the 5 s in-app demo chord. Sparse; the encoder has bits to spare,
  so almost everything survives. Included because it is what the UI shows.

Seed 1234 throughout. ffmpeg 6.1.1, libmp3lame CBR, native `aac`. Measured on
this branch versus `main` at the same seed.

## Results: `light` preset (the default)

### `busy` clip

| engine | codec | SNR of change | A: protected vs orig | B: codec(orig) vs orig | C: codec(prot) vs orig | D: codec(prot) vs codec(orig) | retained | corr | D log-mel low / mid / high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before | mp3 192k | 33.2 dB | -33.2 dB / 0.15 dB | -27.9 dB / 0.36 dB | -27.2 dB / 0.39 dB | -28.8 dB / 0.36 dB | 94% | 0.58 | 0.17 / 0.33 / 0.46 dB |
| **after** | mp3 192k | 28.8 dB | -28.8 dB / 0.20 dB | -27.9 dB / 0.36 dB | -25.1 dB / 0.43 dB | -26.6 dB / 0.39 dB | 96% | 0.77 | 0.23 / 0.37 / 0.50 dB |
| before | mp3 128k | 33.2 dB | -33.2 dB / 0.15 dB | -23.2 dB / 0.65 dB | -23.0 dB / 0.67 dB | -24.9 dB / 0.54 dB | 89% | 0.36 | 0.26 / 0.54 / 0.66 dB |
| **after** | mp3 128k | 28.8 dB | -28.8 dB / 0.20 dB | -23.2 dB / 0.65 dB | -21.9 dB / 0.69 dB | -23.7 dB / 0.56 dB | 94% | 0.55 | 0.31 / 0.55 / 0.67 dB |
| before | aac 128k | 33.2 dB | -33.2 dB / 0.15 dB | -23.7 dB / 0.62 dB | -23.4 dB / 0.63 dB | -22.5 dB / 0.61 dB | 89% | 0.26 | 0.32 / 0.62 / 0.71 dB |
| **after** | aac 128k | 28.8 dB | -28.8 dB / 0.20 dB | -23.7 dB / 0.62 dB | -22.5 dB / 0.65 dB | -21.5 dB / 0.68 dB | 98% | 0.43 | 0.38 / 0.68 / 0.82 dB |

### `demo` clip

| engine | codec | SNR of change | A: protected vs orig | B: codec(orig) vs orig | C: codec(prot) vs orig | D: codec(prot) vs codec(orig) | retained | corr | D log-mel low / mid / high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before | mp3 192k | 33.3 dB | -33.3 dB / 0.16 dB | -30.4 dB / 0.26 dB | -28.2 dB / 0.33 dB | -33.2 dB / 0.23 dB | 95% | 0.98 | 0.17 / 0.24 / 0.25 dB |
| **after** | mp3 192k | 28.7 dB | -28.7 dB / 0.21 dB | -30.4 dB / 0.26 dB | -26.6 dB / 0.38 dB | -28.7 dB / 0.26 dB | 97% | 1.00 | 0.26 / 0.26 / 0.28 dB |
| before | mp3 128k | 33.3 dB | -33.3 dB / 0.16 dB | -25.9 dB / 0.47 dB | -24.8 dB / 0.52 dB | -32.3 dB / 0.37 dB | 91% | 0.85 | 0.18 / 0.42 / 0.51 dB |
| **after** | mp3 128k | 28.7 dB | -28.7 dB / 0.21 dB | -25.9 dB / 0.47 dB | -24.2 dB / 0.56 dB | -28.2 dB / 0.40 dB | 95% | 0.95 | 0.27 / 0.42 / 0.53 dB |
| before | aac 128k | 33.3 dB | -33.3 dB / 0.16 dB | -27.6 dB / 0.30 dB | -26.3 dB / 0.36 dB | -24.7 dB / 0.42 dB | 80% | 0.30 | 0.39 / 0.44 / 0.42 dB |
| **after** | aac 128k | 28.7 dB | -28.7 dB / 0.21 dB | -27.6 dB / 0.30 dB | -24.5 dB / 0.40 dB | -22.9 dB / 0.46 dB | 99% | 0.51 | 0.47 / 0.46 / 0.43 dB |

Cells are `residual dB / mean |log-mel difference| dB (0–15 kHz)`.

### What changed, in one line each

- **Retained change after MP3 128k on the busy clip:** A × retained went from
  about **-34 dB** (-33.2 dB × 89 %) to about **-29 dB** (-28.8 dB × 94 %)
  relative to the signal — roughly 5 dB more of our change is still in the
  decoded file, and it is now the majority of the difference between the two
  MP3s (corr 0.36 → 0.55) rather than a minority under codec noise.
- **Light SNR:** 33.2 → 28.8 dB. The extra change is *not* hiss: masked noise
  was reduced (-12 → -14 dB under the mask) and the >15 kHz component was cut
  by 8 dB. The extra energy is a slow ±0.6 dB (peak) gain wobble and a slow ±3°
  (peak) phase rotation of the track's own components, both co-located with
  the music that masks them. See "listening notes" below.
- **Magnitude spectrogram change is essentially unchanged:** log-mel distance
  0.15 → 0.20 dB before the codec, and D's log-mel is the same before and
  after (0.54 vs 0.56 dB at 128k). Most of the improvement is in the
  waveform / complex spectrum, not in mel features. Be clear about this when
  describing what the tool does.
- **The codec still changes the file more than we do.** At 128 kbps the
  codec-only error (B) is -23 dB on the busy clip; our whole light
  perturbation is -28.8 dB. An inaudible perturbation cannot be louder than a
  codec whose entire design goal is to be just inaudible. What we can do —
  and did — is make our change *persist* through the codec instead of being
  replaced by it.

## Which components survive

`light` preset values, `busy` clip, each component switched on alone:

| component (light values) | A: change added | MP3 192k retained / corr | MP3 128k retained / corr | A log-mel |
| --- | --- | --- | --- | --- |
| all four (light) | -28.8 dB | 96% / 0.77 | 94% / 0.55 | 0.20 dB |
| spectral jitter only | -31.2 dB | 99% / 0.69 | 96% / 0.46 | 0.20 dB |
| phase drift only | -33.5 dB | 96% / 0.62 | 93% / 0.39 | 0.00 dB |
| masked noise only | -39.9 dB | 93% / 0.35 | 89% / 0.20 | 0.02 dB |
| high-band only (>15 kHz) | -56.0 dB | 19% / 0.02 | 12% / 0.01 | 0.00 dB |

Reading this honestly:

- Jitter and phase drift are multiplicative on content the codec reproduces
  accurately, so they come through at 93–99 %. Jitter is the only component
  that moves the magnitude spectrogram at all.
- Masked noise is "retained" in the least-squares sense (the quantiser is
  unbiased on average) but with corr 0.20 at 128k it is buried under codec
  noise that is ~17 dB louder. It contributes almost nothing to spectrogram
  features (0.02 dB). It is kept small at `light` because it is also the only
  component that can sound like hiss.
- The >15 kHz component is removed by libmp3lame's low-pass (≈17 kHz at
  128k). It is now at -56 dB at `light` and is not counted on for anything
  after a re-encode. `test_high_band_only_perturbation_is_removed_by_mp3`
  keeps this as a negative control so the metric is shown to distinguish
  "survived" from "removed".

## `medium` and `strong`

`busy` clip, MP3 128k: medium SNR 26.9 → 23.3 dB, D residual -22.9 → -21.1 dB,
retained 90 → 95 %; strong SNR 22.4 → 19.0 dB, D residual -20.6 → -18.1 dB,
retained 86 → 94 %. Both are moderately stronger than before and both remain
more audible than `light`; nothing about the audibility caveats in LIMITS.md
has improved for them.

## Listening notes

No human listening test was run for this change (or for the original
presets). What follows is reasoning from the construction, plus objective
proxies, and should be read as such.

- The phase drift is a slow per-band phase rotation. Monaural hearing is
  nearly insensitive to it; broadcast "phase rotators" apply far larger
  rotations to speech routinely. The walks are Hann-smoothed; at `light` the
  drift rate averages ~3°/s (99th percentile 9°/s), i.e. a frequency
  deviation of well under 0.05 Hz. Group-delay dispersion across bands is at
  most ~0.2 ms in the bass and microseconds higher up, an order of magnitude
  under published detection thresholds. It is shared between channels, so the
  stereo image does not move (`test_modulation_is_shared_across_channels`).
- The gain jitter is ±0.6 dB peak (0.3 dB std) per knot, drifting over ~0.8 s
  (average slew ~0.5 dB/s). Level JND for a slow change in one band is around 1 dB. The knots are ≥150 Hz
  apart at the bottom so one low partial is not split across two knots.
- Both components are co-located in time-frequency with the content they
  modify, at about -30 dB relative to it, which is the most maskable kind of
  change there is. The reported SNR drop (33 → 29 dB) is real but describes
  this kind of change, not added noise.
- Masked noise (the hiss risk) went *down* at `light`. If the old light preset
  was acceptable on a track, the new one should not be more hissy.
- Sparse, quiet, or solo material remains the risk case for all presets, as
  before. If you hear anything, use `light` (the default) or don't protect
  that track.

## Floors used by the automated tests

`tests/test_codec_roundtrip.py` asserts, for `light` on both synthetic clips
after MP3 192k and 128k round-trips:

| check | floor | measured (busy / demo, 128k) |
| --- | --- | --- |
| D residual dB (decoded protected vs decoded original) | > -36 dB | -23.7 / -28.2 dB |
| retained fraction | ≥ 0.70 | 0.94 / 0.95 |
| light SNR (stays conservative) | ≥ 25 dB | 28.8 / 28.7 dB |
| decoded peak | < 1.0, finite | 0.79 / 0.58 |
| >15 kHz-only control: retained fraction | < 0.5 | 0.12 |

The margins are for encoder-version drift, not because the numbers are
marginal. If you tune presets, re-run `codec-metrics` and update this file and
the floors together.

## What these numbers do not cover

Denoising, source separation, remixing, re-mastering, pitch/tempo change,
re-recording, downsampling, averaging multiple protected copies, or an
attacker who has read the source. Every one of those still weakens or removes
the perturbation. See LIMITS.md.
