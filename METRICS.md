# Metrics: what we measured, and what the numbers do not say

Everything on this page measures **change remaining** or **representation
shift**, not protection efficacy. Nothing here tells you whether any model
copies, imitates, or trains on a track any worse. We have not measured that,
we cannot measure it on a CPU VM in a few seconds, and we do not claim it.
Where a result says "barely matters", that is the result.

No human listening test has been run. Section 1 is objective proxies only.

Reproduce (needs `ffmpeg` with `libmp3lame`; the EnCodec rows additionally
need `pip install torch encodec`, CPU wheel is fine, ~90 MB of weights are
fetched once from Meta's public bucket):

```sh
python -m music_shield audibility   --clip busy            # section 1, any built-in clip or your own WAV/FLAC
python -m music_shield ai-metrics   --clip busy            # section 2
python -m music_shield codec-metrics --clip busy --codecs mp3,aac   # section 3
```

Clips are synthetic and free of third-party material (`music_shield/synth.py`):

- `busy` — 8 s, 44.1 kHz stereo, saw chords to 16 kHz, gated bass, kick/snare/hat
  at 120 BPM. Dense and broadband: the honest codec stress case.
- `demo` — the 5 s in-app demo chord. Sparse-ish, with a quiet noise pad.
- `sparse` — 8 s, quiet (peak -18 dBFS) solo plucked notes with gaps. The
  audibility risk case: almost nothing to hide a perturbation under.
- `sustained` — 8 s steady organ-like chord over an exposed 41 Hz sub-bass, no
  transients, nothing above 3 kHz. Where a slow gain/phase wobble has nowhere
  to hide in time, and where any added high band would be added to silence.

Seed 1234, ffmpeg 6.1.1, libmp3lame CBR, ffmpeg native `aac`, EnCodec 24 kHz.

## 1. Listening notes (objective proxies, no human panel)

### What is measured

For `protected - original` on each clip and preset (`music_shield/audibility.py`):

| proxy | meaning |
| --- | --- |
| SNR | global signal-to-perturbation ratio |
| seg-SNR p05 | 5th percentile of 50 ms segmental SNR over active segments: the worst passages |
| additive NMR mean / p99 | noise-to-mask ratio of the *added* components (masked noise + high band) against the engine's own simplified masking estimate. > 0 dB is above that estimate. Computed on the additive part only because a -30 dB multiplicative change *of* a partial is not noise beside it and reads as a false violation |
| cells > mask | % of active STFT cells whose additive residual is above the mask estimate |
| residual dBFS | absolute RMS level of the residual in <500 Hz / 500 Hz-4 kHz / 4-15 kHz / >15 kHz. What matters when a band was empty in the original |
| log-mel mean / p99 | mean and 99th percentile of the 64-band log-mel deviation (0-15 kHz, cells within 60 dB of the frame's loudest band). p99 is the largest short-term band-level step, the closest thing here to "would you hear the EQ wobble" |
| side residual | residual of L-R relative to the original L-R: stereo image stability |
| risk | a **heuristic** label from fixed thresholds chosen before tuning (likely fine: seg-SNR ≥ 22 dB, NMR p99 ≤ 0 dB, log-mel p99 ≤ 1 dB; possibly audible: ≥ 15 / ≤ 6 / ≤ 2; else likely audible). It is not a listening result |

### Results (current presets)

| clip | preset | SNR | seg-SNR p05 | additive NMR mean / p99 | cells > mask | residual dBFS low / mid / high / >15k | log-mel mean / p99 | side residual | risk (heuristic) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| busy | **light** | 29.0 dB | 24.6 dB | -23.6 / -10.9 dB | 0.0 % | -45 / -63 / -70 / -81 | 0.20 / 0.55 dB | -28.3 dB | likely fine |
| busy | medium | 20.5 dB | 16.3 dB | -21.6 / -8.9 dB | 0.0 % | -37 / -55 / -63 / -74 | 0.42 / 1.17 dB | -21.0 dB | possibly audible |
| busy | strong | 17.1 dB | 12.4 dB | -17.6 / -4.9 dB | 0.0 % | -33 / -51 / -59 / -69 | 0.70 / 1.97 dB | -17.2 dB | likely audible |
| demo | **light** | 28.7 dB | 26.2 dB | -23.8 / -6.6 dB | 0.6 % | -48 / -65 / -84 / -111 | 0.21 / 0.59 dB | -23.2 dB | likely fine |
| demo | medium | 21.5 dB | 17.6 dB | -21.8 / -4.6 dB | 0.7 % | -41 / -60 / -78 / -109 | 0.44 / 1.16 dB | -18.4 dB | possibly audible |
| demo | strong | 17.8 dB | 14.4 dB | -17.8 / -0.6 dB | 0.9 % | -37 / -53 / -72 / -101 | 0.74 / 2.06 dB | -15.3 dB | likely audible |
| sparse | **light** | 28.9 dB | 25.7 dB | -20.6 / +3.0 dB | 2.2 % | -61 / -66 / -128 / -135 | 0.34 / 2.88 dB | -26.4 dB | likely audible |
| sparse | medium | 22.8 dB | 17.7 dB | -18.6 / +5.0 dB | 3.1 % | -55 / -59 / -122 / -130 | 0.58 / 3.75 dB | -21.6 dB | likely audible |
| sparse | strong | 16.6 dB | 12.1 dB | -14.6 / +9.0 dB | 6.4 % | -48 / -55 / -120 / -128 | 1.00 / 6.34 dB | -15.8 dB | likely audible |
| sustained | **light** | 28.2 dB | 25.0 dB | -21.4 / -2.2 dB | 0.6 % | -45 / -64 / -159 / -167 | 0.21 / 1.43 dB | -25.3 dB | possibly audible |
| sustained | medium | 20.5 dB | 17.6 dB | -19.4 / -0.2 dB | 0.9 % | -37 / -56 / -133 / -140 | 0.47 / 2.02 dB | -19.3 dB | likely audible |
| sustained | strong | 16.6 dB | 13.5 dB | -15.4 / +3.8 dB | 2.6 % | -34 / -52 / -127 / -134 | 0.79 / 4.29 dB | -15.9 dB | likely audible |

RMS level change is within ±0.1 dB and crest-factor change within 0.2 dB at
`light` on every clip (1.75 dB crest change at `strong` on `sparse`).

### Reading it honestly

- **`light` on dense or ordinary mixed material (`busy`, `demo`) passes every
  proxy** with margin: worst 50 ms segment still 24-26 dB SNR, the added noise
  is 11 dB (busy) / 7 dB (demo) under the mask estimate at the 99th percentile,
  the largest band-level step is 0.55-0.59 dB (level JND for a slow change in
  one band is about 1 dB), and the stereo image is stable (jitter and phase
  are identical on both channels; only per-channel noise moves L-R, at -23 to
  -28 dB). We think this is likely transparent on normal playback. We have not
  verified it with ears.
- **`light` on steady tonal material (`sustained`) is at the edge.** The
  additive part is fine (NMR p99 -2 dB, nothing added above 3 kHz, the empty
  high bands stay at -159 dBFS), but the log-mel p99 of 1.43 dB is the gain
  wobble itself with nothing in time to hide behind. A slow ±0.6 dB EQ wobble
  on a held organ chord may be noticeable on headphones to an attentive
  listener. Nothing about the additive fixes below changes this; only a
  smaller `jitter_db` would, and that is the component that carries the
  friction (section 2).
- **`light` on quiet sparse solo material (`sparse`) is flagged, and we are
  leaving it flagged.** After the fixes, 2.2 % of active cells still carry
  added noise above the mask estimate (p99 +3 dB), diffusely, in cells more
  than 40 dB below the peak during note decays. Whether that is audible
  depends on whether the crude 32-band, band-uniform masking model is
  conservative or generous in exactly those cells, and we do not know. The
  absolute level of the residual there is -61 to -66 dBFS on a clip peaking
  at -18 dBFS. If you protect solo or ambient material, listen to the
  `light` output on headphones before publishing it, or do not protect it.
- **`medium` and `strong` are expected to be audible** on headphones or on
  anything sparse, and now fail the "likely fine" thresholds on every clip.
  Their descriptions in the UI say so.

### What the proxies changed in the engine

Three problems the proxies exposed, all in the *additive* components (which
section 2 shows contribute almost nothing to friction), fixed in this branch:

| finding | before | after |
| --- | --- | --- |
| The >15 kHz component was calibrated to the track's overall RMS and ignored the mask, so it added energy where the track had none (`sustained`: -76 dBFS of hiss into an empty band at `light`; -60 dBFS at `strong`) and was the source of essentially all above-mask cells on `busy` (additive NMR p99 +10 dB, 7.9 % of cells) | -76 dBFS / +10 dB p99 | It is now ordinary masked noise above 15 kHz at a smaller offset (-10 dB at `light`). Empty bands stay empty (-167 dBFS); `busy` additive NMR p99 -10.9 dB, 0.0 % of cells; `hf_component_applied` is honestly `false` on material with nothing up there |
| Neighbour spreading in the masking model was -10 dB in both directions; real masking spreads roughly -25 dB/Bark upward and steeper downward, so noise leaked into empty bands next to loud ones | -10 / -10 dB | -18 dB upward / -27 dB downward (still generous). `sparse` additive NMR p99 +8.4 → +3.0 dB |
| Pre-echo: a frame's noise is synthesised over its whole 46 ms window, so an attack near the window end got noise ~40 ms *before* it, in the quiet, where backward masking (~5-20 ms) cannot hide it. Frames just before onsets on `sparse` had 10-25 % of cells above the mask | 10-25 % at onsets | Backward running-minimum over the 3 earlier overlapping frames on the masking curve. Cells above mask at offsets -1..+1 around onsets: 0 %. `test_noise_does_not_precede_an_attack_from_silence` asserts noise before an attack from silence is ≥ 40 dB under the noise after it |

The jitter and phase-drift components — the ones that survive codecs and move
model representations — were **not changed at `light`**. Section 3's `light`
numbers are therefore unchanged to within 0.2 dB.

## 2. Copy-friction pipelines (open-source / free, CPU)

### Method

Free, local pipelines a copier or an ML preprocessor might apply, each run on
the original and on the protected file (`music_shield/ai_eval.py`):

| pipeline | what it is | why |
| --- | --- | --- |
| `mp3-denoise` | MP3 128 kbps → ffmpeg `afftdn` FFT denoiser with noise-floor tracking → decode | a crude "clean up the copy" chain |
| `resample16k` | 44.1 kHz → 16 kHz → 44.1 kHz (ffmpeg) | the first step of many ML feature pipelines; removes everything above 8 kHz |
| `encodec-6k`, `encodec-24k` | Meta's **EnCodec** 24 kHz neural codec (MIT licence), 6 and 24 kbps, mono | an actual open model of the class that codec-token generative music models (MusicGen uses a 32 kHz EnCodec variant) consume. We report its re-synthesis *and* what such a model would see: the fraction of RVQ tokens identical to the original's, and the cosine / relative-L2 shift of the continuous encoder latent |

For each: `retained` = least-squares fraction of our change still present
after the pipeline; `corr` = how much of the post-pipeline difference is our
change; `control` = distance(pipeline(original), original), i.e. what the
pipeline alone does; `protected` = distance(pipeline(protected), original).
The EnCodec token/latent **control** is a plain MP3 128k round-trip of the
original, so our shift can be compared with an everyday re-encode.

### Results

`busy` clip:

| pipeline | preset | change added | retained | corr | after: P(prot) vs P(orig) | control: P(orig) vs orig | P(prot) vs orig | tokens same (all / codebook 0) | latent cos / rel-L2 | MP3 control: tokens same | MP3 control: latent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mp3-denoise | light | -29.0 dB | 94 % | 0.57 | -24.1 dB | -23.1 dB / 1.83 dB | -21.8 dB / 1.86 dB | | | | |
| mp3-denoise | medium | -20.5 dB | 96 % | 0.87 | -19.2 dB | -23.1 dB / 1.83 dB | -18.8 dB / 1.97 dB | | | | |
| mp3-denoise | strong | -17.1 dB | 94 % | 0.93 | -16.5 dB | -23.1 dB / 1.83 dB | -16.5 dB / 2.17 dB | | | | |
| resample16k | light | -29.0 dB | 100 % | 1.00 | -29.0 dB | -28.9 dB / 13.09 dB | -25.9 dB / 13.26 dB | | | | |
| resample16k | strong | -17.1 dB | 100 % | 1.00 | -17.1 dB | -28.9 dB / 13.09 dB | -16.8 dB / 13.65 dB | | | | |
| encodec-6k | light | -29.0 dB | 95 % | 0.18 | -14.7 dB | -13.4 dB / 1.42 dB | -13.2 dB / 1.46 dB | 67.2 % / 95.0 % | 1.0000 / 0.010 | 35.2 % / 81.8 % | 0.9993 / 0.037 |
| encodec-6k | medium | -20.5 dB | 91 % | 0.35 | -12.2 dB | -13.4 dB / 1.42 dB | -12.6 dB / 1.50 dB | 42.7 % / 84.8 % | 0.9998 / 0.023 | 35.2 % / 81.8 % | 0.9993 / 0.037 |
| encodec-6k | strong | -17.1 dB | 99 % | 0.51 | -11.3 dB | -13.4 dB / 1.42 dB | -11.9 dB / 1.54 dB | 31.2 % / 80.7 % | 0.9994 / 0.037 | 35.2 % / 81.8 % | 0.9993 / 0.037 |
| encodec-24k | light | -29.0 dB | 103 % | 0.35 | -19.7 dB | -17.9 dB / 0.97 dB | -17.6 dB / 1.00 dB | 23.1 % / 95.0 % | 1.0000 / 0.010 | 9.3 % / 81.8 % | 0.9993 / 0.037 |
| encodec-24k | strong | -17.1 dB | 98 % | 0.81 | -15.3 dB | -17.9 dB / 0.97 dB | -14.5 dB / 1.21 dB | 8.3 % / 80.7 % | 0.9994 / 0.037 | 9.3 % / 81.8 % | 0.9993 / 0.037 |

`demo` and `sparse`, `light` only:

| clip | pipeline | retained | corr | control P(orig) vs orig | P(prot) vs orig | tokens same (all / cb0) | latent cos / rel-L2 | MP3 control tokens | MP3 control latent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| demo | mp3-denoise | 95 % | 0.97 | -23.7 dB / 5.18 dB | -22.6 dB / 5.27 dB | | | | |
| demo | resample16k | 100 % | 1.00 | -59.3 dB / 0.13 dB | -28.7 dB / 0.34 dB | | | | |
| demo | encodec-6k | 97 % | 0.26 | -15.9 dB / 2.63 dB | -15.7 dB / 2.69 dB | 70.7 % / 95.5 % | 1.0000 / 0.007 | 61.2 % / 92.8 % | 1.0000 / 0.010 |
| demo | encodec-24k | 100 % | 0.65 | -21.7 dB / 2.02 dB | -21.5 dB / 2.04 dB | 26.8 % / 95.5 % | 1.0000 / 0.007 | 20.7 % / 92.8 % | 1.0000 / 0.010 |
| sparse | mp3-denoise | 95 % | 1.00 | -25.7 dB / 2.63 dB | -25.6 dB / 2.70 dB | | | | |
| sparse | resample16k | 100 % | 1.00 | -91.6 dB / 0.00 dB | -28.9 dB / 0.34 dB | | | | |
| sparse | encodec-6k | 93 % | 0.33 | -15.1 dB / 4.58 dB | -14.9 dB / 4.61 dB | 81.3 % / 97.3 % | 1.0000 / 0.004 | 73.7 % / 91.8 % | 1.0000 / 0.006 |
| sparse | encodec-24k | 100 % | 0.66 | -19.0 dB / 3.78 dB | -18.5 dB / 3.75 dB | 38.5 % / 97.3 % | 1.0000 / 0.004 | 31.3 % / 91.8 % | 1.0000 / 0.006 |

Cells `x dB / y dB` are residual dB / mean |log-mel difference| (0-15 kHz;
to 12 kHz for EnCodec at 24 kHz). The `resample16k` control log-mel of 13 dB
on `busy` is the 8-15 kHz region being wiped; on `demo`/`sparse` nothing
lives up there so the control is ~0.

### What this says, including the parts that barely matter

- **The change survives all four pipelines**: 91-103 % retained everywhere
  (100 % through resampling — the change lives below 8 kHz; 94-96 % through
  MP3+denoise — an FFT denoiser subtracts an estimated noise floor, and a
  multiplicative gain/phase wobble is not a noise floor). This is the same
  story as section 3: the perturbation is not *removed* by anything cheap.
- **But every pipeline's own error is larger than the perturbation.** EnCodec
  at 6 kbps reconstructs `busy` to -13.4 dB; our whole `light` change is
  -29 dB, 15 dB quieter. The protected file's processed copy ends up
  0.02-0.06 dB (log-mel) further from the master than the unprotected file's
  processed copy at `light`, 0.1-0.3 dB at `strong`. If "copy friction" means
  "the cleaned copy is worse", the honest answer at `light` is: barely.
- **A codec-token model sees less change from `light` than from an MP3.**
  Through EnCodec-6k on `busy`, `light` leaves 67 % of all tokens and 95 % of
  first-codebook tokens identical, and moves the continuous latent by 1.0 %
  (cosine 1.0000). A plain 128 kbps MP3 of the *original* leaves only 35 % /
  82 % of tokens identical and moves the latent by 3.7 %. So for a pipeline
  that tokenises audio with a neural codec, the default preset is a smaller
  disturbance than everyday lossy distribution already is. **Read that as
  "barely matters" for that class of model.** `medium` (43 % / 85 %, 2.3 %)
  is comparable to an MP3; `strong` (31 % / 81 %, 3.7 %) is about equal to
  one. None of them is more than an MP3.
- **Which component does the moving:** at `light` on `busy` through
  EnCodec-6k, jitter alone gives 71 % / 95 % tokens same and latent rel-L2
  0.009; phase drift alone 81 % / 97 % and 0.004; masked noise + high band
  alone 91 % / 99 % and 0.002. Jitter is the only component that matters for
  representation shift; noise and the high band are essentially invisible to
  the model *and* are the only audibility risk. That is the basis for the
  `medium`/`strong` retune in section 3.
- **Not measured, and still the thing that would matter:** whether a model
  trained or prompted on protected audio copies it any worse. Higher
  codebooks flipping (the "all tokens" number) is plausibly noise the model
  ignores; first-codebook tokens (coarse structure) barely move at `light`.
  We will not extrapolate from either number to an efficacy claim.

## 3. Codec round-trip (MP3 / AAC)

### Method

```
original ─── protect ───▶ protected
    │                        │
    ▼ ffmpeg encode/decode   ▼ ffmpeg encode/decode
codec(original)         codec(protected)
```

Decoded audio is aligned to the input by cross-correlation (measured lag 0).
Four comparisons: **A** protected vs original (the change we added); **B**
codec(original) vs original (codec-only control); **C** codec(protected) vs
original; **D** codec(protected) vs codec(original) (change still there).
Per comparison: **residual dB** = `10·log10(Σ(cand-ref)² / Σ ref²)` (= -SNR),
and **log-mel dB** = mean |Δ dB| over a 64-band mel log-spectrogram, 0-15 kHz,
cells within 60 dB of the frame's loudest band. **retained** = `<D,A>/<A,A>`
(least-squares fraction of our specific change still present; codec noise is
uncorrelated with A so it is not inflated by re-randomised quantisation);
**corr** = correlation of D with A.

### Results

`busy` clip:

| preset | codec | SNR | A: protected vs orig | B: codec(orig) vs orig | C: codec(prot) vs orig | D: codec(prot) vs codec(orig) | retained | corr | D log-mel low / mid / high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **light** | mp3 192k | 29.0 dB | -29.0 dB / 0.20 dB | -27.9 dB / 0.36 dB | -25.1 dB / 0.43 dB | -26.8 dB / 0.38 dB | 96 % | 0.77 | 0.23 / 0.36 / 0.48 dB |
| **light** | mp3 128k | 29.0 dB | -29.0 dB / 0.20 dB | -23.2 dB / 0.65 dB | -21.9 dB / 0.68 dB | -23.7 dB / 0.57 dB | 95 % | 0.54 | 0.31 / 0.57 / 0.68 dB |
| **light** | aac 192k | 29.0 dB | -29.0 dB / 0.20 dB | -27.8 dB / 0.37 dB | -24.7 dB / 0.45 dB | -24.1 dB / 0.49 dB | 100 % | 0.58 | 0.29 / 0.47 / 0.59 dB |
| **light** | aac 128k | 29.0 dB | -29.0 dB / 0.20 dB | -23.7 dB / 0.62 dB | -22.6 dB / 0.68 dB | -22.0 dB / 0.59 dB | 97 % | 0.44 | 0.36 / 0.60 / 0.68 dB |
| medium | mp3 192k | 20.5 dB | -20.5 dB / 0.42 dB | -27.9 dB / 0.36 dB | -19.9 dB / 0.57 dB | -20.0 dB / 0.56 dB | 97 % | 0.95 | 0.46 / 0.54 / 0.64 dB |
| medium | mp3 128k | 20.5 dB | -20.5 dB / 0.42 dB | -23.2 dB / 0.65 dB | -18.8 dB / 0.78 dB | -19.0 dB / 0.72 dB | 96 % | 0.85 | 0.50 / 0.70 / 0.83 dB |
| medium | aac 128k | 20.5 dB | -20.5 dB / 0.42 dB | -23.7 dB / 0.62 dB | -18.8 dB / 0.79 dB | -18.5 dB / 0.76 dB | 98 % | 0.78 | 0.55 / 0.77 / 0.83 dB |
| strong | mp3 192k | 17.1 dB | -17.1 dB / 0.70 dB | -27.9 dB / 0.36 dB | -17.0 dB / 0.79 dB | -16.9 dB / 0.79 dB | 97 % | 0.97 | 0.75 / 0.76 / 0.85 dB |
| strong | mp3 128k | 17.1 dB | -17.1 dB / 0.70 dB | -23.2 dB / 0.65 dB | -16.5 dB / 0.96 dB | -16.4 dB / 0.94 dB | 94 % | 0.92 | 0.77 / 0.90 / 1.06 dB |
| strong | aac 128k | 17.1 dB | -17.1 dB / 0.70 dB | -23.7 dB / 0.62 dB | -16.3 dB / 0.97 dB | -15.9 dB / 0.97 dB | 100 % | 0.88 | 0.81 / 0.97 / 1.03 dB |

`demo` clip:

| preset | codec | SNR | A | B | C | D | retained | corr | D log-mel low / mid / high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **light** | mp3 192k | 28.7 dB | -28.7 dB / 0.21 dB | -30.4 dB / 0.26 dB | -26.6 dB / 0.37 dB | -28.7 dB / 0.23 dB | 97 % | 1.00 | 0.26 / 0.22 / 0.24 dB |
| **light** | mp3 128k | 28.7 dB | -28.7 dB / 0.21 dB | -25.9 dB / 0.47 dB | -24.2 dB / 0.55 dB | -28.4 dB / 0.35 dB | 95 % | 0.97 | 0.26 / 0.37 / 0.42 dB |
| **light** | aac 128k | 28.7 dB | -28.7 dB / 0.21 dB | -27.6 dB / 0.30 dB | -25.0 dB / 0.39 dB | -23.3 dB / 0.41 dB | 98 % | 0.53 | 0.43 / 0.42 / 0.39 dB |
| medium | mp3 128k | 21.5 dB | -21.5 dB / 0.44 dB | -25.9 dB / 0.47 dB | -20.3 dB / 0.65 dB | -21.4 dB / 0.54 dB | 95 % | 0.99 | 0.48 / 0.56 / 0.55 dB |
| strong | mp3 128k | 17.8 dB | -17.8 dB / 0.74 dB | -25.9 dB / 0.47 dB | -17.2 dB / 0.87 dB | -17.7 dB / 0.80 dB | 95 % | 1.00 | 0.72 / 0.84 / 0.79 dB |

Cells are `residual dB / mean |log-mel difference| dB (0-15 kHz)`.

### Preset tune: before / after this branch

`light` modulation is unchanged; its SNR moved 28.8 → 29.0 dB because the
additive components got slightly smaller (section 1). `medium` and `strong`
had weight moved from masked noise (invisible to models, the only hiss risk)
to jitter and phase drift (what survives and what models see):

| preset | knobs before → after (noise offset / jitter / phase / HF) | SNR before → after (`busy`) | MP3 128k D before → after | retained / corr before → after | additive NMR p99 before → after | log-mel p99 before → after |
| --- | --- | --- | --- | --- | --- | --- |
| light | -14 / ±0.6 dB / ±3° / RMS -56 dB → -14 / ±0.6 dB / ±3° / mask -10 dB | 28.8 → 29.0 dB | -23.7 → -23.7 dB | 94 % / 0.55 → 95 % / 0.54 | +10.3 → -10.9 dB | 0.55 → 0.55 dB |
| medium | -8 / ±1.0 dB / ±5° / RMS -48 dB → -12 / ±1.3 dB / ±8° / mask -8 dB | 23.3 → 20.5 dB | -21.1 → -19.0 dB | 95 % / 0.78 → 96 % / 0.85 | +18.3 → -8.9 dB | 0.92 → 1.17 dB |
| strong | -4 / ±1.8 dB / ±10° / RMS -40 dB → -8 / ±2.2 dB / ±14° / mask -4 dB | 19.0 → 17.1 dB | -18.1 → -16.4 dB | 94 % / 0.88 → 94 % / 0.92 | +26.3 → -4.9 dB | 1.62 → 1.97 dB |

So `medium` and `strong` now add *less* noise and *more* of the change that
persists and moves representations, at the cost of a somewhat larger
band-level wobble (log-mel p99). Both remain "expected audible" and are
labelled that way.

### Which components survive (`light`, `busy`)

| component alone | A: change added | MP3 192k retained / corr | MP3 128k retained / corr | A log-mel |
| --- | --- | --- | --- | --- |
| all four (light) | -29.0 dB | 96 % / 0.77 | 95 % / 0.54 | 0.20 dB |
| spectral jitter | -31.2 dB | 99 % / 0.69 | 96 % / 0.46 | 0.20 dB |
| phase drift | -33.5 dB | 96 % / 0.62 | 93 % / 0.39 | 0.00 dB |
| masked noise | -42.4 dB | 94 % / 0.28 | 89 % / 0.15 | 0.02 dB |
| high band (>15 kHz) | -78.5 dB | 39 % / 0.00 | 28 % / 0.00 | 0.00 dB |

Jitter is the only component that moves magnitude spectrograms; phase drift
carries much of the waveform-level change and none of the mel change. Masked
noise is retained in the least-squares sense but buried (corr 0.15). The high
band is now -78 dB and removed by the encoder low-pass; it is kept only
because it is free where the track has content up there, and it is gated off
where it does not.

## Floors used by the automated tests

| suite | check | floor | measured |
| --- | --- | --- | --- |
| `test_codec_roundtrip` | light D residual after MP3 192k/128k (busy / demo) | > -36 dB | -23.7 / -28.4 dB |
| `test_codec_roundtrip` | light retained fraction | ≥ 0.70 | 0.95 / 0.95 |
| `test_codec_roundtrip` | light SNR stays conservative | ≥ 25 dB | 29.0 / 28.7 dB |
| `test_codec_roundtrip` | >15 kHz-only control: retained | < 0.5 | 0.28 |
| `test_audibility` | light `busy` seg-SNR p05 | > 20 dB | 24.6 dB |
| `test_audibility` | light `busy` additive NMR p99 | < -3 dB | -10.9 dB |
| `test_audibility` | light `busy` log-mel p99 | < 1.0 dB | 0.55 dB |
| `test_audibility` | noise before an attack from silence vs after | ≥ 40 dB under | passes at `strong` |
| `test_audibility` | `sustained` (nothing >3 kHz): `hf_component_applied` | `False` | `False` |
| `test_ai_eval` | light `busy` retained through mp3-denoise / resample16k | ≥ 0.70 / ≥ 0.90 | 0.94 / 1.00 |
| `test_ai_eval` | light extra log-mel distance over the denoise control | < 0.5 dB | 0.03 dB |
| `test_ai_eval` (skipped without torch/encodec) | EnCodec-6k latent cosine, retained | > 0.99, > 0.5 | 1.0000, 0.95 |

Margins are for encoder-version drift. If you tune presets, re-run all three
commands and update this file and the floors together.

## What these numbers do not cover

Human listening. Real, licensed material (everything above is synthetic).
Source separation, remixing, re-mastering, pitch/tempo change, re-recording,
averaging multiple protected copies, an attacker who has read the source, and
— above all — whether any model actually copies protected audio any worse.
See LIMITS.md.
