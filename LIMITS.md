# Limits and failure modes

Read this before relying on Music Shield for anything. The short version:
it adds friction, and friction can be removed.

## What we are actually claiming

- The output differs from the input in ways that alter the waveform, the
  complex spectrum and (to a smaller degree) magnitude-spectrogram features,
  while trying to keep the change hard to hear at the default preset.
- After a typical MP3/AAC re-encode at 128–192 kbps, an MP3-plus-denoise
  chain, a 16 kHz resample, or a pass through Meta's open EnCodec neural
  codec, most of that change (91–103 %) is still present in the output. We
  measured this on synthetic clips, and only this; see
  [METRICS.md](METRICS.md).
- That's it. We are **not** claiming the output cannot be copied, cannot be
  used to train a model, or proves you own the track.

## What we are not claiming

- **Not AI-proof.** We have not measured how much this degrades any model's
  ability to copy, imitate, or train on a track. What we *did* measure
  (METRICS.md §2) cuts the other way: through EnCodec, the neural codec
  family that codec-token music models consume, the default `light` preset
  moves the model's tokens and latents **less than an ordinary 128 kbps MP3
  re-encode does** (67 % of tokens unchanged vs 35 % for the MP3; latent
  shift 1.0 % vs 3.7 %). For that class of pipeline the honest summary is
  "barely matters". We will not publish an efficacy number we have not
  measured, and you should be suspicious of anyone who does.
- **Not DRM.** Anyone with the file can play it, copy it, and redistribute it.
- **Not ownership proof.** There is no watermark or verifiable payload in v1.
- **Not permanent.** Every transform below can weaken or remove it.

## Known ways to weaken or remove the perturbation

| Attack | Effect on Music Shield |
| --- | --- |
| Lossy re-encoding (MP3, AAC) at 128–192 kbps | **Improved, with a hard ceiling.** The jitter and phase-drift components are multiplicative on the track's own content, which codecs reproduce accurately, so 93–99 % of that change is still in the decoded file (least-squares retained fraction, synthetic clips, MP3 128k/192k and AAC 128k). Masked noise is nominally retained but buried under codec noise about 17 dB louder; the >15 kHz component is low-passed away (12 % retained at MP3 128k) and is no longer counted on. The ceiling: at 128 kbps the codec's *own* error (≈ -23 dB on a dense mix) is louder than the whole `light` perturbation (≈ -29 dB). An inaudible perturbation cannot out-shout a codec whose design goal is to be just inaudible; what we changed is that ours now persists through the codec rather than being replaced by it. Numbers in [METRICS.md](METRICS.md). |
| Lossy re-encoding at low bitrates (≤ 96 kbps MP3/AAC, Opus/Ogg at speech-ish rates) | Not measured. Expect the codec error to grow well past the perturbation and the retained fraction to fall. Do not assume the 128k numbers hold. |
| Downsampling to 16 kHz | **Measured.** 100 % of the change is retained (it lives below 8 kHz); the resampler removes the >15 kHz component and everything else up there, which is why nothing relies on it. |
| Denoising (ffmpeg `afftdn` after MP3 128k) | **Measured.** 94–96 % of the change retained: an FFT denoiser subtracts an estimated noise floor, and a multiplicative gain/phase wobble is not a noise floor. But the denoiser's own damage (log-mel 1.8–5.2 dB) is far larger than ours, so the cleaned protected copy is only 0.03–0.09 dB further from the master than a cleaned unprotected copy at `light`. |
| Neural codec re-synthesis (EnCodec 24 kHz, 6–24 kbps) | **Measured.** 91–103 % of the change retained, but buried: the codec's own reconstruction error (-13 to -18 dB) is 10–15 dB louder than the whole `light` perturbation, and `light` moves the codec's tokens/latents less than an MP3 does. Source separation and re-synthesis models proper were not measured; expect them to discard the change along with everything else they do not model. An adaptive attacker who knows our masking model can target it directly. |
| Remixing, mastering, EQ, compression | A time-varying EQ wobble of ±0.6 dB and a few degrees of phase drift are easily swamped by a mastering chain. Any all-pass or phase-linearising stage rewrites the phase drift outright. |
| Pitch shift, time stretch, tempo change | Re-analysis and resynthesis largely discards the original fine structure, including the phase drift. |
| Averaging multiple protected copies | If you protect the same track more than once, the pseudo-random components partly cancel when averaged. Protect once and distribute that copy. |
| Re-recording (analogue loop, microphone) | Masked noise and high-band content are the first things to go; room acoustics rewrite the phase; only the slow gain wobble has any chance of surviving, and it is small. |
| Mono downmix | Jitter and phase drift are identical on all channels, so a downmix keeps them intact. The per-channel masked noise partly averages out. |
| Adaptive / white-box attacker | The method is open source. Someone who has read `perturb.py` can build a filter tuned to it. We consider this acceptable for a v1 whose purpose is raising the cost of casual bulk scraping, not stopping determined actors. |
| Simply training on it anyway | Modern models are trained on noisy, compressed, badly mastered audio all the time, and §2 of METRICS.md shows `light` is a smaller disturbance to a neural codec than an MP3 is. Expect little to no effect on learning general style; any effect on clean copying is unmeasured. |

## What "spectrogram-level features" now means

Be precise about this when describing the tool. Of the four components, only
the spectral jitter moves magnitude spectrograms (mel, CQT, etc.), and at
`light` it moves them by about 0.2 dB on average — less than an MP3 encoder
at 128 kbps does on its own (≈ 0.65 dB on a dense mix). The phase drift, which
carries much of the codec-robust change, leaves magnitude spectrograms
essentially untouched by construction; it alters the waveform and the complex
spectrum, which matters to raw-audio and neural-codec pipelines and not at all
to a model that looks only at mel spectrograms. Masked noise and the >15 kHz
component contribute almost nothing to magnitude features after a re-encode.

## Audio quality caveats

- **No listening tests have been run**, for these presets or the previous
  ones. The presets are chosen from SNR, construction arguments and objective
  proxies (METRICS.md §1: segmental SNR, noise-to-mask ratio of the added
  components, per-band residual levels, band-level deviation). On the dense
  and mixed synthetic clips `light` passes thresholds we fixed before tuning;
  that is a proxy result, not a listening result. Treat `light` as the only
  preset intended to be transparent, and check it on your own material.
- The masking model is simplified: 32 log-spaced bands, a fixed -20 dB offset,
  neighbour spreading of -18 dB upward / -27 dB downward, an absolute floor,
  and a backward running-minimum so noise cannot precede an attack. It is not
  an MPEG psychoacoustic model and makes no per-listener guarantee.
- **Quiet, sparse solo material is the known weak spot at every preset,
  including `light`.** On the synthetic `sparse` clip about 2 % of active
  time-frequency cells still carry added noise above our masking estimate,
  diffusely, in cells >40 dB below the peak during note decays, at absolute
  levels around -61 to -66 dBFS on a clip peaking at -18 dBFS. Whether that
  is audible depends on how good the crude masking model is in exactly those
  cells, and we do not know. Listen on headphones before publishing protected
  solo, ambient or spoken-word material, or do not protect it.
- **Steady tonal material** (held chords, organ, drones) exposes the gain
  wobble itself: the 99th-percentile band-level step at `light` is 1.4 dB on
  the synthetic `sustained` clip, against a level JND of about 1 dB for a
  slow change in one band. An attentive headphone listener may notice a slow
  EQ movement. Only a smaller `jitter_db` would change this, and jitter is
  the component that carries what friction there is.
- The `light` SNR is ~29 dB. Most of the change is a slow ±0.6 dB gain wobble
  and a slow ±3° phase rotation of the track's own components, not added
  noise. A lower SNR here does not mean "more hiss".
- Phase drift is shared across channels so the stereo image does not move,
  and it is faded to zero between 50 Hz and 20 Hz so sub-bass and DC keep
  their phase. Material with very exposed, sustained
  sub-bass or extremely sharp isolated transients is the case most likely to
  reveal it at `strong`; we have no evidence either way at `light`.
- The >15 kHz component is now ordinary masked noise shaped under the same
  masking estimate as the rest, so a track with nothing above 15 kHz gets
  nothing added there (`hf_component_applied` reports `false`). Previously
  it was calibrated to overall loudness and put -60 to -76 dBFS of hiss into
  empty air bands on sparse material; that is fixed. At `strong` on bright
  material it is only 4 dB under the mask estimate and may still be
  perceptible to young listeners on headphones.
- `medium` and `strong` are expected to be audible on headphones or on sparse
  material (METRICS.md §1 flags them on every clip). In this version they add
  *less* noise than before and *more* gain/phase wobble, because the noise
  was the hiss risk and did nothing measurable for friction.
- If the output would clip, the entire file is scaled down. The UI reports
  this. Your loudness will be slightly lower than the original.
- MP3 input is decoded and returned as WAV. Decoding is lossless with respect
  to the MP3, but the MP3 itself already lost information relative to your
  master; protecting a master is always better than protecting an MP3.

## Operational limits

- Tracks over 15 minutes and uploads over 80 MB are rejected (configurable).
- There is also a **memory guard**, and on a small host it is the limit you
  will actually hit. One job holds the float32 input and the float32 output
  together (8 bytes per sample per channel); everything else is a fixed
  overhead. The server admits `(MUSIC_SHIELD_MEMORY_BUDGET_MB - 256) MiB / 8`
  sample-channels per job and rejects anything bigger with HTTP 413 *before*
  decoding it (from the WAV/FLAC header or ffprobe), instead of being
  OOM-killed halfway through. With the default 512 MB budget that is:
  **~6.3 min of stereo or ~12.7 min of mono at 44.1 kHz** (5.8 / 11.6 min at
  48 kHz). Stereo counts double; a mono upload of the same length always fits
  where the stereo one does not. `/api/info` reports the effective numbers and
  the error message quotes the limit for the file's own rate and layout.
  Measured on the streaming engine: a 200 s stereo MP3 peaks at ~290 MiB of
  server RSS, a file at the guard limit at ~430 MiB. Raise
  `MUSIC_SHIELD_MEMORY_BUDGET_MB` on a bigger host.
- Processing is synchronous and one job runs at a time; a second upload
  waits for the first to finish (the memory guard budgets for a single job).
  A 6-minute stereo 44.1 kHz WAV takes ~3 s on a laptop core and considerably
  longer on a free-tier container.
- Protected files are kept for one hour then deleted. Originals are deleted as
  soon as processing finishes. There is no account, so there is no way to
  recover a file after it expires.
- Authentication is a single shared HTTP Basic credential, enabled only when
  the server is started with `MUSIC_SHIELD_BASIC_PASSWORD` set (see
  DEPLOY.md). Without it the app is open; do not expose an instance like
  that. There are no accounts, roles, or rate limits.

## Things we will not build

- Anything to remove watermarks, fingerprints, or perturbations from audio you
  did not make.
- Anything that promises "unbreakable", "permanent", or "AI-proof" protection.
- Fake efficacy numbers.
