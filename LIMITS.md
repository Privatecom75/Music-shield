# Limits and failure modes

Read this before relying on Music Shield for anything. The short version:
it adds friction, and friction can be removed.

## What we are actually claiming

- The output differs from the input in ways that alter the waveform, the
  complex spectrum and (to a smaller degree) magnitude-spectrogram features,
  while trying to keep the change hard to hear at the default preset.
- After a typical MP3/AAC re-encode at 128–192 kbps, most of that change is
  still present in the decoded file. We measured this on synthetic clips, and
  only this; see [METRICS.md](METRICS.md) for the method and numbers.
- That's it. We are **not** claiming the output cannot be copied, cannot be
  used to train a model, or proves you own the track.

## What we are not claiming

- **Not AI-proof.** We have not measured how much this degrades any specific
  model's ability to copy, imitate, or train on a track. We will not publish a
  number we have not measured, and you should be suspicious of anyone who does.
- **Not DRM.** Anyone with the file can play it, copy it, and redistribute it.
- **Not ownership proof.** There is no watermark or verifiable payload in v1.
- **Not permanent.** Every transform below can weaken or remove it.

## Known ways to weaken or remove the perturbation

| Attack | Effect on Music Shield |
| --- | --- |
| Lossy re-encoding (MP3, AAC) at 128–192 kbps | **Improved, with a hard ceiling.** The jitter and phase-drift components are multiplicative on the track's own content, which codecs reproduce accurately, so 93–99 % of that change is still in the decoded file (least-squares retained fraction, synthetic clips, MP3 128k/192k and AAC 128k). Masked noise is nominally retained but buried under codec noise about 17 dB louder; the >15 kHz component is low-passed away (12 % retained at MP3 128k) and is no longer counted on. The ceiling: at 128 kbps the codec's *own* error (≈ -23 dB on a dense mix) is louder than the whole `light` perturbation (≈ -29 dB). An inaudible perturbation cannot out-shout a codec whose design goal is to be just inaudible; what we changed is that ours now persists through the codec rather than being replaced by it. Numbers in [METRICS.md](METRICS.md). |
| Lossy re-encoding at low bitrates (≤ 96 kbps MP3/AAC, Opus/Ogg at speech-ish rates) | Not measured. Expect the codec error to grow well past the perturbation and the retained fraction to fall. Do not assume the 128k numbers hold. |
| Downsampling to 16 kHz or 22.05 kHz | Removes the high-band component entirely. Many ML pipelines do this as their first step. Jitter and phase drift below the new Nyquist survive resampling; the masked noise is partly filtered. |
| Denoising / spectral subtraction / source separation | Designed to remove exactly what the noise component adds. The multiplicative components are not "noise" and are not removed by subtraction, but a separation or re-synthesis model that re-estimates the source will discard them along with everything else it does not model. An adaptive attacker who knows our masking model can target it directly. |
| Remixing, mastering, EQ, compression | A time-varying EQ wobble of ±0.6 dB and a few degrees of phase drift are easily swamped by a mastering chain. Any all-pass or phase-linearising stage rewrites the phase drift outright. |
| Pitch shift, time stretch, tempo change | Re-analysis and resynthesis largely discards the original fine structure, including the phase drift. |
| Averaging multiple protected copies | If you protect the same track more than once, the pseudo-random components partly cancel when averaged. Protect once and distribute that copy. |
| Re-recording (analogue loop, microphone) | Masked noise and high-band content are the first things to go; room acoustics rewrite the phase; only the slow gain wobble has any chance of surviving, and it is small. |
| Mono downmix | Jitter and phase drift are identical on all channels, so a downmix keeps them intact. The per-channel masked noise partly averages out. |
| Adaptive / white-box attacker | The method is open source. Someone who has read `perturb.py` can build a filter tuned to it. We consider this acceptable for a v1 whose purpose is raising the cost of casual bulk scraping, not stopping determined actors. |
| Simply training on it anyway | Modern models are trained on noisy, compressed, badly mastered audio all the time. A light perturbation may reduce clean copying while having little effect on learning general style. |

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
  proxies (see METRICS.md, "Listening notes"). Treat `light` as the only
  preset intended to be transparent, and check it on your own material.
- The masking model is simplified: 32 log-spaced bands, a fixed -20 dB offset,
  neighbour spreading at -10 dB, an absolute floor. It is not an MPEG
  psychoacoustic model and makes no per-listener guarantee.
- The reported SNR for `light` fell from ~33 dB to ~29 dB in this version.
  The extra change is a slow ±0.6 dB gain wobble and a slow ±3° phase
  rotation of the track's own components — not added noise; the masked noise
  was reduced. A lower SNR here does not mean "more hiss".
- Sparse or very quiet material (solo instruments, ambient, spoken word) has
  less energy to hide noise under; the change is more likely to be audible
  there, especially at `medium` and `strong`.
- Phase drift is shared across channels so the stereo image does not move,
  and it is faded to zero between 50 Hz and 20 Hz so sub-bass and DC keep
  their phase. Material with very exposed, sustained
  sub-bass or extremely sharp isolated transients is the case most likely to
  reveal it at `strong`; we have no evidence either way at `light`.
- On headphones the high-band component may be perceptible to young listeners
  with good hearing at `strong`. It is now 8 dB lower than before at every
  preset.
- If the output would clip, the entire file is scaled down. The UI reports
  this. Your loudness will be slightly lower than the original.
- MP3 input is decoded and returned as WAV. Decoding is lossless with respect
  to the MP3, but the MP3 itself already lost information relative to your
  master; protecting a master is always better than protecting an MP3.

## Operational limits

- Tracks over 15 minutes and uploads over 80 MB are rejected (configurable).
- Processing is synchronous. A 10-minute stereo 48 kHz WAV takes a few seconds
  on a laptop and considerably longer on a free-tier container.
- Protected files are kept for one hour then deleted. Originals are deleted as
  soon as processing finishes. There is no account, so there is no way to
  recover a file after it expires.
- No authentication. Do not expose a public instance without putting it
  behind a login or basic auth.

## Things we will not build

- Anything to remove watermarks, fingerprints, or perturbations from audio you
  did not make.
- Anything that promises "unbreakable", "permanent", or "AI-proof" protection.
- Fake efficacy numbers.
