# Limits and failure modes

Read this before relying on Music Shield for anything. The short version:
it adds friction, and friction can be removed.

## What we are actually claiming

- The output differs from the input in ways that alter spectrogram-level
  features, while trying to keep the change hard to hear at the default preset.
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
| Lossy re-encoding (MP3, AAC, Opus, Ogg) at typical bitrates | Codecs discard exactly the kind of low-level, masked detail the noise component lives in. Expect a large part of the masked noise to be lost; the jitter survives better but is not immune. The high-band component is usually removed outright by low-pass filtering in encoders. |
| Downsampling to 16 kHz or 22.05 kHz | Removes the high-band component entirely. Many ML pipelines do this as their first step. |
| Denoising / spectral subtraction / source separation | Designed to remove exactly what we add. An adaptive attacker who knows our masking model can target it directly. |
| Remixing, mastering, EQ, compression | A time-varying EQ wobble of ±0.4 dB is easily swamped by a mastering chain. |
| Pitch shift, time stretch, tempo change | Re-analysis and resynthesis largely discards the original fine structure. |
| Averaging multiple protected copies | If you protect the same track more than once, the pseudo-random components partly cancel when averaged. Protect once and distribute that copy. |
| Re-recording (analogue loop, microphone) | Masked noise and high-band content are the first things to go. |
| Adaptive / white-box attacker | The method is open source. Someone who has read `perturb.py` can build a filter tuned to it. We consider this acceptable for a v1 whose purpose is raising the cost of casual bulk scraping, not stopping determined actors. |
| Simply training on it anyway | Modern models are trained on noisy, compressed, badly mastered audio all the time. A light perturbation may reduce clean copying while having little effect on learning general style. |

## Audio quality caveats

- The masking model is simplified: 32 log-spaced bands, a fixed -20 dB offset,
  neighbour spreading at -10 dB, an absolute floor. It is not an MPEG
  psychoacoustic model and makes no per-listener guarantee.
- Sparse or very quiet material (solo instruments, ambient, spoken word) has
  less energy to hide noise under; the change is more likely to be audible
  there, especially at `medium` and `strong`.
- On headphones the high-band component may be perceptible to young listeners
  with good hearing at `strong`.
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
