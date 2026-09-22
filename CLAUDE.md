# Project context for Claude Code

## What this is

Multi-species bird classification from audio recordings. University AI course
project (CSE3705), team of ~4 students, running roughly 9 months. The deliverable
is a working system plus a report and a public GitHub repo.

**Not** binary bird-present/absent detection. That was an earlier direction and
was dropped — it isn't a strong enough result on its own. Species classification
is the target.

## Working style — read this first

The student is learning this material, not just shipping it. Default to
**explaining and scaffolding rather than generating finished code**.

- The course has an explicit academic-integrity policy on AI-generated code.
  Work the student can't explain in an oral viva is a liability, not an asset.
- When writing code, comment the *reasoning*, not the syntax. "Why 32kHz" matters
  more than "this loads a file".
- Prefer showing the student how to fix an error over silently fixing it.
- If they ask for a big chunk of code, it's usually better to build it in pieces
  they follow than to hand over a finished module.

Also: the student has gotten overwhelmed when given too many options at once.
Give one clear next action rather than a menu.

**Keep `docs/LOG.md` current.** It is the project's lab notebook: one entry
per step, written when the step is done — what, why, what went wrong, the
numbers with their split and commit. The report is written from it and the
viva is prepared from it. Update it as part of finishing a step, not later.
The "Results so far" table at the top is updated whenever a model is scored.

## Current state

Done (Phase 1 complete):
- `src/config.py` — all audio and dataset parameters live here
- `src/fetch_data.py` — Xeno-canto API v3 survey (v2 is retired; needs a free
  key in the `XC_API_KEY` env var)
- `notebooks/kaggle_download.py` — the download, run on Kaggle. Xeno-canto
  rate-limits the student's home IP to ~13 kB/s (35+ hours); Kaggle gets
  ~780 kB/s (3 hours). 4,500 recordings, 30 species, 150 each, A/B quality,
  mixed mp3/wav.
- `src/preprocess.py` — audio → windows → log-mel, plus `sanity_check()`.
  Verified on real data: call structure clearly visible, confusable pairs
  separable by eye.
- `src/split.py` + `data/splits.csv` — FROZEN recording-level split,
  3150/675/675, stratified by species. Never recompute it; read the CSV.
  Recordist overlap train↔test is 181/217 — noted for a later
  recordist-grouped-split experiment.

- `src/build_cache.py` — one-time log-mel precompute, run on Kaggle
  (notebook "birdsong-cache" v1, mounts at
  `/kaggle/input/notebooks/harshv34546786564356/birdsong-cache/cache`).
  4,500 recordings → 20,483 windows (top 5 by energy), uint8, 13 min.
- `src/dataset.py` — `BirdWindows`: window-level items, multi-hot labels,
  rec_id returned for recording-level scoring.
- `src/evaluate.py` — shared scoring. Recording-level (mean over windows),
  per-class recall worst-first, row-normalised confusion matrix.
- `src/baseline_rf.py` — **RF on MFCCs: test acc 0.597, macro F1 0.597.**
  This is the number to beat. Results in `outputs/rf/`. Hardest: Great Tit
  (huge repertoire), Starling (mimic), House Sparrow. Errors are diffuse,
  not paired; real clusters are high-thin calls, corvids, chatterers.

- `src/model.py` + `src/train.py` — BirdCNN, 4 conv blocks + GAP, 396k
  params. BCE, AdamW, cosine, best-val-F1 checkpoint, test scored once.
  **Test acc 0.883, macro-F1 0.882** (30 epochs, no augmentation, 8 min on
  T4). Best epoch was the last — a longer run is the next experiment.
  Results in `outputs/cnn/`; weights on Kaggle notebook `birdsong-cnn`
  (Quick Save) and local `outputs/cnn/best.pth` (gitignored).

Not started:
- Ablations: longer run, then augmentation one at a time
- Focal→soundscape work (Phase 3), explainability, deployment demo

## Where things live

- **Laptop** (`Downloads/birdsong`): the code. Source of truth. No audio.
- **GitHub** (`harshvsinghnorth/birdsong`): pushed copy. Kaggle clones it.
- **Kaggle**: the audio and the GPU. Audio is the output of notebook
  "Birdoo" Version 3; attach it via Add Input → Your Work → Birdoo and it
  mounts at `/kaggle/input/notebooks/harshv34546786564356/birdoo/data/raw`.
  Every notebook starts with `!git clone` of the repo. Loop is:
  edit locally → push → clone on Kaggle → run → download small results → commit.
- Kaggle secret `XC_API_KEY` holds the API key. It is never in the repo.

## Datasets (three, with distinct roles)

The course requires at least three datasets.

1. **Xeno-canto** — primary training data. Open API, no account needed. Target
   ~4,500 recordings across 30 species. Mostly *focal* recordings: directional
   mic, close range, one bird, relatively clean.
2. **BirdCLEF** (via Kaggle) — test data. Its value is the *soundscape*
   recordings: passive field recorders, distant and overlapping birds, heavy
   noise. Subsample to ~4,500 to keep sizes comparable. Needed around month 3.
3. **NIPS4Bplus** — 687 recordings, 51 species, ~1 hour. Genuinely smaller than
   the other two, and that's fine: it's one of the few sets with *temporal*
   annotations (when each species calls, not just that it's present), so it's
   the fine-grained validation set. Needed around month 5.

Note BirdCLEF's data is largely derived from Xeno-canto — they aren't rival
sources, and using the API directly is normal practice, not a shortcut.

## The core research question

**The focal-to-soundscape gap.** Models trained on clean focal recordings
degrade badly on real soundscapes. Expect something like 85% → 45%.

This gap is the project's actual contribution: measure it, then reduce it.
"We closed a known generalization gap and here's the evidence" is a much
stronger result than any single accuracy number. Keep experiments pointed at
this rather than at leaderboard-chasing on clean data.

## Known hard problems

- **Weak labels.** Xeno-canto labels a whole recording, but the bird may
  vocalize in only a few seconds of it. Most windows are mislabeled noise.
  Current mitigation is energy-based window ranking in `preprocess.py`
  (`rank_windows_by_energy`). Better options later: multiple-instance learning,
  attention pooling over windows.
- **Long-tail imbalance.** Common species have thousands of recordings, rare
  ones a dozen — and rare ones are the conservation-relevant ones. Report
  per-class recall, never just aggregate accuracy.
- **Multi-label, not multi-class.** Real soundscapes have several species at
  once. Use per-class binary cross-entropy, not softmax. Metrics: macro F1, mAP.

## Design decisions already made (and why)

- **32 kHz sample rate** — bird calls carry energy above 8 kHz, so 16 kHz would
  throw away real signal.
- **5-second windows, 50% overlap** — fixed model input; overlap avoids short
  calls being split across a boundary.
- **128 mel bands, fmin 50 Hz, fmax 16 kHz** — 50 Hz floor cuts wind and
  handling noise.
- **Per-example normalization, not dataset-wide** — recordings come from many
  recordists with different gear. Normalizing per example stops the model
  learning "this person's microphone" as a shortcut for the species.
- **30 species, chosen deliberately** — enough to be a real multi-class problem,
  few enough that the confusion matrix is readable and interesting. Including
  some acoustically similar species is a feature, not a bug.

## Critical pitfalls

- **Split by recording, never by window.** Windows from one recording landing in
  both train and test gives inflated accuracy that means nothing. This is the
  single easiest way to silently ruin the project.
- **Preprocessing bugs don't crash.** A wrong `hop_length` or missing
  normalization just makes the model quietly mediocre while you blame the
  architecture. Run `sanity_check()` and *look* at the spectrograms — if call
  structure isn't visible to you, the model can't learn it either.
- Don't add all augmentations at once. Measure each one's contribution
  separately so there's an ablation table for the report.

## Phase plan

| Phase | Weeks | Focus |
|---|---|---|
| 1 | 1–4 | Data pipeline, preprocessing, sanity checks |
| 2 | 5–10 | Baseline (Random Forest on MFCCs) then CNN, 30 species |
| 3 | 11–16 | Focal→soundscape generalization work — the real contribution |
| 4 | 17–19 | Grad-CAM explainability, confusion analysis |
| 5 | 20–22 | Deployment demo, quantization, report, repo cleanup |

Get something working end-to-end early. A real accuracy number in week 3 kills
the drift that otherwise eats months 1–5.

## Stack

Python, PyTorch, librosa, scikit-learn. Free-tier GPU (Kaggle Notebooks give
~30 hrs/week; Colab also fine). No paid infrastructure.
