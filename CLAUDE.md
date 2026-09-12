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

## Current state

Done:
- `src/config.py` — all audio and dataset parameters live here
- `src/fetch_data.py` — Xeno-canto API survey + download
- `src/preprocess.py` — audio → windows → log-mel spectrograms, plus a
  visual sanity check

Not started:
- Dataset class and train/val/test split
- Model (baseline + CNN)
- Training loop
- Evaluation and explainability
- Deployment demo

Immediate next step: run `python -m src.fetch_data survey`, inspect
`data/species_shortlist.csv`, choose 30 species deliberately, then download.

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
