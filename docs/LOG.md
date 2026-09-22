# Project log

A running record of what was done, why, what went wrong, and what the numbers
were. One entry per step, written at the time. The report is generated from
this; the viva is prepared from this.

Rules for keeping it useful:
- Write the entry when the step is done, not later. Memory of *why* fades fast.
- Record problems and dead ends, not just successes. The dead ends are where
  the understanding is.
- Every number goes in with the split it was measured on and the commit that
  produced it.

---

## Results so far

| Model | Features | Val acc | Test acc | Test macro-F1 | Commit |
|---|---|---|---|---|---|
| Random Forest (300 trees) | 20 MFCC mean+std (40-d) | 0.578 | **0.597** | 0.597 | `cea4618` |
| BirdCNN (no augmentation, 30 ep) | log-mel (128×313) | 0.867 | **0.883** | 0.882 | `4dfa86a` |

30 species, 675 test recordings, recording-level scoring throughout. Chance = 0.033.

---

## 2026-09-09 — Project start, first contact with the API

**Goal.** Run the species survey against Xeno-canto to see which of 35
candidate species have enough recordings.

**Problem.** `python -m src.fetch_data survey` failed immediately with
`404 Not Found` on `https://xeno-canto.org/api/2/recordings`. Re-running gave
the identical error; the retry loop retried a 404 three times, which can
never succeed.

**Diagnosis.** Probed both endpoints with `curl`. API v2 is retired (404).
API v3 exists at `/api/3/recordings` but returns 401 without a key. The
project notes' claim that "no account or API key is needed" was true when
written and is not anymore.

**Lesson.** A 4xx is a client error: the request is well-formed and the answer
is "no". Retrying it is wasted time. Retry logic should distinguish
retryable (timeouts, 5xx, 429) from non-retryable (other 4xx).

---

## 2026-09-10 — API v3 migration, quality filter, species selection

**Done.**
- Created a Xeno-canto account, got an API key, stored in the `XC_API_KEY`
  environment variable. Never in source: this repo is public.
- Migrated `fetch_data.py` to v3. Response shape is unchanged from v2
  (`numRecordings`, `numPages`, `recordings[]`), so no parsing changes.
- First survey with the original filter (`q:A`, grade-A only): **only 20 of
  35 species** reached the 100-recording threshold. `config.py` wants 30.

**Problem.** The code's comment said "restrict to A/B" but the code said
`q:A`. A mismatch between a comment and the code it describes.

**Decision.** Widened to `q:">C"` (grades A and B). Blackbird went from 379 to
2,579 recordings; all 35 species now clear the threshold. Trade-off accepted:
grade-B recordings are noisier, which puts more load on the weak-label
mitigation (energy-based window ranking). Not going further to grade C:
that is where labels start being wrong more often than right.

**Species choice.** `download()` caps at 150 per species and every candidate
has more than 150, so recording counts are irrelevant to selection. Chose by
acoustic interest instead. Dropped 5 of 35:
- Cuckoo, Swift, Kestrel — acoustically isolated, near-free correct
  predictions that would inflate accuracy without testing anything.
- Woodpigeon, Collared Dove — lowest data, would mostly test each other.

Kept the confusable groups on purpose: Blackbird/Song Thrush, Great Tit/Blue
Tit/Long-tailed Tit, Chaffinch/Greenfinch/Goldfinch/Serin,
Goldcrest/Treecreeper/Chiffchaff (all very high frequency), Jay/Magpie/Crow.
Starling kept as a known-hard mimic.

**Problem.** Started the local download. 5 files in 13 minutes. See next entry.

---

## 2026-09-11 — The throttle, and moving acquisition to Kaggle

**Diagnosis.** Measured with `curl`:

| Target | Throughput |
|---|---|
| Home connection to Cloudflare | 646–1,557 KB/s |
| Home connection to Xeno-canto (audio) | 13.5 KB/s |
| Home connection to Xeno-canto (metadata API) | 12.3 KB/s |
| 6 parallel connections to Xeno-canto | 12.4 KB/s **aggregate** |

The home line is fine. Xeno-canto caps this IP at ~13 KB/s, on every
endpoint, and the cap is per-IP not per-connection (parallelism gives
nothing). At that rate 4,500 recordings is ~9 days. Two days later it was
unchanged, so not transient load.

**Dead end.** Also found the download endpoint serves the *original* upload,
which is WAV for 43% of recordings (~2.5 MB vs ~0.4 MB for MP3). Added an
MP3-only filter as a bandwidth workaround: cut the estimate to ~35 hours.
Still impractical. This filter was later **removed** (see below) because it
introduces a sampling bias — recordists who upload WAV may systematically
differ (better gear) — and the reason for it disappeared.

**Bug found on the way.** The local code saved every download as `.mp3`
regardless of format, so WAV bytes were sitting in files named `.mp3`.
Worked by accident with soundfile; a trap for anyone reading the data later.
Fixed: files now keep their real extension.

**Solution.** Same request from a Kaggle notebook: **779.8 KB/s**, 56× faster.
Kaggle is where training will happen anyway (free GPU), so the audio never
needs to touch the laptop. `notebooks/kaggle_download.py` written; API key
held in Kaggle Secrets.

**Run.** Notebook `Birdoo`, Version 3, run as a commit (not interactively —
an interactive session's working directory is discarded). **176 min, 4,530
files** = 4,500 recordings + 30 `metadata.jsonl`. 2,976 MP3 + 1,524 WAV.

**Lesson.** Distinguish "my connection", "their server", and "their policy
toward me" by measuring each separately. One baseline against a neutral host
plus one parallelism test answered all three.

---

## 2026-09-13 — Split, GitHub, sanity check, cache

**Layout decided.** Three places, each with one job:
- Laptop: code, source of truth. No audio.
- GitHub `harshvsinghnorth/birdsong`: pushed copy, public deliverable.
- Kaggle: audio + GPU. Every notebook starts with `!git clone`.
Loop: edit locally → push → clone on Kaggle → run → download small results
→ commit.

**Split** (`src/split.py`, commit `b00b4e7`). Stratified by species at the
**recording** level, 70/15/15 → 3150/675/675, every species 105/22–23/22–23.
Frozen to `data/splits.csv` and committed; no notebook ever recomputes it.
Reason: windows from one recording overlap 50% and share mic, background and
gain — splitting them is leakage. Deterministic: verified byte-identical
across two runs.

**Measured, not fixed:** 181 of 217 test recordists also appear in train
(83%). A model could score partly by recognising microphones.
Per-example normalisation is the first defence; a recordist-grouped split is
a planned experiment — if it drops accuracy, the shortcut exists.

**Sanity check** on four real recordings (Blackbird, Song Thrush, Goldcrest,
Treecreeper), plotting the peak-energy window rather than the first window
(the first 5 s is usually the recordist settling in). Call structure clearly
visible; the thrush pair separable by eye (thrush repeats phrases);
Goldcrest and Treecreeper both sit high (~bin 100 of 128 ≈ 7–8 kHz) with
different shapes. Shape `(128, 313)` matches config. Per-example
normalisation confirmed by colourbars at 0→1. One recording had a constant
background tone at ~bin 62 — such things exist in the data and a CNN could
latch onto them.

**Cache** (`src/build_cache.py`, notebook `birdsong-cache` v1). Each recording
→ its top-5 windows by band energy → uint8 `(5, 128, 313)`. **4,500
recordings, 20,483 windows, 0 failures, 13.1 min.** A few MP3s have corrupt
frames; librosa fell back to a second decoder with a `FutureWarning` that the
fallback is removed in librosa 1.0. Noted; nothing to do yet.

Why uint8: a normalised spectrogram is an image; 256 levels over 80 dB is
~0.3 dB per step; 4× smaller than float32 → ~800 MB total, fits in RAM.

---

## 2026-09-13/14 — Dataset, shared evaluation, Random Forest baseline

**Dataset** (`src/dataset.py`). One item = one window, `(1, 128, 313)` float
in [0, 1]. Label is a **multi-hot** 30-vector (all zeros, one 1) rather than
an integer — because the target is multi-label with per-class BCE, so
soundscapes later need no change to model, loss or metrics. Each item also
carries its recording id so evaluation can regroup windows.

**Evaluation** (`src/evaluate.py`). One scorer for every model, or numbers
are not comparable. Mean window probabilities per recording, argmax, then
accuracy, macro-F1, per-class recall printed worst-first, row-normalised
confusion matrix (diagonal = per-class recall).

**Baseline** (`src/baseline_rf.py`, commit `cea4618`). DCT across the mel
axis of the cached spectrograms → 20 MFCCs → mean + std over time → 40
features per window. Random Forest, 300 trees. Trains in 29 s.

**Result.** Val 0.578 / **test 0.597**, macro-F1 0.597. Window-level accuracy
0.513 — recording-level aggregation is worth ~8 points, which is the
weak-label problem measured.

**Reading the per-class recall.** Hardest: Great Tit (0.32 — has one of the
largest song repertoires of any European bird; mean/std features cannot
represent "one of thirty songs"), Starling (0.39 — the mimic), House Sparrow,
Tree Pipit. Easiest: Wren (0.95), Tawny Owl (0.82), Blackcap, Chiffchaff —
species with one distinctive sound.

**Reading the confusion matrix.** Errors are *diffuse*, not paired: MFCC
summaries fail toward "generic bird" rather than toward a specific
look-alike. Real clusters: high-thin calls (Goldcrest ↔ Long-tailed Tit ↔
Treecreeper ↔ Serin), corvids (Jay ↔ Magpie ↔ Crow), chatterers (House
Sparrow ↔ Swallow). Starling's errors go everywhere — what a mimic looks
like. The predicted Blackbird ↔ Song Thrush confusion barely registers.

**Caveat.** 22–23 test recordings per species → each is ~4.5 recall points.
Blackbird at 0.26 val / 0.50 test is noise, not a contradiction. Read the
top and bottom five of the table, not second decimals.

---

## 2026-09-14 — CNN and training loop written

**Model** (`src/model.py`, commit `519a2bb`). Four blocks of Conv3×3 → BN →
ReLU → MaxPool2, channels 32→64→128→256, then global average pooling and one
linear layer. **396k parameters.** Custom and small on purpose: every layer
has a stated reason and fits on a whiteboard. A pretrained ImageNet backbone
is a later ablation ("does photograph pretraining help spectrograms?"), not
the foundation.

Why global average pooling: asks "is this pattern present anywhere in the
window?", so a call at second 1 and at second 4 give the same answer.
Verified: rolling the input by 60 frames changed the logits by 0.0004.

**Training** (`src/train.py`, commit `355db5a`). BCEWithLogitsLoss (30
independent yes/no questions, not softmax), AdamW lr 1e-3 with cosine decay,
batch 64, mixed precision, 30 epochs max. Val scored per recording after
every epoch; best val macro-F1 checkpoint kept; stop after 6 epochs without
improvement. **Test is scored once, at the end, with that checkpoint.**
No augmentation: this run is the first row of the ablation table.

Smoke-tested on a synthetic 3-species cache on CPU: learns it in 2 epochs,
checkpoint saved and reloaded, all output files produced.

**Next.** Run on Kaggle GPU (notebook `birdsong-cnn`). The question: does test
accuracy beat 0.597, and does the bottom of the per-class table change?

---

## 2026-09-15 — First CNN run

**Run.** Notebook `birdsong-cnn`, Kaggle T4, interactive. `src.train` with
defaults: 30 epochs, batch 64, AdamW lr 1e-3 cosine, BCE, no augmentation.
**8.0 minutes.** Results in `outputs/cnn/`; weights kept on Kaggle via Quick
Save and locally as `outputs/cnn/best.pth` (gitignored).

**Result.** Val 0.867 / **test 0.883**, macro-F1 0.882. Window-level 0.827.
Baseline was 0.597: **+28.6 points** from the same cache, split and scorer.
The project notes predicted ~85% on clean focal recordings.

**Best epoch was 30 — the last one.** Early stopping never fired; the run
ended on the epoch budget while val loss was still creeping down
(0.0464 → 0.0461). The cosine schedule had the LR near zero by then. A
longer run (50 epochs) is the obvious next experiment; expect a point or
two, not more.

**The curves** (`training_curves.png`). Train loss falls smoothly and
monotonically to 0.028. Val loss is jumpy through epochs 5–17 (0.089 → 0.066
→ 0.081 → 0.062 → 0.079 …) — the high-LR phase, the optimiser overshooting —
then smooth from epoch 18 once cosine decay shrinks the steps. Val F1 mirrors
it: 0.56 → 0.73 → 0.64 → 0.74 → 0.65, then a clean climb to 0.87. The
train/val loss gap widens over the run (0.028 vs 0.046 at the end): mild
overfitting, the model is starting to memorise train windows. This is the
motivation for augmentation, and the "no augmentation" row of the ablation
table.

**Per-class recall.** The floor rose from 0.32 (RF) to 0.70. Six species at
1.00 on test (Robin, Blue Tit, Treecreeper, Chiffchaff, Wren, Tawny Owl).
Bottom of the test table: Skylark 0.70, Great Tit 0.73, Song Thrush 0.74,
Long-tailed Tit / Jay / House Sparrow 0.77, Starling 0.78. **Great Tit,
Starling and House Sparrow were the RF's hardest species too** — their
difficulty is real, not noise. Skylark is new at the bottom: it sings in
long unbroken streams, so its five windows may all look alike and give the
aggregation nothing to average.

**The confusion matrix** (`confusion_cnn_test.png`) is a strong diagonal
with faint off-diagonals. What survives:
- Corvid pair: Crow ↔ Jay, both directions. Harsh broadband calls.
- Great Tit's errors are still *diffuse* — scattered across Blue Tit,
  Yellowhammer, Robin, Chaffinch, Blackbird. Consistent with a repertoire
  problem: no single look-alike, just many songs.
- Starling's errors are still diffuse — the mimic signature.
- Blackbird ↔ Song Thrush **still barely registers**. The prediction from
  species selection that this pair would be the hard one was wrong for
  both models. Their song *structure* differs enough (thrush repeats, blackbird
  does not) that even the RF's summary statistics half-separated them.

**Val 0.867 vs test 0.883**: 1.6 points apart, test higher. With 675
recordings the headline number carries roughly ±2 points of noise. Report
"about 88%", not 0.883.

**Bug found on the way.** `.gitignore` had `outputs/*.pth`, which does not
match `outputs/cnn/best.pth` (one directory deeper). Would have pushed 1.6 MB
of weights on every retrain. Changed to `*.pth`.

**Next.** Two experiments, run separately so each gets its own ablation row:
(1) 50 epochs, no other change — does the plateau move? (2) augmentation
(SpecAugment-style time/frequency masking, then mixup), one at a time.
Then Phase 3: measure the focal→soundscape gap on BirdCLEF.

---

## 2026-09-15 — This log started

Backfilled from git history and session notes. From here on, every step gets
its entry at the time it happens.
