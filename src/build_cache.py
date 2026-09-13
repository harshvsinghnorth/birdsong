"""
One-time precompute: every recording -> its top-k log-mel windows, cached.

Why a cache at all: librosa.load + resample + mel on 4,500 files costs on the
order of half an hour of CPU. Paying that once and saving the result means
every training run starts in seconds, and -- more importantly -- every model
sees IDENTICAL inputs. No "did I change hop_length between the baseline and
the CNN?" ambiguity in the report.

Output layout (out_dir/):
    XC<id>.npy   uint8 array, shape (n_windows, N_MELS, frames), values 0-255
    index.csv    one row per recording: id, species, split, recordist,
                 n_windows, npy  -- the Dataset class reads this, never the
                 raw audio

uint8 because a normalised spectrogram IS an image. 256 grey levels over the
80 dB range TOP_DB keeps is ~0.3 dB per step, far below anything a CNN can
resolve, and it makes the cache 4x smaller than float32.

Usage:
    python -m src.build_cache <raw_dir> <out_dir> [n_workers]
"""

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import ROOT, TOP_K_WINDOWS
from .preprocess import process_file


def _process_one(task):
    """Worker: one recording -> one .npy. Returns (id, n_windows, error)."""
    rec_id, audio_path, dest = task
    if dest.exists():                      # resumable, like the download
        return rec_id, np.load(dest, mmap_mode="r").shape[0], None
    try:
        specs = process_file(audio_path, top_k=TOP_K_WINDOWS)   # list of (mels, frames) in [0, 1]
        arr = np.stack(specs)
        np.save(dest, np.round(arr * 255).astype(np.uint8))
        return rec_id, len(specs), None
    except Exception as e:                 # one corrupt file must not kill a 30-min job
        return rec_id, 0, f"{type(e).__name__}: {e}"


def build(raw_dir, out_dir, splits_csv, n_workers=4):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tasks = [(r["id"], raw_dir / r["file"], out_dir / f"XC{r['id']}.npy") for r in rows]
    print(f"{len(tasks)} recordings, top-{TOP_K_WINDOWS} windows each, {n_workers} workers", flush=True)

    results, t0 = {}, time.time()
    with Pool(n_workers) as pool:
        for i, (rec_id, n, err) in enumerate(pool.imap_unordered(_process_one, tasks), 1):
            results[rec_id] = (n, err)
            if i % 250 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    # Index: everything the Dataset class needs, nothing it doesn't.
    failed = []
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "species", "split", "recordist", "n_windows", "npy"])
        w.writeheader()
        for r in rows:
            n, err = results[r["id"]]
            if err:
                failed.append((r["id"], err))
                continue
            w.writerow({"id": r["id"], "species": r["species"], "split": r["split"],
                        "recordist": r["recordist"], "n_windows": n, "npy": f"XC{r['id']}.npy"})

    total = sum(n for n, err in results.values() if not err)
    print(f"\n{len(rows) - len(failed)} recordings cached, {total} windows, "
          f"{len(failed)} failed, {(time.time() - t0) / 60:.1f} min")
    for rec_id, err in failed[:10]:
        print(f"  FAILED XC{rec_id}: {err}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m src.build_cache <raw_dir> <out_dir> [n_workers]")
    build(sys.argv[1], sys.argv[2], ROOT / "data" / "splits.csv",
          n_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 4)
