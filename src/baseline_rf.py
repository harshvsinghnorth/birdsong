"""
Baseline: Random Forest on MFCC summary features.

Why a baseline at all: the CNN's accuracy means nothing on its own. "87%" is
only impressive relative to what a simple, well-understood method gets on
the same split with the same preprocessing. This is that method.

Why MFCCs: they were the standard audio feature for two decades before
CNNs -- a DCT across the mel axis decorrelates the bands and compresses each
frame to a few coefficients. Mean and std over time then collapse a
(128 x 313) window to 40 numbers. Everything about *when* in the window a
call happens is thrown away; that is exactly the information a CNN keeps,
so the gap between the two is a measure of how much temporal structure
matters for these species.

The features are computed FROM THE CACHE, not from audio, so baseline and
CNN see identical preprocessing.

Usage:
    python -m src.baseline_rf <cache_dir> <out_dir>
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.fft import dct
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_SEED
from .dataset import BirdWindows
from .evaluate import (aggregate_by_recording, recording_labels, report,
                       plot_confusion, save_results)

N_MFCC = 20


def mfcc_features(specs_uint8, n_mfcc=N_MFCC):
    """(N, mels, frames) uint8 -> (N, 2*n_mfcc) float32.

    DCT-II along the mel axis is the definition of MFCC given a log-mel
    input. Keeping the first n_mfcc coefficients keeps the broad spectral
    envelope and discards fine detail (and most noise). Mean+std over frames
    gives a fixed-size vector regardless of window content.
    """
    x = specs_uint8.astype(np.float32) / 255.0
    c = dct(x, type=2, axis=1, norm="ortho")[:, :n_mfcc, :]     # (N, n_mfcc, frames)
    return np.concatenate([c.mean(axis=2), c.std(axis=2)], axis=1)


def run(cache_dir, out_dir):
    out_dir = Path(out_dir)
    train = BirdWindows(cache_dir, "train")
    species = train.species
    val = BirdWindows(cache_dir, "val", species=species)
    test = BirdWindows(cache_dir, "test", species=species)
    print(f"windows  train={len(train)}  val={len(val)}  test={len(test)}  classes={len(species)}")

    t0 = time.time()
    X_train = mfcc_features(train.specs)
    # Balanced data, so no class_weight. 300 trees is past the point where
    # more trees change the answer; n_jobs=-1 uses every core Kaggle gives.
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED)
    rf.fit(X_train, train.labels)
    print(f"trained in {time.time() - t0:.0f}s")

    results = {}
    for name, ds in [("val", val), ("test", test)]:
        probs = rf.predict_proba(mfcc_features(ds.specs))        # (N_windows, n_classes)

        # Window-level number for reference only -- the recording-level one
        # is the one that gets reported.
        win_acc = float((probs.argmax(1) == ds.labels).mean())

        rec_ids, rec_probs = aggregate_by_recording(probs, ds.rec_ids)
        y_true = recording_labels(ds.labels, ds.rec_ids)
        y_pred = rec_probs.argmax(1)

        r = report(y_true, y_pred, species, title=f"RF baseline -- {name} (per recording)")
        r["window_accuracy"] = win_acc
        print(f"(window-level accuracy for reference: {win_acc:.3f})")
        results[name] = r
        plot_confusion(y_true, y_pred, species, out_dir / f"confusion_rf_{name}.png")

    save_results(results, out_dir / "results_rf.json")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m src.baseline_rf <cache_dir> <out_dir>")
    run(sys.argv[1], sys.argv[2])
