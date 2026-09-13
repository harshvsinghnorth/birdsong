"""
PyTorch Dataset over the spectrogram cache built by build_cache.py.

One item = one 5s log-mel window, shaped (1, N_MELS, frames) as float32 in
[0, 1], plus a MULTI-HOT label vector and the recording id it came from.

Why multi-hot rather than an integer class: the project's target is
multi-label (soundscapes contain several species at once), trained with
per-class BCE. Xeno-canto recordings happen to have exactly one species each,
so today every label vector has a single 1 -- but emitting multi-hot from the
start means the BirdCLEF soundscapes slot in later without touching the
model, the loss, or the metrics.

Why the recording id is returned: training happens on windows, but accuracy
is reported per RECORDING. Evaluation groups windows by rec_id and averages
their predictions. Without the id you cannot get back to the unit the split
was made on.
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_index(cache_dir):
    with open(Path(cache_dir) / "index.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def species_list(index_rows):
    """The one species->index mapping used everywhere. Sorted so it is
    deterministic and identical across splits and across runs."""
    return sorted({r["species"] for r in index_rows})


class BirdWindows(Dataset):
    def __init__(self, cache_dir, split, species=None):
        """
        cache_dir: directory holding index.csv and the XC*.npy files
        split:     "train" | "val" | "test"
        species:   fixed species list. Pass the SAME list to every split;
                   if None it is derived from the full index (all splits),
                   which is also fine since the index is shared.
        """
        cache_dir = Path(cache_dir)
        rows = load_index(cache_dir)
        self.species = species or species_list(rows)
        self.n_classes = len(self.species)
        sp_to_idx = {s: i for i, s in enumerate(self.species)}

        rows = [r for r in rows if r["split"] == split]
        if not rows:
            raise ValueError(f"no recordings with split={split!r} in {cache_dir}")

        # Load every window of this split into RAM as uint8. ~40 KB per
        # window, so the whole train split is well under 1 GB. Keeping it
        # uint8 here and converting per item keeps RAM 4x smaller than
        # converting up front.
        specs, labels, rec_ids = [], [], []
        for r in rows:
            arr = np.load(cache_dir / r["npy"])          # (n_windows, mels, frames) uint8
            specs.append(arr)
            labels.extend([sp_to_idx[r["species"]]] * len(arr))
            rec_ids.extend([int(r["id"])] * len(arr))

        self.specs = np.concatenate(specs)              # (N, mels, frames) uint8
        self.labels = np.asarray(labels, dtype=np.int64)
        self.rec_ids = np.asarray(rec_ids, dtype=np.int64)
        self.split = split

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        # uint8 -> float in [0, 1], and add the channel axis a 2D CNN expects:
        # (mels, frames) -> (1, mels, frames). A spectrogram is a 1-channel image.
        x = torch.from_numpy(self.specs[i].astype(np.float32) / 255.0).unsqueeze(0)

        # Multi-hot: zeros everywhere, 1 at this window's species. Float,
        # because BCEWithLogitsLoss wants targets in the same dtype as logits.
        y = torch.zeros(self.n_classes, dtype=torch.float32)
        y[self.labels[i]] = 1.0

        return x, y, int(self.rec_ids[i])

    def class_counts(self):
        """Windows per species -- print this; imbalance is a thing to KNOW."""
        return np.bincount(self.labels, minlength=self.n_classes)
