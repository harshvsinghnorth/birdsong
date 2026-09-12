"""
Train / validation / test split -- BY RECORDING, never by window.

Why this file exists as a separate step, and why it writes a CSV:

A recording gets chopped into overlapping 5s windows (preprocess.py). Adjacent
windows overlap by 50%, so they share half their samples; and even non-adjacent
windows from one file share the same bird, mic, background and gain. If windows
from one recording land in both train and test, the model is scored on data it
has effectively already seen. Accuracy looks great and means nothing. This is
the single easiest way to silently ruin the project, so the split is decided
HERE, at the recording level, before any window exists.

The split is frozen to a CSV rather than recomputed at train time. Recomputing
means every notebook must get the same seed, the same file order, the same
library version -- any drift and train/test quietly bleed together. A CSV
committed to the repo is a fact, not a procedure.

Usage:
    python -m src.split                       # uses config paths
    python -m src.split <raw_dir> <out_csv>   # e.g. on Kaggle
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from .config import RAW_DIR, RANDOM_SEED

# 70 / 15 / 15. With 150 recordings per species that is ~105 / 22 / 23 each --
# enough test recordings per class that per-class recall is a real number and
# not the outcome of two or three files.
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def load_metadata(raw_dir):
    """One row per recording from every <species>/metadata.jsonl.

    Sorted glob + file order gives a deterministic row order. This matters:
    train_test_split with a fixed seed only reproduces if the input order is
    identical, and filesystem order is not guaranteed.
    """
    rows = []
    for meta in sorted(Path(raw_dir).glob("*/metadata.jsonl")):
        with open(meta, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({
                    "id": r["id"],
                    "file": f"{meta.parent.name}/{r['file']}",
                    "species": r["species"],
                    "recordist": r.get("recordist") or "",
                })
    return rows


def make_split(rows, seed=RANDOM_SEED):
    """Assign each recording to train / val / test, stratified by species.

    Stratified so every species has the same train/val/test proportions. With
    a balanced 150-per-species download this barely matters, but the moment
    the dataset becomes long-tailed (it will, once soundscapes arrive) an
    unstratified split can leave a rare species with zero test recordings.

    Two-stage because train_test_split is two-way: carve off test first, then
    carve val from what remains. The val fraction is rescaled so it is 15% of
    the ORIGINAL total, not 15% of the remaining 85%.
    """
    species = [r["species"] for r in rows]
    trainval, test = train_test_split(
        rows, test_size=TEST_FRAC, stratify=species, random_state=seed)

    tv_species = [r["species"] for r in trainval]
    train, val = train_test_split(
        trainval, test_size=VAL_FRAC / (1 - TEST_FRAC),
        stratify=tv_species, random_state=seed)

    for r in train: r["split"] = "train"
    for r in val:   r["split"] = "val"
    for r in test:  r["split"] = "test"
    return train + val + test


def check_split(rows):
    """Loud verification. Cheap to run, expensive to skip."""
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "a recording appears in more than one split"

    per = defaultdict(Counter)
    for r in rows:
        per[r["species"]][r["split"]] += 1
    print(f"{'species':28} {'train':>6} {'val':>5} {'test':>5}")
    for sp in sorted(per):
        c = per[sp]
        print(f"{sp:28} {c['train']:6} {c['val']:5} {c['test']:5}")
    tot = Counter(r["split"] for r in rows)
    print(f"{'TOTAL':28} {tot['train']:6} {tot['val']:5} {tot['test']:5}")

    # Recordist overlap: NOT prevented by this split, but measured so the
    # number exists. Same recordist in train and test means the model could
    # score partly on recognising a microphone. Per-example normalisation
    # (preprocess.py) is the first defence; a recordist-grouped split is the
    # experiment that would show whether it is enough. That is a report
    # finding, not a bug to fix now.
    by_split = defaultdict(set)
    for r in rows:
        if r["recordist"]:
            by_split[r["split"]].add(r["recordist"])
    shared = by_split["train"] & by_split["test"]
    print(f"\nrecordists in test also in train: {len(shared)} of {len(by_split['test'])}")


def write_csv(rows, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "file", "species", "recordist", "split"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["species"], int(r["id"]))))
    print(f"\nwrote {out_csv}  ({len(rows)} recordings)")


if __name__ == "__main__":
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_DIR
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else RAW_DIR.parent / "splits.csv"

    rows = load_metadata(raw_dir)
    if not rows:
        sys.exit(f"no metadata.jsonl found under {raw_dir}")
    rows = make_split(rows)
    check_split(rows)
    write_csv(rows, out_csv)
