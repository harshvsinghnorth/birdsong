"""
Evaluation shared by every model. Baseline and CNN MUST be scored by the
same code, or their numbers are not comparable and the report's central
claim ("the CNN beats the baseline by X") is meaningless.

Two rules from the project notes are enforced here:

1. Score per RECORDING, not per window. Windows are what the model consumes;
   recordings are the unit the split was made on and the unit a user cares
   about. aggregate_by_recording() averages each recording's window
   probabilities before any metric is computed.

2. Never report aggregate accuracy alone. per-class recall, worst-first, is
   always printed. Rare or acoustically-similar species hide inside a good
   aggregate number, and those are the interesting ones.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score, confusion_matrix


def aggregate_by_recording(probs, rec_ids):
    """
    probs:   (N_windows, n_classes) per-window class probabilities
    rec_ids: (N_windows,) recording id of each window
    returns: (rec_ids_unique, (N_recordings, n_classes) mean probabilities)

    Mean is the simplest pooling. Max would favour a single confident window
    and is worth trying later for soundscapes where the bird is briefly
    present -- another ablation for the report.
    """
    by_rec = defaultdict(list)
    for p, r in zip(probs, rec_ids):
        by_rec[int(r)].append(p)
    ids = sorted(by_rec)
    return np.array(ids), np.stack([np.mean(by_rec[i], axis=0) for i in ids])


def recording_labels(labels, rec_ids):
    """Each recording has one label (all its windows share it). Map back."""
    seen = {}
    for lab, r in zip(labels, rec_ids):
        seen.setdefault(int(r), int(lab))
    return np.array([seen[i] for i in sorted(seen)])


def report(y_true, y_pred, species, title=""):
    """Print accuracy, macro F1, and per-class recall worst-first. Return a
    dict for saving so the report's tables are generated, not retyped."""
    n = len(species)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=range(n), zero_division=0))
    recall = recall_score(y_true, y_pred, average=None, labels=range(n), zero_division=0)

    print(f"\n=== {title} ===  n={len(y_true)}")
    print(f"accuracy  {acc:.3f}")
    print(f"macro F1  {macro_f1:.3f}")
    print("\nper-class recall (worst first):")
    for i in np.argsort(recall):
        print(f"  {species[i]:28} {recall[i]:.2f}")

    return {"title": title, "n": int(len(y_true)), "accuracy": acc, "macro_f1": macro_f1,
            "per_class_recall": {species[i]: float(recall[i]) for i in range(n)}}


def plot_confusion(y_true, y_pred, species, out_path):
    """Row-normalised confusion matrix. Rows = true species, so each row sums
    to 1 and the diagonal IS per-class recall. Off-diagonal bright cells are
    the confusable pairs -- the thing the species list was chosen to expose."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(species)))
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(species))); ax.set_yticks(range(len(species)))
    ax.set_xticklabels(species, rotation=90, fontsize=8)
    ax.set_yticklabels(species, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("wrote", out_path)


def save_results(results, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path)
