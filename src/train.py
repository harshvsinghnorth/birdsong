"""
Train BirdCNN on the spectrogram cache.

    python -m src.train <cache_dir> <out_dir> [--epochs 30] [--batch 64] [--lr 1e-3]

The loop, in order:
  1. forward + BCE loss on a batch of windows
  2. backward + AdamW step, cosine learning-rate decay across the run
  3. after each epoch, score VAL at the recording level (evaluate.py)
  4. keep the checkpoint with the best val macro-F1; stop after 6 epochs
     without improvement
  5. touch TEST exactly once, at the end, with that checkpoint

Val is used to choose the checkpoint. Test is used for nothing except the
final number. That separation is what makes the number honest.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .config import RANDOM_SEED
from .dataset import BirdWindows
from .model import BirdCNN, count_params
from .evaluate import (aggregate_by_recording, recording_labels, report,
                       plot_confusion, save_results)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, loss_fn=None):
    """Run the model over a loader. Returns per-window probabilities, the
    integer label and recording id of each window, and mean loss."""
    model.eval()
    probs, labels, rec_ids, losses = [], [], [], []
    for x, y, rid in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_fn is not None:
            losses.append(loss_fn(logits, y).item())
        probs.append(torch.sigmoid(logits).cpu().numpy())   # sigmoid, not softmax
        labels.append(y.argmax(1).cpu().numpy())            # multi-hot -> index (single-label data)
        rec_ids.append(rid.numpy())
    return (np.concatenate(probs), np.concatenate(labels), np.concatenate(rec_ids),
            float(np.mean(losses)) if losses else None)


def recording_level(probs, labels, rec_ids):
    """Windows -> recordings: mean the probabilities, take the argmax."""
    _, rec_probs = aggregate_by_recording(probs, rec_ids)
    return recording_labels(labels, rec_ids), rec_probs.argmax(1)


def score(model, loader, species, device, loss_fn, title, out_dir, tag):
    """Full recording-level report for one split, via the shared evaluate.py."""
    probs, labels, rec_ids, loss = predict(model, loader, device, loss_fn)
    y_true, y_pred = recording_level(probs, labels, rec_ids)
    r = report(y_true, y_pred, species, title=title)
    r["loss"] = loss
    r["window_accuracy"] = float((probs.argmax(1) == labels).mean())
    plot_confusion(y_true, y_pred, species, Path(out_dir) / f"confusion_cnn_{tag}.png")
    return r


def train_one_epoch(model, loader, opt, loss_fn, scaler, device):
    model.train()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        # Mixed precision: half-precision matmuls on the GPU, ~2x faster on a
        # T4, with the loss scaled so small gradients do not underflow.
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


def plot_history(history, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("BCE loss")
    ax[0].legend()
    ax[1].plot(ep, [h["val_macro_f1"] for h in history])
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("val macro F1 (per recording)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    seed_everything(RANDOM_SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = BirdWindows(args.cache_dir, "train")
    species = train_ds.species
    val_ds = BirdWindows(args.cache_dir, "val", species=species)
    test_ds = BirdWindows(args.cache_dir, "test", species=species)
    print(f"windows  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  classes={len(species)}")

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
    train_dl, val_dl, test_dl = loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)

    model = BirdCNN(n_classes=len(species)).to(device)
    print(f"params: {count_params(model):,}")

    # Per-class BCE: 30 independent yes/no questions per window. Softmax would
    # force the answers to sum to 1, which is wrong once a window can contain
    # two species.
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    history, best_f1, best_epoch, since_best = [], -1.0, 0, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, opt, loss_fn, scaler, device)
        sched.step()

        probs, labels, rec_ids, val_loss = predict(model, val_dl, device, loss_fn)
        y_true, y_pred = recording_level(probs, labels, rec_ids)
        val_f1 = float(f1_score(y_true, y_pred, average="macro",
                                labels=range(len(species)), zero_division=0))
        val_acc = float((y_true == y_pred).mean())

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_f1": val_f1, "val_accuracy": val_acc,
                        "lr": opt.param_groups[0]["lr"]})

        flag = ""
        if val_f1 > best_f1:
            best_f1, best_epoch, since_best = val_f1, epoch, 0
            torch.save({"model": model.state_dict(), "species": species, "epoch": epoch},
                       out_dir / "best.pth")
            flag = "  *best*"
        else:
            since_best += 1

        print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"val_acc {val_acc:.3f}  val_F1 {val_f1:.3f}  [{(time.time() - t0) / 60:.1f} min]{flag}",
              flush=True)
        if since_best >= args.patience:
            print(f"no val improvement for {args.patience} epochs -- stopping")
            break

    # Final scoring with the BEST checkpoint, not the last one.
    print(f"\nbest val macro-F1 {best_f1:.3f} at epoch {best_epoch}")
    ckpt = torch.load(out_dir / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    results = {
        "params": count_params(model),
        "best_epoch": best_epoch,
        "args": vars(args),
        "val": score(model, val_dl, species, device, loss_fn, "CNN -- val (per recording)", out_dir, "val"),
        "test": score(model, test_dl, species, device, loss_fn, "CNN -- test (per recording)", out_dir, "test"),
        "history": history,
    }
    save_results(results, out_dir / "results_cnn.json")
    plot_history(history, out_dir / "training_curves.png")


if __name__ == "__main__":
    main()
