"""
Audio -> mel-spectrogram pipeline.

This is the module that will silently ruin your project if you get it wrong,
because bad preprocessing does not crash -- it just makes your model quietly
mediocre while you blame the architecture. Hence sanity_check() at the bottom:
run it, LOOK at the images, confirm you can see the calls.
"""

import numpy as np
import librosa

from .config import (
    SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_OVERLAP,
    N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX, TOP_DB,
)


def load_audio(path, sr=SAMPLE_RATE):
    """Load a file, resample to a fixed rate, force mono."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def slice_windows(y, window_samples=WINDOW_SAMPLES, overlap=WINDOW_OVERLAP):
    """
    Chop a recording into fixed-length windows.

    Why windows: Xeno-canto labels are per-RECORDING, but a 40s clip may
    contain only a few seconds of the target bird. Training on windows gives
    the model consistent input shapes and lets you later filter to the windows
    that actually contain signal (see rank_windows_by_energy).

    Short recordings are zero-padded rather than dropped.
    """
    step = int(window_samples * (1 - overlap))
    if len(y) <= window_samples:
        padded = np.zeros(window_samples, dtype=y.dtype)
        padded[:len(y)] = y
        return [padded]

    windows = []
    for start in range(0, len(y) - window_samples + 1, step):
        windows.append(y[start:start + window_samples])

    # keep the tail if it is a decent chunk
    tail = y[len(windows) * step:]
    if len(tail) > window_samples * 0.5:
        padded = np.zeros(window_samples, dtype=y.dtype)
        padded[:len(tail)] = tail[:window_samples]
        windows.append(padded)

    return windows


def mel_spectrogram(y, sr=SAMPLE_RATE):
    """
    Convert a waveform window into a log-mel spectrogram.

    Output shape: (N_MELS, time_frames). This is the "image" the CNN sees.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    return mel_db.astype(np.float32)


def normalize(spec):
    """
    Per-example normalisation to roughly [0, 1].

    Per-example (not dataset-wide) matters here: recordings come from many
    different people with different gear and gain settings. Normalising each
    example separately stops the model learning "this recordist's mic" as a
    shortcut feature instead of learning the bird.
    """
    lo, hi = spec.min(), spec.max()
    if hi - lo < 1e-6:
        return np.zeros_like(spec)
    return (spec - lo) / (hi - lo)


def rank_windows_by_energy(windows, sr=SAMPLE_RATE, fmin=1000, fmax=10000):
    """
    Score windows by how much energy sits in the typical bird-call band.

    This is a cheap first attack on the WEAK LABEL problem: a recording is
    labelled with a species, but most windows may be near-silence. Training on
    those teaches the model that silence means Species X. Ranking lets you keep
    the top-k windows per recording instead.

    Returns indices sorted from most to least energetic.
    """
    scores = []
    for w in windows:
        S = np.abs(librosa.stft(w, n_fft=N_FFT, hop_length=HOP_LENGTH))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        band = (freqs >= fmin) & (freqs <= fmax)
        scores.append(float(S[band].mean()))
    return list(np.argsort(scores)[::-1])


def process_file(path, top_k=None):
    """
    Full path: file -> list of normalised log-mel spectrograms.

    top_k: if set, keep only the k most energetic windows.
    """
    y = load_audio(path)
    windows = slice_windows(y)

    if top_k is not None and len(windows) > top_k:
        keep = rank_windows_by_energy(windows)[:top_k]
        windows = [windows[i] for i in sorted(keep)]

    return [normalize(mel_spectrogram(w)) for w in windows]


def sanity_check(paths, out_path):
    """
    Plot waveform + spectrogram for a few files.

    RUN THIS BEFORE TRAINING ANYTHING. If you cannot visually see call
    structure in the spectrogram, the model cannot learn it either, and the
    problem is here -- not in your architecture.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(paths)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, p in enumerate(paths):
        y = load_audio(p)
        # Plot the most energetic window, not the first. The first 5s of a
        # Xeno-canto recording is often the recordist settling in; showing
        # that would tell you nothing about whether the pipeline preserves
        # call structure. This also lets you eyeball the energy ranking.
        windows = slice_windows(y)
        best = rank_windows_by_energy(windows)[0]
        spec = normalize(mel_spectrogram(windows[best]))

        axes[0, i].plot(np.linspace(0, len(y) / SAMPLE_RATE, len(y)), y, lw=0.4)
        axes[0, i].set_title(f"{getattr(p, 'name', str(p))}\nwaveform")
        axes[0, i].set_xlabel("seconds")

        im = axes[1, i].imshow(spec, aspect="auto", origin="lower", cmap="magma")
        axes[1, i].set_title(f"log-mel {spec.shape}  window {best + 1}/{len(windows)} (peak energy)")
        axes[1, i].set_xlabel("frames")
        axes[1, i].set_ylabel("mel bins")
        fig.colorbar(im, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print("wrote", out_path)
