"""
Exploratory data analysis.

Written after the first models rather than before them, which is the wrong
order. It is included anyway because several questions the results raised can
only be answered by looking at the data: why Great Tit is the hardest species,
whether the energy-based window selection actually works, and whether the
recordist overlap we measured is spread thinly or concentrated in a few people.

Four analyses, each answering a question that affects how the results should
be read:

  recordists  Is the dataset dominated by a few prolific contributors, and is
              any single species mostly one person's recordings? Bears directly
              on whether the model can score by recognising a microphone.
  metadata    What KIND of vocalisation is each recording? A species label
              covers song, call, alarm and flight-call, which can be
              acoustically unrelated. Candidate explanation for the species
              that both models found hardest.
  windows     Does rank_windows_by_energy separate signal from silence, or did
              we assume a mitigation that does nothing? Checked by re-reading
              a sample of audio and comparing kept against discarded windows.
  frequency   Where does each species sit on the mel axis, and which species
              overlap? A numerical version of what the sanity check showed by
              eye, and a prediction of which confusions should exist.

Usage (Kaggle, where the audio and cache live):

    python -m src.eda --splits data/splits.csv \\
        --raw  /kaggle/input/notebooks/<user>/birdoo/data/raw \\
        --cache /kaggle/input/notebooks/<user>/birdsong-cache/cache \\
        --out  /kaggle/working/eda

Each stage is skipped with a warning if its input is missing, so the recordist
analysis can be run locally from splits.csv alone.
"""

import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE, N_MELS, FMIN, FMAX

# Vocalisation types are free text written by thousands of recordists
# ("song", "male song", "alarm call, flight call"). Bucket them rather than
# trusting the raw strings.
TYPE_BUCKETS = [
    ("drumming", ["drum"]),
    ("alarm", ["alarm", "scold", "mobbing"]),
    ("flight call", ["flight"]),
    ("subsong", ["subsong", "sub-song"]),
    ("begging", ["begging", "juvenile", "nestling"]),
    ("song", ["song"]),
    ("call", ["call"]),
]


def bucket_type(raw):
    """Map a free-text type field to one bucket. Order matters: 'flight call'
    must be caught before the generic 'call', and 'subsong' before 'song'."""
    t = (raw or "").lower()
    for name, keys in TYPE_BUCKETS:
        if any(k in t for k in keys):
            return name
    return "other/unspecified"


def parse_length(raw):
    """Xeno-canto gives length as 'M:SS' or 'H:MM:SS'. Return seconds."""
    if not raw:
        return None
    try:
        parts = [int(p) for p in str(raw).split(":")]
    except ValueError:
        return None
    secs = 0
    for p in parts:
        secs = secs * 60 + p
    return secs


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ loading

def load_splits(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def attach_metadata(rows, raw_dir):
    """Join splits.csv to the per-species metadata.jsonl written at download
    time. Reports rows that did not match, so a partial download does not
    silently produce a biased analysis."""
    meta = {}
    for mpath in sorted(Path(raw_dir).glob("*/metadata.jsonl")):
        with open(mpath, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                meta[str(r["id"])] = r

    joined, missing = [], 0
    for r in rows:
        m = meta.get(str(r["id"]))
        if m is None:
            missing += 1
            continue
        joined.append({**r, **{k: m.get(k) for k in
                               ("country", "quality", "length", "type",
                                "sample_rate", "licence", "english_name")}})
    if missing:
        print(f"  warning: {missing} recordings in splits.csv had no metadata row")
    return joined


# ------------------------------------------------------------ 1. recordists

def analyse_recordists(rows, out_dir, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = collections.Counter(r["recordist"] for r in rows if r["recordist"])
    n_rec, n_people = len(rows), len(counts)
    ordered = np.array(sorted(counts.values())[::-1])

    def share(k):
        return 100.0 * ordered[:k].sum() / n_rec

    # Per species: what fraction came from that species' single most prolific
    # recordist? A high number means the class is largely one person's gear.
    dominance = []
    for sp in sorted({r["species"] for r in rows}):
        c = collections.Counter(r["recordist"] for r in rows if r["species"] == sp)
        top_n = c.most_common(1)[0][1]
        dominance.append((sp, 100.0 * top_n / sum(c.values()), len(c)))
    dominance.sort(key=lambda x: -x[1])

    # Cross-split leakage restated per species: recordists shared train<->test.
    shared = []
    for sp in sorted({r["species"] for r in rows}):
        tr = {r["recordist"] for r in rows
              if r["species"] == sp and r["split"] == "train"}
        te = {r["recordist"] for r in rows
              if r["species"] == sp and r["split"] == "test" and r["recordist"]}
        shared.append((sp, 100.0 * len(tr & te) / max(len(te), 1)))
    shared.sort(key=lambda x: -x[1])

    stats["recordists"] = {
        "n_recordings": n_rec,
        "n_recordists": n_people,
        "top10_share_pct": round(share(10), 1),
        "top50_share_pct": round(share(50), 1),
        "singletons": int((ordered == 1).sum()),
        "max_by_one_person": int(ordered[0]),
        "per_species_top_recordist_share_pct": {sp: round(p, 1) for sp, p, _ in dominance},
        "per_species_test_recordists_also_in_train_pct": {sp: round(p, 1) for sp, p in shared},
    }

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    cum = 100.0 * np.cumsum(ordered) / n_rec
    ax[0].plot(np.arange(1, len(cum) + 1), cum, color="#1f4e79")
    ax[0].axhline(50, ls="--", lw=0.8, color="grey")
    ax[0].set_xlabel("recordists, most prolific first")
    ax[0].set_ylabel("cumulative % of all recordings")
    ax[0].set_title(f"Contribution concentration\n{n_people} recordists, {n_rec} recordings")
    ax[0].set_xlim(0, len(cum))
    ax[0].set_ylim(0, 100)

    top = dominance[:15]
    ax[1].barh([s for s, _, _ in top][::-1], [p for _, p, _ in top][::-1], color="#c00000")
    ax[1].set_xlabel("% of that species from its single most prolific recordist")
    ax[1].set_title("Per-species recordist dominance (worst 15)")
    ax[1].tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    save_fig(fig, out_dir / "eda_recordists.png")


# -------------------------------------------------------------- 2. metadata

def analyse_metadata(rows, out_dir, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r in rows:
        r["_type"] = bucket_type(r.get("type"))
        r["_secs"] = parse_length(r.get("length"))

    types = collections.Counter(r["_type"] for r in rows)
    countries = collections.Counter(r.get("country") or "unknown" for r in rows)
    quality = collections.Counter(r.get("quality") or "?" for r in rows)
    secs = [r["_secs"] for r in rows if r["_secs"]]

    # Type diversity per species: a species whose recordings spread across
    # several vocalisation types is one label covering several sounds.
    per_species_types = {}
    for sp in sorted({r["species"] for r in rows}):
        c = collections.Counter(r["_type"] for r in rows if r["species"] == sp)
        total = sum(c.values())
        p = np.array([v / total for v in c.values()])
        # Normalised entropy: 0 = a single type, 1 = evenly spread over buckets.
        ent = float(-(p * np.log(p)).sum() / np.log(len(TYPE_BUCKETS) + 1))
        per_species_types[sp] = {
            "counts": dict(c.most_common()),
            "dominant_pct": round(100.0 * c.most_common(1)[0][1] / total, 1),
            "type_entropy": round(ent, 3),
        }

    mixed = sorted(per_species_types.items(), key=lambda kv: -kv[1]["type_entropy"])

    stats["metadata"] = {
        "vocalisation_types": dict(types.most_common()),
        "quality_grades": dict(quality.most_common()),
        "n_countries": len(countries),
        "top_countries": dict(countries.most_common(10)),
        "duration_seconds": {
            "median": float(np.median(secs)) if secs else None,
            "mean": round(float(np.mean(secs)), 1) if secs else None,
            "p90": float(np.percentile(secs, 90)) if secs else None,
            "max": float(np.max(secs)) if secs else None,
        },
        "per_species_types": per_species_types,
        "most_type_mixed_species": [s for s, _ in mixed[:5]],
        "least_type_mixed_species": [s for s, _ in mixed[-5:]],
    }

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    t = types.most_common()
    ax[0][0].bar([k for k, _ in t], [v for _, v in t], color="#1f4e79")
    ax[0][0].set_title("Vocalisation type across the dataset")
    ax[0][0].set_ylabel("recordings")
    ax[0][0].tick_params(axis="x", rotation=30, labelsize=8)

    ax[0][1].hist(secs, bins=40, color="#1f4e79")
    ax[0][1].set_title("Recording duration (download filter was 3-60 s)")
    ax[0][1].set_xlabel("seconds")
    ax[0][1].set_ylabel("recordings")

    c = countries.most_common(12)
    ax[1][0].barh([k for k, _ in c][::-1], [v for _, v in c][::-1], color="#1f4e79")
    ax[1][0].set_title(f"Top countries of {len(countries)} represented")
    ax[1][0].tick_params(axis="y", labelsize=8)

    ent_sorted = sorted(per_species_types.items(), key=lambda kv: kv[1]["type_entropy"])
    ax[1][1].barh([s for s, _ in ent_sorted], [v["type_entropy"] for _, v in ent_sorted],
                  color="#c00000")
    ax[1][1].set_title("Vocalisation-type diversity per species\n"
                       "(higher = one label, many sounds)")
    ax[1][1].tick_params(axis="y", labelsize=6)

    fig.tight_layout()
    save_fig(fig, out_dir / "eda_metadata.png")


# ---------------------------------------------------- 3. window selection

def analyse_windows(rows, cache_dir, raw_dir, out_dir, stats, audio_sample):
    """Two questions: how many windows per recording, and does the top-k
    energy selection actually pick different windows from the ones it drops?

    The cache stores ONLY the kept windows, so the second question cannot be
    answered from it. We re-read a sample of audio and recompute all windows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache_dir = Path(cache_dir)
    with open(cache_dir / "index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))
    n_windows = np.array([int(r["n_windows"]) for r in index])

    result = {
        "n_recordings": len(index),
        "total_windows": int(n_windows.sum()),
        "windows_per_recording": {
            "min": int(n_windows.min()), "max": int(n_windows.max()),
            "mean": round(float(n_windows.mean()), 2),
            "at_cap": int((n_windows == n_windows.max()).sum()),
        },
    }

    kept_e, dropped_e, ratios = [], [], []
    if raw_dir and audio_sample > 0 and Path(raw_dir).exists():
        import librosa
        from .preprocess import load_audio, slice_windows, rank_windows_by_energy
        from .config import N_FFT, HOP_LENGTH

        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
        band = (freqs >= 1000) & (freqs <= 10000)

        rng = np.random.default_rng(0)
        sample = rng.choice(len(rows), size=min(audio_sample, len(rows)), replace=False)
        for i in sample:
            path = Path(raw_dir) / rows[i]["file"]
            if not path.exists():
                continue
            try:
                windows = slice_windows(load_audio(path))
                if len(windows) <= 5:
                    continue
                order = rank_windows_by_energy(windows)
                energies = np.array([
                    float(np.abs(librosa.stft(w, n_fft=N_FFT,
                                              hop_length=HOP_LENGTH))[band].mean())
                    for w in windows
                ])
                k, d = energies[order[:5]], energies[order[5:]]
                kept_e.extend(k.tolist())
                dropped_e.extend(d.tolist())
                if d.mean() > 0:
                    ratios.append(float(k.mean() / d.mean()))
            except Exception as e:
                print(f"  skip {path.name}: {e}")

    if ratios:
        result["energy_selection"] = {
            "recordings_sampled": len(ratios),
            "median_kept_over_dropped_energy_ratio": round(float(np.median(ratios)), 2),
            "pct_recordings_where_kept_louder": round(
                100.0 * float(np.mean([r > 1 for r in ratios])), 1),
            "mean_kept_energy": round(float(np.mean(kept_e)), 5),
            "mean_dropped_energy": round(float(np.mean(dropped_e)), 5),
        }

    stats["windows"] = result

    ncols = 2 if ratios else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.2), squeeze=False)
    ax = axes[0]

    ax[0].hist(n_windows, bins=np.arange(0.5, n_windows.max() + 1.5), color="#1f4e79")
    ax[0].set_xlabel("windows kept per recording")
    ax[0].set_ylabel("recordings")
    ax[0].set_title(f"Windows per recording\n"
                    f"{n_windows.sum():,} windows from {len(index):,} recordings")

    if ratios:
        r = result["energy_selection"]
        ax[1].hist(np.log10(np.array(dropped_e) + 1e-6), bins=50, alpha=0.65,
                   label="discarded windows", color="#999999")
        ax[1].hist(np.log10(np.array(kept_e) + 1e-6), bins=50, alpha=0.65,
                   label="kept (top 5)", color="#c00000")
        ax[1].set_xlabel("log10 mean energy, 1-10 kHz band")
        ax[1].set_ylabel("windows")
        ax[1].set_title("Does energy ranking separate signal from silence?\n"
                        f"median kept/dropped ratio = "
                        f"{r['median_kept_over_dropped_energy_ratio']}x")
        ax[1].legend(fontsize=9)

    fig.tight_layout()
    save_fig(fig, out_dir / "eda_windows.png")


# ------------------------------------------------------- 4. frequency niches

def analyse_frequency(cache_dir, out_dir, stats):
    """Mean mel profile per species: where on the frequency axis does each
    species put its energy? A numerical version of what the sanity check showed
    by eye, and a prediction of which species should be confusable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cache_dir = Path(cache_dir)
    with open(cache_dir / "index.csv", encoding="utf-8") as f:
        index = list(csv.DictReader(f))

    sums = collections.defaultdict(lambda: np.zeros(N_MELS, dtype=np.float64))
    counts = collections.Counter()
    for r in index:
        arr = np.load(cache_dir / r["npy"])            # (n, mels, frames) uint8
        sums[r["species"]] += arr.mean(axis=(0, 2)).astype(np.float64)
        counts[r["species"]] += 1

    species = sorted(sums)
    profiles = np.stack([sums[s] / counts[s] for s in species])     # (30, mels)

    # Centre each profile so similarity reflects SHAPE, not overall loudness,
    # then cosine-compare. High similarity = same frequency niche = confusable.
    centred = profiles - profiles.mean(axis=1, keepdims=True)
    norm = centred / (np.linalg.norm(centred, axis=1, keepdims=True) + 1e-9)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -1.0)

    pairs = []
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            pairs.append((float(sim[i, j]), species[i], species[j]))
    pairs.sort(reverse=True)

    # Mel bins are not linear in Hz, but a linear label is close enough to read
    # the figure; the exact edges are FMIN..FMAX in config.
    mel_hz = np.linspace(FMIN, FMAX, N_MELS)
    peak_hz = {s: int(mel_hz[int(np.argmax(profiles[k]))]) for k, s in enumerate(species)}

    stats["frequency"] = {
        "peak_band_hz_per_species": dict(sorted(peak_hz.items(), key=lambda kv: kv[1])),
        "most_similar_pairs": [{"a": a, "b": b, "cosine": round(c, 3)}
                               for c, a, b in pairs[:10]],
        "least_similar_pairs": [{"a": a, "b": b, "cosine": round(c, 3)}
                                for c, a, b in pairs[-5:]],
    }

    fig, ax = plt.subplots(1, 2, figsize=(14, 7.5),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    order = np.argsort([peak_hz[s] for s in species])
    im = ax[0].imshow(profiles[order], aspect="auto", cmap="magma", origin="lower")
    ax[0].set_yticks(range(len(species)))
    ax[0].set_yticklabels([species[i] for i in order], fontsize=7)
    xt = np.linspace(0, N_MELS - 1, 6).astype(int)
    ax[0].set_xticks(xt)
    ax[0].set_xticklabels([f"{mel_hz[t]/1000:.0f}k" for t in xt])
    ax[0].set_xlabel("approximate frequency (Hz)")
    ax[0].set_title("Mean mel profile per species\n(sorted by peak frequency)")
    fig.colorbar(im, ax=ax[0], fraction=0.035)

    im2 = ax[1].imshow(sim[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
    ax[1].set_xticks(range(len(species)))
    ax[1].set_yticks(range(len(species)))
    ax[1].set_xticklabels([species[i] for i in order], rotation=90, fontsize=6)
    ax[1].set_yticklabels([species[i] for i in order], fontsize=6)
    ax[1].set_title("Spectral-profile similarity\n(bright = same frequency niche)")
    fig.colorbar(im2, ax=ax[1], fraction=0.045)

    fig.tight_layout()
    save_fig(fig, out_dir / "eda_frequency.png")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="data/splits.csv")
    ap.add_argument("--raw", default=None, help="raw audio dir (metadata + energy check)")
    ap.add_argument("--cache", default=None, help="spectrogram cache dir")
    ap.add_argument("--out", default="outputs/eda")
    ap.add_argument("--audio-sample", type=int, default=150,
                    help="recordings to re-read for the energy check; 0 to skip")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    rows = load_splits(args.splits)
    print(f"loaded {len(rows)} recordings from {args.splits}")

    print("\n[1/4] recordists")
    analyse_recordists(rows, out_dir, stats)

    if args.raw and Path(args.raw).exists():
        print("\n[2/4] metadata")
        analyse_metadata(attach_metadata(rows, args.raw), out_dir, stats)
    else:
        print("\n[2/4] metadata -- SKIPPED (no --raw)")

    if args.cache and (Path(args.cache) / "index.csv").exists():
        print("\n[3/4] window selection")
        analyse_windows(rows, args.cache, args.raw, out_dir, stats, args.audio_sample)
        print("\n[4/4] frequency niches")
        analyse_frequency(args.cache, out_dir, stats)
    else:
        print("\n[3/4] [4/4] -- SKIPPED (no --cache)")

    with open(out_dir / "eda_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("\nwrote", out_dir / "eda_stats.json")

    # Headline numbers, so a run is readable without opening the JSON.
    print("\n" + "=" * 64)
    if "recordists" in stats:
        r = stats["recordists"]
        print(f"{r['n_recordists']} recordists for {r['n_recordings']} recordings; "
              f"top 10 = {r['top10_share_pct']}%, top 50 = {r['top50_share_pct']}%")
        worst = max(r["per_species_top_recordist_share_pct"].items(), key=lambda kv: kv[1])
        print(f"most single-recordist-dominated species: {worst[0]} at {worst[1]}%")
    if "metadata" in stats:
        m = stats["metadata"]
        print(f"vocalisation types: {m['vocalisation_types']}")
        print(f"most type-mixed species: {', '.join(m['most_type_mixed_species'])}")
    if "windows" in stats and "energy_selection" in stats["windows"]:
        e = stats["windows"]["energy_selection"]
        print(f"energy selection: kept windows {e['median_kept_over_dropped_energy_ratio']}x "
              f"louder than discarded (median), in "
              f"{e['pct_recordings_where_kept_louder']}% of recordings")
    if "frequency" in stats:
        p = stats["frequency"]["most_similar_pairs"][0]
        print(f"most spectrally similar pair: {p['a']} / {p['b']} (cosine {p['cosine']})")
    print("=" * 64)


if __name__ == "__main__":
    main()
