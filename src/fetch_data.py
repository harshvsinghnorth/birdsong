"""
Fetch bird recordings from Xeno-canto.

Xeno-canto is an open archive, but since API v2 was retired the v3 endpoint
requires a free API key. Get one at https://xeno-canto.org/account and put it
in the XC_API_KEY environment variable -- NEVER hardcode it here, this repo
is public.
API docs: https://xeno-canto.org/explore/api

Two-step workflow:

    python -m src.fetch_data survey     # find which species have the most data
    python -m src.fetch_data download   # actually pull the audio

The survey step writes species_shortlist.csv. LOOK AT THAT FILE and edit it
before downloading -- you want to make a deliberate choice about your species
set, not just take whatever the API returns first.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

from .config import RAW_DIR, N_SPECIES, MIN_RECORDINGS_PER_SPECIES

API_BASE = "https://xeno-canto.org/api/3/recordings"

# v3 authenticates every request with a key. Read from the environment so the
# key never enters version control.
API_KEY = os.environ.get("XC_API_KEY")

# Quality grades: A is cleanest, E is worst. Restricting to A/B gives you
# recordings where the target bird is clearly audible -- important because
# Xeno-canto labels are per-recording, not per-moment.
QUALITY_FILTER = 'q:">C"'

# Recording length filter. Very long files are usually soundscapes with the
# target species appearing briefly; very short ones are often clipped calls.
LENGTH_FILTER = "len:3-60"


def _get(params, retries=3):
    """GET with basic retry. Xeno-canto rate-limits, so we go slowly."""
    if not API_KEY:
        sys.exit(
            "XC_API_KEY is not set. Get a key at "
            "https://xeno-canto.org/account, then set it in PowerShell with: "
            'setx XC_API_KEY "your-key"  (then reopen the terminal)'
        )

    for attempt in range(retries):
        try:
            r = requests.get(API_BASE, params={**params, "key": API_KEY}, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}")
            time.sleep(5)


def survey(candidate_species, out_csv=None):
    """
    For each candidate species, ask the API how many quality recordings exist.

    candidate_species: list of scientific names, e.g. ["Turdus merula", ...]
    """
    out_csv = out_csv or (RAW_DIR.parent / "species_shortlist.csv")
    rows = []

    for i, sp in enumerate(candidate_species, 1):
        query = f'sp:"{sp}" {QUALITY_FILTER} {LENGTH_FILTER}'
        print(f"[{i}/{len(candidate_species)}] {sp} ...", end=" ", flush=True)
        data = _get({"query": query, "page": 1})
        n = int(data.get("numRecordings", 0))
        print(f"{n} recordings")
        rows.append({"scientific_name": sp, "n_recordings": n})
        time.sleep(1)  # be polite to a free community archive

    rows.sort(key=lambda r: r["n_recordings"], reverse=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scientific_name", "n_recordings"])
        w.writeheader()
        w.writerows(rows)

    eligible = [r for r in rows if r["n_recordings"] >= MIN_RECORDINGS_PER_SPECIES]
    print(f"\nWrote {out_csv}")
    print(f"{len(eligible)} species meet the {MIN_RECORDINGS_PER_SPECIES}-recording threshold.")
    print(f"Top {N_SPECIES} will be used unless you edit the file.")
    return rows


def _is_mp3_original(rec):
    """True if the recordist uploaded mp3 rather than wav.

    The v3 download endpoint serves the ORIGINAL file, and Xeno-canto rate
    limits to roughly 13 kB/s per IP. A short wav is ~2.5 MB against ~0.4 MB
    for the same audio as mp3, so wav recordings cost about six times the
    wall-clock download time for no extra information -- we resample to
    SAMPLE_RATE and compute log-mels either way, and mp3 at Xeno-canto
    bitrates preserves everything below our 16 kHz fmax.

    Caveat worth a sentence in the report: this is a sampling bias, not a
    neutral filter. Recordists who upload wav may systematically differ
    (better gear, more deliberate setups). We accept it because it is the
    difference between a nine-day download and an overnight one.
    """
    return rec.get("file-name", "").lower().endswith(".mp3")


def _list_recordings(species, max_recordings):
    """Page through the API collecting recording metadata for one species."""
    query = f'sp:"{species}" {QUALITY_FILTER} {LENGTH_FILTER}'
    collected, page = [], 1

    while len(collected) < max_recordings:
        data = _get({"query": query, "page": page})
        recs = data.get("recordings", [])
        if not recs:
            break
        # Filter AFTER the empty check, so a page with no mp3s does not look
        # like the end of the results and stop paging early.
        collected.extend(r for r in recs if _is_mp3_original(r))
        if page >= int(data.get("numPages", 1)):
            break
        page += 1
        time.sleep(1)

    return collected[:max_recordings]


def download(species_list, per_species=150):
    """
    Download audio for each species into data/raw/<Genus_species>/.

    Also writes metadata.jsonl per species so you keep recordist, licence,
    country and quality -- you need these for attribution in your report.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for sp in species_list:
        folder = RAW_DIR / sp.replace(" ", "_")
        folder.mkdir(exist_ok=True)
        meta_path = folder / "metadata.jsonl"

        print(f"\n=== {sp} ===")
        recs = _list_recordings(sp, per_species)
        print(f"  found {len(recs)} recordings")

        with open(meta_path, "w") as meta_f:
            for j, rec in enumerate(recs, 1):
                xc_id = rec.get("id")
                url = rec.get("file")
                if not url:
                    continue
                if url.startswith("//"):
                    url = "https:" + url

                dest = folder / f"XC{xc_id}.mp3"
                if dest.exists():
                    continue

                try:
                    audio = requests.get(url, timeout=60)
                    audio.raise_for_status()
                    dest.write_bytes(audio.content)
                except Exception as e:
                    print(f"  skip XC{xc_id}: {e}")
                    continue

                meta_f.write(json.dumps({
                    "id": xc_id,
                    "file": dest.name,
                    "species": sp,
                    "english_name": rec.get("en"),
                    "country": rec.get("cnt"),
                    "location": rec.get("loc"),
                    "quality": rec.get("q"),
                    "length": rec.get("length"),
                    "recordist": rec.get("rec"),
                    "licence": rec.get("lic"),
                    "type": rec.get("type"),
                }) + "\n")

                if j % 25 == 0:
                    print(f"  {j}/{len(recs)}")
                time.sleep(0.4)

    print("\nDone. Audio in", RAW_DIR)


# A starting shortlist of widely-recorded species. These are candidates for
# the survey step -- the point is to MEASURE which have enough data, not to
# assume. Replace or extend this list freely.
CANDIDATES = [
    "Turdus merula", "Erithacus rubecula", "Fringilla coelebs",
    "Parus major", "Sylvia atricapilla", "Troglodytes troglodytes",
    "Phylloscopus collybita", "Turdus philomelos", "Cyanistes caeruleus",
    "Emberiza citrinella", "Alauda arvensis", "Sturnus vulgaris",
    "Passer domesticus", "Carduelis carduelis", "Luscinia megarhynchos",
    "Corvus corone", "Columba palumbus", "Streptopelia decaocto",
    "Picus viridis", "Dendrocopos major", "Garrulus glandarius",
    "Pica pica", "Motacilla alba", "Anthus trivialis",
    "Certhia brachydactyla", "Sitta europaea", "Regulus regulus",
    "Aegithalos caudatus", "Chloris chloris", "Serinus serinus",
    "Hirundo rustica", "Apus apus", "Cuculus canorus",
    "Strix aluco", "Falco tinnunculus",
]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "survey"

    if cmd == "survey":
        survey(CANDIDATES)

    elif cmd == "download":
        shortlist = RAW_DIR.parent / "species_shortlist.csv"
        if not shortlist.exists():
            sys.exit("Run 'survey' first.")
        with open(shortlist) as f:
            rows = list(csv.DictReader(f))
        chosen = [r["scientific_name"] for r in rows
                  if int(r["n_recordings"]) >= MIN_RECORDINGS_PER_SPECIES][:N_SPECIES]
        print(f"Downloading {len(chosen)} species\n")
        download(chosen)

    else:
        sys.exit("Usage: python -m src.fetch_data [survey|download]")
