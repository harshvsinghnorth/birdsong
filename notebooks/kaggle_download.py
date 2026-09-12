"""
Xeno-canto download, run on Kaggle instead of locally.

WHY THIS EXISTS: Xeno-canto rate-limits per IP. From home we measured ~13 kB/s,
which put the full 4,500-recording download at roughly 35 hours. The same
request from a Kaggle notebook runs at ~780 kB/s, so the whole thing finishes
in under two hours. Nothing about the data changes -- only where it is fetched.

HOW TO RUN: paste this into one cell of a Kaggle notebook with Internet ON,
then use Save Version -> "Save & Run All (Commit)". Do NOT run it interactively
first: an interactive session's /kaggle/working is discarded, so you would
download everything twice. The commit run persists the output.
"""

import json
import time
from pathlib import Path

import requests
from kaggle_secrets import UserSecretsClient

# Key lives in Add-ons -> Secrets, not in the notebook source. A public Kaggle
# notebook is as exposed as a public repo.
API_KEY = UserSecretsClient().get_secret("XC_API_KEY")

API_BASE = "https://xeno-canto.org/api/3/recordings"
OUT_DIR = Path("/kaggle/working/data/raw")

# A/B grades only. Matches src/config.py -- see the reasoning there.
QUALITY_FILTER = 'q:">C"'
LENGTH_FILTER = "len:3-60"
PER_SPECIES = 150

SPECIES = [
    "Parus major", "Phylloscopus collybita", "Fringilla coelebs",
    "Turdus merula", "Erithacus rubecula", "Sylvia atricapilla",
    "Troglodytes troglodytes", "Dendrocopos major", "Cyanistes caeruleus",
    "Turdus philomelos", "Strix aluco", "Passer domesticus",
    "Sitta europaea", "Garrulus glandarius", "Alauda arvensis",
    "Luscinia megarhynchos", "Emberiza citrinella", "Anthus trivialis",
    "Chloris chloris", "Certhia brachydactyla", "Motacilla alba",
    "Hirundo rustica", "Carduelis carduelis", "Regulus regulus",
    "Aegithalos caudatus", "Sturnus vulgaris", "Pica pica",
    "Corvus corone", "Picus viridis", "Serinus serinus",
]


def api_get(params, retries=3):
    """GET with retry. 4xx is not retried -- a malformed query or a bad key
    will fail identically every time, and retrying it just hides the error."""
    for attempt in range(retries):
        r = requests.get(API_BASE, params={**params, "key": API_KEY}, timeout=60)
        if 400 <= r.status_code < 500 and r.status_code != 429:
            r.raise_for_status()
        try:
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5)


def list_recordings(species, limit):
    query = f'sp:"{species}" {QUALITY_FILTER} {LENGTH_FILTER}'
    collected, page = [], 1
    while len(collected) < limit:
        data = api_get({"query": query, "page": page})
        recs = data.get("recordings", [])
        if not recs:
            break
        collected.extend(recs)
        if page >= int(data.get("numPages", 1)):
            break
        page += 1
        time.sleep(1)
    return collected[:limit]


OUT_DIR.mkdir(parents=True, exist_ok=True)
t_start = time.time()

for i, sp in enumerate(SPECIES, 1):
    folder = OUT_DIR / sp.replace(" ", "_")
    folder.mkdir(exist_ok=True)

    recs = list_recordings(sp, PER_SPECIES)
    print(f"[{i}/{len(SPECIES)}] {sp}: {len(recs)} recordings", flush=True)

    with open(folder / "metadata.jsonl", "w") as meta_f:
        for rec in recs:
            url = rec.get("file")
            if not url:
                continue

            # Save with the ORIGINAL extension. The v3 endpoint serves whatever
            # the recordist uploaded -- writing wav bytes into a .mp3 file works
            # by luck with soundfile and is a trap for anyone reading the data
            # later.
            ext = rec.get("file-name", "").rsplit(".", 1)[-1].lower()
            ext = ext if ext in {"mp3", "wav", "flac", "ogg"} else "mp3"
            dest = folder / f"XC{rec['id']}.{ext}"

            if not dest.exists():
                try:
                    audio = requests.get(url, timeout=180)
                    audio.raise_for_status()
                    dest.write_bytes(audio.content)
                except Exception as e:
                    print(f"  skip XC{rec['id']}: {e}", flush=True)
                    continue

            meta_f.write(json.dumps({
                "id": rec["id"], "file": dest.name, "species": sp,
                "english_name": rec.get("en"), "country": rec.get("cnt"),
                "location": rec.get("loc"), "quality": rec.get("q"),
                "length": rec.get("length"), "recordist": rec.get("rec"),
                "licence": rec.get("lic"), "type": rec.get("type"),
                "sample_rate": rec.get("smp"),
            }) + "\n")

            time.sleep(0.2)  # still be polite to a free community archive

mins = (time.time() - t_start) / 60
n = sum(1 for _ in OUT_DIR.rglob("*") if _.is_file())
print(f"\nDone in {mins:.0f} min. {n} files in {OUT_DIR}")
