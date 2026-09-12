# Bird Species Classification from Acoustic Recordings

Multi-species bird classification from audio, using log-mel spectrograms and CNNs.

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Week 1: get data

Xeno-canto is open — no account or API key needed.

```bash
# 1. Find out which species actually have enough recordings.
python -m src.fetch_data survey
```

This writes `data/species_shortlist.csv`. **Open it and look at it.** Decide
which 30 species you want rather than blindly taking the top rows — you'll be
asked to justify this choice.

```bash
# 2. Download audio for the chosen species.
python -m src.fetch_data download
```

Expect several GB and a few hours. Metadata (recordist, licence, country) is
saved per species in `metadata.jsonl` — you need it for attribution in the report.

## Week 1: verify preprocessing

Before training anything:

```python
from pathlib import Path
from src.preprocess import sanity_check

files = list(Path("data/raw").rglob("*.mp3"))[:3]
sanity_check(files, "outputs/sanity.png")
```

Open the image. **You should be able to see call structure in the spectrograms.**
If you can't see it, the model can't learn it — fix preprocessing before touching
the architecture.

## Layout

```
src/config.py       all audio + dataset parameters (change here, not inline)
src/fetch_data.py   Xeno-canto survey + download
src/preprocess.py   audio -> windows -> log-mel spectrograms
data/raw/           downloaded audio (gitignored)
data/processed/     cached spectrograms (gitignored)
outputs/            figures, models, logs
```

## Design decisions to be able to defend

- **32 kHz sample rate** — bird calls have energy above 8 kHz, so 16 kHz would
  discard real signal.
- **5-second windows** — recordings vary in length; the model needs fixed input.
- **Per-example normalisation** — recordings come from many recordists with
  different gear. Normalising per example prevents the model learning
  "this person's microphone" as a shortcut.
- **Energy-based window ranking** — a recording is labelled with a species but
  most windows may be silence. Ranking by energy in the 1–10 kHz band is a cheap
  first attack on this weak-label problem.

## Not done yet

Dataset class, model, training loop, evaluation, augmentation.
