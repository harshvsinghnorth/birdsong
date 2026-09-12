"""
Central configuration.

Every module imports from here. If you change a value, EVERYTHING downstream
changes consistently -- this is the single most important file for avoiding
the silent preprocessing bugs that quietly ruin audio ML projects.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"          # downloaded audio, untouched
PROCESSED_DIR = ROOT / "data" / "processed"  # cached spectrograms
OUTPUT_DIR = ROOT / "outputs"            # figures, models, logs

# ---------------------------------------------------------------- audio
# 32 kHz keeps bird harmonics up to 16 kHz. Many bird calls have energy well
# above the 8 kHz ceiling you'd get from 16 kHz audio, so don't go lower
# without checking your species' frequency range first.
SAMPLE_RATE = 32000

# Length of one training example, in seconds. Each recording gets chopped
# into windows of this size. 5s is the common choice in BirdCLEF solutions.
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

# Overlap between consecutive windows when slicing a long recording.
# Overlap gives you more training data and reduces the chance a short call
# gets split across a window boundary.
WINDOW_OVERLAP = 0.5  # 50%

# ---------------------------------------------------------------- spectrogram
N_FFT = 2048          # FFT window size
HOP_LENGTH = 512      # samples between frames -> time resolution
N_MELS = 128          # mel bands -> height of the spectrogram "image"
FMIN = 50             # Hz. Below this is mostly wind/handling noise.
FMAX = 16000          # Hz. Nyquist limit for 32 kHz audio.

TOP_DB = 80.0         # dynamic range kept when converting power -> dB

# ---------------------------------------------------------------- dataset
# Start small and deliberate. 30 species is enough to be a real multi-class
# problem, small enough that the confusion matrix is readable.
N_SPECIES = 30
MIN_RECORDINGS_PER_SPECIES = 100

RANDOM_SEED = 42
