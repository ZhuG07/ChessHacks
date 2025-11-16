# training/selfplay/config.py

from pathlib import Path

# Path to your ChessHacks project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Reuse your existing model
MODEL_PATH = PROJECT_ROOT / "training" / "whiteNoise" / "checkpoints_delta_resnet20" / "best.pt"

# Path to Stockfish binary (adjust if needed)
STOCKFISH_PATH = "C:\Users\ethan\Downloads\ChessHacks\ChessHacks2025\training\whiteNoise\stockfish-windows-x86-64-avx2.exe"

# Output directory for self-play datasets
SELFPLAY_OUT_DIR = PROJECT_ROOT / "training" / "data" / "selfplay"
SELFPLAY_OUT_DIR.mkdir(parents=True, exist_ok=True)

# NN / policy config (must match your engine)
NUM_PLANES = 18
NUM_PROMOS = 5
POLICY_DIM = 64 * 64 * NUM_PROMOS
CP_SCALE = 200.0

# Self-play config
TOP_K_MOVES = 5         # how many moves per position for Stockfish evaluation
MAX_MOVES_PER_GAME = 300
NUM_GAMES = 50          # number of self-play games per run (tune)
SF_TIME_LIMIT = 0.05    # seconds per evaluation (tune)
POLICY_TEMP_CP = 120.0  # temperature for cp->policy softmax
