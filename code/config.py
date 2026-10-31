"""Single source of truth for every parameter.

No other script may hard-code these values; import them from here.
"""
from pathlib import Path

# ---- Paths ----
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
RESULTS_DIR: Path = ROOT / "results"
DATA_FILE: Path = DATA_DIR / "wdbc.data"  # official UCI file: id, diagnosis, 30 features, no header
CHECKSUM_FILE: Path = DATA_DIR / "wdbc.data.sha256"
UCI_DATASET_ID: int = 17  # Breast Cancer Wisconsin (Diagnostic)

# ---- Data ----
N_SAMPLES: int = 569
N_FEATURES: int = 30  # d = n_0
_BASE = ("radius", "texture", "perimeter", "area", "smoothness", "compactness",
         "concavity", "concave points", "symmetry", "fractal dimension")
# UCI column order: 10 means, 10 standard errors, 10 worst values
FEATURE_NAMES: tuple[str, ...] = tuple(f"{k} {b}" for k in ("mean", "se", "worst") for b in _BASE)
POSITIVE_LABEL: str = "M"  # malignant -> 1
NEGATIVE_LABEL: str = "B"  # benign    -> 0

# ---- Split (stratified 70/15/15) ----
SPLIT_SEED: int = 42
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
TEST_FRAC: float = 0.15

# ---- Model / training ----
INIT_SEEDS: tuple[int, ...] = tuple(range(10))  # seeds 0-9
LEARNING_RATES: tuple[float, ...] = (0.01, 0.1, 0.5)  # final eta chosen on validation
HIDDEN_WIDTHS: tuple[int, ...] = (4, 8, 16)  # n_1; one hidden layer
LAMBDAS: tuple[float, ...] = (0.0, 1e-3, 1e-2)  # chosen on validation
HIDDEN_ACTIVATION: str = "tanh"
OUTPUT_ACTIVATION: str = "sigmoid"
LOSS: str = "binary_cross_entropy"
MAX_EPOCHS: int = 5000  # full-batch: 1 epoch = 1 iteration
GRAD_TOL: float = 1e-6  # stop when ||grad J||_2 < GRAD_TOL
INIT_SCALE: float = 0.1  # CONFIRMED: "small scale" for random-normal init; biases 0
FD_STEP: float = 1e-5  # finite-difference step h (gradient_check.py)
