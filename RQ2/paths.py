from pathlib import Path


RQ2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RQ2_ROOT.parent

DATA_ROOT = REPO_ROOT / "data"
DATA_ARCHIVED = DATA_ROOT / "Archived"
DATA_TOMIRIS = DATA_ROOT / "Archived_Tomiris"

RQ1_DATA_ROOT = REPO_ROOT / "RQ1" / "data" / "Overfitting"

SAMPLE_ROOT = RQ2_ROOT / "sample_features"
