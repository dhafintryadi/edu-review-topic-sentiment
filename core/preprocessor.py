"""
core/preprocessor.py
====================
Hybrid text preprocessing for Indonesian educational platform reviews.
Wraps the existing ReviewPreprocessor class — 100% logic-preserved.
All resources loaded from core/resources/ — zero dependency on Preprocessing/.
"""
import sys
import logging
from pathlib import Path

# ── Path anchoring ────────────────────────────────────────────────────────────
CORE_DIR      = Path(__file__).resolve().parent       # core/
REPO_ROOT     = CORE_DIR.parent                       # repo root
ASSETS_DIR    = REPO_ROOT / "assets"
RESOURCES_DIR = CORE_DIR / "resources"                # core/resources/

# The preprocessing package is still the authoritative implementation.
# It lives inside core/preprocessing/ after migration.
_PREPROC_PKG = CORE_DIR / "preprocessing"
if str(_PREPROC_PKG) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from preprocessing.pipeline import ReviewPreprocessor  # noqa: E402

LOGGER = logging.getLogger(__name__)

# ── Resource paths (core/resources/) ──────────────────────────────────────────
SLANG_DICT    = RESOURCES_DIR / "slang_dict.json"
TYPO_DICT     = RESOURCES_DIR / "slang_dict.json"
STOPWORD_FILE = RESOURCES_DIR / "stopwords.txt"


def run_preprocessing(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    content_column: str = "content",
    chunksize: int = 5000,
) -> None:
    """
    Run hybrid preprocessing on raw_review.csv.

    Parameters
    ----------
    input_path  : Path to source CSV (defaults to assets/raw_review.csv)
    output_dir  : Directory to write sentiment_preprocessed.csv and
                  topic_preprocessed.csv (defaults to assets/)
    content_column : Column name holding the review text.
    chunksize   : Rows per processing chunk.
    """
    input_path = input_path or (ASSETS_DIR / "raw_review.csv")
    output_dir = output_dir or ASSETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = ReviewPreprocessor(
        slang_path=str(SLANG_DICT),
        stopword_path=str(STOPWORD_FILE),
        typo_path=str(TYPO_DICT),
    )
    preprocessor.preprocess_csv(
        input_path=str(input_path),
        output_dir=str(output_dir),
        content_column=content_column,
        chunksize=chunksize,
    )
    LOGGER.info("Preprocessing complete. Outputs in: %s", output_dir)
