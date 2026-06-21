"""
run_pipeline.py
===============
Native Python orchestrator for the Sekolah Rakyat NLP pipeline.
Imports core modules directly — no subprocess shell hacks.

Usage:
    C:\\Users\\ASUS\\miniconda3\\python.exe run_pipeline.py
"""
import sys
import pickle
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER    = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets"


# ── Stage 0: Asset validation ──────────────────────────────────────────────────
def validate_assets() -> None:
    required = {
        "assets/raw_review.csv":               ASSETS_DIR / "raw_review.csv",
        "assets/baseline_model.pkl":           ASSETS_DIR / "baseline_model.pkl",
        "assets/lda_model_final_k8.gensim":    ASSETS_DIR / "lda_model_final_k8.gensim",
    }
    print("[CHECK] Verifying optimal model binaries and datasets...")
    for label, path in required.items():
        if not path.exists():
            print(f"[ERROR] Missing required asset: {label}", file=sys.stderr)
            sys.exit(1)

    # Binary integrity check
    try:
        with open(ASSETS_DIR / "baseline_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        if "model" not in model_data or "vectorizer" not in model_data:
            raise KeyError("Missing 'model' or 'vectorizer' keys")
    except Exception as e:
        print(f"[ERROR] baseline_model.pkl failed verification: {e}", file=sys.stderr)
        sys.exit(1)

    print("[CHECK] Verifying optimal model binaries and datasets... [OK]")


# ── Stage 1: Preprocessing ────────────────────────────────────────────────────
def stage_preprocessing() -> None:
    from core.preprocessor import run_preprocessing
    try:
        run_preprocessing(
            input_path=ASSETS_DIR / "raw_review.csv",
            output_dir=ASSETS_DIR,
        )
        print("[DEMO] 1. Preprocessing Raw Reviews... [SUCCESS]")
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}", file=sys.stderr)
        sys.exit(1)


# ── Stage 2: Sentiment inference ──────────────────────────────────────────────
def stage_sentiment_inference() -> None:
    from core.sentiment_engine import run_sentiment_inference
    try:
        run_sentiment_inference(
            data_path=ASSETS_DIR / "sentiment_preprocessed.csv",
            model_path=ASSETS_DIR / "baseline_model.pkl",
            output_path=ASSETS_DIR / "sentiment_predictions.csv",
        )
        print("[DEMO] 2. Running Sentiment Inference (Verified Pre-trained Model)... [SUCCESS]")
    except Exception as e:
        print(f"[ERROR] Sentiment inference failed: {e}", file=sys.stderr)
        sys.exit(1)


# ── Stage 3: Topic severity ────────────────────────────────────────────────────
def stage_topic_severity() -> None:
    from core.topic_engine import run_topic_severity
    try:
        run_topic_severity(
            lda_model_path=ASSETS_DIR / "lda_model_final_k8.gensim",
        )
        print("[DEMO] 3. Computing Topic-Sentiment Severity Matrix... [SUCCESS]")
    except Exception as e:
        print(f"[ERROR] Severity calculation failed: {e}", file=sys.stderr)
        sys.exit(1)


# ── Stage 4: SR specification generation ──────────────────────────────────────
def stage_artifact_generation() -> None:
    from core.severity_analyzer import run_severity_mapping
    try:
        run_severity_mapping()
        print("[DEMO] 4. Generating Sekolah Rakyat Validation Artifacts... [SUCCESS]")
    except Exception as e:
        print(f"[ERROR] Artifact generation failed: {e}", file=sys.stderr)
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    validate_assets()
    stage_preprocessing()
    stage_sentiment_inference()
    stage_topic_severity()
    stage_artifact_generation()


if __name__ == "__main__":
    main()
