import sys
import pickle
import subprocess
import json
import pandas as pd
from pathlib import Path

# ─── Repository root (directory containing this script) ──────────────────────
REPO_ROOT = Path(__file__).resolve().parent


def validate_pipeline_assets():
    """[CHECK] Verify binary model integrity and dataset existence."""
    raw_review_path = REPO_ROOT / "Datasets" / "raw_review.csv"
    baseline_model  = REPO_ROOT / "Sentiment-Analysis" / "nn-based" / "baseline_model.pkl"
    lda_model       = REPO_ROOT / "Topic-Modelling" / "phase3b_outputs" / "lda_model_final_k8.gensim"

    print("[CHECK] Verifying optimal model binaries and datasets...")

    for path in [raw_review_path, baseline_model, lda_model]:
        if not path.exists():
            print(f"[ERROR] Diagnostic: Missing file '{path.relative_to(REPO_ROOT)}'",
                  file=sys.stderr)
            sys.exit(1)

    # Binary integrity check — pickle.load must succeed and contain expected keys
    try:
        with open(baseline_model, "rb") as f:
            model_data = pickle.load(f)
        if not isinstance(model_data, dict) \
                or "model" not in model_data \
                or "vectorizer" not in model_data:
            raise KeyError("Serialized model is missing required 'model' or 'vectorizer' keys.")
    except Exception as e:
        print(f"[ERROR] Diagnostic: Binary failed verification: 'baseline_model.pkl' — {e}",
              file=sys.stderr)
        sys.exit(1)

    print("[CHECK] Verifying optimal model binaries and datasets... [OK]")


def run_preprocessing():
    """Stage 2 — Hybrid preprocessing of raw_review.csv."""
    script  = REPO_ROOT / "Preprocessing" / "hybrid_preprocessing" / "run_preprocessing.py"
    input_  = REPO_ROOT / "Datasets" / "raw_review.csv"
    out_dir = REPO_ROOT / "Datasets"

    cmd = [sys.executable, str(script),
           "--input", str(input_),
           "--output-dir", str(out_dir)]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] Preprocessing execution failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("[DEMO] 1. Preprocessing Raw Reviews... [SUCCESS]")


def run_sentiment_inference():
    """Stage 4 — Batch inference using the pre-trained baseline model."""
    script     = REPO_ROOT / "Sentiment-Analysis" / "nn-based" / "batch_inference.py"
    output_csv = REPO_ROOT / "Sentiment-Analysis" / "nn-based" / "sentiment_predictions.csv"

    # Remove stale output to guarantee a fresh run
    if output_csv.exists():
        output_csv.unlink()

    result = subprocess.run([sys.executable, str(script)],
                            capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] Sentiment Inference execution failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Column guardrail — ensure the unified output column exists
    if not output_csv.exists():
        print(f"[ERROR] Sentiment Inference output file not found: {output_csv.name}",
              file=sys.stderr)
        sys.exit(1)
    try:
        df = pd.read_csv(output_csv)
        if "predicted_label_name" not in df.columns:
            print("[ERROR] Column Guardrail Violation: 'predicted_label_name' missing "
                  "from inference output.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Guardrail verification failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("[DEMO] 2. Running Sentiment Inference (Verified Pre-trained Model)... [SUCCESS]")


def run_severity_calculation():
    """Stage 6 — Topic-Sentiment Severity Matrix via pre-trained LDA (phase3b optimal model)."""
    script = REPO_ROOT / "Topic-Modelling" / "run_phase3c.py"
    if not script.exists():
        print(f"[ERROR] Severity script not found: {script.name}", file=sys.stderr)
        sys.exit(1)

    # run_phase3c.py self-anchors via Path(__file__).resolve().parent — no path injection needed
    result = subprocess.run([sys.executable, str(script)],
                            capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"[ERROR] Severity calculation failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print("[DEMO] 3. Computing Topic-Sentiment Severity Matrix... [SUCCESS]")


def run_downstream_validation():
    """Stage 7 — Blueprint gap analysis (Phase 4) and system logic specification (Phase 5)."""
    stages = [
        (
            "Phase 4 Design Implication Synthesis",
            "run_phase4.py",
            REPO_ROOT / "Topic-Modelling" / "phase4_outputs" / "blueprint_gap_analysis.json",
        ),
        (
            "Phase 5 System Spec & AI Adaptive Design",
            "run_phase5.py",
            REPO_ROOT / "Topic-Modelling" / "phase5_outputs" / "sr_system_logic_specification.json",
        ),
    ]

    for label, script_name, artifact in stages:
        script = REPO_ROOT / "Topic-Modelling" / script_name
        if not script.exists():
            print(f"[ERROR] Script not found: {script_name}", file=sys.stderr)
            sys.exit(1)

        result = subprocess.run([sys.executable, str(script)],
                                capture_output=True, text=True, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"[ERROR] {label} failed:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

        if not artifact.exists():
            print(f"[ERROR] Expected artifact not generated: {artifact.name}", file=sys.stderr)
            sys.exit(1)

    print("[DEMO] 4. Generating Sekolah Rakyat Validation Artifacts... [SUCCESS]")


def main():
    validate_pipeline_assets()
    run_preprocessing()
    run_sentiment_inference()
    run_severity_calculation()
    run_downstream_validation()


if __name__ == "__main__":
    main()
