"""
core/sentiment_engine.py
========================
TF-IDF + Logistic Regression batch inference engine.
Logic extracted verbatim from Sentiment-Analysis/nn-based/batch_inference.py.
Data contract: output column is ALWAYS 'predicted_label_name'.
"""
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

LOGGER = logging.getLogger(__name__)

CORE_DIR   = Path(__file__).resolve().parent
REPO_ROOT  = CORE_DIR.parent
ASSETS_DIR = REPO_ROOT / "assets"

# ── Model path (loaded from assets/) ──────────────────────────────────────────
DEFAULT_MODEL_PATH  = ASSETS_DIR / "baseline_model.pkl"
DEFAULT_DATA_PATH   = ASSETS_DIR / "sentiment_preprocessed.csv"
DEFAULT_OUTPUT_PATH = ASSETS_DIR / "sentiment_predictions.csv"


# ── Exact logic from batch_inference.py — zero changes ────────────────────────

def load_model_and_vectorizer(model_path: Path):
    """Load trained model and vectorizer"""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    label_mapping = model_data.get(
        "label_mapping", {"negative": 0, "neutral": 1, "positive": 2}
    )
    return model, vectorizer, label_mapping


def load_data(data_path: Path) -> pd.DataFrame:
    """Load sentiment preprocessed data"""
    df = pd.read_csv(data_path)
    LOGGER.info("Loaded %d samples from %s", len(df), data_path)
    return df


def preprocess_texts(texts, vectorizer):
    """Preprocess texts using the trained vectorizer"""
    return vectorizer.transform(texts)


def batch_predict(model, X, batch_size: int = 1000):
    """Perform batch prediction to avoid memory issues"""
    predictions, probabilities = [], []
    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X[i:end_idx]
        predictions.extend(model.predict(batch_X))
        probabilities.extend(model.predict_proba(batch_X))
        LOGGER.info("Processed %d/%d samples...", end_idx, n_samples)
    return np.array(predictions), np.array(probabilities)


def format_predictions(predictions, probabilities, label_mapping):
    """Format predictions with labels and confidence scores"""
    formatted = []
    for pred, proba in zip(predictions, probabilities):
        confidence = float(np.max(proba))
        prob_dict = {
            cls: round(float(proba[i]), 4)
            for i, cls in enumerate(label_mapping.keys())
        }
        formatted.append({
            "predicted_label": pred,
            "confidence_score": round(confidence, 4),
            "probabilities": prob_dict,
        })
    return formatted


def attach_predictions_to_dataset(df: pd.DataFrame, predictions: list) -> pd.DataFrame:
    """Attach predictions to the original dataset.
    
    DATA CONTRACT: output column name is 'predicted_label_name'.
    """
    df["predicted_label_name"] = [p["predicted_label"] for p in predictions]
    df["calibrated_confidence"] = [p["confidence_score"] for p in predictions]
    classes = list(predictions[0]["probabilities"].keys())
    for cls in classes:
        df[f"probability_{cls}"] = [p["probabilities"].get(cls, 0.0) for p in predictions]
    return df


def save_results(df: pd.DataFrame, output_path: Path) -> None:
    """Save the dataset with predictions"""
    df.to_csv(output_path, index=False, encoding="utf-8")
    LOGGER.info("Results saved to %s — shape: %s", output_path, df.shape)


# ── Public entry point called by run_pipeline.py ───────────────────────────────

def run_sentiment_inference(
    data_path: Path | None = None,
    model_path: Path | None = None,
    output_path: Path | None = None,
    batch_size: int = 1000,
) -> None:
    """
    Run full batch sentiment inference and write output CSV.

    Parameters
    ----------
    data_path   : Path to sentiment_preprocessed.csv (defaults to assets/)
    model_path  : Path to baseline_model.pkl (defaults to assets/)
    output_path : Destination for sentiment_predictions.csv (defaults to assets/)
    batch_size  : Batch size for inference loop.
    """
    data_path   = data_path   or DEFAULT_DATA_PATH
    model_path  = model_path  or DEFAULT_MODEL_PATH
    output_path = output_path or DEFAULT_OUTPUT_PATH

    # Remove stale output to guarantee a fresh run
    if output_path.exists():
        output_path.unlink()

    LOGGER.info("Loading model from %s", model_path)
    model, vectorizer, label_mapping = load_model_and_vectorizer(model_path)

    LOGGER.info("Loading data from %s", data_path)
    df = load_data(data_path)

    texts = df["content_clean"].fillna("").values
    X = preprocess_texts(texts, vectorizer)

    LOGGER.info("Running batch inference (batch_size=%d)...", batch_size)
    predictions, probabilities = batch_predict(model, X, batch_size=batch_size)

    formatted = format_predictions(predictions, probabilities, label_mapping)
    df_out = attach_predictions_to_dataset(df, formatted)

    save_results(df_out, output_path)

    # Column guardrail
    assert "predicted_label_name" in df_out.columns, (
        "DATA CONTRACT VIOLATION: 'predicted_label_name' missing from inference output."
    )
    LOGGER.info(
        "Inference complete. Distribution:\n%s",
        df_out["predicted_label_name"].value_counts().to_string(),
    )
