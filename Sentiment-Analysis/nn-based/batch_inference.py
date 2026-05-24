import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path

def load_model_and_vectorizer(model_path):
    """Load trained model and vectorizer"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    vectorizer = model_data['vectorizer']
    label_mapping = model_data.get('label_mapping', {'negative': 0, 'neutral': 1, 'positive': 2})

    return model, vectorizer, label_mapping

def load_data(data_path):
    """Load sentiment preprocessed data"""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples from {data_path}")
    return df

def preprocess_texts(texts, vectorizer):
    """Preprocess texts using the trained vectorizer"""
    # Transform texts to TF-IDF features
    X = vectorizer.transform(texts)
    return X

def batch_predict(model, X, batch_size=1000):
    """Perform batch prediction to avoid memory issues"""
    predictions = []
    probabilities = []

    n_samples = X.shape[0]
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X[i:end_idx]

        # Get predictions
        batch_pred = model.predict(batch_X)
        predictions.extend(batch_pred)

        # Get prediction probabilities
        batch_proba = model.predict_proba(batch_X)
        probabilities.extend(batch_proba)

        print(f"Processed {end_idx}/{n_samples} samples...")

    return np.array(predictions), np.array(probabilities)

def format_predictions(predictions, probabilities, label_mapping):
    """Format predictions with labels and confidence scores"""
    # Since model.classes_ are strings, predictions are already strings
    formatted_predictions = []
    for pred, proba in zip(predictions, probabilities):
        confidence = float(np.max(proba))

        # Get probabilities dict
        prob_dict = {}
        for i, cls in enumerate(label_mapping.keys()):
            prob_dict[cls] = round(float(proba[i]), 4)

        formatted_predictions.append({
            'predicted_label': pred,  # Already a string
            'confidence_score': round(confidence, 4),
            'probabilities': prob_dict
        })

    return formatted_predictions

def attach_predictions_to_dataset(df, predictions):
    """Attach predictions to the original dataset"""
    # Create new columns
    df['predicted_label_name'] = [p['predicted_label'] for p in predictions]
    df['calibrated_confidence'] = [p['confidence_score'] for p in predictions]

    # Add probability columns for each class
    classes = list(predictions[0]['probabilities'].keys())
    for cls in classes:
        df[f'probability_{cls}'] = [p['probabilities'].get(cls, 0.0) for p in predictions]

    return df

def save_results(df, output_path):
    """Save the dataset with predictions"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to {output_path}")
    print(f"Output shape: {df.shape}")

def main():
    # File paths (dibuat absolut berdasarkan lokasi script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(base_dir, "..", "..", "Datasets", "sentiment_preprocessed.csv"))
    model_path = os.path.join(base_dir, "baseline_model.pkl")
    output_path = os.path.join(base_dir, "sentiment_predictions.csv")

    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model, vectorizer, label_mapping = load_model_and_vectorizer(model_path)

    # Load data
    print("Loading data...")
    df = load_data(data_path)

    # Filter to content_clean column
    texts = df['content_clean'].fillna('').values

    # Preprocess texts
    print("Preprocessing texts...")
    X = preprocess_texts(texts, vectorizer)

    # Batch prediction
    print("Running batch inference...")
    predictions, probabilities = batch_predict(model, X, batch_size=1000)

    # Format predictions
    print("Formatting predictions...")
    formatted_predictions = format_predictions(predictions, probabilities, label_mapping)

    # Attach to dataset
    print("Attaching predictions to dataset...")
    df_with_predictions = attach_predictions_to_dataset(df, formatted_predictions)

    # Save results
    print("Saving results...")
    save_results(df_with_predictions, output_path)

    # Print summary
    print("\n=== Batch Inference Summary ===")
    print(f"Total samples processed: {len(df)}")
    print("Prediction distribution:")
    print(df_with_predictions['predicted_label_name'].value_counts())

    print(f"\nAverage confidence: {df_with_predictions['calibrated_confidence'].mean():.4f}")
    print(f"High confidence (>0.8): {(df_with_predictions['calibrated_confidence'] > 0.8).sum()} samples")

    print("\nBatch inference completed successfully!")

if __name__ == "__main__":
    main()