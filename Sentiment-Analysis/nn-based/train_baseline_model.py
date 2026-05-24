import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import pickle
import json
import os

def load_and_preprocess_data(filepath):
    """Load sentiment preprocessed data and prepare for training"""
    df = pd.read_csv(filepath)

    # Use content_clean for sentiment analysis
    df = df.dropna(subset=['content_clean', 'score'])

    # Map scores to labels
    def map_score_to_label(score):
        try:
            score = int(score)
            if score <= 2:
                return 'negative'
            elif score == 3:
                return 'neutral'
            else:
                return 'positive'
        except (ValueError, TypeError):
            return None  # Handle invalid scores

    df['label'] = df['score'].apply(map_score_to_label)
    df = df.dropna(subset=['label'])  # Remove rows with invalid labels

    # Filter out neutral if too few samples (optional)
    print(f"Class distribution:\n{df['label'].value_counts()}")

    return df[['content_clean', 'label']]

def apply_tfidf_vectorization(X_train, X_test, ngram_range=(1, 2), max_features=10000):
    """Apply TF-IDF vectorization"""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        stop_words='english'  # Could use Indonesian stopwords if available
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec, vectorizer

def apply_class_balancing(X_train, y_train):
    """Apply class balancing using class weights instead of SMOTE (for sparse matrices)"""
    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    print("Class weights:")
    for cls, weight in class_weight_dict.items():
        print(f"{cls}: {weight:.4f}")

    return X_train, y_train, class_weight_dict

def train_logistic_regression(X_train, y_train, class_weight=None):
    """Train Logistic Regression model"""
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='macro')

    # Per-class metrics
    labels = ['negative', 'neutral', 'positive']
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(y_test, y_pred, average=None, labels=labels)

    metrics = {
        'accuracy': round(accuracy, 4),
        'macro_precision': round(precision, 4),
        'macro_recall': round(recall, 4),
        'macro_f1': round(f1, 4),
        'per_class': {
            labels[i]: {
                'precision': round(precision_per_class[i], 4),
                'recall': round(recall_per_class[i], 4),
                'f1': round(f1_per_class[i], 4),
                'support': int(support_per_class[i]) if support_per_class is not None else 0
            } for i in range(len(labels))
        },
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    return metrics

def main():
    # File paths (dibuat absolut berdasarkan lokasi script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.abspath(os.path.join(base_dir, "..", "..", "Datasets", "sentiment_preprocessed.csv"))
    model_file = os.path.join(base_dir, "baseline_model.pkl")
    metrics_file = os.path.join(base_dir, "baseline_metrics.json")

    # Load and preprocess data
    print("Loading data...")
    df = load_and_preprocess_data(data_file)

    # Split data
    X = df['content_clean']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Apply TF-IDF vectorization
    print("Applying TF-IDF vectorization...")
    X_train_vec, X_test_vec, vectorizer = apply_tfidf_vectorization(X_train, X_test)

    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    # Apply class balancing
    print("Applying class balancing...")
    X_train_balanced, y_train_balanced, class_weights = apply_class_balancing(X_train_vec, y_train)

    # Train model
    print("Training Logistic Regression model...")
    model = train_logistic_regression(X_train_balanced, y_train_balanced, class_weight=class_weights)

    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test_vec, y_test)

    # Print results
    print("\nBaseline Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")

    print("\nPer-class metrics:")
    for label, scores in metrics['per_class'].items():
        print(f"{label}: P={scores['precision']:.4f}, R={scores['recall']:.4f}, F1={scores['f1']:.4f} (support={scores['support']})")

    # Save model and vectorizer
    print("Saving model...")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'vectorizer': vectorizer,
            'label_mapping': {'negative': 0, 'neutral': 1, 'positive': 2}
        }, f)

    # Save metrics
    print("Saving metrics...")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nModel saved to: {model_file}")
    print(f"Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()