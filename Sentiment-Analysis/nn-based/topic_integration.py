import pandas as pd
import os
import numpy as np
import json
from collections import defaultdict

def load_sentiment_data(filepath):
    """Load sentiment predictions"""
    df = pd.read_csv(filepath)
    print(f"Loaded sentiment data: {len(df)} rows")
    return df

def load_topic_data(filepath):
    """Load topic preprocessed data"""
    df = pd.read_csv(filepath)
    print(f"Loaded topic data: {len(df)} rows")
    return df

def mock_topic_assignment(content_topic):
    """Mock topic assignment based on keywords in content_topic"""
    if pd.isna(content_topic):
        return np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])

    text = str(content_topic).lower()

    # Topic keywords mapping (based on research output)
    topic_keywords = {
        0: ['kurang lengkap', 'materi', 'content gap', 'tidak lengkap'],  # Incomplete Material
        1: ['paham', 'jelas', 'animasi', 'comprehension'],  # Learning Comprehension
        2: ['harga', 'murah', 'paket', 'premium', 'biaya'],  # Pricing & Affordability
        3: ['bank soal', 'bahas', 'scaffolding', 'dukungan'],  # Scaffolding & Question Banks
        4: ['semangat', 'seru', 'suka', 'motivasi', 'engagement'],  # Motivation & Engagement
        5: ['bug', 'crash', 'login', 'keluar'],  # App Crashes & Login
        6: ['lag', 'update', 'download', 'lambat'],  # Performance Lag
        7: ['sandi', 'authentication', 'error', 'loading']  # Authentication Errors
    }

    # Check for topic keywords
    for topic_id, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            return topic_id

    # Default to random topic
    return np.random.choice([0, 1, 2, 3, 4, 5, 6, 7])

def assign_topics_to_reviews(df):
    """Assign topics to reviews based on content"""
    print("Assigning topics to reviews...")
    df['topic_id'] = df['content_topic'].apply(mock_topic_assignment)

    # Add topic labels and barriers (from research output)
    topic_info = {
        0: {'topic_label': 'Topic 0: Incomplete Material & Content Gaps', 'barrier_category': 'TB-7 Content Quality Mismatch'},
        1: {'topic_label': 'Topic 1: Learning Comprehension & Clarity', 'barrier_category': 'TB-1 Cognitive Difficulty'},
        2: {'topic_label': 'Topic 2: Pricing & Package Affordability', 'barrier_category': 'TB-6 Cost / Affordability Barrier'},
        3: {'topic_label': 'Topic 3: Need for Scaffolding & Question Banks', 'barrier_category': 'TB-3 Lack of Learning Support'},
        4: {'topic_label': 'Topic 4: Learner Motivation & Engagement', 'barrier_category': 'TB-2 Engagement & Motivation Problem'},
        5: {'topic_label': 'Topic 5: Core App Crashes & Login Failures', 'barrier_category': 'TB-4 System Usability Issues'},
        6: {'topic_label': 'Topic 6: Performance Lag & Update Errors', 'barrier_category': 'TB-4 System Usability Issues'},
        7: {'topic_label': 'Topic 7: Authentication & System Access Errors', 'barrier_category': 'TB-4 System Usability Issues'}
    }

    df['topic_label'] = df['topic_id'].map(lambda x: topic_info[x]['topic_label'])
    df['barrier_category'] = df['topic_id'].map(lambda x: topic_info[x]['barrier_category'])

    return df

def join_datasets(sentiment_df, topic_df):
    """Join sentiment and topic datasets by reviewId"""
    print("Joining datasets by reviewId...")

    # Ensure reviewId is string for joining
    sentiment_df['reviewId'] = sentiment_df['reviewId'].astype(str)
    topic_df['reviewId'] = topic_df['reviewId'].astype(str)

    # Left join to keep all sentiment data
    merged_df = sentiment_df.merge(topic_df[['reviewId', 'content_topic']], on='reviewId', how='left')

    print(f"Merged dataset: {len(merged_df)} rows")
    return merged_df

def compute_topic_sentiment_matrix(df):
    """Compute sentiment distribution per topic"""
    print("Computing topic-sentiment matrix...")

    # Group by topic and sentiment
    matrix = df.groupby(['topic_id', 'predicted_sentiment']).size().unstack(fill_value=0)

    # Ensure all sentiment columns exist
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment not in matrix.columns:
            matrix[sentiment] = 0

    # Reorder columns
    matrix = matrix[['negative', 'neutral', 'positive']]

    # Add topic info
    topic_info = {
        0: {'topic_label': 'Incomplete Material & Content Gaps', 'barrier': 'TB-7 Content Quality Mismatch'},
        1: {'topic_label': 'Learning Comprehension & Clarity', 'barrier': 'TB-1 Cognitive Difficulty'},
        2: {'topic_label': 'Pricing & Package Affordability', 'barrier': 'TB-6 Cost / Affordability Barrier'},
        3: {'topic_label': 'Need for Scaffolding & Question Banks', 'barrier': 'TB-3 Lack of Learning Support'},
        4: {'topic_label': 'Learner Motivation & Engagement', 'barrier': 'TB-2 Engagement & Motivation Problem'},
        5: {'topic_label': 'Core App Crashes & Login Failures', 'barrier': 'TB-4 System Usability Issues'},
        6: {'topic_label': 'Performance Lag & Update Errors', 'barrier': 'TB-4 System Usability Issues'},
        7: {'topic_label': 'Authentication & System Access Errors', 'barrier': 'TB-4 System Usability Issues'}
    }

    matrix.reset_index(inplace=True)
    matrix['topic_label'] = matrix['topic_id'].map(lambda x: topic_info[x]['topic_label'])
    matrix['barrier_category'] = matrix['topic_id'].map(lambda x: topic_info[x]['barrier'])

    # Calculate percentages
    matrix['total_reviews'] = matrix[['negative', 'neutral', 'positive']].sum(axis=1)
    matrix['negative_pct'] = (matrix['negative'] / matrix['total_reviews'] * 100).round(2)
    matrix['neutral_pct'] = (matrix['neutral'] / matrix['total_reviews'] * 100).round(2)
    matrix['positive_pct'] = (matrix['positive'] / matrix['total_reviews'] * 100).round(2)

    # Reorder columns
    matrix = matrix[['topic_id', 'topic_label', 'barrier_category', 'total_reviews',
                     'negative', 'neutral', 'positive',
                     'negative_pct', 'neutral_pct', 'positive_pct']]

    return matrix

def save_matrix(matrix, output_path):
    """Save topic-sentiment matrix"""
    matrix.to_csv(output_path, index=False)
    print(f"Topic-sentiment matrix saved to {output_path}")
    print(f"Matrix shape: {matrix.shape}")

def main():
    # File paths (dibuat absolut berdasarkan lokasi script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sentiment_file = os.path.join(base_dir, "sentiment_predictions.csv")
    output_file = os.path.join(base_dir, "topic_sentiment_matrix.csv")

    # Load sentiment data (already includes topic content)
    sentiment_df = load_sentiment_data(sentiment_file)

    # Assign topics
    sentiment_df = assign_topics_to_reviews(sentiment_df)

    # Compute matrix
    matrix = compute_topic_sentiment_matrix(sentiment_df)

    # Save results
    save_matrix(matrix, output_file)

    # Print summary
    print("\n=== Topic-Sentiment Integration Summary ===")
    print(f"Total reviews processed: {len(sentiment_df)}")
    print(f"Topics identified: {len(matrix)}")

    print("\nTop topics by review volume:")
    top_topics = matrix.nlargest(5, 'total_reviews')[['topic_label', 'total_reviews', 'positive_pct', 'negative_pct']]
    for _, row in top_topics.iterrows():
        print(f"- {row['topic_label']}: {row['total_reviews']} reviews "
              f"({row['positive_pct']:.1f}% positive, {row['negative_pct']:.1f}% negative)")

    print("\nTopic-sentiment integration completed successfully!")

if __name__ == "__main__":
    main()