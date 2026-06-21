import pandas as pd
import os
import numpy as np
import json

def load_topic_sentiment_matrix(filepath):
    """Load topic-sentiment matrix"""
    df = pd.read_csv(filepath)
    print(f"Loaded topic-sentiment matrix: {len(df)} rows")
    return df

def compute_severity_scores(df):
    """Compute severity scores using formula: severity = negative_ratio × topic_frequency"""

    # Guard: safe per-row negative ratio — default to 0 when total_reviews == 0
    df['negative_ratio'] = df.apply(
        lambda r: r['negative'] / r['total_reviews'] if r['total_reviews'] > 0 else 0.0,
        axis=1,
    )

    # Topic frequency normalised across all topics
    total_reviews_all_topics = df['total_reviews'].sum()
    df['topic_frequency'] = df['total_reviews'] / total_reviews_all_topics \
        if total_reviews_all_topics > 0 else 0.0

    # Severity score
    df['severity_score'] = df['negative_ratio'] * df['topic_frequency']

    # Round for readability
    df['negative_ratio'] = df['negative_ratio'].round(4)
    df['topic_frequency'] = df['topic_frequency'].round(4)
    df['severity_score'] = df['severity_score'].round(6)

    return df

def rank_topics_by_severity(df):
    """Rank topics by severity score (descending)"""
    df_sorted = df.sort_values('severity_score', ascending=False).reset_index(drop=True)

    # Add rank column
    df_sorted['severity_rank'] = range(1, len(df_sorted) + 1)

    return df_sorted

def add_severity_interpretation(df):
    """Add severity level interpretation"""
    def interpret_severity(score):
        if score >= 0.05:
            return 'Critical'
        elif score >= 0.03:
            return 'High'
        elif score >= 0.015:
            return 'Medium'
        elif score >= 0.005:
            return 'Low'
        else:
            return 'Minimal'

    df['severity_level'] = df['severity_score'].apply(interpret_severity)
    return df

def add_research_insights(df):
    """Add research insights based on severity ranking"""
    insights = {
        'TB-4 System Usability Issues': 'Technical barriers have highest user impact - prioritize system stability',
        'TB-7 Content Quality Mismatch': 'Content gaps affect learning quality - expand material coverage',
        'TB-6 Cost / Affordability Barrier': 'Economic barriers limit access - consider pricing strategy adjustments',
        'TB-1 Cognitive Difficulty': 'Learning comprehension issues - enhance instructional design',
        'TB-3 Lack of Learning Support': 'Support systems needed - increase scaffolding and question banks',
        'TB-2 Engagement & Motivation Problem': 'Motivation barriers - improve engagement features'
    }

    df['research_insight'] = df['barrier_category'].map(insights).fillna('Further investigation needed')
    return df

def save_ranked_topics(df, output_path):
    """Save ranked topics with severity analysis"""
    # Select and reorder columns for output
    columns = [
        'severity_rank', 'topic_id', 'topic_label', 'barrier_category',
        'severity_score', 'severity_level', 'total_reviews',
        'negative_ratio', 'topic_frequency',
        'negative', 'neutral', 'positive',
        'negative_pct', 'neutral_pct', 'positive_pct',
        'research_insight'
    ]

    df_output = df[columns]
    df_output.to_csv(output_path, index=False)
    print(f"Severity-ranked topics saved to {output_path}")

def generate_severity_summary(df):
    """Generate summary statistics"""
    summary = {
        'total_topics': int(len(df)),
        'total_reviews': int(df['total_reviews'].sum()),
        'severity_distribution': {k: int(v) for k, v in df['severity_level'].value_counts().to_dict().items()},
        'top_barrier': df.iloc[0]['barrier_category'],
        'severity_range': {
            'min': float(df['severity_score'].min()),
            'max': float(df['severity_score'].max()),
            'mean': float(df['severity_score'].mean()),
            'median': float(df['severity_score'].median())
        }
    }

    return summary

def save_summary(summary, output_path):
    """Save severity analysis summary"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"Severity summary saved to {output_path}")

def main():
    # File paths (dibuat absolut berdasarkan lokasi script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_file = os.path.join(base_dir, "topic_sentiment_matrix.csv")
    output_file = os.path.join(base_dir, "severity_ranked_topics.csv")
    summary_file = os.path.join(base_dir, "severity_analysis_summary.json")

    # Load data
    df = load_topic_sentiment_matrix(matrix_file)

    # Compute severity scores
    print("Computing severity scores...")
    df = compute_severity_scores(df)

    # Rank topics
    print("Ranking topics by severity...")
    df_ranked = rank_topics_by_severity(df)

    # Add interpretations
    df_ranked = add_severity_interpretation(df_ranked)
    df_ranked = add_research_insights(df_ranked)

    # Save results
    save_ranked_topics(df_ranked, output_file)

    # Generate and save summary
    summary = generate_severity_summary(df_ranked)
    save_summary(summary, summary_file)

    # Print results
    print("\n=== Severity Analysis Results ===")
    print(f"Total topics analyzed: {len(df_ranked)}")
    print(f"Total reviews: {summary['total_reviews']:,}")

    print("\nTop 5 Most Severe Learning Barriers:")
    top_5 = df_ranked.head(5)[['severity_rank', 'topic_label', 'barrier_category',
                               'severity_score', 'severity_level', 'total_reviews']]
    for _, row in top_5.iterrows():
        print(f"{row['severity_rank']}. {row['topic_label']}")
        print(f"   Barrier: {row['barrier_category']}")
        print(f"   Severity: {row['severity_score']:.6f} ({row['severity_level']})")
        print(f"   Reviews: {row['total_reviews']:,}")
        print()

    print("Severity distribution:")
    for level, count in summary['severity_distribution'].items():
        print(f"- {level}: {count} topics")

    print("\nSeverity analysis completed successfully!")

if __name__ == "__main__":
    main()