import pandas as pd
import os
from pathlib import Path

# Anchor to this script's parent directory (Datasets/)
base = Path(__file__).resolve().parent

print('Merging raw reviews...')
df1 = pd.read_csv(base / 'main_review.csv')
df2 = pd.read_csv(base / 'benchmarking_review.csv')
pd.concat([df1, df2], ignore_index=True).to_csv(base / 'raw_review.csv', index=False)
print('raw_review.csv saved.')

print('Merging sentiment_preprocessed...')
df3 = pd.read_csv(base / 'main_review_preprocessed' / 'sentiment_preprocessed.csv')
df4 = pd.read_csv(base / 'benchmark_review_preprocessed' / 'sentiment_preprocessed.csv')
pd.concat([df3, df4], ignore_index=True).to_csv(base / 'sentiment_preprocessed.csv', index=False)
print('sentiment_preprocessed.csv saved.')

print('Merging topic_preprocessed...')
df5 = pd.read_csv(base / 'main_review_preprocessed' / 'topic_preprocessed.csv')
df6 = pd.read_csv(base / 'benchmark_review_preprocessed' / 'topic_preprocessed.csv')
pd.concat([df5, df6], ignore_index=True).to_csv(base / 'topic_preprocessed.csv', index=False)
print('topic_preprocessed.csv saved.')

print('All merges completed successfully!')
