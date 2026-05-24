import pandas as pd
import os

base = r'c:\Users\ASUS\Documents\AITF-2026\PKL\Datasets'

print('Merging raw reviews...')
df1 = pd.read_csv(f'{base}/main_review.csv')
df2 = pd.read_csv(f'{base}/benchmarking_review.csv')
pd.concat([df1, df2], ignore_index=True).to_csv(f'{base}/raw_review.csv', index=False)
print(f'raw_review.csv saved.')

print('Merging sentiment_preprocessed...')
df3 = pd.read_csv(f'{base}/main_review_preprocessed/sentiment_preprocessed.csv')
df4 = pd.read_csv(f'{base}/benchmark_review_preprocessed/sentiment_preprocessed.csv')
pd.concat([df3, df4], ignore_index=True).to_csv(f'{base}/sentiment_preprocessed.csv', index=False)
print(f'sentiment_preprocessed.csv saved.')

print('Merging topic_preprocessed...')
df5 = pd.read_csv(f'{base}/main_review_preprocessed/topic_preprocessed.csv')
df6 = pd.read_csv(f'{base}/benchmark_review_preprocessed/topic_preprocessed.csv')
pd.concat([df5, df6], ignore_index=True).to_csv(f'{base}/topic_preprocessed.csv', index=False)
print(f'topic_preprocessed.csv saved.')

print('All merges completed successfully!')
