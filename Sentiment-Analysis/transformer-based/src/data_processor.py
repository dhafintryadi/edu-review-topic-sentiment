"""
Data processing module for sentiment analysis
- Weak labeling based on score
- Stratified train/validation/test split
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from config import (
    SCORE_TO_SENTIMENT, SENTIMENT_MAPPING, TRAIN_RATIO, 
    VALIDATION_RATIO, TEST_RATIO, RANDOM_STATE, PROCESSED_DATA_DIR
)

logger = logging.getLogger(__name__)


class WeakLabeler:
    """Apply weak labeling to sentiment dataset"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize labeler with dataset"""
        self.df = df.copy()
        self.original_shape = df.shape
        
    def apply_weak_labeling(self) -> pd.DataFrame:
        """
        Apply weak labeling based on score.
        
        Maps:
        - score 1-2 → negative (0)
        - score 3 → neutral (1)
        - score 4-5 → positive (2)
        
        Returns:
            DataFrame with added sentiment_label and sentiment_id columns
        """
        logger.info("Applying weak labeling...")
        
        # Create sentiment_label column
        self.df['sentiment_label'] = self.df['score'].map(SCORE_TO_SENTIMENT)
        
        # Create sentiment_id column
        self.df['sentiment_id'] = self.df['sentiment_label'].map(SENTIMENT_MAPPING)
        
        logger.info(f"Weak labeling completed")
        logger.info(f"New shape: {self.df.shape}")
        
        return self.df
    
    def get_label_distribution(self) -> Dict:
        """
        Get distribution of sentiment labels.
        
        Returns:
            Dictionary with label distribution statistics
        """
        label_counts = self.df['sentiment_label'].value_counts()
        label_pct = (label_counts / len(self.df)) * 100
        
        distribution = {
            'total_samples': len(self.df),
            'negative': {
                'count': int(label_counts.get('negative', 0)),
                'percentage': float(label_pct.get('negative', 0))
            },
            'neutral': {
                'count': int(label_counts.get('neutral', 0)),
                'percentage': float(label_pct.get('neutral', 0))
            },
            'positive': {
                'count': int(label_counts.get('positive', 0)),
                'percentage': float(label_pct.get('positive', 0))
            }
        }
        
        return distribution
    
    def print_label_distribution(self) -> None:
        """Print label distribution to console"""
        dist = self.get_label_distribution()
        
        print("\n" + "="*60)
        print("SENTIMENT LABEL DISTRIBUTION")
        print("="*60)
        print(f"Total samples: {dist['total_samples']}")
        print(f"\nNegative (0):  {dist['negative']['count']:6d} ({dist['negative']['percentage']:6.2f}%)")
        print(f"Neutral  (1):  {dist['neutral']['count']:6d} ({dist['neutral']['percentage']:6.2f}%)")
        print(f"Positive (2):  {dist['positive']['count']:6d} ({dist['positive']['percentage']:6.2f}%)")
        print("="*60 + "\n")


class StratifiedSplitter:
    """Stratified train/validation/test split"""
    
    def __init__(self, df: pd.DataFrame, random_state: int = RANDOM_STATE):
        """Initialize splitter with dataset"""
        self.df = df.copy()
        self.random_state = random_state
        self.split_report = {}
        
    def split_data(self, train_ratio: float = TRAIN_RATIO,
                   val_ratio: float = VALIDATION_RATIO,
                   test_ratio: float = TEST_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified split into train, validation, and test sets.
        
        Args:
            train_ratio: Proportion for training set (default 0.7)
            val_ratio: Proportion for validation set (default 0.15)
            test_ratio: Proportion for test set (default 0.15)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        logger.info(f"Splitting data - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            self.df,
            test_size=(val_ratio + test_ratio),
            stratify=self.df['sentiment_id'],
            random_state=self.random_state
        )
        
        # Second split: split temp into val and test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            stratify=temp_df['sentiment_id'],
            random_state=self.random_state
        )
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        self._generate_report(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
    
    def _generate_report(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame) -> None:
        """Generate split report"""
        total = len(train_df) + len(val_df) + len(test_df)
        
        self.split_report = {
            'total_samples': total,
            'train': {
                'count': len(train_df),
                'percentage': (len(train_df) / total) * 100,
                'label_distribution': train_df['sentiment_id'].value_counts().to_dict()
            },
            'validation': {
                'count': len(val_df),
                'percentage': (len(val_df) / total) * 100,
                'label_distribution': val_df['sentiment_id'].value_counts().to_dict()
            },
            'test': {
                'count': len(test_df),
                'percentage': (len(test_df) / total) * 100,
                'label_distribution': test_df['sentiment_id'].value_counts().to_dict()
            }
        }
    
    def print_split_report(self) -> None:
        """Print split report to console"""
        print("\n" + "="*60)
        print("STRATIFIED SPLIT REPORT")
        print("="*60)
        print(f"Total samples: {self.split_report['total_samples']}")
        print(f"Random state: {self.random_state}\n")
        
        for split_name in ['train', 'validation', 'test']:
            split_info = self.split_report[split_name]
            print(f"{split_name.upper()}:")
            print(f"  Count: {split_info['count']} ({split_info['percentage']:.2f}%)")
            print(f"  Negative: {split_info['label_distribution'].get(0, 0)}")
            print(f"  Neutral:  {split_info['label_distribution'].get(1, 0)}")
            print(f"  Positive: {split_info['label_distribution'].get(2, 0)}")
            print()
        
        print("="*60 + "\n")
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame) -> None:
        """
        Save split datasets to processed data directory.
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
        """
        train_path = PROCESSED_DATA_DIR / "train.csv"
        val_path = PROCESSED_DATA_DIR / "validation.csv"
        test_path = PROCESSED_DATA_DIR / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved train set to {train_path}")
        logger.info(f"Saved validation set to {val_path}")
        logger.info(f"Saved test set to {test_path}")
