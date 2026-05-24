"""
Dataset summary and reporting module for sentiment analysis
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from config import RESULTS_DIR

class DatasetSummaryGenerator:
    """Generate comprehensive dataset summary reports"""
    
    def __init__(self, 
                 combined_df: pd.DataFrame, 
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame, 
                 test_df: pd.DataFrame,
                 validation_report: Dict = None):
        """Initialize summary generator"""
        self.combined_df = combined_df
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.validation_report = validation_report or {}
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive dataset summary.
        
        Returns:
            Dictionary with complete summary statistics
        """
        summary = {
            "project_info": {
                "phase": "Phase 1",
                "stage": "Dataset Preparation",
                "status": "Completed"
            },
            "total_dataset": {
                "total_samples": len(self.combined_df),
                "columns": len(self.combined_df.columns),
                "column_names": list(self.combined_df.columns)
            },
            "data_quality": self._generate_quality_stats(),
            "sentiment_distribution": self._generate_sentiment_distribution(),
            "split_distribution": self._generate_split_distribution(),
            "validation_summary": self.validation_report
        }
        return summary
    
    def _generate_quality_stats(self) -> Dict:
        """Generate data quality statistics"""
        return {
            "missing_values": {
                col: int(self.combined_df[col].isnull().sum())
                for col in self.combined_df.columns
            },
            "duplicate_rows": int(self.combined_df.duplicated().sum()),
            "total_rows": len(self.combined_df)
        }
    
    def _generate_sentiment_distribution(self) -> Dict:
        """Generate sentiment label distribution"""
        if 'sentiment_id' not in self.combined_df.columns:
            return {}
        
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        dist = self.combined_df['sentiment_id'].value_counts().to_dict()
        
        distribution = {}
        for label_id, count in dist.items():
            label_name = label_map.get(label_id, f'unknown_{label_id}')
            distribution[label_name] = {
                'id': int(label_id),
                'count': int(count),
                'percentage': float(count / len(self.combined_df) * 100)
            }
        
        return distribution
    
    def _generate_split_distribution(self) -> Dict:
        """Generate train/val/test split distribution"""
        total = len(self.combined_df)
        
        return {
            "train": {
                "count": len(self.train_df),
                "percentage": float(len(self.train_df) / total * 100),
                "sentiment_distribution": self._count_sentiments(self.train_df)
            },
            "validation": {
                "count": len(self.val_df),
                "percentage": float(len(self.val_df) / total * 100),
                "sentiment_distribution": self._count_sentiments(self.val_df)
            },
            "test": {
                "count": len(self.test_df),
                "percentage": float(len(self.test_df) / total * 100),
                "sentiment_distribution": self._count_sentiments(self.test_df)
            }
        }
    
    def _count_sentiments(self, df: pd.DataFrame) -> Dict:
        """Count sentiment labels in dataframe"""
        if 'sentiment_id' not in df.columns:
            return {}
        
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        counts = df['sentiment_id'].value_counts().to_dict()
        
        result = {}
        for label_id, count in counts.items():
            label_name = label_map.get(label_id, f'unknown_{label_id}')
            result[label_name] = int(count)
        
        return result
    
    def save_json(self) -> Path:
        """Save summary as JSON"""
        summary = self.generate_summary()
        output_path = RESULTS_DIR / "dataset_summary.json"
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return output_path
    
    def save_txt(self) -> Path:
        """Save summary as formatted text"""
        summary = self.generate_summary()
        output_path = RESULTS_DIR / "dataset_summary.txt"
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SENTIMENT ANALYSIS - PHASE 1 DATASET SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Project Info
            f.write("PROJECT INFO\n")
            f.write("-"*70 + "\n")
            for key, value in summary['project_info'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Total Dataset
            f.write("TOTAL DATASET\n")
            f.write("-"*70 + "\n")
            total = summary['total_dataset']
            f.write(f"  Total Samples: {total['total_samples']:,}\n")
            f.write(f"  Total Columns: {total['columns']}\n")
            f.write(f"  Columns: {', '.join(total['column_names'])}\n\n")
            
            # Data Quality
            f.write("DATA QUALITY\n")
            f.write("-"*70 + "\n")
            quality = summary['data_quality']
            f.write(f"  Duplicate Rows: {quality['duplicate_rows']}\n")
            f.write("  Missing Values:\n")
            for col, count in quality['missing_values'].items():
                if count > 0:
                    f.write(f"    {col}: {count}\n")
            f.write("\n")
            
            # Sentiment Distribution
            f.write("SENTIMENT DISTRIBUTION (Total Dataset)\n")
            f.write("-"*70 + "\n")
            for label, stats in summary['sentiment_distribution'].items():
                f.write(f"  {label.upper():10} (ID={stats['id']}): {stats['count']:6,} ({stats['percentage']:6.2f}%)\n")
            f.write("\n")
            
            # Split Distribution
            f.write("TRAIN/VALIDATION/TEST SPLIT\n")
            f.write("-"*70 + "\n")
            for split_name, split_info in summary['split_distribution'].items():
                f.write(f"\n  {split_name.upper()}:\n")
                f.write(f"    Samples: {split_info['count']:,} ({split_info['percentage']:.2f}%)\n")
                f.write(f"    Sentiment Distribution:\n")
                for sentiment, count in split_info['sentiment_distribution'].items():
                    f.write(f"      {sentiment:10}: {count:6,}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        return output_path
    
    def print_summary(self) -> None:
        """Print summary to console"""
        summary = self.generate_summary()
        
        print("\n" + "="*70)
        print("SENTIMENT ANALYSIS - PHASE 1 DATASET SUMMARY")
        print("="*70)
        
        # Total Dataset
        total = summary['total_dataset']
        print(f"\nTOTAL DATASET")
        print(f"  Total Samples: {total['total_samples']:,}")
        print(f"  Total Columns: {total['columns']}")
        
        # Sentiment Distribution
        print(f"\nSENTIMENT DISTRIBUTION")
        for label, stats in summary['sentiment_distribution'].items():
            print(f"  {label.upper():10} (ID={stats['id']}): {stats['count']:6,} ({stats['percentage']:6.2f}%)")
        
        # Split Distribution
        print(f"\nTRAIN/VALIDATION/TEST SPLIT")
        for split_name, split_info in summary['split_distribution'].items():
            print(f"\n  {split_name.upper()}: {split_info['count']:,} ({split_info['percentage']:.2f}%)")
            for sentiment, count in split_info['sentiment_distribution'].items():
                print(f"    {sentiment:10}: {count:6,}")
        
        print("\n" + "="*70 + "\n")
