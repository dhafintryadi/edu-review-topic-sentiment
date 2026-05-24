"""
Data validation module for sentiment analysis
"""

import pandas as pd
import logging
from typing import Dict, List, Tuple
from config import REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for sentiment analysis datasets"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize validator with dataset"""
        self.df = df.copy()
        self.validation_report = {}
        
    def validate_columns(self) -> bool:
        """
        Validate that required columns exist in dataset.
        
        Returns:
            True if all required columns exist
        """
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            self.validation_report['missing_columns'] = missing_cols
            return False
        
        logger.info("All required columns present")
        self.validation_report['missing_columns'] = []
        return True
    
    def check_missing_values(self) -> Dict:
        """
        Check for missing values in dataset.
        
        Returns:
            Dictionary with column names and missing value counts
        """
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        
        missing_info = {}
        for col in missing_data.index:
            if missing_data[col] > 0:
                missing_info[col] = {
                    'count': int(missing_data[col]),
                    'percentage': float(missing_pct[col])
                }
        
        if missing_info:
            logger.warning(f"Missing values found: {missing_info}")
        else:
            logger.info("No missing values detected")
        
        self.validation_report['missing_values'] = missing_info
        return missing_info
    
    def check_duplicates(self) -> Dict:
        """
        Check for duplicate rows in dataset.
        
        Returns:
            Dictionary with duplicate statistics
        """
        total_duplicates = self.df.duplicated().sum()
        duplicates_full = self.df.duplicated(keep=False).sum()
        
        duplicate_info = {
            'total_duplicate_rows': int(total_duplicates),
            'total_rows_in_duplicates': int(duplicates_full),
            'percentage': float((total_duplicates / len(self.df)) * 100)
        }
        
        if total_duplicates > 0:
            logger.warning(f"Duplicates found: {duplicate_info}")
        else:
            logger.info("No duplicate rows detected")
        
        self.validation_report['duplicates'] = duplicate_info
        return duplicate_info
    
    def validate_score_column(self) -> Dict:
        """
        Validate score column (should be 1-5).
        
        Returns:
            Dictionary with score validation results
        """
        if 'score' not in self.df.columns:
            logger.error("Score column not found")
            return {'valid': False, 'reason': 'Score column not found'}
        
        # Convert score to numeric if it's not already
        try:
            self.df['score'] = pd.to_numeric(self.df['score'], errors='coerce')
        except Exception as e:
            logger.error(f"Error converting score to numeric: {e}")
            return {'valid': False, 'reason': f'Cannot convert score to numeric: {e}'}
        
        score_info = {
            'data_type': str(self.df['score'].dtype),
            'min_value': float(self.df['score'].min()),
            'max_value': float(self.df['score'].max()),
            'unique_values': int(self.df['score'].nunique()),
            'value_distribution': self.df['score'].value_counts().to_dict()
        }
        
        # Check if scores are in valid range (1-5)
        valid_scores = (self.df['score'] >= 1) & (self.df['score'] <= 5)
        invalid_count = (~valid_scores).sum()
        
        score_info['invalid_count'] = int(invalid_count)
        
        if invalid_count > 0:
            logger.warning(f"Invalid score values found: {invalid_count}")
            score_info['valid'] = False
        else:
            logger.info("Score column validation passed")
            score_info['valid'] = True
        
        self.validation_report['score_validation'] = score_info
        return score_info
    
    def validate_content_column(self) -> Dict:
        """
        Validate content column.
        
        Returns:
            Dictionary with content validation results
        """
        if 'content' not in self.df.columns:
            logger.error("Content column not found")
            return {'valid': False, 'reason': 'Content column not found'}
        
        empty_content = self.df['content'].astype(str).str.strip() == ''
        empty_count = empty_content.sum()
        
        content_info = {
            'data_type': str(self.df['content'].dtype),
            'total_rows': len(self.df),
            'empty_content_count': int(empty_count),
            'avg_length': float(self.df['content'].astype(str).str.len().mean()),
            'min_length': int(self.df['content'].astype(str).str.len().min()),
            'max_length': int(self.df['content'].astype(str).str.len().max())
        }
        
        if empty_count > 0:
            logger.warning(f"Empty content found: {empty_count} rows")
            content_info['valid'] = False
        else:
            logger.info("Content column validation passed")
            content_info['valid'] = True
        
        self.validation_report['content_validation'] = content_info
        return content_info
    
    def validate_all(self) -> Tuple[bool, Dict]:
        """
        Run all validations.
        
        Returns:
            Tuple of (all_valid: bool, validation_report: dict)
        """
        logger.info("Starting comprehensive dataset validation...")
        
        col_valid = self.validate_columns()
        if not col_valid:
            return False, self.validation_report
        
        self.check_missing_values()
        self.check_duplicates()
        self.validate_score_column()
        self.validate_content_column()
        
        logger.info("Validation complete")
        return True, self.validation_report
    
    def print_report(self) -> None:
        """Print validation report to console"""
        print("\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        print(f"\nDataset shape: {self.df.shape}")
        print(f"Total rows: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")
        
        if 'missing_values' in self.validation_report:
            print(f"\nMissing Values:")
            if self.validation_report['missing_values']:
                for col, info in self.validation_report['missing_values'].items():
                    print(f"  {col}: {info['count']} ({info['percentage']:.2f}%)")
            else:
                print("  None detected ✓")
        
        if 'duplicates' in self.validation_report:
            print(f"\nDuplicates:")
            dup_info = self.validation_report['duplicates']
            print(f"  Duplicate rows: {dup_info['total_duplicate_rows']} ({dup_info['percentage']:.2f}%)")
        
        if 'score_validation' in self.validation_report:
            print(f"\nScore Column Validation:")
            score_info = self.validation_report['score_validation']
            print(f"  Data type: {score_info['data_type']}")
            print(f"  Range: {score_info['min_value']:.0f} - {score_info['max_value']:.0f}")
            print(f"  Unique values: {score_info['unique_values']}")
            print(f"  Invalid count: {score_info['invalid_count']}")
        
        if 'content_validation' in self.validation_report:
            print(f"\nContent Column Validation:")
            content_info = self.validation_report['content_validation']
            print(f"  Data type: {content_info['data_type']}")
            print(f"  Empty content: {content_info['empty_content_count']}")
            print(f"  Avg length: {content_info['avg_length']:.0f} chars")
        
        print("\n" + "="*60 + "\n")
