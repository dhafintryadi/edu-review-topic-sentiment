"""
src/leakage_detector.py
Phase 2C: Cross-Split Leakage Mitigator
Removes overlapping samples between splits, prioritizing test and validation integrity.
Priority: Test > Validation > Train
"""

import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class LeakageMitigator:
    def __init__(self, text_col: str = "content"):
        self.text_col = text_col

    def remove_cross_split_leakage(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """
        Removes identical texts appearing in multiple splits.
        Priority:
        1. Keep in Test. Remove from Train and Validation.
        2. Keep in Validation. Remove from Train.
        """
        logger.info("Starting cross-split leakage mitigation...")
        
        train_texts = set(train_df[self.text_col].dropna().astype(str))
        val_texts = set(val_df[self.text_col].dropna().astype(str))
        test_texts = set(test_df[self.text_col].dropna().astype(str))
        
        # Initial stats
        stats = {
            "initial_overlap_train_val": len(train_texts & val_texts),
            "initial_overlap_train_test": len(train_texts & test_texts),
            "initial_overlap_val_test": len(val_texts & test_texts),
        }
        
        logger.info(f"Initial Overlaps - Train/Val: {stats['initial_overlap_train_val']}, Train/Test: {stats['initial_overlap_train_test']}, Val/Test: {stats['initial_overlap_val_test']}")
        
        # 1. Test set is sacred. 
        # Remove anything in validation or train that is also in test.
        val_clean = val_df[~val_df[self.text_col].isin(test_texts)].copy()
        train_clean = train_df[~train_df[self.text_col].isin(test_texts)].copy()
        
        stats["removed_from_val_due_to_test"] = len(val_df) - len(val_clean)
        stats["removed_from_train_due_to_test"] = len(train_df) - len(train_clean)
        
        # Update validation texts after removing test overlaps
        val_texts_clean = set(val_clean[self.text_col].dropna().astype(str))
        
        # 2. Validation set is second priority.
        # Remove anything in train that is also in validation.
        train_final = train_clean[~train_clean[self.text_col].isin(val_texts_clean)].copy()
        
        stats["removed_from_train_due_to_val"] = len(train_clean) - len(train_final)
        stats["total_removed_from_train"] = len(train_df) - len(train_final)
        
        logger.info(f"Removed from Val (due to Test): {stats['removed_from_val_due_to_test']}")
        logger.info(f"Removed from Train (due to Test): {stats['removed_from_train_due_to_test']}")
        logger.info(f"Removed from Train (due to Val): {stats['removed_from_train_due_to_val']}")
        
        # Final Verification
        final_train_texts = set(train_final[self.text_col].dropna().astype(str))
        final_val_texts = set(val_clean[self.text_col].dropna().astype(str))
        
        assert len(final_train_texts & final_val_texts) == 0, "Train-Val leakage remains!"
        assert len(final_train_texts & test_texts) == 0, "Train-Test leakage remains!"
        assert len(final_val_texts & test_texts) == 0, "Val-Test leakage remains!"
        
        logger.info("Verification passed: 0 cross-split overlaps remain.")
        
        return train_final, val_clean, test_df, stats
