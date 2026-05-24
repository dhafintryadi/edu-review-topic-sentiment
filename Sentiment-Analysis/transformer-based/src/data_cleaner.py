"""
src/data_cleaner.py
Phase 2C: Intra-split Data Cleaner
Performs exact deduplication and near-duplicate auditing for individual dataset splits.
"""

import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, text_col: str = "content"):
        self.text_col = text_col

    def remove_exact_duplicates(self, df: pd.DataFrame, split_name: str) -> Tuple[pd.DataFrame, dict]:
        """
        Removes exact duplicates within a single DataFrame based on the text column.
        Keeps the first occurrence.
        """
        initial_count = len(df)
        
        # Ensure text is string and handle NaNs
        df_clean = df.copy()
        df_clean[self.text_col] = df_clean[self.text_col].fillna("").astype(str)
        
        df_clean = df_clean.drop_duplicates(subset=[self.text_col], keep="first")
        final_count = len(df_clean)
        removed_count = initial_count - final_count
        
        stats = {
            "split": split_name,
            "initial_size": initial_count,
            "final_size": final_count,
            "removed_exact_duplicates": removed_count,
            "duplication_ratio_percent": round((removed_count / initial_count * 100) if initial_count > 0 else 0, 2)
        }
        
        logger.info(f"[{split_name}] Deduplication: {initial_count:,} -> {final_count:,} (Removed {removed_count:,})")
        return df_clean, stats

    def audit_near_duplicates(self, df: pd.DataFrame) -> dict:
        """
        Performs a basic normalized-text duplicate analysis (lowercased, stripped of excessive whitespace).
        Does not remove them, just audits for reporting.
        """
        if self.text_col not in df.columns:
            return {}
            
        normalized = df[self.text_col].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
        near_dups_count = normalized.duplicated(keep="first").sum()
        
        return {
            "near_duplicates_found": int(near_dups_count),
            "near_duplication_ratio_percent": round((near_dups_count / len(df) * 100) if len(df) > 0 else 0, 2)
        }
