"""
Data loading module for sentiment analysis
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Union, Tuple
from config import DATASET_PATH_1, DATASET_PATH_2, RAW_DATA_DIR, COLUMN_ALIASES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV dataset from file path.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        pandas DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid CSV
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset from {file_path}")
        logger.info(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        raise ValueError(f"Failed to load CSV file: {e}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to standard format using aliases.
    
    Handles alternative column names like content_clean -> content.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    
    for standard_col, aliases in COLUMN_ALIASES.items():
        # Find which alias exists in the dataframe
        found_col = None
        for alias in aliases:
            if alias in df.columns:
                found_col = alias
                break
        
        # If an alias is found and it's not already the standard name, rename it
        if found_col and found_col != standard_col:
            df.rename(columns={found_col: standard_col}, inplace=True)
            logger.info(f"Normalized column: {found_col} -> {standard_col}")
    
    return df


def load_both_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both main and benchmark review datasets.
    
    Returns:
        Tuple of (dataset1_df, dataset2_df)
    """
    logger.info("Loading first dataset...")
    dataset1_df = load_dataset(DATASET_PATH_1)
    dataset1_df = normalize_columns(dataset1_df)
    # Convert score to numeric
    dataset1_df['score'] = pd.to_numeric(dataset1_df['score'], errors='coerce')
    
    logger.info("Loading second dataset...")
    dataset2_df = load_dataset(DATASET_PATH_2)
    dataset2_df = normalize_columns(dataset2_df)
    # Convert score to numeric
    dataset2_df['score'] = pd.to_numeric(dataset2_df['score'], errors='coerce')
    
    return dataset1_df, dataset2_df


def combine_datasets(dataset1_df: pd.DataFrame, dataset2_df: pd.DataFrame, 
                    add_source: bool = True) -> pd.DataFrame:
    """
    Combine two datasets.
    
    Args:
        dataset1_df: First dataset
        dataset2_df: Second dataset
        add_source: Whether to add source column (dataset1/dataset2)
        
    Returns:
        Combined pandas DataFrame
    """
    if add_source:
        dataset1_copy = dataset1_df.copy()
        dataset2_copy = dataset2_df.copy()
        dataset1_copy['source'] = 'dataset1'
        dataset2_copy['source'] = 'dataset2'
        combined_df = pd.concat([dataset1_copy, dataset2_copy], ignore_index=True)
    else:
        combined_df = pd.concat([dataset1_df, dataset2_df], ignore_index=True)
    
    logger.info(f"Combined dataset shape: {combined_df.shape}")
    return combined_df


def save_raw_datasets(dataset1_df: pd.DataFrame, dataset2_df: pd.DataFrame, 
                     combined_df: pd.DataFrame) -> None:
    """
    Save raw datasets to data/raw directory.
    
    Args:
        dataset1_df: First dataset
        dataset2_df: Second dataset
        combined_df: Combined dataset
    """
    dataset1_path = RAW_DATA_DIR / "dataset1.csv"
    dataset2_path = RAW_DATA_DIR / "dataset2.csv"
    combined_path = RAW_DATA_DIR / "combined.csv"
    
    dataset1_df.to_csv(dataset1_path, index=False)
    dataset2_df.to_csv(dataset2_path, index=False)
    combined_df.to_csv(combined_path, index=False)
    
    logger.info(f"Saved dataset1 to {dataset1_path}")
    logger.info(f"Saved dataset2 to {dataset2_path}")
    logger.info(f"Saved combined dataset to {combined_path}")
