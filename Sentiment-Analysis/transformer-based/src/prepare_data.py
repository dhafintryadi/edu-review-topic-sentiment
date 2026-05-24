"""
Main data preparation pipeline for sentiment analysis
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import LOGS_DIR
from data_loader import load_both_datasets, combine_datasets, save_raw_datasets
from data_validator import DataValidator
from data_processor import WeakLabeler, StratifiedSplitter
from dataset_summary import DatasetSummaryGenerator

# Setup logging
log_file = LOGS_DIR / "data_preparation.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run complete data preparation pipeline"""
    
    logger.info("="*70)
    logger.info("STARTING DATA PREPARATION PIPELINE")
    logger.info("="*70)
    
    try:
        # Step 1: Load datasets
        logger.info("\n[STEP 1] Loading datasets...")
        dataset1_df, dataset2_df = load_both_datasets()
        
        # Step 2: Combine datasets
        logger.info("\n[STEP 2] Combining datasets...")
        combined_df = combine_datasets(dataset1_df, dataset2_df, add_source=True)
        
        # Step 3: Save raw datasets
        logger.info("\n[STEP 3] Saving raw datasets...")
        save_raw_datasets(dataset1_df, dataset2_df, combined_df)
        
        # Step 4: Validate combined dataset
        logger.info("\n[STEP 4] Validating dataset...")
        validator = DataValidator(combined_df)
        all_valid, report = validator.validate_all()
        validator.print_report()
        
        if not all_valid:
            logger.warning("Dataset validation found issues, but proceeding...")
        
        # Step 5: Apply weak labeling
        logger.info("\n[STEP 5] Applying weak labeling...")
        labeler = WeakLabeler(combined_df)
        labeled_df = labeler.apply_weak_labeling()
        labeler.print_label_distribution()
        
        # Step 5b: Clean dataset - remove rows with missing scores
        logger.info("\n[STEP 5b] Cleaning dataset - removing rows with missing scores...")
        initial_count = len(labeled_df)
        labeled_df = labeled_df.dropna(subset=['score', 'sentiment_id'])
        removed_count = initial_count - len(labeled_df)
        logger.info(f"Removed {removed_count} rows with missing scores")
        logger.info(f"Dataset size after cleaning: {len(labeled_df)}")
        
        # Step 6: Show sample data
        logger.info("\n[STEP 6] Sample labeled data:")
        print("\n" + "="*60)
        print("SAMPLE DATA (First 5 rows)")
        print("="*60)
        sample_cols = ['content', 'score', 'sentiment_label', 'sentiment_id']
        if 'source' in labeled_df.columns:
            sample_cols.append('source')
        print(labeled_df[sample_cols].head())
        print("="*60 + "\n")
        
        # Step 7: Show score distribution
        logger.info("\n[STEP 7] Score distribution:")
        print("\n" + "="*60)
        print("SCORE DISTRIBUTION")
        print("="*60)
        score_dist = labeled_df['score'].value_counts().sort_index()
        for score, count in score_dist.items():
            pct = (count / len(labeled_df)) * 100
            print(f"Score {score}: {count:6d} ({pct:6.2f}%)")
        print("="*60 + "\n")
        
        # Step 8: Stratified split
        logger.info("\n[STEP 8] Performing stratified split...")
        splitter = StratifiedSplitter(labeled_df, random_state=42)
        train_df, val_df, test_df = splitter.split_data()
        splitter.print_split_report()
        
        # Step 9: Save splits
        logger.info("\n[STEP 9] Saving split datasets...")
        splitter.save_splits(train_df, val_df, test_df)
        
        # Step 10: Generate summary report
        logger.info("\n[STEP 10] Generating dataset summary report...")
        summary_generator = DatasetSummaryGenerator(
            combined_df=labeled_df,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            validation_report=validator.validation_report
        )
        summary_generator.print_summary()
        summary_json_path = summary_generator.save_json()
        summary_txt_path = summary_generator.save_txt()
        logger.info(f"Saved JSON summary to {summary_json_path}")
        logger.info(f"Saved TXT summary to {summary_txt_path}")
        
        # Step 11: Final summary statistics
        logger.info("\n[STEP 11] Final verification:")
        print("\n" + "="*70)
        print("FINAL DATASET SUMMARY")
        print("="*70)
        print(f"Total samples: {len(labeled_df):,}")
        print(f"Training samples: {len(train_df):,} ({len(train_df)/len(labeled_df)*100:.2f}%)")
        print(f"Validation samples: {len(val_df):,} ({len(val_df)/len(labeled_df)*100:.2f}%)")
        print(f"Test samples: {len(test_df):,} ({len(test_df)/len(labeled_df)*100:.2f}%)")
        print(f"\nColumns: {list(labeled_df.columns)}")
        print(f"\nOutput files:")
        print(f"  Raw datasets: data/raw/")
        print(f"  Processed splits: data/processed/")
        print(f"  Summary reports: results/")
        print("="*70 + "\n")
        
        logger.info("="*70)
        logger.info("DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
