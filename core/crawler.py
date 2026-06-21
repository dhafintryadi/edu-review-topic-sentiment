"""
core/crawler.py
===============
Standalone utility module to crawl reviews from the Google Play Store.
Exists strictly for research transparency and live demo safety.
Does NOT run as part of the main deterministic pipeline (run_pipeline.py).
"""
import logging
import os
import time
from pathlib import Path
from typing import List, Dict

import pandas as pd
from google_play_scraper import Sort, reviews

LOGGER = logging.getLogger(__name__)

CSV_COLUMNS = [
    "reviewId",
    "userName",
    "score",
    "content",
    "at",
    "thumbsUpCount",
    "replyContent",
    "repliedAt",
    "appVersion",
]

def run_standalone_crawler(
    app_id: str = "com.ruangguru.livestudents",
    max_reviews: int = 100,
    batch_size: int = 50,
    delay_seconds: float = 1.0,
    max_retries: int = 3,
) -> Path:
    """
    Standalone Google Play scraper utility for research transparency.
    Pulls up to max_reviews for a target app_id and saves it dynamically 
    to assets/crawled_review_example-{k}.csv without overwriting the production raw_review.csv.
    """
    CORE_DIR = Path(__file__).resolve().parent
    REPO_ROOT = CORE_DIR.parent
    ASSETS_DIR = REPO_ROOT / "assets"
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine target filename with dynamic versioning (k-iterable)
    k = 1
    while True:
        target_path = ASSETS_DIR / f"crawled_review_example-{k}.csv"
        if not target_path.exists():
            break
        k += 1
    
    LOGGER.info("Starting standalone crawling for %s (Target: %s)", app_id, target_path.name)
    
    # 2. Fetch reviews in batches
    all_reviews = []
    continuation_token = None
    
    try:
        while len(all_reviews) < max_reviews:
            count = min(batch_size, max_reviews - len(all_reviews))
            attempt = 0
            
            while True:
                try:
                    review_batch, continuation_token = reviews(
                        app_id,
                        lang="id",
                        country="id",
                        sort=Sort.NEWEST,
                        count=count,
                        continuation_token=continuation_token,
                    )
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt > max_retries:
                        LOGGER.error("Max retries reached for %s: %s", app_id, exc)
                        raise
                    backoff = delay_seconds * attempt
                    LOGGER.warning("Retry %d/%d for %s after: %s. Sleeping %.1fs.", attempt, max_retries, app_id, exc, backoff)
                    time.sleep(backoff)
            
            if not review_batch:
                LOGGER.info("No more reviews returned from Play Store API.")
                break
                
            all_reviews.extend(review_batch)
            LOGGER.info("Fetched %d reviews (total %d/%d)", len(review_batch), len(all_reviews), max_reviews)
            
            if continuation_token is None or continuation_token.token is None:
                break
                
            time.sleep(delay_seconds)
            
    except Exception as exc:
        LOGGER.error("Crawling halted due to error: %s. Saving partial reviews fetched so far (%d).", exc, len(all_reviews))
    
    if not all_reviews:
        LOGGER.warning("No reviews were fetched. Output file not created.")
        return Path()
        
    # 3. Save as CSV
    df = pd.DataFrame(all_reviews[:max_reviews])
    df = df.reindex(columns=CSV_COLUMNS)
    df.to_csv(target_path, index=False)
    LOGGER.info("Saved %d reviews successfully to %s", len(df), target_path)
    return target_path

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    print("=== Standalone Google Play Scraper (Demo Mode) ===")
    run_standalone_crawler(max_reviews=10)
