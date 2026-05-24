import logging
import os
import re
import time
from typing import Dict, List

import pandas as pd
from google_play_scraper import Sort, app, reviews

APP_METADATA = {
    "ruangguru": {
        "app_id": "com.ruangguru.livestudents",
        "output_file": "ruangguru_reviews.csv",
        "lang": "id",
        "country": "id",
    },
    "quipper": {
        "app_id": "com.quipper.school.assignment",
        "output_file": "quipper_reviews.csv",
        "lang": "id",
        "country": "id",
    },
    "pahamify": {
        "app_id": "com.pahamify.android",
        "output_file": "pahamify_reviews.csv",
        "lang": "id",
        "country": "id",
    },
    "duolingo": {
        "app_id": "com.duolingo",
        "output_file": "duolingo_reviews.csv",
        "lang": "id",
        "country": "id",
    },
    "khan_academy": {
        "app_id": "org.khanacademy.android",
        "output_file": "khanacademy_reviews.csv",
        "lang": "id",
        "country": "id",
    },
    "cerebrum": {
        "app_id": "id.cerebrum.app",
        "output_file": "cerebrum_reviews.csv",
        "lang": "id",
        "country": "id",
    },
}

INDONESIAN_REVIEW_REGEX = re.compile(
    r"\b(?:dan|yang|untuk|ini|saya|kamu|adalah|aplikasi|bagus|kurang|tidak|terima|kasih|keren|belajar|lebih|mantap)\b",
    re.IGNORECASE,
)

OUTPUT_DIR = "datasets"
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


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_app(app_id: str, lang: str = "en", country: str = "us") -> Dict[str, object]:
    return app(app_id, lang=lang, country=country)


def fetch_reviews_for_app(
    app_id: str,
    lang: str = "en",
    country: str = "us",
    max_reviews: int = 5000,
    batch_size: int = 200,
    delay_seconds: float = 1.0,
    max_retries: int = 3,
) -> List[Dict[str, object]]:
    all_reviews = []
    continuation_token = None

    while len(all_reviews) < max_reviews:
        count = min(batch_size, max_reviews - len(all_reviews))
        attempt = 0

        while True:
            try:
                review_batch, continuation_token = reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=Sort.NEWEST,
                    count=count,
                    continuation_token=continuation_token,
                )
                break
            except Exception as exc:
                attempt += 1
                if attempt > max_retries:
                    logging.error(
                        "Maximum retries reached for %s after %s attempts: %s",
                        app_id,
                        attempt,
                        exc,
                    )
                    raise
                backoff = delay_seconds * attempt
                logging.warning(
                    "Retry %s/%s for %s after error: %s. Waiting %.1fs.",
                    attempt,
                    max_retries,
                    app_id,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        if not review_batch:
            logging.info("No more reviews returned for %s.", app_id)
            break

        all_reviews.extend(review_batch)
        logging.info(
            "Fetched %s reviews for %s (total %s).",
            len(review_batch),
            app_id,
            len(all_reviews),
        )

        if continuation_token is None or continuation_token.token is None:
            break

        time.sleep(delay_seconds)

    return all_reviews[:max_reviews]


def filter_indonesian_reviews(reviews_data: List[Dict[str, object]]) -> List[Dict[str, object]]:
    filtered = []
    for review in reviews_data:
        content = review.get("content", "")
        if isinstance(content, str) and INDONESIAN_REVIEW_REGEX.search(content):
            filtered.append(review)
    return filtered


def save_reviews(app_key: str, reviews_data: List[Dict[str, object]], output_file: str) -> None:
    df = pd.DataFrame(reviews_data)
    df = df.reindex(columns=CSV_COLUMNS)
    output_path = os.path.join(OUTPUT_DIR, output_file)
    df.to_csv(output_path, index=False)
    logging.info("Saved %s reviews for %s to %s.", len(df), app_key, output_path)


def main() -> None:
    configure_logging()
    ensure_output_dir()

    summary = {}

    for app_key, metadata in APP_METADATA.items():
        app_id = metadata["app_id"]
        output_file = metadata["output_file"]
        lang = metadata.get("lang", "en")
        country = metadata.get("country", "us")

        logging.info("Starting review collection for %s (%s) [%s/%s].", app_key, app_id, lang, country)

        try:
            app_info = validate_app(app_id, lang=lang, country=country)
            logging.info(
                "Validated app %s: %s.",
                app_key,
                app_info.get("title", "(title unavailable)"),
            )
        except Exception as exc:
            logging.error("Unable to validate app %s (%s): %s", app_key, app_id, exc)
            summary[app_key] = 0
            continue

        reviews_data = fetch_reviews_for_app(app_id, lang=lang, country=country)
        reviews_data = filter_indonesian_reviews(reviews_data)
        logging.info(
            "Filtered %s Indonesian-language reviews for %s.",
            len(reviews_data),
            app_key,
        )
        save_reviews(app_key, reviews_data, output_file)
        summary[app_key] = len(reviews_data)

    logging.info("Review collection complete.")
    for app_key, count in summary.items():
        print(f"{app_key}: {count} reviews collected")


if __name__ == "__main__":
    main()
