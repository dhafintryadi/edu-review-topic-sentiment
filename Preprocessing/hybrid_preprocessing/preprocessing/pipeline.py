import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

from .cleaning import basic_clean_text, normalize_excessive_punctuation, normalize_whitespace
from .normalization import (
    emoji_to_text,
    normalize_slang,
    normalize_typo,
    reduce_repeated_characters,
    stem_tokens,
)
from .tokenizer import remove_stopwords, tokenize

LOGGER = logging.getLogger(__name__)

DEFAULT_NEGATIONS = {"tidak", "bukan", "belum", "kurang"}
DEFAULT_MAX_REVIEW_LENGTH = 2000


def load_json(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_stopwords(path: str) -> Set[str]:
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


class ReviewPreprocessor:
    def __init__(
        self,
        slang_path: str,
        stopword_path: str,
        typo_path: Optional[str] = None,
        negations: Optional[Set[str]] = None,
        max_review_length: int = DEFAULT_MAX_REVIEW_LENGTH,
        enable_topic_stemming: bool = False,
    ):
        self.slang_dict = load_json(slang_path)
        self.typo_dict = load_json(typo_path) if typo_path else {}
        self.stopwords = load_stopwords(stopword_path)
        self.negations = negations or DEFAULT_NEGATIONS
        self.max_review_length = max_review_length
        self.enable_topic_stemming = enable_topic_stemming

    def truncate_text(self, text: str) -> str:
        if text is None:
            return ""
        if len(text) > self.max_review_length:
            return text[: self.max_review_length]
        return text

    def preprocess_review(self, text: str) -> str:
        if text is None:
            return ""

        text = self.truncate_text(text)
        text = basic_clean_text(text)
        text = normalize_excessive_punctuation(text)
        text = normalize_whitespace(text)
        text = emoji_to_text(text)
        text = reduce_repeated_characters(text)

        tokens = tokenize(text)
        tokens = normalize_slang(tokens, self.slang_dict)
        tokens = normalize_typo(tokens, self.typo_dict)

        return " ".join(tokens).strip()

    def preprocess_topic(self, text: str) -> str:
        if text is None:
            return ""

        text = self.truncate_text(text)
        text = basic_clean_text(text)
        text = normalize_excessive_punctuation(text)
        text = normalize_whitespace(text)
        text = emoji_to_text(text)
        text = reduce_repeated_characters(text)

        tokens = tokenize(text)
        tokens = normalize_slang(tokens, self.slang_dict)
        tokens = normalize_typo(tokens, self.typo_dict)
        tokens = remove_stopwords(tokens, self.stopwords, self.negations)
        if self.enable_topic_stemming:
            tokens = stem_tokens(tokens)
        tokens = [tok for tok in tokens if tok and not tok.isspace() and len(tok) >= 3]
        if len(tokens) < 3:
            return ""
        return " ".join(tokens)

    def process_dataframe(self, df: pd.DataFrame, content_column: str = "content") -> pd.DataFrame:
        df = df.copy()
        df[content_column] = df[content_column].astype(str)
        df = df.drop_duplicates(subset=[content_column], keep="first")
        df["content_clean"] = df[content_column].map(self.preprocess_review)
        df["content_topic"] = df[content_column].map(self.preprocess_topic)
        return df

    def preprocess_csv(
        self,
        input_path: str,
        output_dir: str,
        content_column: str = "content",
        chunksize: int = 5000,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        sentiment_path = os.path.join(output_dir, "sentiment_preprocessed.csv")
        topic_path = os.path.join(output_dir, "topic_preprocessed.csv")

        # ── Overwrite protection: delete stale outputs before writing ──────────
        # Append mode is used inside the chunk loop for incremental writes, but
        # without this pre-run cleanup each successive pipeline run would double
        # the row count instead of producing a fresh file.
        for _stale in (sentiment_path, topic_path):
            if os.path.exists(_stale):
                os.remove(_stale)
                LOGGER.info("Removed stale output before rewrite: %s", _stale)

        seen_contents = set()
        chunk_iter = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
        )

        sentiment_written = False
        topic_written = False


        for idx, chunk in enumerate(tqdm(chunk_iter, desc=f"Processing {os.path.basename(input_path)}")):
            chunk[content_column] = chunk[content_column].astype(str)
            chunk = chunk[~chunk[content_column].isin(seen_contents)]
            seen_contents.update(chunk[content_column].tolist())

            if chunk.empty:
                LOGGER.info("Chunk %d empty after deduplication, skipping.", idx)
                continue

            chunk_start = time.perf_counter()
            chunk["content_clean"] = chunk[content_column].map(self.preprocess_review)
            review_done = time.perf_counter()
            chunk["content_topic"] = chunk[content_column].map(self.preprocess_topic)
            topic_done = time.perf_counter()

            sentiment_chunk = chunk.copy()
            sentiment_chunk = sentiment_chunk.rename(columns={content_column: "content_raw"})
            sentiment_chunk.to_csv(
                sentiment_path,
                index=False,
                mode="a",
                header=not sentiment_written,
            )
            sentiment_written = True

            topic_chunk = chunk.copy()
            topic_chunk = topic_chunk.rename(columns={content_column: "content_raw"})
            topic_chunk.to_csv(
                topic_path,
                index=False,
                mode="a",
                header=not topic_written,
            )
            topic_written = True

            review_time = review_done - chunk_start
            topic_time = topic_done - review_done
            row_count = len(chunk)
            LOGGER.info(
                "Chunk %d processed rows=%d review_time=%.3fs topic_time=%.3fs review_per=%.4fs topic_per=%.4fs",
                idx,
                row_count,
                review_time,
                topic_time,
                review_time / row_count,
                topic_time / row_count,
            )
            if review_time / row_count > 0.05 or topic_time / row_count > 0.1:
                LOGGER.warning(
                    "Chunk %d slow per-row: review_per=%.4fs topic_per=%.4fs",
                    idx,
                    review_time / row_count,
                    topic_time / row_count,
                )

        LOGGER.info("Finished preprocessing %s. Sentiment: %s; Topic: %s", input_path, sentiment_path, topic_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid preprocessing for Indonesian Google Play review datasets.")
    parser.add_argument("--input", required=True, help="CSV file containing reviews")
    parser.add_argument("--output-dir", required=True, help="Directory to write preprocessed outputs")
    parser.add_argument("--content-column", default="content", help="Review text column name")
    parser.add_argument("--slang-dict", default=None, help="Custom slang dictionary JSON file")
    parser.add_argument("--typo-dict", default=None, help="Custom typo dictionary JSON file")
    parser.add_argument("--stopword-file", default=None, help="Stopword text file")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))
    slang_dict = args.slang_dict or os.path.join(base_dir, "resources", "slang_dict.json")
    typo_dict = args.typo_dict or os.path.join(base_dir, "resources", "slang_dict.json")
    stopword_file = args.stopword_file or os.path.join(base_dir, "resources", "stopwords.txt")

    preprocessor = ReviewPreprocessor(slang_path=slang_dict, stopword_path=stopword_file, typo_path=typo_dict)
    preprocessor.preprocess_csv(args.input, args.output_dir, content_column=args.content_column)


if __name__ == "__main__":
    main()
