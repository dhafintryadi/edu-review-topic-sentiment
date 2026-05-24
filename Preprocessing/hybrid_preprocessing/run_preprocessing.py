import argparse
import logging
import os

from preprocessing.pipeline import ReviewPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute hybrid Indonesian review preprocessing.")
    parser.add_argument("--input", required=True, help="Input CSV file path")
    parser.add_argument("--output-dir", required=True, help="Output directory for preprocessed files")
    parser.add_argument("--content-column", default="content", help="CSV column with review text")
    parser.add_argument("--slang-dict", default=None, help="Custom slang dictionary JSON file")
    parser.add_argument("--stopword-file", default=None, help="Stopwords file")
    parser.add_argument("--typo-dict", default=None, help="Custom typo dictionary JSON file")
    parser.add_argument("--chunksize", type=int, default=5000, help="Number of rows per read chunk")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    package_root = os.path.dirname(os.path.abspath(__file__))
    slang_dict = args.slang_dict or os.path.join(package_root, "resources", "slang_dict.json")
    stopword_file = args.stopword_file or os.path.join(package_root, "resources", "stopwords.txt")
    typo_dict = args.typo_dict or slang_dict

    preprocessor = ReviewPreprocessor(
        slang_path=slang_dict,
        stopword_path=stopword_file,
        typo_path=typo_dict,
    )

    preprocessor.preprocess_csv(
        input_path=args.input,
        output_dir=args.output_dir,
        content_column=args.content_column,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
