import pandas as pd
from pathlib import Path

# Define datasets folder path
datasets_folder = Path("datasets")

# List of review files to merge
review_files = [
    "ruangguru_reviews.csv",
    "quipper_reviews.csv",
    "pahamify_reviews.csv",
    "cerebrum_reviews.csv"
]

# Read and combine all CSV files
dfs = []
for file in review_files:
    file_path = datasets_folder / file
    if file_path.exists():
        df = pd.read_csv(file_path)
        print(f"Loaded {file}: {len(df)} rows")
        dfs.append(df)
    else:
        print(f"Warning: {file} not found")

# Concatenate all dataframes
main_review = pd.concat(dfs, ignore_index=True)

# Save to main_review.csv
output_path = datasets_folder / "main_review.csv"
main_review.to_csv(output_path, index=False)

print(f"\n✓ Successfully merged {len(review_files)} files")
print(f"✓ Total rows: {len(main_review)}")
print(f"✓ Saved to: {output_path}")
