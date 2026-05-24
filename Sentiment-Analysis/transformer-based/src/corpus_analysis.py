"""
src/corpus_analysis.py
Task 4E & 4F: Generates corpus-level sentiment statistics and visualizations 
from the final inference output.
"""

import os, sys, json, logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

RESULTS = os.path.join(ROOT, "results")
VIS_DIR = os.path.join(ROOT, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}


def plot_text_table(df, save_path, title):
    """Utility to render a small dataframe as an image table."""
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1.5))
    ax.axis("off")
    ax.set_title(title, weight="bold", size=14)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Format column widths
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        if col == 0:
            cell.set_width(0.6) # Text column wider
        else:
            cell.set_width(0.15)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    logger.info("=" * 60)
    logger.info("PHASE 4 — CORPUS SENTIMENT ANALYSIS & VISUALIZATION")
    logger.info("=" * 60)

    # 1. Load data
    parquet_path = os.path.join(RESULTS, "final_sentiment_inference.parquet")
    csv_path = os.path.join(RESULTS, "final_sentiment_inference.csv")
    
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        logger.error("No inference output found! Run final_inference.py first.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(df):,} rows.")

    # 2. Overall Sentiment Distribution
    dist = df["predicted_label_name"].value_counts()
    dist_pct = df["predicted_label_name"].value_counts(normalize=True) * 100
    
    dist_dict = {
        "count": dist.to_dict(),
        "percentage": dist_pct.round(2).to_dict()
    }
    
    with open(os.path.join(RESULTS, "sentiment_distribution.json"), "w") as f:
        json.dump(dist_dict, f, indent=2)
        
    logger.info("Saved sentiment_distribution.json")

    # 3. Source-level Breakdown
    if "source" in df.columns:
        source_dist = pd.crosstab(df["source"], df["predicted_label_name"])
        source_dist_pct = pd.crosstab(df["source"], df["predicted_label_name"], normalize="index") * 100
        
        # Save as JSON
        with open(os.path.join(RESULTS, "source_sentiment_breakdown.json"), "w") as f:
            json.dump({
                "count": source_dist.to_dict(orient="index"),
                "percentage": source_dist_pct.round(2).to_dict(orient="index")
            }, f, indent=2)

        # Plot source breakdown
        ax = source_dist_pct.plot(
            kind="bar", stacked=True, 
            color=[COLORS.get(c, "#333") for c in source_dist_pct.columns],
            figsize=(10, 6)
        )
        plt.title("Sentiment Distribution by Dataset Source", fontsize=14, weight="bold")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Source")
        plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc='upper left')
        for c in ax.containers:
            ax.bar_label(c, label_type='center', fmt='%.1f%%', color='white', weight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, "source_sentiment_breakdown.png"), dpi=300)
        plt.close()
        logger.info("Saved source_sentiment_breakdown.png")

    # 4. Overall Percentage Chart
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = dist_pct.index
    sizes = dist_pct.values
    colors = [COLORS.get(l, "#333") for l in labels]
    
    ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    ax.set_title(f"Final Sentiment Distribution (N={len(df):,})", fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "sentiment_percentage_chart.png"), dpi=300)
    plt.close()
    
    # 5. Bar Chart Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.countplot(data=df, x="predicted_label_name", order=["negative", "neutral", "positive"], palette=COLORS, ax=ax)
    ax.set_title("Corpus Sentiment Absolute Counts", fontsize=14, weight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    for p in bars.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "final_sentiment_distribution.png"), dpi=300)
    plt.close()

    # 6. Confidence Statistics
    conf_stats = df.groupby("predicted_label_name")["calibrated_confidence"].describe()
    with open(os.path.join(RESULTS, "confidence_statistics.json"), "w") as f:
        json.dump(conf_stats.to_dict(orient="index"), f, indent=2)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x="calibrated_confidence", hue="predicted_label_name", fill=True, common_norm=False, palette=COLORS, ax=ax)
    ax.set_title("Calibrated Confidence Distribution by Sentiment", fontsize=14, weight="bold")
    ax.set_xlabel("Calibrated Confidence")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "confidence_distribution.png"), dpi=300)
    plt.close()

    # 7. Extremes Analysis (Top Confident & Ambiguous)
    text_col = "content_clean" if "content_clean" in df.columns else "content_raw"
    cols_to_export = [text_col, "predicted_label_name", "calibrated_confidence", "score", "source"]
    
    df_pos = df[df["predicted_label_name"] == "positive"]
    df_neg = df[df["predicted_label_name"] == "negative"]
    df_neu = df[df["predicted_label_name"] == "neutral"]
    
    top_pos = df_pos.sort_values("calibrated_confidence", ascending=False).head(20)[cols_to_export]
    top_neg = df_neg.sort_values("calibrated_confidence", ascending=False).head(20)[cols_to_export]
    
    # Ambiguous = Neutral predictions with the LOWEST confidence (borderline cases)
    ambiguous = df_neu.sort_values("calibrated_confidence", ascending=True).head(20)[cols_to_export]
    
    top_pos.to_csv(os.path.join(RESULTS, "high_confidence_positive.csv"), index=False)
    top_neg.to_csv(os.path.join(RESULTS, "high_confidence_negative.csv"), index=False)
    ambiguous.to_csv(os.path.join(RESULTS, "ambiguous_reviews.csv"), index=False)
    
    logger.info("Saved high_confidence_positive/negative.csv and ambiguous_reviews.csv")

    # 8. Top Sentiment Examples Visualization (Optional textual image)
    try:
        combined_examples = pd.concat([
            top_pos.head(3),
            top_neg.head(3),
            ambiguous.head(3)
        ]).reset_index(drop=True)
        # truncate text for plotting
        combined_examples[text_col] = combined_examples[text_col].apply(lambda x: str(x)[:80] + "..." if len(str(x)) > 80 else str(x))
        plot_text_table(
            combined_examples[[text_col, "predicted_label_name", "calibrated_confidence"]], 
            os.path.join(VIS_DIR, "top_sentiment_examples.png"),
            "Top Sentiment Examples & Ambiguous Cases"
        )
    except Exception as e:
        logger.warning(f"Could not generate text table image: {e}")

    logger.info("PHASE 4 CORPUS ANALYSIS COMPLETE.")

if __name__ == "__main__":
    main()
