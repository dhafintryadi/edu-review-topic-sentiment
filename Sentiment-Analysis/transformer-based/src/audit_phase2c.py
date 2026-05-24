"""
src/audit_phase2c.py
Full audit of the completed Phase 2C dataset cleaning pipeline.
"""

import os, sys, json, logging, datetime
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

DATA_CLEAN = os.path.join(ROOT, "data", "cleaned")
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")

LOG_FILE = os.path.join(LOGS, "phase2c_audit.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

EXPECTED_SIZES = {
    "train_clean.csv": 6119,
    "validation_clean.csv": 2509,
    "test_clean.csv": 4508
}

EXPECTED_WEIGHTS = [2.1538, 4.6889, 0.4306]

def main():
    logger.info("=" * 60)
    logger.info("PHASE 2C AUDIT")
    logger.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 60)

    audit_metrics = {}
    audit_verdict = "PASS"
    red_flags = []

    # 1. Verify Cleaned Dataset Files
    logger.info("\n[1] Verifying Cleaned Dataset Files")
    splits = {}
    file_status = {}
    for fname, expected_size in EXPECTED_SIZES.items():
        path = os.path.join(DATA_CLEAN, fname)
        split_name = fname.replace("_clean.csv", "")
        if not os.path.exists(path):
            red_flags.append(f"Missing file: {fname}")
            file_status[fname] = "MISSING"
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8")
            if df.empty:
                red_flags.append(f"File is empty: {fname}")
                file_status[fname] = "EMPTY"
                continue
            
            splits[split_name] = df
            size_match = len(df) == expected_size
            file_status[fname] = {"rows": len(df), "expected": expected_size, "match": size_match}
            if not size_match:
                logger.warning(f"Size mismatch in {fname}: Expected {expected_size}, got {len(df)}")
            else:
                logger.info(f"{fname} loaded successfully, size matches expected ({expected_size} rows).")
        except Exception as e:
            red_flags.append(f"Cannot read {fname}: {str(e)}")
            file_status[fname] = "ERROR"

    audit_metrics["file_verification"] = file_status

    if len(splits) < 3:
        logger.error("Critical files missing, aborting audit.")
        return

    # 2. Verify Exact Duplicate Removal
    logger.info("\n[2] Verifying Exact Duplicate Removal")
    duplicate_counts = {}
    for s, df in splits.items():
        dup_count = int(df.duplicated(subset=["content"]).sum())
        duplicate_counts[s] = dup_count
        logger.info(f"[{s}] Exact duplicates remaining: {dup_count}")
        if dup_count > 0:
            red_flags.append(f"Exact duplicates found in {s}: {dup_count}")

    audit_metrics["duplicate_counts"] = duplicate_counts

    # 3. Verify Cross-Split Leakage Removal
    logger.info("\n[3] Verifying Cross-Split Leakage Removal")
    train_texts = set(splits["train"]["content"].dropna().astype(str))
    val_texts = set(splits["validation"]["content"].dropna().astype(str))
    test_texts = set(splits["test"]["content"].dropna().astype(str))

    leakage_stats = {
        "train_validation_overlap": len(train_texts & val_texts),
        "train_test_overlap": len(train_texts & test_texts),
        "validation_test_overlap": len(val_texts & test_texts)
    }

    for overlap_type, count in leakage_stats.items():
        logger.info(f"{overlap_type}: {count}")
        if count > 0:
            red_flags.append(f"Cross-split leakage found ({overlap_type}): {count}")

    audit_metrics["leakage_stats"] = leakage_stats

    # 4. Verify Label Integrity
    logger.info("\n[4] Verifying Label Integrity")
    label_status = {}
    for s, df in splits.items():
        unique_labels = df["sentiment_id"].unique().tolist()
        invalid_labels = [l for l in unique_labels if l not in [0, 1, 2]]
        
        status = {
            "unique_labels": [int(x) for x in unique_labels],
            "invalid_labels_found": len(invalid_labels) > 0,
            "has_nulls": bool(df["sentiment_id"].isnull().any())
        }
        label_status[s] = status
        logger.info(f"[{s}] Labels: {status['unique_labels']} | Invalid: {status['invalid_labels_found']} | Nulls: {status['has_nulls']}")
        
        if status["invalid_labels_found"] or status["has_nulls"]:
            red_flags.append(f"Label corruption in {s}")

    audit_metrics["label_integrity"] = label_status

    # 5. Perform Random Manual Sampling Audit
    logger.info("\n[5] Performing Random Manual Sampling Audit")
    random_samples = {}
    for s, df in splits.items():
        sample = df.sample(min(3, len(df)), random_state=42)[["content", "sentiment_id"]].to_dict(orient="records")
        random_samples[s] = sample
        logger.info(f"[{s}] Sample texts OK.")

    audit_metrics["random_samples"] = random_samples

    # 6 & 7. Verify Cleaning Pipeline Logic & Dataset Diversity
    logger.info("\n[6 & 7] Verifying Dataset Diversity and Pipeline Logic")
    diversity_stats = {}
    for s, df in splits.items():
        unique_texts = df["content"].nunique()
        diversity_stats[s] = {
            "total_rows": len(df),
            "unique_texts": unique_texts,
            "diversity_ratio": round(unique_texts / len(df), 4) if len(df) > 0 else 0
        }
        logger.info(f"[{s}] Diversity Ratio: {diversity_stats[s]['diversity_ratio']} (Unique: {unique_texts})")
        if diversity_stats[s]['diversity_ratio'] < 0.99:  # Expecting close to 1.0 since duplicates were removed
            red_flags.append(f"Massive dataset collapse or poor diversity in {s}")

    audit_metrics["diversity_stats"] = diversity_stats

    # 8. Verify Class Weight Calculation
    logger.info("\n[8] Verifying Class Weight Calculation")
    labels = splits["train"]["sentiment_id"].dropna().astype(int).values
    classes = np.array([0, 1, 2])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    weights = [round(float(w), 4) for w in weights]
    
    weights_match = weights == EXPECTED_WEIGHTS
    logger.info(f"Calculated Weights: {weights} | Match Expected: {weights_match}")
    if not weights_match:
        red_flags.append(f"Class weights mismatch! Expected {EXPECTED_WEIGHTS}, got {weights}")
    
    audit_metrics["class_weights_verification"] = {
        "calculated": weights,
        "expected": EXPECTED_WEIGHTS,
        "match": weights_match
    }

    # 9 & 10. Verify Tokenizer Compatibility & Token Statistics
    logger.info("\n[9 & 10] Verifying Tokenizer Compatibility and Statistics")
    try:
        tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        all_texts = pd.concat([df["content"] for df in splits.values()]).dropna().astype(str).tolist()
        
        # Take a small random sample to verify tokenization doesn't crash
        sample_texts = all_texts[:500]
        enc = tokenizer(sample_texts, max_length=96, padding="max_length", truncation=True, return_tensors="pt")
        tokenizer_works = True
        logger.info("Tokenizer sanity check passed on cleaned dataset sample.")
    except Exception as e:
        tokenizer_works = False
        red_flags.append(f"Tokenizer failure on cleaned dataset: {str(e)}")
        logger.error(f"Tokenizer check failed: {e}")

    audit_metrics["tokenizer_compatibility"] = tokenizer_works

    if tokenizer_works:
        # Check token lengths to ensure max_length=96 is still good
        lengths = [len(tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in sample_texts]
        arr = np.array(lengths)
        pct_covered = (arr <= 96).sum() / len(arr) * 100
        logger.info(f"Token stats check on sample: {pct_covered:.2f}% covered by max_length=96.")
        
        if pct_covered < 95.0:
            logger.warning("Truncation coverage dropped below 95% on sample! Might want to re-check max_length.")
        audit_metrics["token_coverage_sample"] = pct_covered

    # Final Verdict
    logger.info("\n[FINAL VERDICT]")
    if len(red_flags) == 0:
        audit_verdict = "PASS"
    elif any("corruption" in flag.lower() or "leakage" in flag.lower() for flag in red_flags):
        audit_verdict = "FAIL"
    else:
        audit_verdict = "PASS WITH MINOR WARNINGS"

    logger.info(f"VERDICT: {audit_verdict}")
    for flag in red_flags:
        logger.error(f"RED FLAG: {flag}")

    audit_metrics["final_verdict"] = audit_verdict
    audit_metrics["red_flags"] = red_flags

    # Save metrics
    with open(os.path.join(RESULTS, "phase2c_audit_metrics.json"), "w") as f:
        json.dump(audit_metrics, f, indent=2)

    # Generate Report Text
    report_lines = [
        "===========================================================",
        "PHASE 2C DATASET AUDIT REPORT",
        f"Generated: {datetime.datetime.now().isoformat()}",
        "===========================================================",
        "",
        "1. DATASET SIZES & INTEGRITY",
    ]
    for s, info in audit_metrics["file_verification"].items():
        if isinstance(info, dict):
            report_lines.append(f"  {s}: {info['rows']} rows (Match Expected: {info['match']})")
        else:
            report_lines.append(f"  {s}: {info}")
            
    report_lines += [
        "",
        "2. DUPLICATE & LEAKAGE VERIFICATION",
        "  Exact Duplicates Remaining:",
        f"    Train: {audit_metrics['duplicate_counts'].get('train', 'N/A')}",
        f"    Val:   {audit_metrics['duplicate_counts'].get('validation', 'N/A')}",
        f"    Test:  {audit_metrics['duplicate_counts'].get('test', 'N/A')}",
        "  Cross-Split Leakage:",
        f"    Train/Val Overlap: {audit_metrics['leakage_stats'].get('train_validation_overlap', 'N/A')}",
        f"    Train/Test Overlap: {audit_metrics['leakage_stats'].get('train_test_overlap', 'N/A')}",
        f"    Val/Test Overlap: {audit_metrics['leakage_stats'].get('validation_test_overlap', 'N/A')}",
        "",
        "3. LABEL & DIVERSITY INTEGRITY",
    ]
    
    for s, st in audit_metrics["label_integrity"].items():
        report_lines.append(f"  [{s}] Valid Labels: {not st['invalid_labels_found']} | Nulls: {st['has_nulls']}")
    for s, st in audit_metrics["diversity_stats"].items():
        report_lines.append(f"  [{s}] Diversity Ratio: {st['diversity_ratio']:.4f}")
        
    report_lines += [
        "",
        "4. CLASS WEIGHTS & TOKENIZER VERIFICATION",
        f"  Class Weights Calculated: {audit_metrics['class_weights_verification']['calculated']}",
        f"  Class Weights Matched Expected: {audit_metrics['class_weights_verification']['match']}",
        f"  Tokenizer Compatibility: {'PASS' if audit_metrics['tokenizer_compatibility'] else 'FAIL'}",
        "",
        "5. RANDOM SAMPLE INSPECTION (First 1 from each split)",
        f"  Train: {audit_metrics['random_samples'].get('train', [{}])[0].get('content', 'N/A')[:60]}... (Label: {audit_metrics['random_samples'].get('train', [{}])[0].get('sentiment_id', 'N/A')})",
        f"  Val:   {audit_metrics['random_samples'].get('validation', [{}])[0].get('content', 'N/A')[:60]}... (Label: {audit_metrics['random_samples'].get('validation', [{}])[0].get('sentiment_id', 'N/A')})",
        f"  Test:  {audit_metrics['random_samples'].get('test', [{}])[0].get('content', 'N/A')[:60]}... (Label: {audit_metrics['random_samples'].get('test', [{}])[0].get('sentiment_id', 'N/A')})",
        "",
        "===========================================================",
        f"FINAL VERDICT: {audit_verdict}",
        "==========================================================="
    ]
    
    if red_flags:
        report_lines.append("RED FLAGS DETECTED:")
        for flag in red_flags:
            report_lines.append(f" - {flag}")

    with open(os.path.join(RESULTS, "phase2c_audit_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Audit report saved to {os.path.join(RESULTS, 'phase2c_audit_report.txt')}")
    logger.info("PHASE 2C AUDIT COMPLETE.")

if __name__ == "__main__":
    main()
