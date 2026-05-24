"""
PHASE 2B - Validation & Summary Report
Validates tokenizer module, dataset encoder, HuggingFace Dataset integration,
and generates logs/phase2b.log + results/phase2b_summary.txt
"""

import os, sys, json, logging, datetime
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
sys.path.insert(0, BASE)

RESULTS = os.path.join(ROOT, "results")
LOGS    = os.path.join(ROOT, "logs")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS,    exist_ok=True)

LOG_FILE     = os.path.join(LOGS,    "phase2b.log")
SUMMARY_FILE = os.path.join(RESULTS, "phase2b_summary.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

results = {}

def chk(name):
    def dec(fn):
        def wrapper(*a, **kw):
            try:
                detail = fn(*a, **kw)
                results[name] = {"status": "PASS", "detail": str(detail) if detail else "OK"}
                log.info(f"  [PASS] {name}: {results[name]['detail']}")
            except Exception as e:
                results[name] = {"status": "FAIL", "detail": str(e)}
                log.error(f"  [FAIL] {name}: {e}")
        return wrapper
    return dec

log.info("=" * 60)
log.info("PHASE 2B - Validation Run")
log.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
log.info("=" * 60)

# ── Import modules ──────────────────────────────────────────
log.info("\n[1] Module Imports")

@chk("import_tokenizer_module")
def test_import_tok():
    from tokenizer import SentimentTokenizer, build_tokenizer
    return "OK"

@chk("import_dataset_encoder_module")
def test_import_enc():
    from dataset_encoder import (
        load_csv_splits, build_dataset_dict,
        encode_dataset, validate_encoded_dataset, build_encoded_datasets
    )
    return "OK"

test_import_tok()
test_import_enc()

# ── Tokenizer module validation ─────────────────────────────
log.info("\n[2] SentimentTokenizer Validation")
from tokenizer import build_tokenizer

@chk("tokenizer_instantiation")
def test_tok_init():
    global tok
    tok = build_tokenizer(max_length=128)
    return repr(tok)

@chk("tokenizer_single_text")
def test_tok_single():
    enc = tok(["Produk ini sangat bagus dan membantu!"])
    assert len(enc["input_ids"])     == 1
    assert len(enc["input_ids"][0])  == 128
    assert len(enc["attention_mask"][0]) == 128
    return f"input_ids len={len(enc['input_ids'][0])}"

@chk("tokenizer_batch_text")
def test_tok_batch():
    texts = ["Bagus!", "Jelek sekali", "Biasa aja", "Sangat membantu"]
    enc = tok(texts)
    assert len(enc["input_ids"]) == 4
    assert all(len(ids) == 128 for ids in enc["input_ids"])
    return f"batch=4, each len=128"

@chk("tokenizer_get_lengths")
def test_tok_lengths():
    lengths = tok.get_token_lengths_batch(["Halo", "Ini adalah review yang cukup panjang sekali"])
    assert len(lengths) == 2
    assert all(l > 0 for l in lengths)
    return f"lengths={lengths}"

@chk("tokenizer_encode_for_map")
def test_tok_map_fn():
    examples = {"content": ["Bagus", "Jelek"]}
    enc = tok.encode_for_map(examples, text_col="content")
    assert "input_ids" in enc and "attention_mask" in enc
    return "encode_for_map compatible"

@chk("tokenizer_vocab_size")
def test_tok_vocab():
    assert tok.vocab_size == 31923
    return f"vocab_size={tok.vocab_size}"

test_tok_init()
test_tok_single()
test_tok_batch()
test_tok_lengths()
test_tok_map_fn()
test_tok_vocab()

# ── Dataset encoder validation ──────────────────────────────
log.info("\n[3] Dataset Encoder Validation")

from dataset_encoder import (
    load_csv_splits, build_dataset_dict,
    encode_dataset, validate_encoded_dataset
)

DATA_DIR = os.path.join(ROOT, "data", "processed")

@chk("csv_load_splits")
def test_load_csv():
    global frames
    frames = load_csv_splits(DATA_DIR)
    assert "train" in frames and "validation" in frames and "test" in frames
    sizes = {s: len(df) for s, df in frames.items()}
    return f"sizes={sizes}"

@chk("dataset_dict_build")
def test_build_dict():
    global raw_ds
    raw_ds = build_dataset_dict(frames)
    assert "train" in raw_ds and "validation" in raw_ds and "test" in raw_ds
    return f"train={len(raw_ds['train'])}, val={len(raw_ds['validation'])}, test={len(raw_ds['test'])}"

@chk("dataset_columns")
def test_columns():
    assert "text" in raw_ds["train"].column_names
    assert "label" in raw_ds["train"].column_names
    return f"columns={raw_ds['train'].column_names}"

@chk("dataset_encode_small_batch")
def test_encode_small():
    """Encode first 50 samples of each split to validate pipeline (CPU-friendly)."""
    global enc_small
    small_ds = {
        split: ds.select(range(min(50, len(ds))))
        for split, ds in raw_ds.items()
    }
    from datasets import DatasetDict
    small_dict = DatasetDict(small_ds)
    enc_small = encode_dataset(small_dict, tok, batch_size=16)
    return f"encoded small: train={len(enc_small['train'])}"

@chk("encoded_columns_present")
def test_enc_cols():
    cols = enc_small["train"].column_names
    assert "input_ids"     in cols, f"Missing input_ids, got {cols}"
    assert "attention_mask" in cols, f"Missing attention_mask, got {cols}"
    assert "labels"         in cols, f"Missing labels, got {cols}"
    return f"columns={cols}"

@chk("encoded_tensor_shapes")
def test_tensor_shapes():
    sample = enc_small["train"][:4]
    assert sample["input_ids"].shape     == (4, 128), f"Got {sample['input_ids'].shape}"
    assert sample["attention_mask"].shape == (4, 128), f"Got {sample['attention_mask'].shape}"
    assert sample["labels"].shape         == (4,),    f"Got {sample['labels'].shape}"
    return (f"input_ids={tuple(sample['input_ids'].shape)}, "
            f"mask={tuple(sample['attention_mask'].shape)}, "
            f"labels={tuple(sample['labels'].shape)}")

@chk("label_values_valid")
def test_label_values():
    labels = [int(x) for x in enc_small["train"]["labels"]]
    invalid = [l for l in labels if l not in [0, 1, 2]]
    assert len(invalid) == 0, f"Invalid labels found: {set(invalid)}"
    return f"all labels in {{0,1,2}}, sample={labels[:8]}"

@chk("dataset_structural_validation")
def test_struct_validation():
    report = validate_encoded_dataset(enc_small, max_length=128)
    failed = [s for s, r in report.items() if not r["all_pass"]]
    assert len(failed) == 0, f"Failed splits: {failed}"
    return "all splits PASS"

test_load_csv()
test_build_dict()
test_columns()
test_encode_small()
test_enc_cols()
test_tensor_shapes()
test_label_values()
test_struct_validation()

# ── Result files check ──────────────────────────────────────
log.info("\n[4] Analysis Output Files Check")

required_files = {
    "results/duplicate_analysis.json":        os.path.join(RESULTS, "duplicate_analysis.json"),
    "results/token_length_analysis.json":     os.path.join(RESULTS, "token_length_analysis.json"),
    "results/truncation_impact_analysis.json":os.path.join(RESULTS, "truncation_impact_analysis.json"),
    "visualizations/token_length_distribution.png": os.path.join(ROOT, "visualizations", "token_length_distribution.png"),
    "visualizations/truncation_tradeoff.png":       os.path.join(ROOT, "visualizations", "truncation_tradeoff.png"),
    "visualizations/duplicate_distribution.png":    os.path.join(ROOT, "visualizations", "duplicate_distribution.png"),
}

for label, path in required_files.items():
    exists = os.path.exists(path)
    status = "PASS" if exists else "FAIL"
    results[f"file_{label}"] = {"status": status, "detail": "exists" if exists else "MISSING"}
    log.info(f"  [{status}] {label}")

# ── Load analysis results for summary ──────────────────────
token_analysis   = json.load(open(os.path.join(RESULTS, "token_length_analysis.json")))
truncation_data  = json.load(open(os.path.join(RESULTS, "truncation_impact_analysis.json")))
dup_data         = json.load(open(os.path.join(RESULTS, "duplicate_analysis.json")))

recommended_max  = truncation_data["recommendation"]["recommended_max_length"]

# ── Summary report ──────────────────────────────────────────
log.info("\n[5] Writing Summary")

total  = len(results)
passed = sum(1 for v in results.values() if v["status"] == "PASS")
failed = total - passed

lines = [
    "=" * 65,
    "PHASE 2B SUMMARY — Token Length Analysis & Tokenization Pipeline",
    f"Generated: {datetime.datetime.now().isoformat()}",
    "=" * 65,
    "",
    "── ENVIRONMENT ─────────────────────────────────────────────────",
    f"  Python          : {sys.version.split()[0]}",
    f"  Device          : CPU",
    f"  Model           : indolem/indobertweet-base-uncased",
    "",
    "── DATASET AUDIT ───────────────────────────────────────────────",
]
for split in ["train", "validation", "test"]:
    d = dup_data.get(split, {})
    lines.append(f"  {split:<12}: {d.get('total_rows',0):>6,} rows | "
                 f"{d.get('duplicate_rows',0):>5,} duplicates ({d.get('duplicate_ratio',0):.1f}%)")

lk = dup_data.get("cross_split_leakage", {})
lines += [
    f"  Cross-split leakage:",
    f"    train↔val  : {lk.get('train_val', 'N/A')}",
    f"    train↔test : {lk.get('train_test', 'N/A')}",
    f"    val↔test   : {lk.get('val_test', 'N/A')}",
    "",
    "── TOKEN LENGTH ANALYSIS ───────────────────────────────────────",
    f"  Total samples analyzed : {token_analysis['total_samples']:,}",
    f"  Mean token length       : {token_analysis['mean']}",
    f"  Median token length     : {token_analysis['median']}",
    f"  Std deviation           : {token_analysis['std']}",
    f"  Min / Max               : {token_analysis['min']} / {token_analysis['max']}",
    f"  P75 / P90 / P95 / P99  : "
    f"{token_analysis['p75']} / {token_analysis['p90']} / "
    f"{token_analysis['p95']} / {token_analysis['p99']}",
    "",
    "── TRUNCATION IMPACT ANALYSIS ──────────────────────────────────",
]
for ml_str, td in truncation_data.items():
    if ml_str == "recommendation":
        continue
    lines.append(f"  max_length={int(ml_str):<5}: "
                 f"{td['pct_covered']:>6.2f}% covered, "
                 f"{td['pct_truncated']:>5.2f}% truncated "
                 f"({td['truncated_count']:,} samples)")

lines += [
    "",
    "── RECOMMENDATION ──────────────────────────────────────────────",
    f"  Recommended max_length : {recommended_max}",
    f"  Rationale              : {truncation_data['recommendation']['rationale']}",
    "",
    "── CPU TRAINING NOTES ──────────────────────────────────────────",
]
for note in truncation_data["recommendation"]["cpu_notes"]:
    lines.append(f"  - {note}")

lines += [
    "",
    "── VALIDATION RESULTS ──────────────────────────────────────────",
    f"  Total checks : {total}",
    f"  Passed       : {passed}",
    f"  Failed       : {failed}",
    "",
]
for name, info in results.items():
    icon = "[PASS]" if info["status"] == "PASS" else "[FAIL]"
    lines.append(f"  {icon} {name:<45} {info['detail'][:60]}")

lines += [
    "",
    "── CONCLUSION ──────────────────────────────────────────────────",
    f"  Phase 2B Status: {'COMPLETE' if failed == 0 else 'INCOMPLETE — see failed checks'}",
    "=" * 65,
]

summary = "\n".join(lines)
with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write(summary)

log.info(f"\nSummary written to: {SUMMARY_FILE}")
log.info("\n" + summary)
log.info(f"\n{'PHASE 2B VALIDATION COMPLETE' if failed == 0 else f'PHASE 2B: {failed} check(s) failed'}")
