"""
PHASE 2A - Comprehensive Transformer Ecosystem Validation
Model: indolem/indobertweet-base-uncased
"""

import sys
import os
import logging
import datetime
import subprocess

# Fix Windows console encoding
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────
# Setup Logging
# ──────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "transformer_setup.log")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "phase2a_summary.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

MODEL_NAME = "indolem/indobertweet-base-uncased"
results = {}

def check(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                status = "PASS"
                detail = result if result else "OK"
                log.info(f"  ✅ {name}: {detail}")
            except Exception as e:
                status = "FAIL"
                detail = str(e)
                log.error(f"  ❌ {name}: {detail}")
            results[name] = {"status": status, "detail": detail}
            return status == "PASS"
        return wrapper
    return decorator


# ──────────────────────────────────────────────
# STEP 1: Import Checks
# ──────────────────────────────────────────────
log.info("=" * 60)
log.info("PHASE 2A - Transformer Ecosystem Validation")
log.info(f"Timestamp: {datetime.datetime.now().isoformat()}")
log.info(f"Python: {sys.version}")
log.info("=" * 60)

log.info("\n[1] Import Checks")

@check("import_transformers")
def test_import_transformers():
    import transformers
    return f"v{transformers.__version__}"

@check("import_torch")
def test_import_torch():
    import torch
    return f"v{torch.__version__}"

@check("import_datasets")
def test_import_datasets():
    import datasets
    return f"v{datasets.__version__}"

@check("import_accelerate")
def test_import_accelerate():
    import accelerate
    return f"v{accelerate.__version__}"

@check("import_evaluate")
def test_import_evaluate():
    import evaluate
    return f"v{evaluate.__version__}"

@check("import_sentencepiece")
def test_import_sentencepiece():
    import sentencepiece
    return f"v{sentencepiece.__version__}"

test_import_transformers()
test_import_torch()
test_import_datasets()
test_import_accelerate()
test_import_evaluate()
test_import_sentencepiece()

# ──────────────────────────────────────────────
# STEP 2: Device / CUDA Check
# ──────────────────────────────────────────────
log.info("\n[2] Device Availability")

import torch

@check("device_check")
def test_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
    return f"device={device}, cuda_version={cuda_ver}"

test_device()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# STEP 3: Tokenizer Validation
# ──────────────────────────────────────────────
log.info("\n[3] Tokenizer Validation")

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

@check("tokenizer_load")
def test_tokenizer_load():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return f"vocab_size={tokenizer.vocab_size}"

@check("tokenizer_vocab_size")
def test_vocab_size():
    assert tokenizer.vocab_size > 0, "Vocab size must be > 0"
    return f"{tokenizer.vocab_size}"

@check("tokenization_basic")
def test_tokenization():
    text = "Saya sangat senang hari ini!"
    tokens = tokenizer(text, return_tensors="pt")
    ids = tokens["input_ids"]
    assert ids.shape[1] > 0, "No tokens generated"
    return f"tokens={ids.shape[1]}, input_ids shape={list(ids.shape)}"

@check("tokenization_batch")
def test_tokenization_batch():
    texts = [
        "Produk ini sangat bagus",
        "Pelayanan buruk sekali",
        "Biasa saja, tidak istimewa",
    ]
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    assert tokens["input_ids"].shape[0] == 3, "Batch size mismatch"
    return f"batch={tokens['input_ids'].shape}"

@check("tokenization_special_tokens")
def test_special_tokens():
    assert tokenizer.cls_token is not None, "No CLS token"
    assert tokenizer.sep_token is not None, "No SEP token"
    return f"cls={tokenizer.cls_token}, sep={tokenizer.sep_token}, pad={tokenizer.pad_token}"

test_tokenizer_load()
test_vocab_size()
test_tokenization()
test_tokenization_batch()
test_special_tokens()

# ──────────────────────────────────────────────
# STEP 4: Model Loading
# ──────────────────────────────────────────────
log.info("\n[4] Model Loading")

@check("automodel_load")
def test_automodel_load():
    global base_model
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model.eval()
    n_params = sum(p.numel() for p in base_model.parameters())
    return f"params={n_params:,}"

@check("model_config")
def test_model_config():
    cfg = base_model.config
    return (
        f"hidden_size={cfg.hidden_size}, "
        f"num_layers={cfg.num_hidden_layers}, "
        f"num_heads={cfg.num_attention_heads}"
    )

@check("model_to_device")
def test_model_to_device():
    base_model.to(DEVICE)
    return f"model on {DEVICE}"

test_automodel_load()
test_model_config()
test_model_to_device()

# ──────────────────────────────────────────────
# STEP 5: Forward Pass
# ──────────────────────────────────────────────
log.info("\n[5] Forward Pass")

@check("tensor_generation")
def test_tensor_generation():
    global sample_inputs
    text = "Aplikasi ini sangat membantu pekerjaan saya sehari-hari"
    sample_inputs = tokenizer(text, return_tensors="pt")
    sample_inputs = {k: v.to(DEVICE) for k, v in sample_inputs.items()}
    return f"input_ids shape={list(sample_inputs['input_ids'].shape)}"

@check("forward_pass")
def test_forward_pass():
    import torch
    with torch.no_grad():
        outputs = base_model(**sample_inputs)
    last_hidden = outputs.last_hidden_state
    assert last_hidden.shape[0] == 1, "Batch dim mismatch"
    return f"last_hidden_state shape={list(last_hidden.shape)}"

@check("simple_inference")
def test_simple_inference():
    import torch
    texts = ["bagus", "jelek", "oke"]
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = base_model(**enc)
    pooled = out.last_hidden_state[:, 0, :]  # [CLS] token
    assert pooled.shape == (3, base_model.config.hidden_size), "Pooled shape mismatch"
    return f"pooled CLS shape={list(pooled.shape)}"

test_tensor_generation()
test_forward_pass()
test_simple_inference()

# ──────────────────────────────────────────────
# STEP 6: AutoModelForSequenceClassification
# ──────────────────────────────────────────────
log.info("\n[6] AutoModelForSequenceClassification")

@check("seq_cls_init")
def test_seq_cls_init():
    import torch
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
    )
    clf_model.eval()
    clf_model.to(DEVICE)
    text = "Saya suka produk ini"
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = clf_model(**enc).logits
    assert logits.shape == (1, 3), f"Logits shape wrong: {logits.shape}"
    return f"logits shape={list(logits.shape)}, OK for 3-class classification"

test_seq_cls_init()

# ──────────────────────────────────────────────
# STEP 7: Get Installed Packages
# ──────────────────────────────────────────────
log.info("\n[7] Installed Package Versions")

key_packages = [
    "transformers", "torch", "datasets", "accelerate",
    "evaluate", "sentencepiece", "tokenizers", "safetensors",
    "huggingface_hub", "numpy", "scikit-learn", "pyarrow"
]

pkg_versions = {}
for pkg in key_packages:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                ver = line.split(":", 1)[1].strip()
                pkg_versions[pkg] = ver
                log.info(f"  {pkg}: {ver}")
                break
    except Exception:
        pkg_versions[pkg] = "unknown"

# ──────────────────────────────────────────────
# STEP 8: Write Summary
# ──────────────────────────────────────────────
log.info("\n[8] Writing Summary")

import transformers
total = len(results)
passed = sum(1 for v in results.values() if v["status"] == "PASS")
failed = total - passed

summary_lines = [
    "=" * 60,
    "PHASE 2A VALIDATION SUMMARY",
    f"Generated: {datetime.datetime.now().isoformat()}",
    "=" * 60,
    "",
    "── ENVIRONMENT ─────────────────────────────────────────────",
    f"Python Version  : {sys.version.split()[0]}",
    f"Platform        : {sys.platform}",
    f"Device          : {'CUDA' if torch.cuda.is_available() else 'CPU'}",
    f"CUDA Available  : {torch.cuda.is_available()}",
    "",
    "── PACKAGE VERSIONS ────────────────────────────────────────",
]
for pkg, ver in pkg_versions.items():
    summary_lines.append(f"  {pkg:<25} {ver}")

summary_lines += [
    "",
    "── VALIDATION RESULTS ──────────────────────────────────────",
    f"Total Checks : {total}",
    f"Passed       : {passed}",
    f"Failed       : {failed}",
    "",
]

for check_name, info in results.items():
    icon = "✅" if info["status"] == "PASS" else "❌"
    summary_lines.append(f"  {icon} {check_name:<35} {info['detail'][:80]}")

summary_lines += [
    "",
    "── CONCLUSION ──────────────────────────────────────────────",
    f"  Phase 2A Status: {'✅ COMPLETE' if failed == 0 else '❌ INCOMPLETE - See failed checks above'}",
    "=" * 60,
]

summary_text = "\n".join(summary_lines)

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    f.write(summary_text)

log.info(f"\nSummary written to: {SUMMARY_FILE}")
log.info(f"Log written to    : {LOG_FILE}")
log.info("\n" + summary_text)
log.info("\n✅ PHASE 2A VALIDATION COMPLETE" if failed == 0 else f"\n❌ PHASE 2A: {failed} check(s) failed")
