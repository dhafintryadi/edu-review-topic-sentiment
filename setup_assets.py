"""
setup_assets.py
===============
One-time setup: copy pre-trained binaries into assets/ directory.
Run this once before running run_pipeline.py.

Usage:
    C:\\Users\\ASUS\\miniconda3\\python.exe setup_assets.py
"""
import shutil
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
(REPO_ROOT / "core" / "resources").mkdir(parents=True, exist_ok=True)

copies = [
    # (source, dest_filename)
    (REPO_ROOT / "Datasets" / "raw_review.csv",
     ASSETS_DIR / "raw_review.csv"),

    (REPO_ROOT / "Sentiment-Analysis" / "nn-based" / "baseline_model.pkl",
     ASSETS_DIR / "baseline_model.pkl"),

    (REPO_ROOT / "Topic-Modelling" / "phase3b_outputs" / "lda_model_final_k8.gensim",
     ASSETS_DIR / "lda_model_final_k8.gensim"),

    # Gensim saves companion files alongside the main .gensim file
    (REPO_ROOT / "Topic-Modelling" / "phase3b_outputs" / "lda_model_final_k8.gensim.expElogbeta.npy",
     ASSETS_DIR / "lda_model_final_k8.gensim.expElogbeta.npy"),

    (REPO_ROOT / "Topic-Modelling" / "phase3b_outputs" / "lda_model_final_k8.gensim.id2word",
     ASSETS_DIR / "lda_model_final_k8.gensim.id2word"),

    (REPO_ROOT / "Topic-Modelling" / "phase3b_outputs" / "lda_model_final_k8.gensim.state",
     ASSETS_DIR / "lda_model_final_k8.gensim.state"),

    (REPO_ROOT / "Topic-Modelling" / "phase2_outputs" / "lda_dictionary.gensim",
     ASSETS_DIR / "lda_dictionary.gensim"),

    (REPO_ROOT / "Topic-Modelling" / "phase3b_outputs" / "phase3b_topic_taxonomy.json",
     ASSETS_DIR / "phase3b_topic_taxonomy.json"),

    (REPO_ROOT / "Topic-Modelling" / "phase1_outputs" / "lda_ready_corpus.csv",
     ASSETS_DIR / "lda_ready_corpus.csv"),

    (REPO_ROOT / "Topic-Modelling" / "phase3d_outputs" / "sr_blueprint_validation.json",
     ASSETS_DIR / "sr_blueprint_validation.json"),

    # Preprocessing resources to core/resources/
    (REPO_ROOT / "Preprocessing" / "hybrid_preprocessing" / "resources" / "slang_dict.json",
     REPO_ROOT / "core" / "resources" / "slang_dict.json"),

    (REPO_ROOT / "Preprocessing" / "hybrid_preprocessing" / "resources" / "stopwords.txt",
     REPO_ROOT / "core" / "resources" / "stopwords.txt"),
]

print("Setting up assets/ directory...")
for src, dst in copies:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  [OK] {dst.name}")
    else:
        print(f"  [SKIP] Source not found: {src.relative_to(REPO_ROOT)}")

print(f"\nassets/ contents:")
for f in sorted(ASSETS_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
print("\nSetup complete. Run: python run_pipeline.py")
