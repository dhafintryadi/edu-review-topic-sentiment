import shutil
from pathlib import Path

def copy_assets():
    src_root = Path(r"c:\Users\ASUS\Documents\AITF-2026\PKL")
    dest_root = Path(r"C:\Users\ASUS\Documents\2027 S.Si.D\Github Portofolio\Projects Repository\edu-review-topic-sentiment")
    
    files_to_copy = [
        # Datasets
        "Datasets/raw_review.csv",
        "Datasets/sentiment_preprocessed.csv",
        "Datasets/topic_preprocessed.csv",
        # NN Model
        "Sentiment-Analysis/nn-based/baseline_model.pkl",
        # LDA Preprocessing Outputs
        "Topic-Modelling/phase1_outputs/lda_ready_corpus.csv",
        # LDA Dictionary
        "Topic-Modelling/phase2_outputs/lda_dictionary.gensim",
        # LDA Models
        "Topic-Modelling/phase3b_outputs/lda_model_final_k8.gensim",
        "Topic-Modelling/phase3b_outputs/lda_model_final_k8.gensim.expElogbeta.npy",
        "Topic-Modelling/phase3b_outputs/lda_model_final_k8.gensim.id2word",
        "Topic-Modelling/phase3b_outputs/lda_model_final_k8.gensim.state",
        "Topic-Modelling/phase3b_outputs/phase3b_topic_taxonomy.json",
    ]
    
    for relative_path in files_to_copy:
        src = src_root / relative_path
        dest = dest_root / relative_path
        
        if not src.exists():
            print(f"[SKIP] Source file does not exist: {src}")
            continue
            
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying {src} -> {dest}")
        shutil.copy2(src, dest)
        
    print("Done copying all assets!")

if __name__ == "__main__":
    copy_assets()
