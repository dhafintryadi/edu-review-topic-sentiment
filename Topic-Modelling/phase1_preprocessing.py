"""
PHASE_1: Topic Preprocessing Optimization — Sekolah Rakyat
==========================================================
Objectives:
  1. Remove punctuation artifacts & ultra-generic fillers
  2. Normalize Indonesian slang (preserve semantics)
  3. Educational-domain stopword filtering
  4. Phrase detection → educational bigrams
  5. Low-information review filtering (preserve short complaints)
  6. Semantic validation on high-confidence negative reviews
  7. Export LDA-ready corpus

Input  : PKL/Datasets/main_review_preprocessed/topic_preprocessed.csv
         PKL/Sentiment-Analysis/.../final_sentiment_inference.csv
Output : PKL/Topic-Modelling/phase1_outputs/
"""

import re
import json
import warnings
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim.models.phrases import Phrases, Phraser

warnings.filterwarnings("ignore")

# ── Paths (Updated for Clean Data) ───────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
# Menggunakan sentiment_predictions.csv yang baru saja kita bersihkan dan validasi
SENTI_CSV = ROOT / "Sentiment-Analysis/nn-based/sentiment_predictions.csv"
MAIN_CSV  = SENTI_CSV # Data sentimen kita sekarang sudah berisi kolom content_topic yang dibutuhkan
OUT_DIR   = Path(__file__).parent / "phase1_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("  PHASE 1 — Topic Preprocessing Optimization")
print("=" * 65)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A. Indonesian slang normalization dictionary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLANG_MAP = {
    # Pronouns / common shortenings
    "gw": "saya", "gue": "saya", "aku": "saya", "sy": "saya",
    "lo": "kamu", "lu": "kamu", "kalian": "kamu",
    # Negations
    "gak": "tidak", "ga": "tidak", "enggak": "tidak", "nggak": "tidak",
    "ngga": "tidak", "gaada": "tidak ada", "gada": "tidak ada",
    # Common verbs / filler
    "udah": "sudah", "udh": "sudah", "sdh": "sudah",
    "blm": "belum", "belom": "belum",
    "mau": "ingin", "pengen": "ingin", "pgn": "ingin",
    "bisa": "bisa",  # keep
    "bikin": "membuat", "bikinin": "membuat",
    "pake": "pakai", "dipake": "dipakai", "dipakein": "dipakai",
    "liat": "lihat", "ngliat": "melihat",
    "nonton": "menonton", "tonton": "menonton",
    "dapet": "dapat", "dpet": "dapat",
    "beli": "beli",   # keep
    "kasih": "beri",  "dikasih": "diberi",
    "tolong": "tolong",  # keep — complaint signal
    "coba": "coba",   # keep
    "banget": "sekali",
    "bgt": "sekali", "bngt": "sekali",
    "aja": "saja", "doang": "saja",
    "kaya": "seperti", "kayak": "seperti",
    "gimana": "bagaimana", "gmn": "bagaimana",
    "kenapa": "mengapa", "knpa": "mengapa",
    "apk": "aplikasi", "app": "aplikasi",
    "hp": "handphone",
    "ngelag": "lag", "nge-lag": "lag", "ngelag": "lag",
    "lemot": "lambat", "lelet": "lambat",
    "eror": "error", "errror": "error",
    "ngebug": "bug", "nge-bug": "bug",
    "loading": "loading",   # keep as barrier term
    "stuck": "stuck",       # keep
    "freeze": "freeze",     # keep
    "crash": "crash",       # keep
    # Payment / access
    "bayar": "bayar",       # keep
    "mahal": "mahal",       # keep
    "gratis": "gratis",     # keep
    "langganan": "berlangganan",
    # Teaching / learning
    "ajar": "belajar",
    "blajar": "belajar", "bljar": "belajar",
    "ngajar": "mengajar",
    "ngerti": "mengerti", "ngerti": "mengerti",
    "paham": "paham",       # keep
    "materi": "materi",     # keep
    "soal": "soal",         # keep
    "guru": "guru",         # keep
    "murid": "murid",       # keep
    "sekolah": "sekolah",   # keep
    "kurikulum": "kurikulum",  # keep
    # Login / auth
    "login": "login",       # keep
    "daftar": "daftar",     # keep
    "akun": "akun",         # keep
    "sandi": "sandi",       # keep
    # Misc
    "trs": "terus", "terus": "terus",
    "kok": "mengapa",
    "deh": "", "sih": "", "loh": "", "dong": "",
    "yuk": "", "yah": "", "ya": "",
    "wkwk": "", "hehe": "", "haha": "",
    "plis": "tolong", "pliss": "tolong",
    "minn": "admin", "min": "admin",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# B. Educational domain stopwords
#    Keep: technical, barrier, curriculum, learning terms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EDU_STOPWORDS = {
    # Punctuation artifacts remaining
    ",", ".", "!", "?", ":", ";", "(", ")", "-", "+", "=",
    '"', "'", "/", "\\", "|", "★", "•", "–",
    # Ultra-generic fillers
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan",
    "ini", "itu", "ada", "juga", "kami", "mereka", "kita",
    "sudah", "sudah", "belum", "akan", "bisa", "ingin",
    "sangat", "sekali", "lebih", "paling", "cukup",
    "karena", "jadi", "jika", "kalau", "tapi", "tetapi",
    "namun", "meskipun", "walaupun", "bahwa",
    "saya", "kamu", "anda", "dia",
    "sama", "buat", "saat", "pas", "waktu",
    "lagi", "masih", "sudah", "baru", "lama",
    "banyak", "semua", "beberapa", "setiap",
    "hal", "cara", "bagian", "tempat",
    "kali", "hari", "menit", "jam",
    "banget_removed",  # placeholder — handled via slang
    # Emoji descriptors from preprocessing
    "double_exclamation_mark", "folded_hands", "thumbs_up", "thumbs_down",
    "smiling_face", "crying_face", "loudly_crying_face", "red_heart",
    "blue_heart", "white_heart", "star_struck", "clapping_hands",
    "face_with_tears_of_joy", "smiling_face_with_hearts",
    "smiling_face_with_smiling_eyes", "smiling_face_with_heart",
    "beaming_face", "grinning_face", "neutral_face", "pensive_face",
    "slightly_smiling_face", "disappointed_face", "winking_face",
    "face_with_hand_over_mouth", "relieved_face", "melting_face",
    "person_facepalming", "sparkles", "fire", "star", "books",
    "victory_hand", "raising_hands", "folded_hands_light_skin_tone",
    "thumbs_up_light_skin_tone", "heart_suit", "sparkling_heart",
    "revolving_hearts", "growing_heart", "heart_with_ribbon",
    "t_rex", "alien_monster", "moai", "smiling_cat_with_heart",
    "smiling_face_with_sunglasses", "bouquet",
    # Generic positive words that add no topical value
    "bagus", "baik", "keren", "mantap", "oke",
    "terima", "kasih", "terimakasih", "makasih",
    "semoga", "harap", "mohon", "tolong",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C. Critical learning-barrier vocabulary to NEVER remove
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRESERVE_TERMS = {
    # Technical barriers
    "bug", "error", "crash", "lag", "loading", "stuck", "freeze",
    "lambat", "offline", "server", "update", "restart", "blank",
    # Login / access
    "login", "daftar", "akun", "sandi", "password", "verifikasi",
    # Paywall
    "bayar", "mahal", "premium", "berlangganan", "gratis", "biaya",
    # Content
    "materi", "soal", "latihan", "video", "kurikulum", "pelajaran",
    "quiz", "kuis", "latihan", "rangkuman", "pembahasan",
    # Navigation/UX
    "tampilan", "menu", "fitur", "navigasi", "antarmuka",
    # Teaching quality
    "guru", "mengajar", "penjelasan", "paham", "mengerti",
    # System
    "notifikasi", "iklan", "chat", "whatsapp",
    # Important negations when combined
    "tidak", "tidak_bisa", "tidak_muncul", "tidak_tersedia",
    "gak_bisa", "tidak_ada",
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# D. Preprocessing functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def normalize_slang(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        mapped = SLANG_MAP.get(t, t)
        if mapped:  # drop empty-string mappings (filler removal)
            out.append(mapped)
    return out


def apply_stopwords(tokens: list[str]) -> list[str]:
    return [
        t for t in tokens
        if t not in EDU_STOPWORDS or t in PRESERVE_TERMS
    ]


# Emoji descriptor pattern — any token that is a lowercase word joined
# by underscores AND not in PRESERVE_TERMS (emoji names from preprocessing)
_EMOJI_RE = re.compile(
    r"^(face|smiling|grinning|crying|laughing|beaming|angry|enraged"
    r"|thumbs|clapping|raising|folded|sparkling|revolving|growing"
    r"|heart|star|fire|bouquet|books|victory|hundred|person"
    r"|neutral|pensive|relieved|melting|slightly|winking|disappointed"
    r"|moai|alien|t_rex|cat|palm|woman|man|loudly|slightly)"
    r".*"
)

def is_emoji_token(t: str) -> bool:
    """Return True if token looks like an emoji descriptor."""
    return bool(_EMOJI_RE.match(t)) and t not in PRESERVE_TERMS


def clean_token(t: str) -> str:
    """Strip residual punctuation from token edges."""
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t.lower().strip()


def preprocess_text(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    tokens = text.lower().split()
    tokens = [clean_token(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]   # drop 1-char
    tokens = [t for t in tokens if not is_emoji_token(t)]  # purge emoji descriptors
    tokens = normalize_slang(tokens)
    tokens = apply_stopwords(tokens)
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Load data
print("\n[1] Loading data …")
main_df = pd.read_csv(MAIN_CSV, low_memory=False)
senti_df = main_df.copy() # File yang sama
senti_main = senti_df.copy() # Tidak perlu filter 'source' lagi

usable = main_df[
    main_df["content_topic"].notna() &
    (main_df["content_topic"].str.strip() != "")
].copy()
print(f"  Usable rows: {len(usable):,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Apply preprocessing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[2] Applying preprocessing pipeline …")
usable["tokens_clean"] = usable["content_topic"].apply(preprocess_text)
usable["token_count_clean"] = usable["tokens_clean"].str.len()

before_filter = len(usable)
# Low-information filter: drop docs with < 2 tokens UNLESS they contain
# a preserve term (short but meaningful complaint)
def is_meaningful(tokens):
    if len(tokens) >= 2:
        return True
    if any(t in PRESERVE_TERMS for t in tokens):
        return True
    return False

usable = usable[usable["tokens_clean"].apply(is_meaningful)].copy()
after_filter = len(usable)
print(f"  Filtered {before_filter - after_filter:,} low-information docs -> {after_filter:,} remain")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Bigram / phrase detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[3] Training bigram phrase model …")
token_lists = usable["tokens_clean"].tolist()

# Pass 1 — broad bigrams (lower threshold catches barrier pairs)
phrases_model = Phrases(
    token_lists,
    min_count=10,    # appear >=10 times
    threshold=5,     # lower = more permissive bigram formation
    connector_words=frozenset(),
)
bigram_phraser = Phraser(phrases_model)

# Pass 2 — trigrams from bigrams
bigram_lists = [bigram_phraser[toks] for toks in token_lists]
trigram_model = Phrases(bigram_lists, min_count=8, threshold=5)
trigram_phraser = Phraser(trigram_model)

usable["tokens_bigram"] = usable["tokens_clean"].apply(
    lambda toks: trigram_phraser[bigram_phraser[toks]]
)

# Collect all detected bigrams for inspection
all_bigram_tokens = [t for toks in usable["tokens_bigram"] for t in toks]
bigram_only = [t for t in all_bigram_tokens if "_" in t]
bigram_counter = Counter(bigram_only)
top_bigrams = pd.DataFrame(bigram_counter.most_common(60),
                            columns=["phrase", "count"])
top_bigrams.to_csv(OUT_DIR / "detected_bigrams.csv", index=False)

print(f"  Unique bigrams detected : {len(bigram_counter):,}")
print("  Top 30 bigrams:")
print(top_bigrams.head(30).to_string(index=False))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Validate candidate barrier phrases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CANDIDATE_PHRASES = [
    "login_gagal", "tidak_bisa_login", "tidak_bisa_masuk",
    "loading_lama", "loading_terus", "stuck_loading",
    "video_error", "video_tidak_muncul", "video_tidak_bisa",
    "notif_ganggu", "dihubungi_terus", "chat_terus",
    "tidak_bisa_buka", "keluar_sendiri", "force_close",
    "tidak_tersedia", "materi_kurang", "soal_tidak_muncul",
    "harus_bayar", "harus_premium", "tidak_gratis",
    "bug_terus", "sering_error", "sering_lag",
    "tidak_paham", "sulit_dipahami", "tidak_mengerti",
]

print("\n[4] Candidate barrier phrase presence:")
found_phrases = {}
for phrase in CANDIDATE_PHRASES:
    count = bigram_counter.get(phrase, 0)
    found_phrases[phrase] = count
    status = "✓" if count > 0 else "✗"
    print(f"  {status}  {phrase:<35} count={count}")

phrase_validation = pd.DataFrame(
    [{"phrase": k, "count": v, "detected": v > 0}
     for k, v in found_phrases.items()]
)
phrase_validation.to_csv(OUT_DIR / "candidate_phrase_validation.csv", index=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Semantic validation on high-confidence negative reviews
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[5] Semantic validation — high-confidence negatives …")
HIGH_CONF_THRESHOLD = 0.90

neg_high = senti_main[
    (senti_main["predicted_label_name"] == "negative") &
    (senti_main["calibrated_confidence"] >= HIGH_CONF_THRESHOLD)
][["reviewId", "predicted_label_name", "calibrated_confidence",
   "content_topic", "content_raw"]].copy()

print(f"  High-confidence negatives: {len(neg_high):,}")

# Merge with processed tokens
val_df = neg_high.merge(
    usable[["reviewId", "tokens_bigram", "token_count_clean"]],
    on="reviewId", how="inner"
)
print(f"  Matched after preprocessing: {len(val_df):,}")

# Check preserve-term coverage
val_df["has_barrier_term"] = val_df["tokens_bigram"].apply(
    lambda toks: any(
        t in PRESERVE_TERMS or "_" in t for t in toks
    )
)
coverage = val_df["has_barrier_term"].mean() * 100
print(f"  Barrier/bigram term coverage: {coverage:.1f}%")

# Sample 10 for qualitative review
sample_val = val_df[["content_raw", "tokens_bigram",
                      "calibrated_confidence", "has_barrier_term"]].sample(
    min(10, len(val_df)), random_state=42
)
sample_val["tokens_bigram"] = sample_val["tokens_bigram"].apply(
    lambda x: " | ".join(x)
)
sample_val.to_csv(OUT_DIR / "semantic_validation_sample.csv", index=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Token length stats after preprocessing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[6] Post-preprocessing token stats …")
usable["token_count_bigram"] = usable["tokens_bigram"].str.len()
post_stats = usable["token_count_bigram"].describe(
    percentiles=[.25, .5, .75, .90, .95]
).round(2)
print(post_stats.to_string())

# Vocabulary size after cleaning
final_vocab = Counter(t for toks in usable["tokens_bigram"] for t in toks)
print(f"\n  Final vocabulary size : {len(final_vocab):,}")
print(f"  Total tokens          : {sum(final_vocab.values()):,}")

# Plot before/after token distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(main_df["content_topic"].dropna().str.split().str.len().dropna(),
             bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
axes[0].set_title("Before — Raw content_topic", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Token count"); axes[0].set_ylabel("Frequency")

axes[1].hist(usable["token_count_bigram"], bins=50,
             color="#27ae60", edgecolor="white", alpha=0.8)
axes[1].set_title("After — Cleaned + Bigrams", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Token count")

plt.suptitle("Token Length Distribution: Before vs After Phase 1",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "token_length_before_after.png", dpi=150)
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Top vocabulary after cleaning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
top_final = pd.DataFrame(final_vocab.most_common(60),
                          columns=["term", "frequency"])
top_final.to_csv(OUT_DIR / "top60_final_vocabulary.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(top_final["term"][:40][::-1],
        top_final["frequency"][:40][::-1],
        color="#27ae60", edgecolor="white")
ax.set_title("Top 40 Terms After Phase 1 Preprocessing",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(OUT_DIR / "top40_final_vocabulary.png", dpi=150)
plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Export corpus for LDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[7] Exporting LDA-ready corpus …")

# Merge sentiment labels (as metadata — NOT used in topic model input)
corpus_df = usable[["reviewId", "score", "content_raw",
                     "content_topic", "tokens_bigram",
                     "token_count_bigram"]].copy()

corpus_df = corpus_df.merge(
    senti_main[["reviewId", "predicted_label_name",
                "calibrated_confidence", "probability_negative"]],
    on="reviewId", how="left"
)

# Serialise token list to space-joined string for CSV
corpus_df["lda_text"] = corpus_df["tokens_bigram"].apply(
    lambda toks: " ".join(toks) if isinstance(toks, list) else ""
)

# Drop rows where lda_text is empty
corpus_df = corpus_df[corpus_df["lda_text"].str.strip() != ""].reset_index(drop=True)

corpus_df.to_csv(OUT_DIR / "lda_ready_corpus.csv", index=False)
print(f"  Exported {len(corpus_df):,} documents -> lda_ready_corpus.csv")

# Save token lists as JSON for gensim direct loading
token_lists_out = corpus_df["tokens_bigram"].tolist()
with open(OUT_DIR / "lda_token_lists.json", "w", encoding="utf-8") as f:
    json.dump(token_lists_out, f, ensure_ascii=False)

# Save phase summary
summary = {
    "phase": "PHASE_1_TOPIC_PREPROCESSING_OPTIMIZATION",
    "input_docs": int(before_filter),
    "after_quality_filter": int(after_filter),
    "final_lda_corpus": int(len(corpus_df)),
    "final_vocab_size": len(final_vocab),
    "total_tokens_final": int(sum(final_vocab.values())),
    "bigrams_detected": int(len(bigram_counter)),
    "barrier_coverage_pct": round(coverage, 2),
    "post_preprocessing_token_stats": post_stats.to_dict(),
    "candidate_phrases_detected": int(phrase_validation["detected"].sum()),
    "candidate_phrases_total": int(len(phrase_validation)),
}
with open(OUT_DIR / "phase1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 65)
print("  PHASE 1 COMPLETE")
print(f"  Outputs → {OUT_DIR}")
print("=" * 65)
for p in sorted(OUT_DIR.iterdir()):
    print(f"    {p.name:<50} {p.stat().st_size/1024:>7.1f} KB")
