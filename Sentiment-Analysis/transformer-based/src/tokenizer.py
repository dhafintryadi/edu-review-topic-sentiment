"""
src/tokenizer.py
Reusable tokenizer pipeline for IndoBERTweet sentiment analysis.
Phase 2B — CPU-efficient, configurable, logging-supported.
"""

import logging
from typing import List, Union, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

DEFAULT_MODEL      = "indolem/indobertweet-base-uncased"
DEFAULT_MAX_LENGTH = 128


class SentimentTokenizer:
    """
    Reusable tokenizer wrapper for IndoBERTweet.
    Supports single/batch tokenization with configurable max_length,
    padding, truncation, and return format.
    """

    def __init__(
        self,
        model_name: str         = DEFAULT_MODEL,
        max_length: int         = DEFAULT_MAX_LENGTH,
        padding: Union[bool, str] = "max_length",
        truncation: bool         = True,
        return_tensors: Optional[str] = None,
    ):
        """
        Args:
            model_name:     HuggingFace model identifier.
            max_length:     Maximum token sequence length.
            padding:        Padding strategy — True / "max_length" / "longest" / False.
            truncation:     Whether to truncate sequences exceeding max_length.
            return_tensors: None (lists) | "pt" (PyTorch) | "np" (NumPy).
        """
        self.model_name     = model_name
        self.max_length     = max_length
        self.padding        = padding
        self.truncation     = truncation
        self.return_tensors = return_tensors

        logger.info(f"[SentimentTokenizer] Loading tokenizer: {model_name}")
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        logger.info(
            f"[SentimentTokenizer] Ready — vocab_size={self._tokenizer.vocab_size}, "
            f"max_length={max_length}, padding={padding}, truncation={truncation}"
        )

    # ── Public API ──────────────────────────────────────────

    def __call__(
        self,
        texts: Union[str, List[str]],
        return_tensors: Optional[str] = None,
    ) -> dict:
        """
        Tokenize one or more texts.

        Args:
            texts:          Single string or list of strings.
            return_tensors: Override instance default.

        Returns:
            dict with keys: input_ids, attention_mask, token_type_ids (optional)
        """
        rt = return_tensors if return_tensors is not None else self.return_tensors
        return self._tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=rt,
        )

    def tokenize_batch(self, texts: List[str], return_tensors: Optional[str] = None) -> dict:
        """Alias for batch tokenization."""
        return self(texts, return_tensors=return_tensors)

    def get_token_length(self, text: str) -> int:
        """Return token count for a single text (with special tokens, no truncation)."""
        ids = self._tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        return len(ids)

    def get_token_lengths_batch(self, texts: List[str], batch_size: int = 512) -> List[int]:
        """
        Compute raw token lengths (no truncation) for a list of texts.
        Memory-efficient: processes in batches.
        """
        lengths = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc   = self._tokenizer(batch, add_special_tokens=True, truncation=False)
            lengths.extend(len(ids) for ids in enc["input_ids"])
        return lengths

    # ── HuggingFace Dataset .map() compatible function ──────

    def encode_for_map(self, examples: dict, text_col: str = "content") -> dict:
        """
        Tokenization function compatible with HuggingFace Dataset.map().

        Usage:
            dataset = dataset.map(
                lambda x: tokenizer.encode_for_map(x, text_col="content"),
                batched=True,
            )
        """
        return self(examples[text_col])

    # ── Properties ──────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def cls_token_id(self) -> int:
        return self._tokenizer.cls_token_id

    @property
    def sep_token_id(self) -> int:
        return self._tokenizer.sep_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def underlying(self) -> PreTrainedTokenizerBase:
        """Access the raw HuggingFace tokenizer."""
        return self._tokenizer

    def __repr__(self) -> str:
        return (
            f"SentimentTokenizer(model={self.model_name!r}, "
            f"max_length={self.max_length}, padding={self.padding!r}, "
            f"truncation={self.truncation})"
        )


# ── Factory shortcut ────────────────────────────────────────

def build_tokenizer(
    model_name: str = DEFAULT_MODEL,
    max_length: int = DEFAULT_MAX_LENGTH,
    **kwargs,
) -> SentimentTokenizer:
    """
    Convenience factory for creating a SentimentTokenizer.

    Example:
        tok = build_tokenizer(max_length=128)
        encoded = tok(["Produk ini bagus!", "Jelek sekali"])
    """
    return SentimentTokenizer(model_name=model_name, max_length=max_length, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    tok = build_tokenizer(max_length=128)
    samples = [
        "Aplikasi ini sangat membantu dan mudah digunakan!",
        "Pelayanan buruk dan lambat, tidak rekomendasikan.",
        "Biasa saja.",
    ]
    enc = tok(samples)
    print(f"input_ids shape   : ({len(enc['input_ids'])}, {len(enc['input_ids'][0])})")
    print(f"attention_mask OK : {all(len(m) == len(enc['input_ids'][0]) for m in enc['attention_mask'])}")
    print(f"Token lengths (no truncation): {tok.get_token_lengths_batch(samples)}")
    print(repr(tok))
