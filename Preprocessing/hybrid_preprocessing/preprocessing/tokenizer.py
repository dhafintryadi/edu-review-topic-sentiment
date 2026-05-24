from nltk.tokenize import TweetTokenizer
from typing import List, Set

_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=True)


def tokenize(text: str) -> List[str]:
    if text is None:
        return []
    return [tok for tok in _tokenizer.tokenize(text) if tok.strip()]


def remove_stopwords(tokens: List[str], stopwords: Set[str], negations: Set[str]) -> List[str]:
    return [tok for tok in tokens if tok not in stopwords or tok in negations]
