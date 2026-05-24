import re
from typing import List, Dict

import emoji
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

REPEATED_CHAR_PATTERN = re.compile(r"([a-zA-Z])\1{2,}")
_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()
_STEM_CACHE: Dict[str, str] = {}


def reduce_repeated_characters(text: str) -> str:
    return REPEATED_CHAR_PATTERN.sub(r"\1", text)


def emoji_to_text(text: str) -> str:
    if text is None:
        return ""
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r":([a-z0-9_]+):", r"\1", text)
    return text


def normalize_slang(tokens: List[str], slang_dict: Dict[str, str]) -> List[str]:
    return [slang_dict.get(token, token) for token in tokens]


def normalize_typo(tokens: List[str], typo_dict: Dict[str, str]) -> List[str]:
    return [typo_dict.get(token, token) for token in tokens]


def stem_tokens(tokens: List[str]) -> List[str]:
    stemmed = []

    for token in tokens:
        if token.isalpha():
            stemmed.append(_STEM_CACHE.setdefault(token, _stemmer.stem(token)))
        else:
            stemmed.append(token)

    return stemmed
