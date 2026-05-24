import re
import unicodedata

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r"<.*?>")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
MULTI_PUNCT_PATTERN = re.compile(r"([!?.,…]){2,}")


def remove_control_characters(text: str) -> str:
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C"
    )


def basic_clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = URL_PATTERN.sub("", text)
    text = HTML_PATTERN.sub(" ", text)
    text = remove_control_characters(text)
    text = MULTI_PUNCT_PATTERN.sub(r"\1", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


def normalize_excessive_punctuation(text: str) -> str:
    text = re.sub(r"([?!]){2,}", r"\1", text)
    text = re.sub(r"(\.){2,}", r".", text)
    text = re.sub(r"(,){2,}", r",", text)
    text = re.sub(r"(\"){2,}", r'"', text)
    return text
