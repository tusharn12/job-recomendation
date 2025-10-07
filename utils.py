"""Utility helpers: text cleaning, tokenization, and logging setup."""
import re
import logging
from typing import List
import nltk
from nltk.corpus import stopwords

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Ensure stopwords are available (quietly attempt download)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Normalize whitespace and basic punctuation in text."""
    if not isinstance(text, str):
        logger.debug("clean_text received non-str input.")
        return ""
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def simple_tokenize(text: str) -> List[str]:
    """Lowercase tokenization with stopword removal; returns tokens."""
    text = clean_text(text).lower()
    tokens = re.findall(r"\b[a-z0-9+#\.\-]+\b", text)
    return [t for t in tokens if t not in STOPWORDS]
