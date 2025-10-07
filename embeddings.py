"""Embeddings helper with lazy model loading and caching."""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
from .config import MODEL_NAME, EMBEDDING_BATCH
from .utils import logger

_MODEL = None

def get_model(name: str = MODEL_NAME):
    global _MODEL
    if _MODEL is None:
        logger.info(f"Loading embedding model: {name}")
        _MODEL = SentenceTransformer(name)
    return _MODEL

def embed_texts(texts: Union[str, List[str]], batch_size: Optional[int] = EMBEDDING_BATCH) -> List[np.ndarray]:
    model = get_model()
    if isinstance(texts, str):
        texts = [texts]
    try:
        embs = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        return embs
    except Exception as e:
        logger.error(f"Embedding failure: {e}")
        return np.zeros((len(texts), model.get_sentence_embedding_dimension()))

def cosine_sim(a: np.ndarray, b: np.ndarray):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.ndim == 1 and b.ndim == 1:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    # matrix multiplication path
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(a_norm, b_norm.T)
