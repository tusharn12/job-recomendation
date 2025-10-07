"""Matcher implements hybrid scoring (TF-IDF + embeddings) with keyword boosting."""
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from .embeddings import embed_texts, cosine_sim
from .config import TFIDF_MAX_FEATURES, DEFAULT_HYBRID_ALPHA, KEYWORD_BOOST
from .utils import logger

class Matcher:
    def __init__(self, jobs: List[Dict] = None):
        self.job_texts = []
        self.job_meta = []
        self.job_embs = None
        self.tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1,2))
        if jobs:
            self.fit_jobs(jobs)

    def fit_jobs(self, jobs: List[Dict]):
        """Fit TF-IDF on job texts and pre-compute embeddings."""
        logger.info(f"Fitting matcher on {len(jobs)} jobs.")
        self.job_meta = jobs
        self.job_texts = [j['text'] for j in jobs]
        if not self.job_texts:
            logger.warning("No job texts given to fit_jobs.")
            return
        self.tfidf.fit(self.job_texts)
        self.job_embs = embed_texts(self.job_texts)

    def _keyword_boost(self, resume_text: str, job_text: str, base_score: float) -> float:
        """Boost score if explicit skills/keywords exist in resume matching job description."""
        resume_low = resume_text.lower()
        boost = 0.0
        # simple heuristic: count exact token overlaps for words longer than 2 chars
        job_tokens = set(t for t in job_text.lower().split() if len(t) > 2)
        overlap = sum(1 for tok in job_tokens if tok in resume_low)
        if overlap > 0:
            boost = KEYWORD_BOOST * min(1.0, overlap / 10.0)
        return base_score + boost

    def match(self, resume_text: str, top_k: int = 5, alpha: float = DEFAULT_HYBRID_ALPHA) -> List[Dict]:
        """Return top_k matching jobs with scores and metadata."""
        if not resume_text:
            logger.warning("Empty resume_text provided to match().")
            return []
        if not self.job_texts:
            logger.warning("Matcher has no jobs fitted; returning empty list.")
            return []

        # TF-IDF similarity
        try:
            resume_tfidf = self.tfidf.transform([resume_text])
            job_tfidf = self.tfidf.transform(self.job_texts)
            tfidf_sim = (job_tfidf @ resume_tfidf.T).toarray().squeeze()
        except Exception as e:
            logger.error(f"TF-IDF similarity failed: {e}")
            tfidf_sim = np.zeros(len(self.job_texts))

        # Embedding similarity
        try:
            resume_emb = embed_texts(resume_text)
            emb_sim = cosine_sim(self.job_embs, resume_emb)
            if emb_sim.ndim > 1:
                emb_sim = emb_sim.squeeze()
        except Exception as e:
            logger.error(f"Embedding similarity failed: {e}")
            emb_sim = np.zeros(len(self.job_texts))

        def normalize(x: np.ndarray) -> np.ndarray:
            x = np.array(x, dtype=float)
            if x.max() - x.min() < 1e-9:
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())

        tfidf_norm = normalize(tfidf_sim)
        emb_norm = normalize(emb_sim)

        raw_scores = alpha * emb_norm + (1.0 - alpha) * tfidf_norm

        # apply keyword boosting per job
        boosted = []
        for i, s in enumerate(raw_scores):
            boosted_score = self._keyword_boost(resume_text, self.job_texts[i], s)
            boosted.append(boosted_score)

        boosted = np.array(boosted)
        indices = np.argsort(boosted)[::-1][:top_k]
        results = []
        for idx in indices:
            meta = self.job_meta[idx]
            results.append({
                'job_id': meta.get('id') or meta.get('job_id'),
                'title': meta.get('title') or meta.get('job_title'),
                'company': meta.get('company'),
                'score': float(boosted[idx])
            })
        return results
