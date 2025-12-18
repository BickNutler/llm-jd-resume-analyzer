from __future__ import annotations
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..preprocessing.clean_text import clean_text

@dataclass
class TfidfSimilarity:
    vectorizer: TfidfVectorizer

    def score(self, a: str, b: str) -> float:
        a2, b2 = clean_text(a), clean_text(b)
        X = self.vectorizer.fit_transform([a2, b2])
        sim = cosine_similarity(X[0], X[1])[0][0]
        return float(sim)

def default_tfidf() -> TfidfSimilarity:
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        max_features=5000,
        stop_words="english"
    )
    return TfidfSimilarity(vectorizer=vec)
