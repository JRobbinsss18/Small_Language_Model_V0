import pickle, os
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self, path: str):
        self.path = path
        self.vectorizer: TfidfVectorizer | None = None
        self.docs: List[Dict[str, str]] | None = None
        self.legacy_upgraded = False

    def _upgrade_legacy(self, loaded_docs: Any) -> List[Dict[str, str]]:
        if isinstance(loaded_docs, list) and loaded_docs and isinstance(loaded_docs[0], str):
            self.legacy_upgraded = True
            return [{'text': t, 'source': 'legacy'} for t in loaded_docs]
        if isinstance(loaded_docs, list) and (not loaded_docs or isinstance(loaded_docs[0], dict)):
            return loaded_docs
        raise RuntimeError("Unsupported retriever docs format")

    def build_or_load(self, chunks: List[Dict[str,str]] | None):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                self.vectorizer, loaded_docs = pickle.load(f)
            self.docs = self._upgrade_legacy(loaded_docs)
            with open(self.path, 'wb') as f:
                pickle.dump((self.vectorizer, self.docs), f)
            return
        if chunks is None:
            raise RuntimeError('chunks required to build retriever')
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)
        self.docs = chunks
        self.vectorizer.fit([c['text'] for c in self.docs])
        with open(self.path, 'wb') as f:
            pickle.dump((self.vectorizer, self.docs), f)

    def rebuild(self, chunks: List[Dict[str,str]]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True)
        self.docs = chunks
        self.vectorizer.fit([c['text'] for c in self.docs])
        with open(self.path, 'wb') as f:
            pickle.dump((self.vectorizer, self.docs), f)
        self.legacy_upgraded = False

    def topk(self, query: str, k: int = 5) -> List[Dict[str,str]]:
        if self.vectorizer is None or self.docs is None:
            raise RuntimeError("retriever not initialized")
        qv = self.vectorizer.transform([query])
        dv = self.vectorizer.transform([c['text'] for c in self.docs])
        sims = cosine_similarity(qv, dv)[0]
        idxs = np.argsort(-sims)[:k]
        return [self.docs[i] for i in idxs]
