import numpy as np
import math
import re
from typing import List, Dict, Any
from collections import Counter

class BM25Okapi:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Native BM25Okapi implementation from scratch.
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0
        self.doc_lengths = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_contents = []

    def _advanced_tokenize(self, text: str) -> List[str]:
        """
        Advanced tokenizer using regex to capture words and handle punctuation.
        """
        text = text.lower()
        # Find all alphanumeric sequences (handles standard tokens well)
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def fit(self, documents: List[Dict[str, Any]]):
        """
        Indexes a list of document chunks.
        Documents should be dicts containing at least "content" and "chunk_id".
        """
        self.doc_contents = documents
        self.corpus_size = len(documents)
        
        tokenized_corpus = [self._advanced_tokenize(doc["content"]) for doc in documents]
        
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size else 0

        # Calculate document frequencies
        df = Counter()
        for document in tokenized_corpus:
            self.doc_freqs.append(Counter(document))
            df.update(set(document))

        # Calculate IDF
        for word, freq in df.items():
            # Standard BM25 IDF formula
            numerator = self.corpus_size - freq + 0.5
            denominator = freq + 0.5
            self.idf[word] = math.log(numerator / denominator + 1)

    def score_corpus(self, query: str) -> List[float]:
        """
        Scores all documents in the corpus against the query.
        """
        query_tokens = self._advanced_tokenize(query)
        scores = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.idf:
                continue
            
            idf = self.idf[token]
            for doc_idx, doc_freqs in enumerate(self.doc_freqs):
                f = doc_freqs.get(token, 0)
                if f > 0:
                    dl = self.doc_lengths[doc_idx]
                    numerator = f * (self.k1 + 1)
                    denominator = f + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                    scores[doc_idx] += idf * (numerator / denominator)

        return scores

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'k1': self.k1,
                'b': self.b,
                'corpus_size': self.corpus_size,
                'avg_doc_len': self.avg_doc_len,
                'doc_lengths': self.doc_lengths,
                'doc_freqs': self.doc_freqs,
                'idf': self.idf,
                'doc_contents': self.doc_contents
            }, f)

    def load(self, path: str) -> bool:
        import pickle
        import os
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.k1 = data['k1']
            self.b = data['b']
            self.corpus_size = data['corpus_size']
            self.avg_doc_len = data['avg_doc_len']
            self.doc_lengths = data['doc_lengths']
            self.doc_freqs = data['doc_freqs']
            self.idf = data['idf']
            self.doc_contents = data['doc_contents']
        return True

    def get_scores(self, query: str) -> np.ndarray:
        """
        Calculates BM25 scores for all documents given a query.
        """
        return np.array(self.score_corpus(query))

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Returns the top_k chunks for the given query.
        """
        if self.corpus_size == 0:
            return []
            
        scores = self.get_scores(query)
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0: # Only return if there is some overlap
                res = self.doc_contents[idx].copy()
                res["bm25_score"] = scores[idx]
                results.append(res)
                
        return results
