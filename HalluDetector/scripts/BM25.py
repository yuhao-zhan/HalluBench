import re
import math
from typing import List, Dict, Tuple
from utils import _min_max_norm


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9_\-]{2,}", text)


class BM25Retriever:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}

        for tokens in self.tokenized_docs:
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts.keys():
                self.df[t] = self.df.get(t, 0) + 1

        self.N = len(self.tokenized_docs)
        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> List[float]:
        q_tokens = tokenize(query)
        q_counts: Dict[str, int] = {}
        for qt in q_tokens:
            q_counts[qt] = q_counts.get(qt, 0) + 1

        scores = [0.0] * self.N
        for idx in range(self.N):
            dl = self.doc_lengths[idx]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-8)))
            tf_counts = self.tf[idx]
            s = 0.0
            for term in q_counts.keys():
                if term not in self.idf:
                    continue
                f = tf_counts.get(term, 0)
                if f == 0:
                    continue
                num = f * (self.k1 + 1)
                denom = f + denom_norm
                s += self.idf[term] * (num / (denom + 1e-12))
            scores[idx] = s
        # Normalize the scores using min-max normalization
        scores, _, _ = _min_max_norm(scores)
        return scores
