#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

# Local imports
from utils import OptimizedContextLocator, NumpyEncoder
from claim_checking import nli_score


def tokenize(text: str) -> List[str]:
    text = text.lower()
    # Keep words of length >= 2
    return re.findall(r"[a-z0-9_\-]{2,}", text)


class BM25Retriever:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.tokenized_docs = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0

        # Term frequencies per document
        self.tf: List[Dict[str, int]] = []
        # Document frequency per term
        self.df: Dict[str, int] = {}

        for tokens in self.tokenized_docs:
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)

            # Update DF (once per term per doc)
            for t in counts.keys():
                self.df[t] = self.df.get(t, 0) + 1

        self.N = len(self.tokenized_docs)

        # Precompute IDF using BM25+ style smoothing
        # idf(t) = log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf: Dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> List[float]:
        q_tokens = tokenize(query)
        # Aggregate duplicate query terms weights
        q_term_counts: Dict[str, int] = {}
        for qt in q_tokens:
            q_term_counts[qt] = q_term_counts.get(qt, 0) + 1

        scores = [0.0] * self.N
        for idx in range(self.N):
            dl = self.doc_lengths[idx]
            denom_norm = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-8)))
            tf_counts = self.tf[idx]
            s = 0.0
            for term, qtf in q_term_counts.items():
                if term not in self.idf:
                    continue
                f = tf_counts.get(term, 0)
                if f == 0:
                    continue
                # BM25 term contribution
                num = f * (self.k1 + 1)
                denom = f + denom_norm
                s += self.idf[term] * (num / (denom + 1e-12))
            scores[idx] = s
        return scores

    def top_k(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        scores = self.score(query)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores[:k]


def load_cached_web_content(cache_path: str) -> Dict[str, str]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_corpus_from_cache(cache: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
    concatenated_parts: List[str] = []
    stats = {"total": 0, "skipped_error": 0, "used": 0}
    for url, content in cache.items():
        stats["total"] += 1
        if not content:
            stats["skipped_error"] += 1
            continue
        upper = content.upper()
        if "ERROR" in upper or upper.startswith("[ERROR]"):
            stats["skipped_error"] += 1
            continue
        concatenated_parts.append(content)
        stats["used"] += 1
    combined = "\n\n".join(concatenated_parts)
    return combined, stats


def main():
    parser = argparse.ArgumentParser(description="BM25 retrieval followed by NLI scoring over cached web content.")
    parser.add_argument(
        "--cache_file",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "web_content_cache",
            "cache_gemini_PhD_jobs.json",
        ),
        help="Path to cached web content JSON",
    )
    parser.add_argument(
        "--claims",
        type=str,
        nargs="*",
        default=[
            "OpenAI has featured roles available.",
            "Google DeepMind has open Research Scientist roles in the US.",
            "Google DeepMind may have Machine Learning Engineer roles that need further exploration.",
            "Meta has an AI careers page for job searches.",
            "Meta's job search functionality can help identify specific job titles and locations."
        ],
        help="Claims to evaluate",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top K chunks by BM25 to score with NLI")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
            "test_claim_checking_results_bm25.json",
        ),
        help="Output JSON path",
    )

    args = parser.parse_args()

    # 1) Load cache and concatenate usable content
    cache = load_cached_web_content(args.cache_file)
    combined_text, cache_stats = build_corpus_from_cache(cache)

    # 2) Chunk the combined text
    locator = OptimizedContextLocator()
    chunks = locator.extract_sentences(combined_text)
    chunk_texts = [c["chunk_text"] for c in chunks]

    # 3) Build BM25 index
    retriever = BM25Retriever(chunk_texts, k1=1.5, b=0.75)

    # 4) For each claim: retrieve top-K and run NLI
    results = {
        "meta": {
            "cache_file": os.path.abspath(args.cache_file),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "bm25": {"k1": retriever.k1, "b": retriever.b, "avgdl": retriever.avgdl, "num_docs": retriever.N},
            "cache_stats": cache_stats,
            "num_chunks": len(chunk_texts),
        },
        "claims": [],
    }

    for claim in args.claims:
        topk = retriever.top_k(claim, k=args.top_k)
        top_entries = []
        for idx, bm25_score in topk:
            premise = chunk_texts[idx]
            scores = nli_score(premise, claim)
            top_entries.append(
                {
                    "chunk_index": idx,
                    "bm25_score": float(bm25_score),
                    "chunk_text": premise if len(premise) <= 400 else premise[:400] + "...",
                    "nli": {
                        "entailment": scores.get("entailment", 0.0),
                        "neutral": scores.get("neutral", 0.0),
                        "contradiction": scores.get("contradiction", 0.0),
                    },
                }
            )

        results["claims"].append(
            {
                "claim": claim,
                "top_k": args.top_k,
                "retrieved_chunks": top_entries,
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"Saved BM25+NLI results to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()


