#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute noise domination without query weights.

Inputs mirror noise_domination_detector.detect_noise_domination:
- claim: str
- relevant_chunks: List[Dict]
- related_queries: List[Dict] (expects items with 'query_idx')
- query_scores: Dict returned by chunk_scoring.py (expects 'detailed_chunk_scores')

Behavior:
- For each chunk, compute final_score as the average of its combined scores
  over ONLY the related query indices. No weight is applied.
- Rank chunks by final_score and check if any entailment chunk appears in the top 10%.
"""

import numpy as np
from typing import List, Dict, Any


def _get_related_query_indices(related_queries: List[Dict[str, Any]], detailed_chunk_scores: List[Dict[str, Any]]) -> List[str]:
    indices: List[str] = []
    for query_info in related_queries:
        if isinstance(query_info, dict) and "query_idx" in query_info:
            indices.append(str(query_info["query_idx"]))

    if indices:
        return indices

    # Fallback: use all available indices from first chunk if none provided
    if detailed_chunk_scores:
        first = detailed_chunk_scores[0]
        # Prefer original_query_scores; fallback to expanded_query_scores if present
        scores_map = first.get("original_query_scores") or first.get("expanded_query_scores") or {}
        return [str(k) for k in scores_map.keys()]

    return []


def _lookup_score(scores_map: Dict[Any, float], query_idx: str) -> float:
    # Be robust to key types (str vs int)
    if query_idx in scores_map:
        return float(scores_map[query_idx])
    try:
        as_int = int(query_idx)
        if as_int in scores_map:
            return float(scores_map[as_int])
        as_str = str(as_int)
        if as_str in scores_map:
            return float(scores_map[as_str])
    except Exception:
        pass
    return 0.0


def detect_noise_domination(
    claim: str,
    relevant_chunks: List[Dict[str, Any]],
    related_queries: List[Dict[str, Any]],
    query_scores: Dict[str, Any],
) -> Dict[str, Any]:
    detailed_chunk_scores = query_scores.get("detailed_chunk_scores", [])

    if not detailed_chunk_scores:
        return {
            "claim": claim,
            "is_noise_dominated": True,
            "reason": "No chunk scores available",
            "document_level_noise": True,
            "chunk_level_noise": True,
            "entailment_chunks": [],
            "top_10_percent_threshold": 0,
            "total_chunks": 0,
        }

    related_query_indices = _get_related_query_indices(related_queries, detailed_chunk_scores)
    print(f"{len(related_query_indices)} related query indices")
    # Output the related queries
    for query in related_queries:
        print(f"Query: {query['query']}")

    # Compute final scores (unweighted average over related queries)
    chunk_final_scores: List[Dict[str, Any]] = []
    for chunk in detailed_chunk_scores:
        chunk_id = chunk.get("chunk_id")
        chunk_text = chunk.get("chunk_text", "")
        url = chunk.get("url")

        # Prefer original_query_scores if present; otherwise try expanded_query_scores
        query_scores_map = chunk.get("original_query_scores") or chunk.get("expanded_query_scores") or {}

        related_scores: List[float] = []
        for q_idx in related_query_indices:
            related_scores.append(_lookup_score(query_scores_map, q_idx))

        final_score = float(np.mean(related_scores)) if related_scores else 0.0

        chunk_final_scores.append(
            {
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "url": url,
                "final_score": final_score,
                "related_scores": related_scores,
                "related_query_indices": related_query_indices,
            }
        )

    # Rank by final score (descending) - DOCUMENT-LEVEL ranking
    chunk_final_scores.sort(key=lambda x: x["final_score"], reverse=True)

    # Identify entailment chunks from relevant_chunks
    entailment_chunks: List[Dict[str, Any]] = []
    for chunk in relevant_chunks:
        if chunk.get("score", 0) > 0:
            entailment_chunks.append(chunk)

    # Calculate top 10% threshold for document-level
    total_chunks = len(chunk_final_scores)
    if total_chunks == 0:
        return {
            "claim": claim,
            "is_noise_dominated": True,
            "reason": "No chunks available for ranking",
            "document_level_noise": True,
            "chunk_level_noise": True,
            "entailment_chunks": [],
            "top_10_percent_threshold": 0,
            "total_chunks": 0,
        }

    top_10_percent_count = max(1, int(np.ceil(total_chunks * 0.1)))
    top_10_percent_threshold = (
        chunk_final_scores[top_10_percent_count - 1]["final_score"]
        if top_10_percent_count <= total_chunks
        else chunk_final_scores[-1]["final_score"]
    )
    print(f"========= DOCUMENT-LEVEL NOISE DOMINATION CHECK =========")
    print(f"Top 10% threshold: {top_10_percent_threshold}")

    # Output the top 10% chunk text
    # print()
    # print(f"Top 10% chunk text (Document-level):")
    # for index, chunk in enumerate(chunk_final_scores[:top_10_percent_count]):
    #     print(f"Chunk {index + 1}: {chunk['chunk_text']}")
    #     print("-"*100)

    # Check if entailment chunks are in top 10% (DOCUMENT-LEVEL check)
    entailment_chunks_in_top_10: List[Dict[str, Any]] = []
    document_level_noise = True

    for entailment_chunk in entailment_chunks:
        chunk_id = entailment_chunk.get("chunk_id")
        chunk_url = entailment_chunk.get("source_url")
        # print(f"Entailment chunk: {chunk_id} - {chunk_url}")
        

        matching_chunk = None
        for ranked_chunk in chunk_final_scores:
            if ranked_chunk["url"] == chunk_url and ranked_chunk["chunk_id"] == chunk_id:
                matching_chunk = ranked_chunk
                break

        if not matching_chunk:
            # Fallback: try URL and text overlap
            entailment_text = entailment_chunk.get("chunk_text", "").lower()
            for ranked_chunk in chunk_final_scores:
                if ranked_chunk["url"] == chunk_url:
                    ranked_text = ranked_chunk.get("chunk_text", "").lower()
                    if entailment_text in ranked_text or ranked_text in entailment_text:
                        matching_chunk = ranked_chunk
                        break

        if matching_chunk:
            # print(f"Entailment chunk score: {matching_chunk['final_score']}")
            # print(f"Top 10% threshold: {top_10_percent_threshold}")
            is_in_top_10 = matching_chunk["final_score"] >= top_10_percent_threshold
            entailment_chunks_in_top_10.append(
                {
                    "chunk_id": chunk_id,
                    "source_url": chunk_url,
                    "score": entailment_chunk.get("score", 0),
                    "final_score": matching_chunk["final_score"],
                    "is_in_top_10_percent": is_in_top_10,
                    "rank": chunk_final_scores.index(matching_chunk) + 1,
                }
            )
            if is_in_top_10:
                document_level_noise = False

        else:
            print("Found no matching chunk")

    if not entailment_chunks_in_top_10:
        document_level_noise = True

    # CHUNK-LEVEL noise domination check
    chunk_level_noise = True
    chunk_level_results = []
    
    # print(f"\n{'='*50}")
    print("========= CHUNK-LEVEL NOISE DOMINATION CHECK =========")
    # print(f"{'='*50}")
    
    for entailment_chunk in entailment_chunks:
        chunk_id = entailment_chunk.get("chunk_id")
        chunk_url = entailment_chunk.get("source_url")
        # print(f"\nChecking chunk-level noise for: {chunk_id} - {chunk_url}")
        
        # Get all chunks from the same URL/document
        same_url_chunks = [chunk for chunk in chunk_final_scores if chunk["url"] == chunk_url]
        # print(f"Found {len(same_url_chunks)} chunks in the same document")
        
        if len(same_url_chunks) == 0:
            print("No chunks found in the same document")
            chunk_level_results.append({
                "chunk_id": chunk_id,
                "source_url": chunk_url,
                "is_chunk_level_noise": True,
                "reason": "No chunks found in same document",
                "document_chunk_count": 0,
                "rank_in_document": None,
                "document_top_10_threshold": None
            })
            continue
        
        # Sort chunks from the same document by score
        same_url_chunks.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Calculate top 10% threshold for this document
        doc_chunk_count = len(same_url_chunks)
        doc_top_10_count = max(1, int(np.ceil(doc_chunk_count * 0.1)))
        doc_top_10_threshold = (
            same_url_chunks[doc_top_10_count - 1]["final_score"]
            if doc_top_10_count <= doc_chunk_count
            else same_url_chunks[-1]["final_score"]
        )
        
        # Find the entailment chunk in the same-url chunks
        matching_doc_chunk = None
        for doc_chunk in same_url_chunks:
            if doc_chunk["chunk_id"] == chunk_id:
                matching_doc_chunk = doc_chunk
                break
        
        if not matching_doc_chunk:
            # Fallback: try text overlap
            entailment_text = entailment_chunk.get("chunk_text", "").lower()
            for doc_chunk in same_url_chunks:
                doc_chunk_text = doc_chunk.get("chunk_text", "").lower()
                if entailment_text in doc_chunk_text or doc_chunk_text in entailment_text:
                    matching_doc_chunk = doc_chunk
                    break
        
        if matching_doc_chunk:
            # Check if this chunk is in top 10% of its document
            is_in_doc_top_10 = matching_doc_chunk["final_score"] >= doc_top_10_threshold
            rank_in_document = same_url_chunks.index(matching_doc_chunk) + 1
            
            # print(f"  Chunk score: {matching_doc_chunk['final_score']}")
            print(f"  Document top 10% threshold: {doc_top_10_threshold}")
            print(f"  Rank in document: {rank_in_document}/{doc_chunk_count}")
            print(f"  Is in document top 10%: {is_in_doc_top_10}")
            
            if is_in_doc_top_10:
                chunk_level_noise = False
            
            chunk_level_results.append({
                "chunk_id": chunk_id,
                "source_url": chunk_url,
                "is_chunk_level_noise": not is_in_doc_top_10,
                "reason": "Not in top 10% of document" if not is_in_doc_top_10 else "In top 10% of document",
                "document_chunk_count": doc_chunk_count,
                "rank_in_document": rank_in_document,
                "document_top_10_threshold": doc_top_10_threshold,
                "chunk_score": matching_doc_chunk["final_score"]
            })
        else:
            print("  Could not find matching chunk in document")
            chunk_level_results.append({
                "chunk_id": chunk_id,
                "source_url": chunk_url,
                "is_chunk_level_noise": True,
                "reason": "Could not find matching chunk in document",
                "document_chunk_count": doc_chunk_count,
                "rank_in_document": None,
                "document_top_10_threshold": doc_top_10_threshold
            })
    
    # Overall noise domination (either document-level or chunk-level)
    is_noise_dominated = document_level_noise or chunk_level_noise
    
    return {
        "claim": claim,
        "is_noise_dominated": is_noise_dominated,
        "document_level_noise": document_level_noise,
        "chunk_level_noise": chunk_level_noise,
        "entailment_chunks": entailment_chunks_in_top_10,
        "chunk_level_results": chunk_level_results,
        "top_10_percent_threshold": top_10_percent_threshold,
        "total_chunks": total_chunks,
        "related_query_indices": related_query_indices,
        "ranked_chunks": chunk_final_scores[:top_10_percent_count],
        "all_chunk_scores": chunk_final_scores,
    }


