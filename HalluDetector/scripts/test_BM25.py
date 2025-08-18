import json
import sys
import os
import numpy as np
from typing import List, Dict, Tuple

# Add the parent directory to path to import BM25
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BM25 import BM25Retriever, tokenize

def load_chunks_from_cache(cache_file_path: str) -> List[Tuple[str, str, str]]:
    """Load chunks from the JSON cache file."""
    chunks = []
    
    try:
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract chunks from the chunk_score structure
        if 'chunk_score' in data:
            for chunk_key, chunk_data in data['chunk_score'].items():
                chunk_text = chunk_data.get('chunk_text', '')
                url = chunk_data.get('url', '')
                chunk_id = chunk_data.get('chunk_id_original', chunk_key)
                
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append((chunk_id, chunk_text, url))
        
        print(f"Loaded {len(chunks)} chunks from cache")
        
    except FileNotFoundError:
        print(f"Error: Cache file not found at {cache_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in cache file: {e}")
        return []
    except Exception as e:
        print(f"Error loading cache file: {e}")
        return []
    
    return chunks


def test_bm25_retrieval():
    """Test the improved BM25 retrieval system."""
    
    # Test query
    query = "OpenAI's careers page lists the role 'ML Infrastructure Engineer' with a location in San Francisco."
    
    print("=" * 80)
    print("TESTING IMPROVED BM25 RETRIEVAL SYSTEM")
    print("=" * 80)
    print(f"Query: {query}")
    print()
    
    # Load chunks from cache
    cache_file_path = "/data2/yuhaoz/DeepResearch/HalluBench/HalluDetector/json_cache/cache_gemini_PhD_jobs.json"
    chunks = load_chunks_from_cache(cache_file_path)
    
    if not chunks:
        print("No chunks loaded. Exiting.")
        return

    
    # Extract chunk texts for BM25
    chunk_texts = [chunk[1] for chunk in chunks]
    
    # Test tokenization on query
    print("\nQUERY TOKENIZATION:")
    print("-" * 40)
    query_tokens = tokenize(query)
    print(f"Original query: {query}")
    print(f"Tokenized query: {query_tokens}")
    print()
   
    # Initialize BM25 retriever
    print("INITIALIZING BM25 RETRIEVER:")
    print("-" * 40)
    bm25 = BM25Retriever(chunk_texts)
    print(f"Total documents: {bm25.N}")
    print(f"Average document length: {bm25.avgdl:.2f}")
    print()
    
    # Get BM25 scores
    print("BM25 SCORING AND RETRIEVAL:")
    print("-" * 40)
    scored_chunks, scores = bm25.score(query)

    # Sort scored_chunks and scores by scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    scored_chunks = [scored_chunks[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    print(f"length of scored_chunks: {len(scored_chunks)}")
    for i in range(min(10, len(scored_chunks))):
        print(f"Rank {i+1}: Score = {scores[i]:.6f}")
        print(f"Chunk ID: {scored_chunks[i]}")
        print("-" * 60)


if __name__ == "__main__":
    test_bm25_retrieval()
