# input: claim, relevant_chunks, related sub-queries, each chunk's scores for relevance and quality against the every sub-query
# Processing: 
# 1. Compute the final score for each chunk aggregated across ONLY the related sub-queries via weight averaging
# 2. Rank the chunks based on the final score
# 3. See whether the entail_chunk is in the top 10% of the ranked chunks
# Output: check whether the claim is dominated by noise. If NOT in the top 10%, then the claim is dominated by noise.

import numpy as np
from typing import List, Dict, Any, Tuple
from utils import WeightComputer


def detect_noise_domination(claim: str, 
                          relevant_chunks: List[Dict], 
                          related_queries: List[Dict], 
                          query_scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect whether a claim is dominated by noise based on chunk ranking.
    
    Args:
        claim: The claim to evaluate
        relevant_chunks: List of relevant chunks from claim_checking.py
        related_queries: List of related queries from claim_link_to_query.py
        query_scores: Output from chunk_scoring.py containing detailed chunk scores
        
    Returns:
        Dictionary containing noise domination analysis results
    """
    
    # Extract detailed chunk scores from query_scores
    detailed_chunk_scores = query_scores.get('detailed_chunk_scores', [])
    
    if not detailed_chunk_scores:
        return {
            'claim': claim,
            'is_noise_dominated': True,
            'reason': 'No chunk scores available',
            'entailment_chunks': [],
            'top_10_percent_threshold': 0,
            'total_chunks': 0
        }
    
    # Get metadata from query_scores
    metadata = query_scores.get('metadata', {})
    expanded_queries = metadata.get('expanded_queries', [])
    query_mapping = metadata.get('query_mapping', {})
    query_clusters = metadata.get('query_clusters', {})
    entity_weights = metadata.get('entity_weights', {})
    documents = metadata.get('documents', [])
    
    # Get the query indices for related queries
    related_query_indices = []
    for query_info in related_queries:
        if 'query_idx' in query_info:
            related_query_indices.append(query_info['query_idx'])
    
    # If no related queries, use all queries
    if not related_query_indices:
        # Extract all available query indices from the first chunk
        if detailed_chunk_scores:
            first_chunk = detailed_chunk_scores[0]
            related_query_indices = list(first_chunk.get('expanded_query_scores', {}).keys())
    
    # Compute query weights for expanded queries
    weight_computer = WeightComputer(c=6.0)  # Use default c parameter
    expanded_query_weights = {}
    
    for query_idx, expanded_query in enumerate(expanded_queries):
        # Find the original query index for this expanded query
        original_query_idx = query_mapping.get(query_idx, query_idx)
        
        # Get the query clusters for the original query
        original_query_clusters = query_clusters.get(original_query_idx, {})
        
        # Compute weight for this expanded query
        query_weight = weight_computer.compute_query_weight(expanded_query, original_query_clusters, entity_weights)
        expanded_query_weights[query_idx] = query_weight
    
    # Compute original query weights by taking maximum weight among expanded variants
    original_query_weights = {}
    for original_query_idx in set(query_mapping.values()):
        # Find all expanded queries that map to this original query
        expanded_indices_for_original = [exp_idx for exp_idx, orig_idx in query_mapping.items() if orig_idx == original_query_idx]
        
        # Take the maximum weight among all expanded variants
        weights_for_original = [expanded_query_weights.get(exp_idx, 1.0) for exp_idx in expanded_indices_for_original]
        original_query_weights[original_query_idx] = max(weights_for_original) if weights_for_original else 1.0
    
    # Compute final scores for each chunk using only related queries with weights
    chunk_final_scores = []
    
    for chunk_data in detailed_chunk_scores:
        chunk_id = chunk_data['chunk_id']
        chunk_text = chunk_data['chunk_text']
        url = chunk_data['url']
        
        # Get scores and weights for related queries only
        related_scores = []
        related_weights = []
        
        for query_idx in related_query_indices:
            if query_idx in chunk_data.get('expanded_query_scores', {}):
                score = chunk_data['expanded_query_scores'][query_idx]
                # Get the original query index for this expanded query
                original_query_idx = query_mapping.get(query_idx, query_idx)
                # Get the weight for the original query
                weight = original_query_weights.get(original_query_idx, 1.0)
                
                related_scores.append(score)
                related_weights.append(weight)
        
        # Compute weighted average across related queries
        if related_scores and related_weights:
            # Normalize weights
            total_weight = sum(related_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in related_weights]
                final_score = sum(score * weight for score, weight in zip(related_scores, normalized_weights))
            else:
                final_score = np.mean(related_scores)
        else:
            final_score = 0.0
        
        chunk_final_scores.append({
            'chunk_id': chunk_id,
            'chunk_text': chunk_text,
            'url': url,
            'final_score': final_score,
            'related_scores': related_scores,
            'related_weights': related_weights,
            'related_query_indices': related_query_indices
        })
    
    # Rank chunks by final score (descending)
    chunk_final_scores.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Identify entailment chunks from relevant_chunks
    entailment_chunks = []
    for chunk in relevant_chunks:
        if chunk.get('score', 0) > 0:  # Assuming positive scores indicate entailment
            entailment_chunks.append(chunk)
    
    # Calculate top 10% threshold
    total_chunks = len(chunk_final_scores)
    top_10_percent_count = max(1, int(np.ceil(total_chunks * 0.1)))
    
    if total_chunks == 0:
        return {
            'claim': claim,
            'is_noise_dominated': True,
            'reason': 'No chunks available for ranking',
            'entailment_chunks': [],
            'top_10_percent_threshold': 0,
            'total_chunks': 0
        }
    
    # Get the score threshold for top 10%
    if top_10_percent_count <= len(chunk_final_scores):
        top_10_percent_threshold = chunk_final_scores[top_10_percent_count - 1]['final_score']
    else:
        top_10_percent_threshold = chunk_final_scores[-1]['final_score'] if chunk_final_scores else 0
    
    # Check if entailment chunks are in top 10%
    entailment_chunks_in_top_10 = []
    is_noise_dominated = True
    
    for entailment_chunk in entailment_chunks:
        # Find the corresponding chunk in our ranked list
        chunk_id = entailment_chunk.get('chunk_id')
        chunk_url = entailment_chunk.get('source_url')
        
        # Find matching chunk by URL and chunk_id or by URL and text similarity
        matching_chunk = None
        for ranked_chunk in chunk_final_scores:
            if (ranked_chunk['url'] == chunk_url and 
                ranked_chunk['chunk_id'] == chunk_id):
                matching_chunk = ranked_chunk
                break
        
        # If no exact match, try to find by URL and text similarity
        if not matching_chunk:
            for ranked_chunk in chunk_final_scores:
                if ranked_chunk['url'] == chunk_url:
                    # Simple text similarity check
                    ranked_text = ranked_chunk['chunk_text'].lower()
                    entailment_text = entailment_chunk.get('chunk_text', '').lower()
                    if entailment_text in ranked_text or ranked_text in entailment_text:
                        matching_chunk = ranked_chunk
                        break
        
        if matching_chunk:
            is_in_top_10 = matching_chunk['final_score'] >= top_10_percent_threshold
            entailment_chunks_in_top_10.append({
                'chunk_id': chunk_id,
                'source_url': chunk_url,
                'score': entailment_chunk.get('score', 0),
                'final_score': matching_chunk['final_score'],
                'is_in_top_10_percent': is_in_top_10,
                'rank': chunk_final_scores.index(matching_chunk) + 1
            })
            
            # If any entailment chunk is in top 10%, the claim is not noise dominated
            if is_in_top_10:
                is_noise_dominated = False
    
    # If no entailment chunks found, the claim is noise dominated
    if not entailment_chunks_in_top_10:
        is_noise_dominated = True
    
    return {
        'claim': claim,
        'is_noise_dominated': is_noise_dominated,
        'reason': 'Entailment chunks not in top 10%' if is_noise_dominated else 'Entailment chunks found in top 10%',
        'entailment_chunks': entailment_chunks_in_top_10,
        'top_10_percent_threshold': top_10_percent_threshold,
        'total_chunks': total_chunks,
        'related_query_indices': related_query_indices,
        'original_query_weights': original_query_weights,
        'expanded_query_weights': expanded_query_weights,
        'ranked_chunks': chunk_final_scores[:top_10_percent_count],  # Top 10% chunks for reference
        'all_chunk_scores': chunk_final_scores  # All chunks with their final scores
    }









