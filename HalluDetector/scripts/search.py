import json
import logging
import numpy as np
from typing import List, Dict, Any
import re
import asyncio
import os


from action_checking import check_actions_against_observations
from claim_checking import process_claims_and_urls
from no_weight_noise_domination_detector import detect_noise_domination


# 全局日志配置
logging.basicConfig(level=logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Note: Global scorer is no longer needed since scoring is done upfront in evaluate.py

async def single_iteration(query: str, observation_memory: str, action_list: List[str], search_list: List[str], claim_list: List[str], web_content_cache: Dict[str, str], cache_file: str = None, url_mapping: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Process a single iteration of the evaluation pipeline with memory optimization.
    
    Args:
        observation_memory: Current observation memory
        action_list: List of actions to check
        search_list: List of search URLs
        claim_list: List of claims to verify
        web_content_cache: Cached web content
        cache_file: Optional cache file path
        url_mapping: Global URL-to-index mapping for consistent chunk IDs
        
    Returns:
        Dictionary containing all results from this iteration
    """

    result = {
        'action_judgments': {
            'observation_memory_judgments': [],
            'query_judgments': []
        },
        'claim_results': [],
        'noise_results': [],
        'observation_memory': observation_memory
    }

    # Step 1: Check whether action is consistent with observation_memory and query
    print("Step 1: Checking action consistency...")
    action_check_results = check_actions_against_observations(query, observation_memory, action_list)
    
    # Check for contradictions against observation memory
    for action_judgment in action_check_results:
        if action_judgment and action_judgment['judgment'] == 'contradiction' and action_judgment['judgment_score'] > 0.5:
            print(f"❌  [H1]: Action Contradiction with Observation Memory: {action_judgment['action']}")
            print(f"Observation Memory: {query}\n{observation_memory}")
            print()
    
    # Step 2: Load pre-computed chunk scores from cache file (scoring done upfront in evaluate.py)
    print("Step 2: Loading pre-computed chunk scores from cache...")
    
    # Load scoring results from cache file since scoring was done upfront
    scoring_results = {}
    # url_against_query = {}
    # titles_against_query = {}
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            scoring_results = cache_data.get('chunk_score', {})
            # Load url_against_query and titles_against_query from cache
            # url_against_query = cache_data.get('url_against_query', {})
            # titles_against_query = cache_data.get('titles_against_query', {})
            print(f"✅ Loaded {len(scoring_results)} pre-computed chunk scores from cache")
        except Exception as e:
            print(f"⚠️ Error loading cache file: {e}")
            scoring_results = {}
    else:
        print("⚠️ No cache file provided, scoring results will be empty")
    
    result['scoring_results'] = scoring_results

    # Step 3: Claim checking
    print("Step 3: Running claim checking...")
    claim_results, updated_observation_memory = await process_claims_and_urls(claim_list, search_list, web_content_cache, observation_memory, cache_file)
    result['claim_results'] = claim_results
    
    # Update the local observation memory with the updated version from claim checking
    observation_memory = updated_observation_memory
    # result['observation_memory'] = observation_memory
    print(f"📝 Observation memory updated from claim checking: {len(observation_memory)} characters")
    
    # Process each claim result
    for claim_result in claim_results:
        if isinstance(claim_result, dict):
            final_judgment = claim_result.get('final_judgment', 'unknown')
            claim_text = claim_result.get('claim', 'Unknown claim')

            # print('-'*50)
            # print(f"Claim: {claim_text}")
            # print(f"Final Judgment: {final_judgment}")
            # print("Relevant Chunks:")
            # for chunk in claim_result.get('relevant_chunks', []):
            #     print(f"  - {chunk['chunk_text'][:200]}...")
            
            # if final_judgment == 'neutral':
            #     print(f"❌ [H3] Fact Fabrication: {claim_text}")
            # elif final_judgment == 'contradiction':
            #     print(f"❌ [H4] Fact Contradiction: {claim_text}")
            if final_judgment == 'entailment':
                # Observation memory is already updated in process_claims_and_urls
                # Just log that this claim was already added to memory
                # print(f"✅ Claim already added to observation memory: {claim_text}")

                # Get relevant chunks from claim results
                relevant_chunks = claim_result.get('relevant_chunks', [])
                if relevant_chunks[0]['score'] == -1.0:
                    print(f"Entailed by observation memory, skip noise detection!")
                    continue
                    
                # Check whether the claim is supported by noise information
                # Use pre-computed claim-to-query mappings from cache (no need to call find_relevant_queries_for_claims)
                selected_queries = {}
                if cache_file and os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        claim_query_mappings = cache_data.get('related_query', {})
                        selected_queries = claim_query_mappings.get(claim_text, {})
                    except Exception as e:
                        print(f"⚠️ Error loading claim-query mappings from cache: {e}")
                        selected_queries = {}
                
                # Since scoring was done upfront, we need to reconstruct the scoring_results structure
                # that noise domination detector expects
                if cache_file and os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        
                        # Get chunk scores and query list from cache
                        chunk_scores = cache_data.get('chunk_score', {})
                        query_list_from_cache = cache_data.get('query_list', [])
                        
                        # Reconstruct detailed_chunk_scores structure
                        detailed_chunk_scores = []
                        for chunk_id, chunk_data in chunk_scores.items():
                            if isinstance(chunk_data, dict) and 'scores' in chunk_data and chunk_data.get('url', '') in search_list:
                                chunk_summary = {
                                    'chunk_id': chunk_id,
                                    'url': chunk_data.get('url', ''),
                                    'chunk_text': chunk_data.get('chunk_text', ''),
                                    'position': chunk_data.get('position', 0),
                                    'original_query_scores': {},
                                    'bm25_norm_scores': {},
                                    'reranker_norm_scores': {}
                                }
                                
                                # Extract scores for each query
                                for q_idx_str, q_scores in chunk_data['scores'].items():
                                    if isinstance(q_scores, dict):
                                        # chunk_score_on_current_query_url = url_against_query.get(chunk_data.get('url', ''), {}).get(q_idx_str, 0.0)
                                        # chunk_score_on_current_query_title = titles_against_query.get(chunk_data.get('url', ''), {}).get(q_idx_str, 0.0)
                                        # print('*'*50)
                                        # print(f"Chun score before url and title: {q_scores.get('combined', 0.0)}")
                                        # print(f"Chunk score on current query url: {chunk_score_on_current_query_url}")
                                        # print(f"Chunk score on current query title: {chunk_score_on_current_query_title}")
                                        # chunk_summary['original_query_scores'][q_idx_str] = (0.7 * q_scores.get('combined', 0.0) + 0.15 * chunk_score_on_current_query_url + 0.15 * chunk_score_on_current_query_title)
                                        # print(f"Chunk score after url and title: {chunk_summary['original_query_scores'][q_idx_str]}")
                                        # print('*'*50)
                                        chunk_summary['original_query_scores'][q_idx_str] = q_scores.get('combined', 0.0)
                                        chunk_summary['bm25_norm_scores'][q_idx_str] = q_scores.get('bm25_norm', 0.0)
                                        chunk_summary['reranker_norm_scores'][q_idx_str] = q_scores.get('reranker_norm', 0.0)
                                
                                detailed_chunk_scores.append(chunk_summary)
                        
                        # Create scoring_results structure for noise detection
                        scoring_results_for_noise = {
                            'detailed_chunk_scores': detailed_chunk_scores,
                            'query_weights': {}
                        }
                        
                        print(f"  📊 Reconstructed {len(detailed_chunk_scores)} chunk scores for noise detection in iteration")
                        
                    except Exception as e:
                        print(f"⚠️ Error loading cache for noise detection: {e}")
                        scoring_results_for_noise = {'detailed_chunk_scores': [], 'query_weights': {}}
                else:
                    scoring_results_for_noise = {'detailed_chunk_scores': [], 'query_weights': {}}
                
                # Get query weights from scoring results
                query_weights = scoring_results_for_noise.get('query_weights', {})
                
                # Handle case where selected_queries is empty
                relevant_queries = []
                if selected_queries and isinstance(selected_queries, dict):
                    relevant_queries = selected_queries.get('relevant_queries', [])
                
                noise_result = detect_noise_domination(
                    claim_text, 
                    relevant_chunks, 
                    relevant_queries, 
                    scoring_results_for_noise
                )
                
                if noise_result['is_noise_dominated']:
                    # Check both types of noise domination
                    doc_noise = noise_result.get('document_level_noise', False)
                    chunk_noise = noise_result.get('chunk_level_noise', False)
                    
                    if doc_noise and chunk_noise:
                        print(f"❌ [H5] Document & Chunk-Level Noise Domination: {claim_text}")
                    elif doc_noise:
                        print(f"Claim: {claim_text}")
                        print(f"✅ Chunk-level Top 10% BUT")
                        observation_memory += claim_text + "\n\n"
                        print(f"✅ Claim already added to observation memory: {claim_text}")
                        print(f"❌ [H5] Document-Level Noise Domination")
                    elif chunk_noise:
                        print(f"Claim: {claim_text}")
                        print(f"✅ Document-level Top 10% BUT")
                        observation_memory += claim_text + "\n\n"
                        print(f"✅ Claim already added to observation memory: {claim_text}")
                        print(f"❌ [H5] Chunk-Level Noise Domination: {claim_text}")
                    else:
                        print(f"❌ [H5] Noise Domination: {claim_text}")
                    
                    result['noise_results'].append({
                        'claim': claim_text,
                        'noise_result': noise_result
                    })
                else:
                    print(f"✅  Fact Entailment [After Noise Domination]: {claim_text}")
                    observation_memory += claim_text + "\n\n"
                    print(f"✅ Claim already added to observation memory: {claim_text}")

    result['observation_memory'] = observation_memory
    
    return result