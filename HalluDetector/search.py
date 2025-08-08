import json
import logging
import numpy as np
from typing import List, Dict, Any
import re
import asyncio


from action_checking import check_actions_against_observations
from chunk_scoring import IntegratedChunkScorer
from claim_checking import process_claims_and_urls
from noise_domination_detector import detect_noise_domination
from claim_link_to_query import find_relevant_queries_for_claims

# 全局日志配置
logging.basicConfig(level=logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def single_iteration(query_list: List[str], observation_memory: str, action_list: List[str], search_list: List[str], claim_list: List[str], web_content_cache: Dict[str, str]) -> Dict[str, Any]:
    """
    Process a single iteration of the evaluation pipeline.
    
    Args:
        query_list: List of queries
        observation_memory: Current observation memory
        action_list: List of actions to check
        search_list: List of search URLs
        claim_list: List of claims to verify
        
    Returns:
        Dictionary containing all results from this iteration
    """
    result = {
        'action_judgments': [],
        'claim_results': [],
        'noise_results': [],
        'observation_memory': observation_memory
    }

    # Step 1: Check whether action is consistent with observation_memory
    print("Step 1: Checking action consistency...")
    judgments = check_actions_against_observations(observation_memory, action_list)
    result['action_judgments'] = judgments
    
    for judgment in judgments:
        if judgment['judgment'] == 'contradiction':
            print(f"❌  [H1]: Action Contradiction: {judgment['action']}")

    # Step 2: Scoring each chunks against each sub-query
    print("Step 2: Running integrated chunk scoring...")
    scorer = IntegratedChunkScorer(
        sbert_model='all-MiniLM-L6-v2',
        ner_threshold=0.5,
        c=6.0,
        num_gpus=4
    )
    
    # Run scoring
    scoring_results = await scorer.score_chunks(query_list, search_list, web_content_cache)
    result['scoring_results'] = scoring_results

    # Step 3: Claim checking
    print("Step 3: Running claim checking...")
    claim_results = await process_claims_and_urls(claim_list, search_list, web_content_cache)
    result['claim_results'] = claim_results
    
    # Process each claim result
    for claim_result in claim_results:
        if isinstance(claim_result, dict):
            final_judgment = claim_result.get('final_judgment', 'unknown')
            claim_text = claim_result.get('claim', 'Unknown claim')
            
            if final_judgment == 'neutral':
                print(f"❌ [H3] Fact Fabrication: {claim_text}")
            elif final_judgment == 'contradiction':
                print(f"❌ [H4] Fact Contradiction: {claim_text}")
            elif final_judgment == 'entailment':
                # Add the entailment atomic claim to the observation memory
                result['observation_memory'] += claim_text + '\n'
                
                # Check whether the claim is supported by noise information
                # Link claim to related sub-queries
                selected_queries = find_relevant_queries_for_claims(claim_text, query_list)
                
                # Get relevant chunks from claim results
                relevant_chunks = claim_result.get('relevant_chunks', [])
                
                # Get query weights from scoring results
                query_weights = scoring_results.get('query_weights', {})
                
                noise_result = detect_noise_domination(
                    claim_text, 
                    relevant_chunks, 
                    selected_queries.get('relevant_queries', []), 
                    query_weights
                )
                
                if noise_result['is_noise_dominated']:
                    print(f"❌ [H5] Noise Domination: {claim_text}")
                    result['noise_results'].append({
                    'claim': claim_text,
                    'noise_result': noise_result
                })
                else:
                    print(f"✅  Fact Entailment: {claim_text}")
    
    return result