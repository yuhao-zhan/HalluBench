import argparse
from fileinput import filename
import json
import os
from typing import List, Dict, Any
import asyncio
import aiohttp
import logging
import time

from decomposition import decompose_workflow_to_cache, decompose_report_to_cache
from search import single_iteration
from claim_checking_url_and_titles import process_claims_and_urls
from utils import fetch_all_urls_and_cache, is_url
from fixed_thre_claim_link_to_query import find_relevant_queries_for_claims
from no_weight_noise_domination_detector import detect_noise_domination
from url_and_title_NLI_plus_BM25_chunk_score import IntegratedChunkScorer
from title_url_scoring import TitleURLScorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sepcify process name
import setproctitle
setproctitle.setproctitle('Yuhao_evaluate')

# Global BGEScorer instance to avoid loading reranker models multiple times
# This ensures that the expensive reranker model loading happens only once,
# and the same instance is reused across all scoring operations
_global_bge_scorer = None

def get_global_bge_scorer():
    """Get or create global BGEScorer instance to prevent multiple model loading."""
    global _global_bge_scorer
    if _global_bge_scorer is None:
        print("🔧 Initializing global BGEScorer (this will load reranker models once)...")
        print("🎯 This is the ONLY time you'll see 'initial target device' messages!")
        from reranker_scoring import BGEScorer
        _global_bge_scorer = BGEScorer(num_gpus=4)
        print("✅ Global BGEScorer initialized successfully")
    return _global_bge_scorer


def monitor_memory_usage():
    """Monitor and log current memory usage to help prevent OOM."""
    try:
        import psutil
        import torch
        
        # System memory
        memory = psutil.virtual_memory()
        print(f"💾 System Memory: {memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB ({memory.percent:.1f}%)")
        
        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"🎮 GPU {i}: {allocated:.2f}GB / {total:.2f}GB used")
                
                # Warning if GPU memory is high
                if allocated / total > 0.8:
                    print(f"⚠️ GPU {i} memory usage is high ({allocated/total*100:.1f}%)")
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024**3
        print(f"🔄 Process Memory: {process_memory:.2f}GB")
        
    except ImportError:
        print("📊 Memory monitoring not available (psutil not installed)")


_global_models = {}

def initialize_global_models():
    """Initialize models once and reuse across iterations."""
    global _global_models
    if not _global_models:
        print("🔧 Initializing global models for memory efficiency…")
    # Models will be initialized lazily when first needed
    _global_models['initialized'] = True


def create_global_url_mapping(all_urls: List[str]) -> Dict[str, int]:
    """Create a global URL-to-index mapping based on all_urls for consistent chunk IDs."""
    url_mapping = {}
    for idx, url in enumerate(all_urls):
        url_mapping[url] = idx
    print(f"🌐 Created global URL mapping for {len(all_urls)} URLs")
    return url_mapping





def initialize_results_file(output_file: str, query: str, report: str, all_urls: List[str], urls: List[str]):
    """Initialize the results JSON file with basic structure."""
    initial_structure = {
        'query': query,
        'report': report,
        'all_source_links': all_urls,
        'summary_citations': urls,
        'chain_of_research_results': [],
        'report_results': [],
        'summary': {
            'total_iterations': 0,
            'total_paragraphs': 0,
            'processed_iterations': 0,
            'processed_paragraphs': 0
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_structure, f, ensure_ascii=False, indent=2)
    
    print(f"📝 Initialized results file: {output_file}")


def append_iteration_result(output_file: str, iteration_result: Dict[str, Any], iteration_index: int):
    """Append a single iteration result to the JSON file."""
    try:
        # Read current file
        with open(output_file, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        
        # Append the iteration result
        current_data['chain_of_research_results'].append(iteration_result)
        current_data['summary']['total_iterations'] = len(current_data['chain_of_research_results'])
        current_data['summary']['processed_iterations'] = len([r for r in current_data['chain_of_research_results'] if 'error' not in r])
        
        # Write back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved iteration {iteration_index + 1} result to file")
        
    except Exception as e:
        print(f"❌ Error saving iteration {iteration_index + 1} result: {str(e)}")


def append_report_paragraph_result(output_file: str, paragraph_result: Dict[str, Any], paragraph_index: int):
    """Append a single report paragraph result to the JSON file."""
    try:
        # Read current file
        with open(output_file, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
        
        # Append the paragraph result
        current_data['report_results'].append(paragraph_result)
        current_data['summary']['total_paragraphs'] = len(current_data['report_results'])
        current_data['summary']['processed_paragraphs'] = len([r for r in current_data['report_results'] if 'error' not in r])
        
        # Write back to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved paragraph {paragraph_index} result to file")
        
    except Exception as e:
        print(f"❌ Error saving paragraph {paragraph_index} result: {str(e)}")


async def process_cache_data(cache_data: dict, web_content_cache: Dict[str, str], query: str, cache_file: str, output_file: str, url_mapping: Dict[str, int]):
    """
    Process cache data through single_iteration for each iteration.
    Note: Chunk scoring is now done upfront, so this function only processes iterations.

    Args:
        cache_data: Dictionary containing iterations data
        sub_queries: List of sub-queries
        web_content_cache: Cached web content to use instead of fetching URLs
        query: The original query
        cache_file: Path to the cache file containing pre-computed chunk scores
        output_file: Path to the output JSON file for saving results incrementally
        url_mapping: Global URL-to-index mapping for consistent chunk IDs
        # Note: claim-query mappings are loaded from cache_file when needed
        
    Returns:
        List of results from each iteration
    """
    # Initialize global models once
    initialize_global_models()

    results = []
    observation_memory = ""

    for i, iteration in enumerate(cache_data['iterations']):
        # Memory check before each iteration
        print(f"\n{'='*50}")
        print(f"ITERATION {i+1} MEMORY CHECK")
        print(f"{'='*50}")
        monitor_memory_usage()
        
        # Extract data for this iteration using numbered keys
        action_list = iteration.get(f'action_list_{i+1}', [])
        search_list = iteration.get(f'search_list_{i+1}', [])
        claim_list = iteration.get(f'claim_list_{i+1}', [])

        # If the relevant queries is empty and not a url, remove the claim from the claim list
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            claim_query_mappings = cache_data.get('related_query', {})
            for claim in claim_list:
                if claim_query_mappings.get(claim, {}).get('relevant_queries', []) == [] and not is_url(claim):
                    print(f"‼️ No relevant queries found for claim: {claim}")
                    claim_list.remove(claim)
        
        # If not the firsr search claim, the search list should add all previous search list items
        if i > 0:
            for j in range(i):
                search_list = search_list + [item for item in cache_data['iterations'][j].get(f'search_list_{j+1}', []) if item not in search_list]

        # Process this iteration with memory management
        # Note: scoring_results will be loaded from cache file since scoring was done upfront
        # No reranker models are loaded here - they're already loaded globally
        result = await single_iteration(
            query=query,
            observation_memory=observation_memory,
            action_list=action_list,
            search_list=search_list,
            claim_list=claim_list,
            web_content_cache=web_content_cache,
            cache_file=cache_file,
            url_mapping=url_mapping
        )
        
        # Update observation memory for next iteration
        observation_memory = result['observation_memory']
        
        results.append(result)
        
        # IMMEDIATELY save this iteration result to file
        append_iteration_result(output_file, result, i)
        
        # MEMORY CLEANUP after each iteration
        print(f"\n🧹 MEMORY CLEANUP after iteration {i+1}")
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Memory check after cleanup
        print(f"Memory status after cleanup:")
        monitor_memory_usage()

    return results


async def process_report_paragraphs(urls: List[str], web_content_cache: Dict[str, str], cache_data: dict = None, observation_memory: str = "", output_file: str = "", cache_file: str = "") -> List[Dict[str, Any]]:
    """
    Process each paragraph of the report using cached atomic claims or decompose_observation.
    
    Args:
        urls: List of URLs to use for claim checking
        web_content_cache: Cached web content
        cache_data: Optional cache data containing pre-decomposed atomic claims
        query: The original query
        output_file: Path to the output JSON file for saving results incrementally
        # Note: claim-query mappings are loaded from cache_data when needed
        
    Returns:
        List of results for each paragraph
    """
    print(f"\n{'='*50}")
    print("PROCESSING REPORT PARAGRAPHS")
    print(f"{'='*50}")
    
    # Check if we have cached atomic claims
    if cache_data and 'report' in cache_data:
        print(f"✅ Using cached atomic claims from cache file")
        print(f"💾 No reranker models loaded here - using pre-computed scores from cache")
        cached_paragraphs = cache_data['report']
        print(f"Found {len(cached_paragraphs)} cached paragraphs with atomic claims")
        
        # Memory check before processing
        print(f"\n💾 Memory status before report processing:")
        monitor_memory_usage()
        
        report_results = []
        
        for i, cached_para in enumerate(cached_paragraphs, 1):
            print(f"\nProcessing cached paragraph {i}/{len(cached_paragraphs)}...")
            
            # Memory check every 10 paragraphs
            if i % 10 == 0:
                print(f"\n💾 Memory check at paragraph {i}:")
                monitor_memory_usage()
            
            try:
                # Use cached atomic claims
                atomic_claims = cached_para.get('atomic_claims', [])
                paragraph_text = cached_para.get('paragraph_text', '')
                
                if not atomic_claims:
                    print(f"  No atomic claims found in cached paragraph {i}")
                    paragraph_result = {
                        'paragraph_index': i,
                        'paragraph_text': paragraph_text,
                        'claims': [],
                        'claim_results': [],
                        'error': 'No atomic claims found in cache'
                    }
                    report_results.append(paragraph_result)
                    
                    # IMMEDIATELY save this paragraph result to file
                    if output_file:
                        append_report_paragraph_result(output_file, paragraph_result, i)
                    
                    continue

                # If the relevant queries is empty, remove the claim from the claim list
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    claim_query_mappings = cache_data.get('related_query', {})
                    for claim in atomic_claims:
                        if claim_query_mappings.get(claim, {}).get('relevant_queries', []) == [] and not is_url(claim):
                            print(f"‼️ No relevant queries found for claim: {claim}")
                            atomic_claims.remove(claim)
                
                print(f"  Using {len(atomic_claims)} cached claims from paragraph {i}")
                
                # Process claims and URLs using process_claims_and_urls with cached content
                claim_results, updated_observation_memory = await process_claims_and_urls(atomic_claims, urls, web_content_cache, observation_memory, cache_file)

                # Update observation memory with the updated version from claim checking
                observation_memory = updated_observation_memory
                print(f"  📝 Observation memory updated from claim checking: {len(observation_memory)} characters")

                # Process each claim result
                result = {
                    'claim_results': claim_results,  # Store the original claim results
                    'noise_results': [],
                    'observation_memory': observation_memory
                }
                
                # Load chunk scores from cache file since they were already computed upfront
                chunk_scores = cache_data.get('chunk_score', {}) if cache_data else {}
                
                print(f"  📊 Loaded {len(chunk_scores)} chunk scores from cache for noise detection")
                
                                        # Reconstruct detailed_chunk_scores structure that noise domination detector expects
                detailed_chunk_scores = []
                for chunk_id, chunk_data in chunk_scores.items():
                    if isinstance(chunk_data, dict) and 'scores' in chunk_data:
                        # Convert the raw chunk_score format to detailed_chunk_scores format
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
                                chunk_summary['original_query_scores'][q_idx_str] = q_scores.get('combined', 0.0)
                                chunk_summary['bm25_norm_scores'][q_idx_str] = q_scores.get('bm25_norm', 0.0)
                                chunk_summary['reranker_norm_scores'][q_idx_str] = q_scores.get('reranker_norm', 0.0)
                        
                        detailed_chunk_scores.append(chunk_summary)
                
                print(f"  📊 Reconstructed {len(detailed_chunk_scores)} detailed chunk scores for noise detection")
                
                # Create scoring_results structure that noise domination detector expects
                scoring_results = {
                    'detailed_chunk_scores': detailed_chunk_scores,
                    'query_weights': {}  # Empty for now, can be populated if needed
                }
                
                for claim_result in claim_results:
                    if isinstance(claim_result, dict):
                        final_judgment = claim_result.get('final_judgment', 'unknown')
                        claim_text = claim_result.get('claim', 'Unknown claim')

                        print('-'*50)
                        print(f"Claim: {claim_text}")
                        print(f"Final Judgment: {final_judgment}")
                        print("Relevant Chunks:")
                        for chunk in claim_result.get('relevant_chunks', []):
                            print(f"  - {chunk['chunk_text']}")
                        
                        if final_judgment == 'neutral':
                            print(f"❌ [H3] Fact Fabrication: {claim_text}")
                        elif final_judgment == 'contradiction':
                            print(f"❌ [H4] Fact Contradiction: {claim_text}")
                        elif final_judgment == 'entailment':
                            # Get relevant chunks from claim results
                            relevant_chunks = claim_result.get('relevant_chunks', [])
                            if relevant_chunks and relevant_chunks[0]['score'] == -1.0:
                                print(f"Entailed by observation memory, skip noise detection!")
                                continue
                                
                            # Check whether the claim is supported by noise information
                            # Use pre-computed claim-to-query mappings from cache
                            claim_query_mappings = cache_data.get('related_query', {}) if cache_data else {}
                            selected_queries = claim_query_mappings.get(claim_text, {})
                            
                            # Get query weights from scoring results
                            query_weights = scoring_results.get('query_weights', {})
                            
                            # Handle case where selected_queries is empty
                            relevant_queries = []
                            if selected_queries and isinstance(selected_queries, dict):
                                relevant_queries = selected_queries.get('relevant_queries', [])
                            
                            noise_result = detect_noise_domination(
                                claim_text, 
                                relevant_chunks, 
                                relevant_queries, 
                                scoring_results
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
                                    print(f"❌ [H5] Document-Level Noise Domination")
                                elif chunk_noise:
                                    print(f"Claim: {claim_text}")
                                    print(f"✅ Chunk-level Top 10% BUT")
                                    print(f"❌ [H5] Chunk-Level Noise Domination")
                                else:
                                    print(f"❌ [H5] Noise Domination: {claim_text}")
                                
                                result['noise_results'].append({
                                    'claim': claim_text,
                                    'noise_result': noise_result
                                })
                            else:
                                print(f"✅  Fact Entailment: {claim_text}")
                
                # Store results for this paragraph
                paragraph_result = {
                    'paragraph_index': i,
                    # 'paragraph_text': paragraph_text,
                    'claims': atomic_claims,
                    'urls': urls,
                    'claim_results': claim_results,
                    'processed_claim_results': result['claim_results'],
                    'noise_results': result['noise_results'],
                    # 'observation_memory': result['observation_memory']
                }
                
                report_results.append(paragraph_result)
                
                # IMMEDIATELY save this paragraph result to file
                if output_file:
                    append_report_paragraph_result(output_file, paragraph_result, i)
                
                print(f"  ✅ Cached paragraph {i} processed successfully")
                
                # Memory cleanup every 20 paragraphs to prevent accumulation
                if i % 20 == 0:
                    print(f"  🧹 Memory cleanup at paragraph {i}")
                    import gc
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"  ❌ Error processing cached paragraph {i}: {str(e)}")
                paragraph_result = {
                    'paragraph_index': i,
                    'paragraph_text': cached_para.get('paragraph_text', ''),
                    'claims': [],
                    'claim_results': [],
                    'error': str(e)
                }
                report_results.append(paragraph_result)
                
                # IMMEDIATELY save this error result to file
                if output_file:
                    append_report_paragraph_result(output_file, paragraph_result, i)
    else:
        # Fallback to original method if no cache
        print(f"⚠️ No cached atomic claims found!")
        
    
    # Final memory cleanup
    print(f"\n🧹 Final memory cleanup after report processing")
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n✅ Report processing completed. Processed {len(report_results)} paragraphs.")
    print(f"Final memory status:")
    monitor_memory_usage()
    
    return report_results


async def score_all_chunks_upfront(
    sub_queries: List[str], 
    all_urls: List[str], 
    web_content_cache: Dict[str, str], 
    cache_file: str, 
    url_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Score all chunks from all URLs against all sub-queries upfront.
    This eliminates the need for repeated scoring during iterations.
    
    Args:
        sub_queries: List of sub-queries to score against
        all_urls: List of all URLs to process
        web_content_cache: Cached web content
        cache_file: Path to the cache file for saving chunk scores
        url_mapping: Global URL-to-index mapping for consistent chunk IDs
        
    Returns:
        Dictionary containing scoring results for all chunks
    """

    # If the cache file exists, skip scoring
    if os.path.exists(cache_file):
        print(f"✅ Skipping chunk scoring because cache file already exists: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_content = json.load(f)
            if 'chunk_score' in all_content:
                return all_content['chunk_score']
            else:
                print(f"❌ No chunk scores found in cache file: {cache_file}")


    print(f"\n{'='*50}")
    print("SCORING ALL CHUNKS UPFRONT")
    print(f"{'='*50}")
    
    # Memory check before scoring
    print(f"💾 Memory status before chunk scoring:")
    monitor_memory_usage()
    
    # Initialize the global scorer with the pre-loaded BGEScorer instance
    print("🔧 Initializing IntegratedChunkScorer for upfront scoring...")
    print("📝 Note: Using global BGEScorer instance to avoid reloading reranker models")
    print("🚀 This ensures 'initial target device' appears only once!")
    
    # Get the global BGEScorer instance (loaded only once)
    global_bge_scorer = get_global_bge_scorer()
    
    scorer = IntegratedChunkScorer(
        sbert_model='all-MiniLM-L6-v2',
        ner_threshold=0.5,
        c=6.0,
        num_gpus=4,
        reranker_instance=global_bge_scorer  # Pass the global instance directly
    )
    
    print(f"📊 Scoring {len(sub_queries)} queries against chunks from {len(all_urls)} URLs...")
    print(f"💾 Web content cache contains {len(web_content_cache)} documents")
    
    # Score all chunks against all queries
    print(f"🚀 Starting upfront scoring of all chunks...")
    print(f"💾 Using pre-loaded BGEScorer instance (no model reloading)")
    scoring_results = await scorer.score_chunks(
        queries=sub_queries,
        urls=all_urls,  # Use all_urls to score everything upfront
        web_content_cache=web_content_cache,
        cache_file=cache_file,
        url_mapping=url_mapping
    )
    
    print(f"✅ Upfront scoring completed!")
    print(f"📊 Scored {len(scoring_results.get('detailed_chunk_scores', []))} chunks")
    print(f"📊 Results saved to cache file: {cache_file}")
    
    # Memory check after scoring
    print(f"💾 Memory status after chunk scoring:")
    monitor_memory_usage()
    
    # Memory cleanup after scoring
    print(f"🧹 Memory cleanup after upfront scoring")
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return scoring_results


async def map_all_claims_to_queries_upfront(
    cache_data: dict,
    cache_file: str
) -> Dict[str, Any]:
    """
    Map all extracted claims to their related queries upfront.
    This eliminates the need for repeated calls to find_relevant_queries_for_claims.
    
    Args:
        cache_data: Dictionary containing cached data with claims
        cache_file: Path to the cache file for saving/loading claim-query mappings
        
    Returns:
        Dictionary containing claim-to-query mappings
    """
    # Check if claim-query mappings already exist in cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                all_content = json.load(f)
            if 'related_query' in all_content:
                print(f"✅ Skipping claim-to-query mapping because 'related_query' already exists in cache")
                return all_content['related_query']
        except Exception as e:
            print(f"⚠️ Error checking cache for claim-query mappings: {e}")
    
    print(f"\n{'='*50}")
    print("MAPPING ALL CLAIMS TO QUERIES UPFRONT")
    print(f"{'='*50}")
    
    # Get all claims from cache data
    all_claims = []
    
    # Extract claims from iterations
    iterations = cache_data.get('iterations', [])
    for i, iteration in enumerate(iterations):
        claim_list = iteration.get(f'claim_list_{i+1}', [])
        all_claims.extend(claim_list)
    
    # Extract claims from report paragraphs
    report_paragraphs = cache_data.get('report', [])
    for para in report_paragraphs:
        atomic_claims = para.get('atomic_claims', [])
        all_claims.extend(atomic_claims)
    
    # Remove duplicates while preserving order
    unique_claims = []
    seen_claims = set()
    for claim in all_claims:
        if claim not in seen_claims:
            unique_claims.append(claim)
            seen_claims.add(claim)
    
    print(f"📊 Found {len(unique_claims)} unique claims to map")
    
    # Get query list from cache
    query_list = cache_data.get('query_list', [])
    print(f"📊 Mapping against {len(query_list)} queries")
    
    # Map all claims to their related queries in one call
    print(f"  📝 Mapping {len(unique_claims)} claims to queries...")
    
    try:
        # Find relevant queries for all claims at once
        selected_queries = find_relevant_queries_for_claims(unique_claims, query_list, fixed_threshold=0.01)
        
        # Convert the result format to use claim text as keys
        claim_query_mappings = {}
        for claim_idx, claim_info in selected_queries.items():
            claim_text = claim_info['claim']
            claim_query_mappings[claim_text] = claim_info
        
        print(f"✅ Completed mapping {len(claim_query_mappings)} claims to queries")
        
    except Exception as e:
        print(f"⚠️ Error mapping claims to queries: {e}")
        claim_query_mappings = {}
    
    # Save the mappings to cache file
    # The cache structure will now include:
    # - chunk_score: Pre-computed chunk scores
    # - related_query: Pre-computed claim-query mappings
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_content = json.load(f)
        
        # Add the new claim-query mappings
        all_content['related_query'] = claim_query_mappings
        
        # Write back to cache file
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_content, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Claim-query mappings saved to cache file: {cache_file}")
        print(f"📊 Cache now contains both chunk scores and claim-query mappings")
        
    except Exception as e:
        print(f"❌ Error saving claim-query mappings to cache: {e}")
    
    return claim_query_mappings


async def score_all_urls_and_titles_upfront(
    cache_data: dict,
    web_content_cache: Dict[str, str],
    cache_file: str
) -> Dict[str, Any]:
    """
    Score all URLs and titles against queries and claims upfront.
    This eliminates the need for repeated scoring during iterations.
    
    Args:
        cache_data: Dictionary containing cached data with queries and claims
        web_content_cache: Cached web content to extract titles from
        cache_file: Path to the cache file for saving/loading URL and title scores
        
    Returns:
        Dictionary containing URL and title scores
    """
    # Check if URL and title scores already exist in cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                all_content = json.load(f)
            
            # Check if all required attributes exist and are not empty
            required_attrs = ['url_against_query', 'url_against_claim', 'titles_against_query', 'titles_against_claim']
            all_exist = all(attr in all_content and all_content[attr] for attr in required_attrs)
            
            if all_exist:
                print(f"✅ Skipping URL and title scoring because all required attributes already exist in cache")
                print(f"📊 Found in cache:")
                for attr in required_attrs:
                    count = len(all_content[attr]) if all_content[attr] else 0
                    print(f"  - {attr}: {count} URLs")
                return {attr: all_content[attr] for attr in required_attrs}
        except Exception as e:
            print(f"⚠️ Error checking cache for URL and title scores: {e}")
    
    print(f"\n{'='*50}")
    print("SCORING ALL URLS AND TITLES UPFRONT")
    print(f"{'='*50}")
    
    # Get queries and claims from the cache data (following the pattern from map_all_claims_to_queries_upfront)
    all_queries = cache_data.get('query_list', [])
    
    # Get all claims from cache data
    all_claims = []
    
    # Extract claims from iterations
    iterations = cache_data.get('iterations', [])
    for i, iteration in enumerate(iterations):
        claim_list = iteration.get(f'claim_list_{i+1}', [])
        all_claims.extend(claim_list)
    
    # Extract claims from report paragraphs
    report_paragraphs = cache_data.get('report', [])
    for para in report_paragraphs:
        atomic_claims = para.get('atomic_claims', [])
        all_claims.extend(atomic_claims)
    
    # Remove duplicates while preserving order
    unique_claims = []
    seen_claims = set()
    for claim in all_claims:
        if claim not in seen_claims:
            unique_claims.append(claim)
            seen_claims.add(claim)
    
    print(f"📊 Found {len(all_queries)} queries and {len(unique_claims)} unique claims for scoring")
    
    # Initialize TitleURLScorer with GPU support
    title_url_scorer = TitleURLScorer(num_gpus=4)
    
    # Compute all scores upfront
    title_url_scores = title_url_scorer.compute_all_scores_upfront(
        web_content_cache=web_content_cache,
        queries=all_queries,
        claims=unique_claims
    )
    
    # Save the scores to the json cache file (following the same pattern as other functions)
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_content = json.load(f)
        
        # Add the new scoring results
        all_content.update(title_url_scores)
        
        # Write back to cache file
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_content, f, ensure_ascii=False, indent=2)
        
        print(f"✅ URL and Title scores saved to json cache file: {cache_file}")
        print(f"📊 Cache now contains:")
        print(f"  - url_against_query: {len(title_url_scores.get('url_against_query', {}))} URLs")
        print(f"  - url_against_claim: {len(title_url_scores.get('url_against_claim', {}))} URLs")
        print(f"  - titles_against_query: {len(title_url_scores.get('titles_against_query', {}))} URLs")
        print(f"  - titles_against_claim: {len(title_url_scores.get('titles_against_claim', {}))} URLs")
        
    except Exception as e:
        print(f"❌ Error saving URL and Title scores to cache: {e}")
    
    return title_url_scores


async def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description='Semantic Fact Check Evaluation Pipeline'
    )
    parser.add_argument('input_json', help='Path to input JSON file')
    args = parser.parse_args()

    # Initial memory check
    print(f"\n{'='*50}")
    print("INITIAL MEMORY STATUS")
    print(f"{'='*50}")
    monitor_memory_usage()
    
    # Initialize global BGEScorer early to avoid repeated model loading
    print(f"\n{'='*50}")
    print("INITIALIZING GLOBAL BGESCORER (ONCE ONLY)")
    print(f"{'='*50}")
    get_global_bge_scorer()  # This will load the reranker models once

    # read the query and report from the input json
    with open(args.input_json, 'r', encoding='utf-8') as f:
        text = json.load(f)
        query = text.get('query', '')
        report = text.get('final_report', '')
        all_urls = text.get('all_source_links', [])
        urls = text.get('summary_citations', [])
        
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Create cache file paths
    web_content_cache_file = f"../web_content_cache/cache_{os.path.basename(args.input_json)}"
    json_cache_file = f"../json_cache/new_cache_{os.path.basename(args.input_json)}"
    
    # Create output file path
    output_file = f"../results/new_results_{os.path.basename(args.input_json)}"
    
    # Create global URL mapping for consistent chunk IDs
    url_mapping = create_global_url_mapping(all_urls)
    
    # Initialize the results file with basic structure
    initialize_results_file(output_file, query, report, all_urls, urls)
    
    # Fetch all URLs and cache them
    print(f"\n📥 Fetching and caching web content...")
    web_content_cache = await fetch_all_urls_and_cache(all_urls, web_content_cache_file)
    
    # Memory check after web content fetching
    print(f"\n💾 Memory status after web content fetching:")
    monitor_memory_usage()

    # Decompose the query to get the atomic actions
    print(f"\n🔍 Decomposing workflow...")
    decompose_workflow_to_cache(args.input_json)

    # Read the cache file
    cache_file = json_cache_file
    
    # Decompose report paragraphs to cache if not already done
    print(f"\n📝 Decomposing report paragraphs to cache...")
    decompose_report_to_cache(report, query, cache_file)
    
    # Read the updated cache file
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    # Get sub-queries from cache file
    sub_queries = cache_data.get('query_list', [])

    # NEW FLOW: Score all chunks upfront against all queries
    # This eliminates the need for repeated scoring during iterations
    print(f"\n🔄 Scoring all chunks upfront...")
    scoring_results = await score_all_chunks_upfront(sub_queries, all_urls, web_content_cache, cache_file, url_mapping)

    # NEW FLOW: Score all URLs and titles upfront against queries and claims
    # This eliminates the need for repeated scoring during iterations
    # Results are saved to the same json cache file as chunk scores and claim-query mappings
    print(f"\n🔄 Scoring URLs and Titles upfront against queries and claims...")
    title_url_scores = await score_all_urls_and_titles_upfront(cache_data, web_content_cache, cache_file)
    
    # Memory cleanup after scoring
    print(f"\n🧹 Memory cleanup after URL and Title scoring")
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Memory check after scoring
    print(f"💾 Memory status after URL and Title scoring:")
    monitor_memory_usage()

    # NEW FLOW: Map all claims to their related queries upfront
    # This eliminates the need for repeated calls to find_relevant_queries_for_claims
    # The cache file will now contain both chunk scores and claim-query mappings
    # Note: No need to pass claim_query_mappings as parameter - functions load directly from cache
    print(f"\n🔄 Mapping all claims to their related queries upfront...")
    claim_query_mappings = await map_all_claims_to_queries_upfront(cache_data, cache_file)

    # Process the cache data to get data for each iteration
    # Note: Both chunk scoring and claim-query mapping have been done upfront
    # Iterations now only need to load from cache, eliminating repeated computations
    # Functions load claim-query mappings directly from cache_file when needed
    print(f"\n🔄 Processing Chain of Research iterations...")
    results = await process_cache_data(cache_data, web_content_cache, query, cache_file, output_file, url_mapping)
   
    # After processing the Chain of Research, we need to process the report
    print(f"\n{'='*50}")
    print("PROCESSING REPORT")
    print(f"{'='*50}")
    
    # Process each paragraph of the report using cached web content and claim-query mappings
    # No need to call find_relevant_queries_for_claims here - using pre-computed mappings from cache
    # Functions load claim-query mappings directly from cache_data when needed

    # Load the observation memory from the results
    observation_memory = results[-1]['observation_memory']
    report_results = await process_report_paragraphs(urls, web_content_cache, cache_data, observation_memory, output_file, cache_file)

    # Final memory status
    print(f"\n💾 Final memory status:")
    monitor_memory_usage()
    
    print(f"\n✅ Complete results saved to: {output_file}")
    
    # Read final file to show summary
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        
        print(f"📊 Final Summary:")
        print(f"  - Chain of Research iterations: {final_data['summary']['total_iterations']}")
        print(f"  - Report paragraphs: {final_data['summary']['total_paragraphs']}")
        print(f"  - Successfully processed iterations: {final_data['summary']['processed_iterations']}")
        print(f"  - Successfully processed paragraphs: {final_data['summary']['processed_paragraphs']}")
        
    except Exception as e:
        print(f"❌ Error reading final results file: {str(e)}")

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time) / 60} minutes")


if __name__ == '__main__':
    asyncio.run(main())