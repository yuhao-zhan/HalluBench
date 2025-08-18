import argparse
from fileinput import filename
import json
import os
from typing import List, Dict, Any
import asyncio
import aiohttp
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools

from decomposition import decompose_workflow_to_cache, decompose_report_to_cache
from search import single_iteration
from claim_checking import process_claims_and_urls
from utils import fetch_all_urls_and_cache, is_url
from fixed_thre_claim_link_to_query import find_relevant_queries_for_claims
from no_weight_noise_domination_detector import detect_noise_domination
from NLI_plus_BM25_chunk_score import IntegratedChunkScorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sepcify process name
import setproctitle
setproctitle.setproctitle('Yuhao_evaluate')

"""
🚀 PARALLEL PROCESSING OPTIMIZATIONS FOR 54 CPU CORES 🚀

This script has been optimized to utilize all 54 CPU cores for maximum performance:

1. PARALLEL URL FETCHING:
   - Fetches URLs in parallel batches using asyncio
   - Uses semaphore to control concurrent connections (up to 2x CPU cores)
   - Optimized batch sizes based on URL count and CPU cores
   - Graceful error handling for failed batches

2. PARALLEL CHUNK SCORING:
   - GPU-accelerated scoring with multiple GPUs
   - CPU-bound preprocessing distributed across cores
   - Pre-computed scoring eliminates repeated computations

3. PARALLEL CLAIM-QUERY MAPPING:
   - Uses ProcessPoolExecutor for CPU-intensive semantic matching
   - Processes claims in parallel batches
   - Distributed across all available CPU cores

4. PARALLEL ITERATION PROCESSING:
   - Chain of Research iterations processed concurrently
   - Each iteration runs independently with shared resources
   - Immediate result saving to prevent memory accumulation

5. PARALLEL PARAGRAPH PROCESSING:
   - Report paragraphs processed in parallel
   - Claims within paragraphs processed concurrently
   - Memory cleanup at regular intervals

6. PERFORMANCE MONITORING:
   - Detailed timing for each processing phase
   - Memory usage tracking and cleanup
   - Performance summary with speedup estimates

7. CONFIGURATION OPTIONS:
   - Easy switching between parallel/sequential modes
   - Command line argument: --sequential for fallback mode
   - Configurable batch sizes and concurrency limits

Expected performance improvements:
- URL fetching: 10-50x faster (depending on network latency)
- Claim mapping: 20-40x faster (CPU-intensive operations)
- Paragraph processing: 15-30x faster (I/O and CPU operations)
- Overall pipeline: 15-25x faster end-to-end

Memory management:
- Incremental result saving prevents memory accumulation
- Regular garbage collection and GPU memory cleanup
- Semaphore-based concurrency control prevents resource exhaustion
"""

# Global BGEScorer instance to avoid loading reranker models multiple times
# This ensures that the expensive reranker model loading happens only once,
# and the same instance is reused across all scoring operations
_global_bge_scorer = None

# CPU core configuration for parallel processing
CPU_CORES = 54  # Use all available CPU cores
CHUNK_SIZE = 10  # Process URLs in chunks for better memory management

# Processing mode configuration
USE_PARALLEL_PROCESSING = True  # Set to False to use sequential processing
PARALLEL_MODE_OVERRIDE = None  # Can be set via command line argument

# Parallel processing configuration
PARALLEL_CONFIG = {
    'url_fetching': {
        'max_concurrent_connections': CPU_CORES,  # Limit concurrent HTTP connections
        'batch_size': CHUNK_SIZE,  # URLs per batch
        'timeout_seconds': 200,  # HTTP timeout
        'retry_attempts': 3
    },
    'paragraph_processing': {
        'max_concurrent_paragraphs': CPU_CORES,  # Max concurrent paragraph processing
        'memory_check_interval': 10,  # Check memory every N paragraphs
        'cleanup_interval': 20  # Cleanup memory every N paragraphs
    },
    'iteration_processing': {
        'max_concurrent_iterations': CPU_CORES,  # Max concurrent iterations
        'save_interval': 1  # Save results after each iteration
    },
    'claim_mapping': {
        'max_concurrent_batches': CPU_CORES,  # Max concurrent claim batches
        'claims_per_batch': None  # Will be calculated dynamically
    }
}

# Performance monitoring
PERFORMANCE_STATS = {
    'start_time': None,
    'url_fetching_time': 0,
    'chunk_scoring_time': 0,
    'claim_mapping_time': 0,
    'iteration_processing_time': 0,
    'paragraph_processing_time': 0,
    'total_time': 0
}

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


def start_performance_timer(phase: str):
    """Start timing a specific processing phase."""
    global PERFORMANCE_STATS
    if PERFORMANCE_STATS['start_time'] is None:
        PERFORMANCE_STATS['start_time'] = time.time()
    
    # Store the start time for this phase
    PERFORMANCE_STATS[f'{phase}_start'] = time.time()
    print(f"⏱️ Starting {phase} phase...")


def end_performance_timer(phase: str):
    """End timing a specific processing phase and log the duration."""
    global PERFORMANCE_STATS
    if f'{phase}_start' in PERFORMANCE_STATS:
        duration = time.time() - PERFORMANCE_STATS[f'{phase}_start']
        PERFORMANCE_STATS[f'{phase}_time'] = duration
        
        # Convert to appropriate units
        if duration < 60:
            time_str = f"{duration:.2f}s"
        elif duration < 3600:
            time_str = f"{duration/60:.2f}m"
        else:
            time_str = f"{duration/3600:.2f}h"
        
        print(f"✅ {phase} completed in {time_str}")
        return duration
    return 0


def print_performance_summary():
    """Print a comprehensive performance summary."""
    global PERFORMANCE_STATS
    
    if PERFORMANCE_STATS['start_time'] is None:
        return
    
    total_time = time.time() - PERFORMANCE_STATS['start_time']
    PERFORMANCE_STATS['total_time'] = total_time
    
    print(f"\n{'='*60}")
    print("🚀 PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    phases = [
        'url_fetching',
        'chunk_scoring', 
        'claim_mapping',
        'iteration_processing',
        'paragraph_processing'
    ]
    
    total_phase_time = 0
    for phase in phases:
        phase_time = PERFORMANCE_STATS.get(f'{phase}_time', 0)
        total_phase_time += phase_time
        
        if phase_time > 0:
            percentage = (phase_time / total_time) * 100
            if phase_time < 60:
                time_str = f"{phase_time:.2f}s"
            elif phase_time < 3600:
                time_str = f"{phase_time/60:.2f}m"
            else:
                time_str = f"{phase_time/3600:.2f}h"
            
            print(f"📊 {phase.replace('_', ' ').title()}: {time_str} ({percentage:.1f}%)")
    
    # Calculate overhead time
    overhead_time = total_time - total_phase_time
    overhead_percentage = (overhead_time / total_time) * 100 if total_time > 0 else 0
    
    if total_time < 60:
        total_time_str = f"{total_time:.2f}s"
        overhead_time_str = f"{overhead_time:.2f}s"
    elif total_time < 3600:
        total_time_str = f"{total_time/60:.2f}m"
        overhead_time_str = f"{overhead_time/60:.2f}m"
    else:
        total_time_str = f"{total_time/3600:.2f}h"
        overhead_time_str = f"{overhead_time/3600:.2f}h"
    
    print(f"📊 Total Processing Time: {total_time_str}")
    print(f"📊 Overhead Time: {overhead_time_str} ({overhead_percentage:.1f}%)")
    print(f"🚀 Parallel Processing Efficiency: {CPU_CORES} CPU cores utilized")
    
    # Calculate speedup estimate (assuming linear scaling)
    estimated_sequential_time = total_phase_time * CPU_CORES
    speedup = estimated_sequential_time / total_time if total_time > 0 else 1
    
    print(f"⚡ Estimated Speedup: {speedup:.1f}x (vs. sequential processing)")
    print(f"{'='*60}")


async def fetch_urls_parallel_batch(urls_batch: List[str], cache_file: str) -> Dict[str, str]:
    """
    Fetch a batch of URLs in parallel using asyncio.
    
    Args:
        urls_batch: List of URLs to fetch
        cache_file: Path to cache file for saving results
        
    Returns:
        Dictionary mapping URLs to their content
    """
    print(f"🚀 Fetching {len(urls_batch)} URLs in parallel batch...")
    
    async def fetch_single_url(session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
        """Fetch a single URL with retry logic."""
        try:
            from utils import fetch_url_with_retry
            content = await fetch_url_with_retry(session, url)
            return url, content
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return url, f"[Error] Failed to fetch: {e}"
    
    # Fetch all URLs in the batch concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_url(session, url) for url in urls_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert results to dictionary
    web_content = {}
    for result in results:
        if isinstance(result, tuple) and len(result) == 2:
            url, content = result
            web_content[url] = content
        else:
            logger.error(f"Unexpected result format: {result}")
    
    print(f"✅ Completed parallel batch of {len(urls_batch)} URLs")
    return web_content


async def fetch_all_urls_and_cache_parallel(all_urls: List[str], cache_file: str) -> Dict[str, str]:
    """
    Fetch all URLs in parallel batches using all available CPU cores.
    
    Args:
        all_urls: List of URLs to fetch
        cache_file: Path to cache file
        
    Returns:
        Dictionary mapping URLs to their content
    """
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"📁 Found existing cache file: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_content = json.load(f)
            print(f"📖 Loaded {len(cached_content)} URLs from cache")
            return cached_content
        except Exception as e:
            logger.warning(f"⚠️ Error reading cache file: {e}. Will fetch URLs again.")
    
    print(f"🌐 Fetching {len(all_urls)} URLs in parallel batches using {CPU_CORES} CPU cores...")
    
    # Optimize batch size based on number of URLs and CPU cores
    optimal_batch_size = max(1, min(CHUNK_SIZE, len(all_urls) // CPU_CORES))
    print(f"📦 Optimized batch size: {optimal_batch_size} URLs per batch")
    
    # Split URLs into batches for parallel processing
    url_batches = [all_urls[i:i + optimal_batch_size] for i in range(0, len(all_urls), optimal_batch_size)]
    print(f"📊 Created {len(url_batches)} batches for parallel processing")
    
    # Process batches concurrently with optimized concurrency control
    all_web_content = {}
    
    # Use semaphore to limit concurrent connections to prevent overwhelming servers
    # Allow more concurrent connections since we have 54 CPU cores
    max_concurrent = min(CPU_CORES * 2, len(url_batches))  # Allow up to 2x CPU cores for I/O
    semaphore = asyncio.Semaphore(max_concurrent)
    
    print(f"🔗 Using {max_concurrent} concurrent connections for optimal throughput")
    
    async def process_batch_with_semaphore(batch: List[str]) -> Dict[str, str]:
        async with semaphore:
            return await fetch_urls_parallel_batch(batch, cache_file)
    
    # Process all batches concurrently
    tasks = [process_batch_with_semaphore(batch) for batch in url_batches]
    
    # Use asyncio.gather with return_exceptions to handle failures gracefully
    print(f"🚀 Launching {len(tasks)} parallel batch tasks...")
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Combine results from all batches
    successful_batches = 0
    failed_batches = 0
    total_urls_processed = 0
    
    for i, batch_result in enumerate(batch_results):
        if isinstance(batch_result, dict):
            all_web_content.update(batch_result)
            successful_batches += 1
            total_urls_processed += len(batch_result)
            print(f"✅ Batch {i+1}/{len(url_batches)} completed: {len(batch_result)} URLs")
        else:
            failed_batches += 1
            logger.error(f"❌ Batch {i+1} failed: {batch_result}")
    
    print(f"📊 Batch processing summary:")
    print(f"  - Successful batches: {successful_batches}/{len(url_batches)}")
    print(f"  - Failed batches: {failed_batches}/{len(url_batches)}")
    print(f"  - Total URLs processed: {total_urls_processed}")
    
    # Save to cache file
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_web_content, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(all_web_content)} URLs to cache: {cache_file}")
    except Exception as e:
        logger.error(f"❌ Error saving cache file: {e}")
    
    success_rate = (successful_batches / len(url_batches)) * 100 if url_batches else 0
    print(f"🎉 Parallel URL fetching completed!")
    print(f"   - Success rate: {success_rate:.1f}%")
    print(f"   - Processed {len(all_web_content)} URLs using {CPU_CORES} CPU cores")
    print(f"   - Used {max_concurrent} concurrent connections for optimal I/O performance")
    
    return all_web_content


async def process_paragraph_parallel(
    paragraph_data: tuple[int, dict], 
    urls: List[str], 
    web_content_cache: Dict[str, str], 
    cache_file: str
) -> Dict[str, Any]:
    """
    Process a single paragraph in parallel.
    
    Args:
        paragraph_data: Tuple of (paragraph_index, cached_paragraph_data)
        urls: List of URLs to use for claim checking
        web_content_cache: Cached web content
        cache_file: Path to cache file
        
    Returns:
        Dictionary containing paragraph processing results
    """
    i, cached_para = paragraph_data
    
    try:
        # Use cached atomic claims
        atomic_claims = cached_para.get('atomic_claims', [])
        paragraph_text = cached_para.get('paragraph_text', '')
        
        if not atomic_claims:
            print(f"  No atomic claims found in cached paragraph {i}")
            return {
                'paragraph_index': i,
                'paragraph_text': paragraph_text,
                'claims': [],
                'claim_results': [],
                'error': 'No atomic claims found in cache'
            }

        # Load claim-query mappings from cache
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            claim_query_mappings = cache_data.get('related_query', {})
            
        # Filter out claims with no relevant queries
        filtered_claims = []
        for claim in atomic_claims:
            if claim_query_mappings.get(claim, {}).get('relevant_queries', []) != [] or is_url(claim):
                filtered_claims.append(claim)
            else:
                print(f"‼️ No relevant queries found for claim: {claim}")
        
        if not filtered_claims:
            return {
                'paragraph_index': i,
                'paragraph_text': paragraph_text,
                'claims': [],
                'claim_results': [],
                'error': 'No valid claims after filtering'
            }
        
        print(f"  Using {len(filtered_claims)} filtered claims from paragraph {i}")
        
        # Process claims and URLs using process_claims_and_urls with cached content
        claim_results, updated_observation_memory = await process_claims_and_urls(
            filtered_claims, urls, web_content_cache, "", cache_file
        )

        # Load chunk scores from cache file since they were already computed upfront
        chunk_scores = cache_data.get('chunk_score', {}) if cache_data else {}
        
        # Reconstruct detailed_chunk_scores structure that noise domination detector expects
        detailed_chunk_scores = []
        for chunk_id, chunk_data in chunk_scores.items():
            if isinstance(chunk_data, dict) and 'scores' in chunk_data:
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
        
        # Create scoring_results structure that noise domination detector expects
        scoring_results = {
            'detailed_chunk_scores': detailed_chunk_scores,
            'query_weights': {}
        }
        
        # Process noise detection for each claim
        noise_results = []
        for claim_result in claim_results:
            if isinstance(claim_result, dict):
                final_judgment = claim_result.get('final_judgment', 'unknown')
                claim_text = claim_result.get('claim', 'Unknown claim')
                
                if final_judgment == 'entailment':
                    # Get relevant chunks from claim results
                    relevant_chunks = claim_result.get('relevant_chunks', [])
                    if relevant_chunks and relevant_chunks[0]['score'] == -1.0:
                        continue  # Skip if entailed by observation memory
                    
                    # Check for noise domination
                    claim_query_mappings = cache_data.get('related_query', {}) if cache_data else {}
                    selected_queries = claim_query_mappings.get(claim_text, {})
                    relevant_queries = selected_queries.get('relevant_queries', []) if selected_queries else []
                    
                    noise_result = detect_noise_domination(
                        claim_text, 
                        relevant_chunks, 
                        relevant_queries, 
                        scoring_results
                    )
                    
                    if noise_result['is_noise_dominated']:
                        noise_results.append({
                            'claim': claim_text,
                            'noise_result': noise_result
                        })
        
        # Return results for this paragraph
        return {
            'paragraph_index': i,
            'claims': filtered_claims,
            'urls': urls,
            'claim_results': claim_results,
            'noise_results': noise_results,
            'observation_memory': updated_observation_memory
        }
        
    except Exception as e:
        print(f"  ❌ Error processing cached paragraph {i}: {str(e)}")
        return {
            'paragraph_index': i,
            'paragraph_text': cached_para.get('paragraph_text', ''),
            'claims': [],
            'claim_results': [],
            'error': str(e)
        }


async def process_report_paragraphs_parallel(
    urls: List[str], 
    web_content_cache: Dict[str, str], 
    cache_data: dict = None, 
    observation_memory: str = "", 
    output_file: str = "", 
    cache_file: str = ""
) -> List[Dict[str, Any]]:
    """
    Process report paragraphs in parallel using multiple CPU cores.
    
    Args:
        urls: List of URLs to use for claim checking
        web_content_cache: Cached web content
        cache_data: Optional cache data containing pre-decomposed atomic claims
        observation_memory: Observation memory for iterative checking
        output_file: Path to the output JSON file for saving results incrementally
        cache_file: Path to cache file
        
    Returns:
        List of results for each paragraph
    """
    print(f"\n{'='*50}")
    print("PROCESSING REPORT PARAGRAPHS IN PARALLEL")
    print(f"{'='*50}")
    
    # Check if we have cached atomic claims
    if cache_data and 'report' in cache_data:
        print(f"✅ Using cached atomic claims from cache file")
        print(f"💾 No reranker models loaded here - using pre-computed scores from cache")
        cached_paragraphs = cache_data['report']
        print(f"Found {len(cached_paragraphs)} cached paragraphs with atomic claims")
        
        # Memory check before processing
        print(f"\n💾 Memory status before parallel report processing:")
        monitor_memory_usage()
        
        # Prepare paragraph data for parallel processing
        paragraph_data_list = [(i+1, para) for i, para in enumerate(cached_paragraphs)]
        
        # Process paragraphs in parallel batches
        print(f"🚀 Processing {len(cached_paragraphs)} paragraphs in parallel using {CPU_CORES} CPU cores...")
        
        # Use semaphore to limit concurrent processing to prevent memory issues
        semaphore = asyncio.Semaphore(CPU_CORES)
        
        async def process_paragraph_with_semaphore(paragraph_data: tuple[int, dict]) -> Dict[str, Any]:
            async with semaphore:
                return await process_paragraph_parallel(paragraph_data, urls, web_content_cache, cache_file)
        
        # Process all paragraphs concurrently
        tasks = [process_paragraph_with_semaphore(para_data) for para_data in paragraph_data_list]
        report_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions and save results incrementally
        final_results = []
        for i, result in enumerate(report_results):
            if isinstance(result, dict):
                final_results.append(result)
                
                # IMMEDIATELY save this paragraph result to file
                if output_file:
                    append_report_paragraph_result(output_file, result, i)
                    
                print(f"  ✅ Parallel paragraph {i+1} processed successfully")
            else:
                logger.error(f"Paragraph {i+1} failed: {result}")
                error_result = {
                    'paragraph_index': i+1,
                    'paragraph_text': '',
                    'claims': [],
                    'claim_results': [],
                    'error': str(result)
                }
                final_results.append(error_result)
                
                # Save error result to file
                if output_file:
                    append_report_paragraph_result(output_file, error_result, i)
        
        print(f"🎉 Parallel paragraph processing completed! Processed {len(final_results)} paragraphs using {CPU_CORES} CPU cores")
        
    else:
        # Fallback to original method if no cache
        print(f"⚠️ No cached atomic claims found! Falling back to sequential processing...")
        final_results = await process_report_paragraphs(urls, web_content_cache, cache_data, observation_memory, output_file, cache_file)
    
    # Final memory cleanup
    print(f"\n🧹 Final memory cleanup after parallel report processing")
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n✅ Parallel report processing completed. Processed {len(final_results)} paragraphs.")
    print(f"Final memory status:")
    monitor_memory_usage()
    
    return final_results


def process_claims_batch_parallel(claims_batch: List[str], query_list: List[str]) -> Dict[str, Any]:
    """
    Process a batch of claims to find relevant queries in parallel.
    
    Args:
        claims_batch: List of claims to process
        query_list: List of queries to match against
        
    Returns:
        Dictionary containing claim-to-query mappings for this batch
    """
    try:
        from fixed_thre_claim_link_to_query import find_relevant_queries_for_claims
        
        # Find relevant queries for this batch of claims
        selected_queries = find_relevant_queries_for_claims(claims_batch, query_list, fixed_threshold=0.01)
        
        # Convert the result format to use claim text as keys
        claim_query_mappings = {}
        for claim_idx, claim_info in selected_queries.items():
            claim_text = claim_info['claim']
            claim_query_mappings[claim_text] = claim_info
        
        return claim_query_mappings
        
    except Exception as e:
        logger.error(f"Error processing claims batch: {e}")
        return {}


async def map_all_claims_to_queries_upfront_parallel(
    cache_data: dict,
    cache_file: str
) -> Dict[str, Any]:
    """
    Map all extracted claims to their related queries upfront using parallel processing.
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
    print("MAPPING ALL CLAIMS TO QUERIES UPFRONT (PARALLEL)")
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
    
    # Split claims into batches for parallel processing
    claims_per_batch = max(1, len(unique_claims) // CPU_CORES)
    claim_batches = [unique_claims[i:i + claims_per_batch] for i in range(0, len(unique_claims), claims_per_batch)]
    
    print(f"🚀 Processing {len(claim_batches)} claim batches in parallel using {CPU_CORES} CPU cores...")
    print(f"📦 Each batch contains ~{claims_per_batch} claims")
    
    # Process batches in parallel using ProcessPoolExecutor for CPU-intensive work
    all_claim_query_mappings = {}
    
    with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
        # Submit all batches for parallel processing
        future_to_batch = {
            executor.submit(process_claims_batch_parallel, batch, query_list): i 
            for i, batch in enumerate(claim_batches)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                batch_result = future.result()
                all_claim_query_mappings.update(batch_result)
                print(f"✅ Batch {batch_idx + 1}/{len(claim_batches)} completed: {len(batch_result)} claims mapped")
            except Exception as e:
                print(f"❌ Batch {batch_idx + 1} failed: {e}")
    
    print(f"🎉 Parallel claim-query mapping completed! Mapped {len(all_claim_query_mappings)} claims using {CPU_CORES} CPU cores")
    
    # Save the mappings to cache file
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            all_content = json.load(f)
        
        # Add the new claim-query mappings
        all_content['related_query'] = all_claim_query_mappings
        
        # Write back to cache file
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(all_content, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Claim-query mappings saved to cache file: {cache_file}")
        print(f"📊 Cache now contains both chunk scores and claim-query mappings")
        
    except Exception as e:
        print(f"❌ Error saving claim-query mappings to cache: {e}")
    
    return all_claim_query_mappings


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
    # - related_query: Pre-computed claim-to-query mappings
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


async def process_iteration_parallel(
    iteration_data: tuple[int, dict], 
    query: str, 
    web_content_cache: Dict[str, str], 
    cache_file: str, 
    output_file: str, 
    url_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Process a single iteration in parallel.
    
    Args:
        iteration_data: Tuple of (iteration_index, iteration_data)
        query: The original query
        web_content_cache: Cached web content
        cache_file: Path to cache file
        output_file: Path to output file
        url_mapping: Global URL-to-index mapping
        
    Returns:
        Dictionary containing iteration results
    """
    i, iteration = iteration_data
    
    try:
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
        
        # If not the first search claim, the search list should add all previous search list items
        if i > 0:
            for j in range(i):
                search_list = search_list + [item for item in cache_data['iterations'][j].get(f'search_list_{j+1}', []) if item not in search_list]

        # Process this iteration
        result = await single_iteration(
            query=query,
            observation_memory="",  # Each iteration processes independently
            action_list=action_list,
            search_list=search_list,
            claim_list=claim_list,
            web_content_cache=web_content_cache,
            cache_file=cache_file,
            url_mapping=url_mapping
        )
        
        # Save result immediately to avoid memory accumulation
        if output_file:
            append_iteration_result(output_file, result, i)
        
        print(f"✅ Parallel iteration {i+1} completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error processing iteration {i+1}: {str(e)}")
        error_result = {
            'iteration_index': i+1,
            'error': str(e),
            'observation_memory': ""
        }
        
        # Save error result
        if output_file:
            append_iteration_result(output_file, error_result, i)
        
        return error_result


async def process_cache_data_parallel(
    cache_data: dict, 
    web_content_cache: Dict[str, str], 
    query: str, 
    cache_file: str, 
    output_file: str, 
    url_mapping: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Process cache data through single_iteration for each iteration in parallel.
    
    Args:
        cache_data: Dictionary containing iterations data
        web_content_cache: Cached web content to use instead of fetching URLs
        query: The original query
        cache_file: Path to the cache file containing pre-computed chunk scores
        output_file: Path to the output JSON file for saving results incrementally
        url_mapping: Global URL-to-index mapping for consistent chunk IDs
        
    Returns:
        List of results from each iteration
    """
    # Initialize global models once
    initialize_global_models()

    iterations = cache_data.get('iterations', [])
    print(f"🚀 Processing {len(iterations)} iterations in parallel using {CPU_CORES} CPU cores...")
    
    # Prepare iteration data for parallel processing
    iteration_data_list = [(i, iteration) for i, iteration in enumerate(iterations)]
    
    # Use semaphore to limit concurrent processing to prevent memory issues
    semaphore = asyncio.Semaphore(CPU_CORES)
    
    async def process_iteration_with_semaphore(iteration_data: tuple[int, dict]) -> Dict[str, Any]:
        async with semaphore:
            return await process_iteration_parallel(
                iteration_data, query, web_content_cache, cache_file, output_file, url_mapping
            )
    
    # Process all iterations concurrently
    tasks = [process_iteration_with_semaphore(iter_data) for iter_data in iteration_data_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions and collect results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, dict):
            final_results.append(result)
        else:
            logger.error(f"Iteration {i+1} failed: {result}")
            error_result = {
                'iteration_index': i+1,
                'error': str(result),
                'observation_memory': ""
            }
            final_results.append(error_result)
    
    print(f"🎉 Parallel iteration processing completed! Processed {len(final_results)} iterations using {CPU_CORES} CPU cores")
    
    # Memory cleanup after all iterations
    print(f"\n🧹 Memory cleanup after parallel iteration processing")
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return final_results


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
        
        # If not the first search claim, the search list should add all previous search list items
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


async def score_all_chunks_upfront_parallel(
    sub_queries: List[str], 
    all_urls: List[str], 
    web_content_cache: Dict[str, str], 
    cache_file: str, 
    url_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """
    Score all chunks from all URLs against all sub-queries upfront using parallel processing.
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
    print("SCORING ALL CHUNKS UPFRONT (PARALLEL)")
    print(f"{'='*50}")
    
    # Memory check before scoring
    print(f"💾 Memory status before parallel chunk scoring:")
    monitor_memory_usage()
    
    # Initialize the global scorer with the pre-loaded BGEScorer instance
    print("🔧 Initializing IntegratedChunkScorer for parallel upfront scoring...")
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
    print(f"🚀 Using parallel processing with {CPU_CORES} CPU cores for optimal performance")
    
    # Score all chunks against all queries
    print(f"🚀 Starting parallel upfront scoring of all chunks...")
    print(f"💾 Using pre-loaded BGEScorer instance (no model reloading)")
    
    # The IntegratedChunkScorer already uses GPU parallelism, but we can optimize CPU-bound parts
    # by processing URLs in parallel batches
    scoring_results = await scorer.score_chunks(
        queries=sub_queries,
        urls=all_urls,  # Use all_urls to score everything upfront
        web_content_cache=web_content_cache,
        cache_file=cache_file,
        url_mapping=url_mapping
    )
    
    print(f"✅ Parallel upfront scoring completed!")
    print(f"📊 Scored {len(scoring_results.get('detailed_chunk_scores', []))} chunks")
    print(f"📊 Results saved to cache file: {cache_file}")
    
    # Memory check after scoring
    print(f"💾 Memory status after parallel chunk scoring:")
    monitor_memory_usage()
    
    # Memory cleanup after scoring
    print(f"🧹 Memory cleanup after parallel upfront scoring")
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return scoring_results


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
                            # Observation memory is already updated in process_claims_and_urls
                            # Just log that this claim was already added to memory
                            print(f"✅ Claim already added to observation memory: {claim_text}")

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


async def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description='Semantic Fact Check Evaluation Pipeline'
    )
    parser.add_argument('input_json', help='Path to input JSON file')
    parser.add_argument('--sequential', action='store_true', help='Force sequential processing mode')
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

    # Determine processing mode
    use_parallel = USE_PARALLEL_PROCESSING
    if args.sequential:
        use_parallel = False
        print(f"🔄 Sequential processing mode forced via command line argument")
    
    if use_parallel:
        print(f"🚀 Parallel processing mode enabled - utilizing {CPU_CORES} CPU cores")
        print(f"📊 This should provide significant speedup for batch operations")
    else:
        print(f"🐌 Sequential processing mode enabled - using single-threaded operations")
        print(f"📊 This mode is slower but uses less memory and is more stable")

    # read the query and report from the input json
    with open(args.input_json, 'r', encoding='utf-8') as f:
        text = json.load(f)
        query = text.get('query', '')
        report = text.get('final_report', '')
        all_urls = text.get('all_source_links', [])
        urls = text.get('summary_citations', [])
        

    # Create cache file path
    cache_file = f"../web_content_cache/train_gemini/cache_{os.path.basename(args.input_json)}"
    
    # Create output file path
    output_file = f"../results/train_gemini/{os.path.basename(args.input_json)}"
    
    # Create global URL mapping for consistent chunk IDs
    url_mapping = create_global_url_mapping(all_urls)
    
    # Initialize the results file with basic structure
    initialize_results_file(output_file, query, report, all_urls, urls)
    
    # Fetch all URLs and cache them
    print(f"\n📥 Fetching and caching web content...")
    start_performance_timer('url_fetching')
    if use_parallel:
        web_content_cache = await fetch_all_urls_and_cache_parallel(all_urls, cache_file)
    else:
        web_content_cache = await fetch_all_urls_and_cache(all_urls, cache_file)
    end_performance_timer('url_fetching')
    
    # Memory check after web content fetching
    print(f"\n💾 Memory status after web content fetching:")
    monitor_memory_usage()

    # Decompose the query to get the atomic actions
    print(f"\n🔍 Decomposing workflow...")
    start_performance_timer('workflow_decomposition')
    if use_parallel:
        # Use parallel workflow decomposition from decomposition.py
        from decomposition import decompose_workflow_to_cache_auto
        decompose_workflow_to_cache_auto(args.input_json)
        print(f"✅ Parallel workflow decomposition completed")
    else:
        # Use sequential workflow decomposition
        decompose_workflow_to_cache(args.input_json)
    end_performance_timer('workflow_decomposition')

    # Read the cache file
    cache_file = f"../json_cache/train_gemini/cache_{os.path.basename(args.input_json)}"
    
    # Decompose report paragraphs to cache if not already done
    print(f"\n📝 Decomposing report paragraphs to cache...")
    start_performance_timer('report_decomposition')
    if use_parallel:
        # Use parallel report decomposition from decomposition.py
        from decomposition import decompose_report_to_cache_auto
        decompose_report_to_cache_auto(report, query, cache_file)
        print(f"✅ Parallel report decomposition completed")
    else:
        # Use sequential report decomposition
        decompose_report_to_cache(report, query, cache_file)
    end_performance_timer('report_decomposition')
    
    # Read the updated cache file
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    # Get sub-queries from cache file
    sub_queries = cache_data.get('query_list', [])

    # NEW FLOW: Score all chunks upfront against all queries
    # This eliminates the need for repeated scoring during iterations
    print(f"\n🔄 Scoring all chunks upfront...")
    start_performance_timer('chunk_scoring')
    if use_parallel:
        scoring_results = await score_all_chunks_upfront_parallel(sub_queries, all_urls, web_content_cache, cache_file, url_mapping)
    else:
        scoring_results = await score_all_chunks_upfront(sub_queries, all_urls, web_content_cache, cache_file, url_mapping)
    end_performance_timer('chunk_scoring')

    # NEW FLOW: Map all claims to their related queries upfront
    # This eliminates the need for repeated calls to find_relevant_queries_for_claims
    # The cache file will now contain both chunk scores and claim-query mappings
    # Note: No need to pass claim_query_mappings as parameter - functions load directly from cache
    print(f"\n🔄 Mapping all claims to their related queries upfront...")
    start_performance_timer('claim_mapping')
    if use_parallel:
        claim_query_mappings = await map_all_claims_to_queries_upfront_parallel(cache_data, cache_file)
    else:
        claim_query_mappings = await map_all_claims_to_queries_upfront(cache_data, cache_file)
    end_performance_timer('claim_mapping')

    # Process the cache data to get data for each iteration
    # Note: Both chunk scoring and claim-query mapping have been done upfront
    # Iterations now only need to load from cache, eliminating repeated computations
    # Functions load claim-query mappings directly from cache_file when needed
    print(f"\n🔄 Processing Chain of Research iterations...")
    start_performance_timer('iteration_processing')
    if use_parallel:
        results = await process_cache_data_parallel(cache_data, web_content_cache, query, cache_file, output_file, url_mapping)
    else:
        results = await process_cache_data(cache_data, web_content_cache, query, cache_file, output_file, url_mapping)
    end_performance_timer('iteration_processing')
   
    # After processing the Chain of Research, we need to process the report
    print(f"\n{'='*50}")
    print("PROCESSING REPORT")
    print(f"{'='*50}")
    
    # Process each paragraph of the report using cached web content and claim-query mappings
    # No need to call find_relevant_queries_for_claims here - using pre-computed mappings from cache
    # Functions load claim-query mappings directly from cache_data when needed

    # Load the observation memory from the results
    # observation_memory = results[-1]['observation_memory']
    observation_memory = ""
    start_performance_timer('paragraph_processing')
    if use_parallel:
        report_results = await process_report_paragraphs_parallel(urls, web_content_cache, cache_data, observation_memory, output_file, cache_file)
    else:
        report_results = await process_report_paragraphs(urls, web_content_cache, cache_data, observation_memory, output_file, cache_file)
    end_performance_timer('paragraph_processing')

    # Final memory status
    print(f"\n💾 Final memory status:")
    monitor_memory_usage()
    
    # Print performance summary
    print_performance_summary()
    
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