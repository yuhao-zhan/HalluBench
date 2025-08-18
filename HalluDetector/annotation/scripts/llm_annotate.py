import argparse
from fileinput import filename
import json
import os
from typing import List, Dict, Any
import asyncio
import aiohttp
import logging
import time
import sys


from ..scripts.decomposition import decompose_workflow_to_cache, decompose_report_to_cache
from ..scripts.search import single_iteration
from ..scripts.claim_checking import process_claims_and_urls
from ..scripts.utils import fetch_all_urls_and_cache, is_url
from ..scripts.fixed_thre_claim_link_to_query import find_relevant_queries_for_claims
from ..scripts.no_weight_noise_domination_detector import detect_noise_domination
from ..scripts.NLI_plus_BM25_chunk_score import IntegratedChunkScorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sepcify process name
import setproctitle
setproctitle.setproctitle('Yuhao_annotate')

def main():
    start_time = time.time()
    raw_input_json = "/data2/yuhaoz/DeepResearch/HalluBench/scripts/gemini_PhD_jobs.json"
    cache_input_json = "/data2/yuhaoz/DeepResearch/HalluBench/HalluDetector/json_cache/train_gemini/cache_gemini_PhD_jobs.json"

    # read the query and report from the input json
    with open(raw_input_json, 'r', encoding='utf-8') as f:
        text = json.load(f)
        query = text.get('query', '')

    with open(cache_input_json, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
        all_urls = cache_data.get('all_source_links', [])
        urls = cache_data.get('summary_citations', [])
        report = cache_data.get('final_report', '')

    # Create cache file path
    cache_file = f"../web_content_cache/xinxuan_report_70/cache_{os.path.basename(input_json)}"
    
    # Create output file path
    output_file = f"../results/xinxuan_report_70/results_{os.path.basename(input_json)}"
    
    # Create global URL mapping for consistent chunk IDs
    url_mapping = create_global_url_mapping(all_urls)
    
    # Initialize the results file with basic structure
    initialize_results_file(output_file, query, report, all_urls, urls)
    
    # Fetch all URLs and cache them
    print(f"\n📥 Fetching and caching web content...")
    web_content_cache = await fetch_all_urls_and_cache(all_urls, cache_file)
    
    # Memory check after web content fetching
    print(f"\n💾 Memory status after web content fetching:")
    monitor_memory_usage()

    # Decompose the query to get the atomic actions
    print(f"\n🔍 Decomposing workflow...")
    # decompose_workflow_to_cache(input_json)

    # Read the cache file
    cache_file = f"../json_cache/xinxuan_report_70/cache_{os.path.basename(input_json)}"
    
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
    # observation_memory = results[-1]['observation_memory']
    observation_memory = ""
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
    # Read the input json cache file to read the query_list, 

if __name__ == "__main__":
    main()