import argparse
from fileinput import filename
import json
import os
from typing import List, Dict, Any
import asyncio
import aiohttp
import logging

from decomposition import decompose_workflow_to_cache, decompose_observation
from search import single_iteration
from claim_checking import process_claims_and_urls
from utils import fetch_all_urls_and_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def process_cache_data(cache_data: dict, sub_queries: List[str], web_content_cache: Dict[str, str]):
    """
    Process cache data through single_iteration for each iteration.
    
    Args:
        cache_data: Dictionary containing iterations data
        sub_queries: List of sub-queries
        web_content_cache: Cached web content to use instead of fetching URLs
        
    Returns:
        List of results from each iteration
    """
    results = []
    observation_memory = ""
    
    for i, iteration in enumerate(cache_data['iterations']):
        # Extract data for this iteration using numbered keys
        action_list = iteration.get(f'action_list_{i+1}', [])
        search_list = iteration.get(f'search_list_{i+1}', [])
        claim_list = iteration.get(f'claim_list_{i+1}', [])
        
        print(f"\nProcessing Iteration {i+1}...")
        print(f"Action list: {len(action_list)} items")
        print(f"Search list: {len(search_list)} items")
        print(f"Claim list: {len(claim_list)} items")
        
        # Process this iteration
        result = await single_iteration(
            query_list=sub_queries,
            observation_memory=observation_memory,
            action_list=action_list,
            search_list=search_list,
            claim_list=claim_list,
            web_content_cache=web_content_cache
        )
        
        # Update observation memory for next iteration
        observation_memory = result['observation_memory']
        
        results.append(result)
    
    return results


async def process_report_paragraphs(report: str, query: str, urls: List[str], web_content_cache: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Process each paragraph of the report using decompose_observation and process_claims_and_urls.
    
    Args:
        report: The full report text
        query: The original query
        urls: List of URLs to use for claim checking
        web_content_cache: Cached web content
        
    Returns:
        List of results for each paragraph
    """
    # Split report into paragraphs
    paragraphs = [p.strip() for p in report.split('\n\n') if p.strip()]
    
    print(f"\n{'='*50}")
    print("PROCESSING REPORT PARAGRAPHS")
    print(f"{'='*50}")
    print(f"Found {len(paragraphs)} paragraphs to process")
    
    report_results = []
    
    for i, paragraph in enumerate(paragraphs, 1):
        print(f"\nProcessing paragraph {i}/{len(paragraphs)}...")
        print(f"Paragraph length: {len(paragraph)} characters")
        
        try:
            # Decompose the paragraph into atomic claims
            atomic_claims = decompose_observation(paragraph, query)
            
            if not atomic_claims:
                print(f"  No extractable claims found in paragraph {i}")
                report_results.append({
                    'paragraph_index': i,
                    'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                    'claims': [],
                    'claim_results': [],
                    'error': 'No extractable claims found'
                })
                continue
            
            print(f"  Extracted {len(atomic_claims)} claims from paragraph {i}")
            
            # Process claims and URLs using process_claims_and_urls with cached content
            claim_results = await process_claims_and_urls_with_cache(atomic_claims, urls, web_content_cache)
            
            # Store results for this paragraph
            paragraph_result = {
                'paragraph_index': i,
                'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                'claims': atomic_claims,
                'urls': urls,
                'claim_results': claim_results
            }
            
            report_results.append(paragraph_result)
            print(f"  ✅ Paragraph {i} processed successfully")
            
        except Exception as e:
            print(f"  ❌ Error processing paragraph {i}: {str(e)}")
            report_results.append({
                'paragraph_index': i,
                'paragraph_text': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                'claims': [],
                'claim_results': [],
                'error': str(e)
            })
    
    print(f"\n✅ Report processing completed. Processed {len(report_results)} paragraphs.")
    return report_results


async def process_claims_and_urls_with_cache(claims: List[str], urls: List[str], web_content_cache: Dict[str, str]) -> Dict[str, Any]:
    """
    Process claims and URLs using cached web content instead of fetching URLs again.
    
    Args:
        claims: List of claims to check
        urls: List of URLs to use for claim checking
        web_content_cache: Cached web content
        
    Returns:
        Dictionary containing claim checking results
    """
    from claim_checking import NLIClaimChecker
    
    checker = NLIClaimChecker()
    
    # Filter web content to only include the requested URLs
    filtered_web_content = {url: web_content_cache.get(url, f"[Error] URL not found in cache: {url}") 
                           for url in urls}
    
    # Use the cached content instead of fetching URLs again
    return await checker.process_claims_and_urls_with_content(claims, filtered_web_content)


async def main():
    parser = argparse.ArgumentParser(
        description='Semantic Fact Check Evaluation Pipeline'
    )
    parser.add_argument('input_json', help='Path to input JSON file')
    args = parser.parse_args()

    # read the query and report from the input json
    with open(args.input_json, 'r', encoding='utf-8') as f:
        text = json.load(f)
        query = text.get('query', '')
        report = text.get('report', '')
        all_urls = text.get('all_source_links', [])
        urls = text.get('summary_citations', [])
        
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Create cache file path
    cache_file = f"web_content_cache/cache_{os.path.basename(args.input_json)}"
    
    # Fetch all URLs and cache them
    web_content_cache = await fetch_all_urls_and_cache(all_urls, cache_file)

    # Decompose the query to get the atomic actions
    decompose_workflow_to_cache(args.input_json)

    # Read the cache file
    cache_file = f"json_cache/cache_{os.path.basename(args.input_json)}"
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    # Get sub-queries from cache file
    sub_queries = cache_data.get('query_list', [])

    # Process the cache data to get data for each iteration
    results = await process_cache_data(cache_data, sub_queries, web_content_cache)
   
    # Write the updated cache file
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Updated cache file with evaluation results: {cache_file}")

    # After processing the Chain of Research, we need to process the report
    print(f"\n{'='*50}")
    print("PROCESSING REPORT")
    print(f"{'='*50}")
    
    # Process each paragraph of the report using cached web content
    report_results = await process_report_paragraphs(report, query, urls, web_content_cache)
    
    # Combine all results
    final_results = {
        'chain_of_research_results': results,
        'report_results': report_results,
        'summary': {
            'total_iterations': len(results),
            'total_paragraphs': len(report_results),
            'processed_iterations': len([r for r in results if 'error' not in r]),
            'processed_paragraphs': len([r for r in report_results if 'error' not in r])
        }
    }
    
    # Save the complete results
    output_file = f"results/results_{os.path.basename(args.input_json)}"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Complete results saved to: {output_file}")
    print(f"📊 Summary:")
    print(f"  - Chain of Research iterations: {final_results['summary']['total_iterations']}")
    print(f"  - Report paragraphs: {final_results['summary']['total_paragraphs']}")
    print(f"  - Successfully processed iterations: {final_results['summary']['processed_iterations']}")
    print(f"  - Successfully processed paragraphs: {final_results['summary']['processed_paragraphs']}")


if __name__ == '__main__':
    asyncio.run(main())