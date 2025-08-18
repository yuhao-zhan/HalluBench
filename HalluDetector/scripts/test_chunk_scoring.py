#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import os
from datetime import datetime

from chunk_scoring import IntegratedChunkScorer
from utils import NumpyEncoder


async def test_chunk_scoring():
    """Test the integrated chunk scoring system."""
    
    # Test URLs and queries
    urls = [
        "https://openai.com/careers/",
        "https://boards.greenhouse.io/deepmind/jobs/6541427",  
        # "https://www.metacareers.com/teams/technology?tab=AI"
    ]

    queries = [
        # "Full-time job opportunities are sought.",
        "Job opportunities are at OpenAI.",
        # "Job opportunities are at Google DeepMind.",
        "Job opportunities are at Meta.",
        "Roles are Research Scientist.",
        # "Roles are Machine Learning Engineer.",
        # "Roles are Research Engineer (or equivalent).",
        # "Roles involve working on AI.",
        "Positions are based in the United States."
    ]
    
    print("="*80)
    print("TESTING INTEGRATED CHUNK SCORING SYSTEM")
    print("="*80)
    print(f"URLs: {len(urls)}")
    print(f"Queries: {len(queries)}")
    print()
    
    # Initialize integrated scorer
    scorer = IntegratedChunkScorer(
        sbert_model='all-MiniLM-L6-v2',
        ner_threshold=0.5,
        c=6.0,
        num_gpus=4
    )

    with open("../web_content_cache/cache_gemini_PhD_jobs.json", 'r', encoding='utf-8') as f:
        test_web_content = json.load(f)
        # get the web_content for the urls
        web_content = {url: test_web_content[url] for url in urls}
        
    # Run scoring
    print("Running integrated chunk scoring...")
    # Create a simple URL mapping for testing
    url_mapping = {url: idx for idx, url in enumerate(urls)}
    results = await scorer.score_chunks(queries, urls, web_content, None, url_mapping)  # No cache file for testing
    
    # Save results
    output_file = "../results/test_chunk_scoring_results.json"
    os.makedirs("../results", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"Results saved to: {output_file}")
    print()
    
    # Display detailed results
    display_detailed_results(results, queries)


def display_detailed_results(results, queries):
    """Display detailed results from the chunk scoring system."""
    
    print("="*80)
    print("DETAILED RESULTS ANALYSIS")
    print("="*80)
    
    # Metadata
    metadata = results['metadata']
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Number of queries: {metadata['num_queries']}")
    print(f"Number of expanded queries: {metadata['num_expanded_queries']}")
    print(f"Number of documents: {metadata['num_documents']}")
    print()
    
    # Display queries
    print("ORIGINAL QUERIES:")
    for i, query in enumerate(queries):
        print(f"  {i}: {query}")
    print()
    
    # Display question queries
    print("QUESTION QUERIES:")
    question_queries = metadata['question_queries']
    for i, query in enumerate(question_queries):
        print(f"  {i}: {query}")
    print()
    
    # Display expanded queries
    print("EXPANDED QUERIES:")
    for i, query in enumerate(metadata['expanded_queries']):
        original_idx = metadata['query_mapping'][i]
        print(f"  {i} (from {original_idx}): {query}")
    print()
    
    # Display entity weights
    print("ENTITY WEIGHTS (Top 10):")
    entity_weights = metadata['entity_weights']
    sorted_entities = sorted(entity_weights.items(), key=lambda x: x[1], reverse=True)
    for entity, weight in sorted_entities[:10]:
        print(f"  {entity}: {weight:.4f}")
    print()
    
    # Display query clusters
    print("QUERY CLUSTERS:")
    query_clusters = metadata['query_clusters']
    for query_idx, cluster_info in query_clusters.items():
        print(f"  Query {query_idx}: {cluster_info['query']}")
        print(f"    Key terms: {cluster_info['key_query_terms']}")
        for cluster_key, cluster_data in cluster_info['clusters'].items():
            print(f"    Cluster '{cluster_key}':")
            print(f"      Key term: {cluster_data['key_query_term']}")
            similar_entities = [e['text'] for e in cluster_data['similar_entities'][:3]]
            print(f"      Similar entities: {similar_entities}")
        print()
    
    # Display quality scores
    print("QUALITY SCORES:")
    quality_scores = results['quality_scores']
    for query_idx, score in quality_scores.items():
        print(f"  Query {query_idx}: {score:.4f}")
    print()
    
    # Display relevance scores
    print("RELEVANCE SCORES:")
    relevance_scores = results['relevance_scores']
    for query_idx, score in relevance_scores.items():
        print(f"  Query {query_idx}: {score:.4f}")
    print()
    
    # Display combined scores (computed from quality and relevance scores)
    print("COMBINED SCORES (Quality + Relevance):")
    combined_scores = {}
    for query_idx in range(len(queries)):
        quality_score = quality_scores.get(query_idx, 0.0)
        relevance_score = relevance_scores.get(query_idx, 0.0)
        combined_scores[query_idx] = quality_score + relevance_score
    
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    for query_idx, score in sorted_combined:
        quality_score = quality_scores.get(query_idx, 0.0)
        relevance_score = relevance_scores.get(query_idx, 0.0)
        print(f"  Query {query_idx}: {score:.4f} (Quality: {quality_score:.4f}, Relevance: {relevance_score:.4f})")
    print()
    
    # Display top scoring queries
    print("TOP SCORING QUERIES:")
    for i, (query_idx, score) in enumerate(sorted_combined[:5], 1):
        print(f"  {i}. Query {query_idx}: {queries[query_idx]}")
        print(f"     Combined Score: {score:.4f}")
        print(f"     Quality Score: {quality_scores.get(query_idx, 0.0):.4f}")
        print(f"     Relevance Score: {relevance_scores.get(query_idx, 0.0):.4f}")
        print()
    
    # Display collection stats
    print("COLLECTION STATISTICS:")
    collection_stats = metadata['collection_stats']
    print(f"  Average document length: {collection_stats['avdl']:.2f}")
    print(f"  Total documents: {collection_stats['total_docs']}")
    print(f"  Total length: {collection_stats['total_length']}")
    print()
    
    # Display model info
    print("MODEL INFORMATION:")
    model_info = metadata['model_info']
    print(f"  SBERT Model: {model_info['sbert_model']}")
    print(f"  NER Threshold: {model_info['ner_threshold']}")
    print(f"  C Parameter: {model_info['c_parameter']}")
    print(f"  Log-logistic Model: {model_info['log_logistic_model']}")
    print()
    
    # Summary statistics
    print("SUMMARY STATISTICS:")
    quality_scores_list = list(quality_scores.values())
    relevance_scores_list = list(relevance_scores.values())
    combined_scores_list = list(combined_scores.values())
    
    print(f"  Quality Scores - Min: {min(quality_scores_list):.4f}, Max: {max(quality_scores_list):.4f}, Avg: {sum(quality_scores_list)/len(quality_scores_list):.4f}")
    print(f"  Relevance Scores - Min: {min(relevance_scores_list):.4f}, Max: {max(relevance_scores_list):.4f}, Avg: {sum(relevance_scores_list)/len(relevance_scores_list):.4f}")
    print(f"  Combined Scores - Min: {min(combined_scores_list):.4f}, Max: {max(combined_scores_list):.4f}, Avg: {sum(combined_scores_list)/len(combined_scores_list):.4f}")
    print()
    
    # Display detailed chunk scores in descending order
    print("="*80)
    print("DETAILED CHUNK SCORES (DESCENDING ORDER)")
    print("="*80)
    
    # Get aggregated chunk scores from the main system
    aggregated_chunk_scores = results.get('aggregated_chunk_scores', [])
    
    # Display top chunks
    print(f"Total 4-sentence chunks scored: {len(aggregated_chunk_scores)}")
    print()
    
    print("TOP CHUNKS BY COMBINED SCORE (Relevance + Quality Contribution):")
    for i, chunk_score in enumerate(aggregated_chunk_scores[:20], 1):
        print(f"{i:2d}. CHUNK {chunk_score['chunk_id']}")
        print(f"    Combined Score: {chunk_score['combined_score']:.4f}")
        print(f"    Relevance Score: {chunk_score['relevance_score']:.4f}")
        print(f"    Quality Contribution: {chunk_score['quality_contribution']:.4f}")
        print(f"    Query {chunk_score['query_idx']}: {queries[chunk_score['query_idx']]}")
        print(f"    URL: {chunk_score['url']}")
        print(f"    Text: {chunk_score['chunk_text'][:150]}{'...' if len(chunk_score['chunk_text']) > 150 else ''}")
        
        if chunk_score['matching_contexts']:
            print(f"    Matching 1-sentence contexts ({len(chunk_score['matching_contexts'])}):")
            for j, context in enumerate(chunk_score['matching_contexts'], 1):
                print(f"      {j}. \"{context['context_text']}\" (Quality: {context['quality_score']:.4f}, Contrib: {context['contribution']:.4f})")
        else:
            print(f"    No matching 1-sentence contexts found")
        print()
    
    # Display statistics
    print("CHUNK SCORE STATISTICS:")
    relevance_scores = [c['relevance_score'] for c in aggregated_chunk_scores]
    quality_contributions = [c['quality_contribution'] for c in aggregated_chunk_scores]
    combined_scores = [c['combined_score'] for c in aggregated_chunk_scores]
    
    print(f"  Relevance Scores - Min: {min(relevance_scores):.4f}, Max: {max(relevance_scores):.4f}, Avg: {sum(relevance_scores)/len(relevance_scores):.4f}")
    print(f"  Quality Contributions - Min: {min(quality_contributions):.4f}, Max: {max(quality_contributions):.4f}, Avg: {sum(quality_contributions)/len(quality_contributions):.4f}")
    print(f"  Combined Scores - Min: {min(combined_scores):.4f}, Max: {max(combined_scores):.4f}, Avg: {sum(combined_scores)/len(combined_scores):.4f}")
    
    # Count chunks with quality contributions
    chunks_with_quality = sum(1 for c in aggregated_chunk_scores if c['quality_contribution'] > 0)
    print(f"  Chunks with quality contributions: {chunks_with_quality}/{len(aggregated_chunk_scores)}")
    
    print()
    
    print("="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_chunk_scoring()) 