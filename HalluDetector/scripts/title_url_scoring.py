#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from typing import List, Dict, Any, Tuple
from FlagEmbedding import FlagReranker
import torch
import warnings
import concurrent.futures
import threading



warnings.filterwarnings('ignore')

# Global GPU locks to prevent concurrent access to the same GPU
_gpu_locks = {}
_gpu_lock_creation_lock = threading.Lock()

def get_gpu_lock(gpu_id: int) -> threading.Lock:
    """Get or create a lock for a specific GPU to prevent concurrent access."""
    if gpu_id not in _gpu_locks:
        with _gpu_lock_creation_lock:
            if gpu_id not in _gpu_locks:
                _gpu_locks[gpu_id] = threading.Lock()
    return _gpu_locks[gpu_id]

def _normalize_scores_preserve_mapping(scores_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize scores using min-max normalization while preserving the claim-to-score mapping.
    
    Args:
        scores_dict: Dictionary mapping claims/queries to scores
        
    Returns:
        Dictionary with normalized scores, preserving the same keys
    """
    if not scores_dict:
        return {}
    
    values = list(scores_dict.values())
    vmin = min(values)
    vmax = max(values)
    
    if vmax - vmin < 1e-12:
        # All scores are the same, return 0.0 for all
        return {key: 0.0 for key in scores_dict.keys()}
    
    # Normalize while preserving the mapping
    normalized_scores = {}
    for key, score in scores_dict.items():
        normalized_scores[key] = float((score - vmin) / (vmax - vmin))
    
    return normalized_scores

class TitleURLScorer:
    """Handles scoring of URLs and titles against queries using BGE Reranker."""
    
    def __init__(self, num_gpus: int = 0):
        """Initialize TitleURLScorer with BGE Reranker."""
        self.available_gpus = 0
        self.rerankers = []
        self.reranker = None
        
        if num_gpus > 0 and torch.cuda.is_available():
            self.available_gpus = min(num_gpus, torch.cuda.device_count())
            print(f"🔧 Using {self.available_gpus} GPUs")
            
            # Initialize reranker on each available GPU
            for gpu_id in range(self.available_gpus):
                try:
                    # Set device before initializing reranker
                    torch.cuda.set_device(gpu_id)
                    
                    # Initialize reranker on this specific GPU
                    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device=f'cuda:{gpu_id}')
                    
                    # Store (gpu_id, reranker) tuple
                    self.rerankers.append((gpu_id, reranker))
                    print(f"✅ Initialized reranker on GPU {gpu_id}")
                    
                except Exception as e:
                    print(f"❌ Failed to initialize reranker on GPU {gpu_id}: {e}")
                    continue
            
            if self.rerankers:
                print(f"🎯 Successfully initialized {len(self.rerankers)} GPU rerankers")
            else:
                print("⚠️ Failed to initialize any GPU rerankers, falling back to CPU")
                self.available_gpus = 0
        
        # Initialize CPU fallback reranker
        if self.available_gpus == 0:
            print("🖥️ Initializing CPU reranker as fallback")
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False, device='cpu')
    
    def extract_titles_from_content(self, content: str) -> List[str]:
        """Extract all titles and headings from web content."""
        titles = []
        
        # Extract main title from the Title field
        title_match = re.search(r'Title: (.+?)\n', content)
        if title_match:
            titles.append(title_match.group(1).strip())
        
        # Extract headings (h1-h6) from markdown content
        heading_patterns = [
            r'^# (.+?)$',           # h1
            r'^## (.+?)$',          # h2
            r'^### (.+?)$',         # h3
            r'^#### (.+?)$',        # h4
            r'^##### (.+?)$',       # h5
            r'^###### (.+?)$',      # h6
        ]
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    title = match.group(1).strip()
                    if title and title not in titles:
                        titles.append(title)
        
        
        return titles
    
    def combine_titles_into_catalog(self, titles: List[str]) -> str:
        """Combine all titles into a catalog-like format."""
        if not titles:
            return ""
        
        # Remove duplicates while preserving order
        unique_titles = []
        seen = set()
        for title in titles:
            if title not in seen:
                unique_titles.append(title)
                seen.add(title)
        
        # Combine into a catalog format
        catalog = ""
        for i, title in enumerate(unique_titles, 1):
            catalog += f"{i}. {title}\n"
        
        return catalog
    
    def _distribute_work_across_gpus(self, num_tasks: int) -> List[List[int]]:
        """Distribute work across GPUs for parallel processing."""
        if not self.rerankers:
            return []
        
        num_gpus = len(self.rerankers)
        
        # Simple round-robin distribution
        gpu_tasks = [[] for _ in range(num_gpus)]
        for task_id in range(num_tasks):
            gpu_idx = task_id % num_gpus
            gpu_tasks[gpu_idx].append(task_id)
        
        return gpu_tasks
    
    def _score_with_gpu_parallel_processing(self, query_url_pairs: List[List[str]]) -> List[float]:
        """Score query-URL pairs using GPU parallel processing."""
        if not self.rerankers:
            return [0.0] * len(query_url_pairs)
        
        total_pairs = len(query_url_pairs)
        num_gpus = len(self.rerankers)
        
        print(f"🚀 GPU PARALLEL PROCESSING: {total_pairs} pairs using {num_gpus} GPUs")
        
        # Distribute work across GPUs
        gpu_task_distribution = self._distribute_work_across_gpus(total_pairs)
        
        all_scores = [0.0] * total_pairs
        
        def process_gpu_batch(gpu_idx: int, task_indices: List[int]) -> List[Tuple[int, float]]:
            """Process a GPU's batch of tasks."""
            if not task_indices:
                return []
            
            gpu_id, reranker = self.rerankers[gpu_idx]
            
            # Get GPU lock to prevent concurrent access
            gpu_lock = get_gpu_lock(gpu_id)
            
            try:
                with gpu_lock:  # Ensure only one process uses this GPU at a time
                    torch.cuda.set_device(gpu_id)
                    
                    # Extract the pairs for this GPU
                    gpu_pairs = [query_url_pairs[i] for i in task_indices]
                    
                    print(f"🎯 GPU {gpu_id} processing {len(gpu_pairs)} pairs")
                    
                    # Process the batch on this GPU
                    gpu_scores = reranker.compute_score(gpu_pairs, normalize=True)
                    
                    # Return (global_index, score) pairs
                    results = []
                    for local_idx, score in enumerate(gpu_scores):
                        global_idx = task_indices[local_idx]
                        results.append((global_idx, float(score)))
                    
                    print(f"✅ GPU {gpu_id} completed {len(gpu_pairs)} pairs")
                    return results
                    
            except Exception as e:
                print(f"❌ Error processing on GPU {gpu_id}: {e}")
                return [(idx, 0.0) for idx in task_indices]
        
        # Process all GPUs in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            # Submit all GPU tasks
            future_to_gpu = {
                executor.submit(process_gpu_batch, gpu_idx, task_indices): gpu_idx
                for gpu_idx, task_indices in enumerate(gpu_task_distribution)
                if task_indices
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_gpu):
                gpu_idx = future_to_gpu[future]
                try:
                    results = future.result()
                    # Map results back to global positions
                    for global_idx, score in results:
                        if global_idx < len(all_scores):
                            all_scores[global_idx] = score
                except Exception as e:
                    print(f"❌ Error collecting results from GPU {gpu_idx}: {e}")
        
        print(f"🎉 GPU parallel processing completed for {total_pairs} pairs")
        return all_scores
    
    def score_urls_against_queries(self, urls: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        """Score all URLs against all queries using GPU parallel processing."""
        if not urls or not queries:
            return {}
        
        print(f"🌐 Scoring {len(urls)} URLs against {len(queries)} queries...")
        
        # Create all query-URL pairs
        query_url_pairs = []
        url_query_mapping = []  # Maps pair index back to (url_idx, query_idx)
        
        for url_idx, url in enumerate(urls):
            for query_idx, query in enumerate(queries):
                query_url_pairs.append([query, url])
                url_query_mapping.append((url_idx, query_idx))
        
        print(f"🎯 Created {len(query_url_pairs)} query-URL pairs for scoring")
        
        # Score all pairs using GPU parallel processing
        if self.available_gpus > 0:
            all_scores = self._score_with_gpu_parallel_processing(query_url_pairs)
        else:
            # CPU fallback
            all_scores = []
            for pair in query_url_pairs:
                try:
                    scores = self.reranker.compute_score([pair], normalize=True)
                    all_scores.append(float(scores[0]))
                except Exception as e:
                    print(f"❌ Error scoring URL: {e}")
                    all_scores.append(0.0)
        
        # Reconstruct results by URL
        url_scores = {}
        for pair_idx, score in enumerate(all_scores):
            url_idx, query_idx = url_query_mapping[pair_idx]
            url = urls[url_idx]
            query = queries[query_idx]
            
            if url not in url_scores:
                url_scores[url] = {}
            url_scores[url][query] = score
        
        print(f"✅ Completed scoring URLs against queries")
        # Normalize the scores while preserving the query-to-score mapping
        for url in url_scores:
            url_scores[url] = _normalize_scores_preserve_mapping(url_scores[url])
        return url_scores
    
    def score_urls_against_claims(self, urls: List[str], claims: List[str]) -> Dict[str, Dict[str, float]]:
        """Score all URLs against all claims using GPU parallel processing."""
        if not urls or not claims:
            return {}
        
        print(f"🌐 Scoring {len(urls)} URLs against {len(claims)} claims...")
        
        # Create all claim-URL pairs
        claim_url_pairs = []
        url_claim_mapping = []  # Maps pair index back to (url_idx, claim_idx)
        
        for url_idx, url in enumerate(urls):
            for claim_idx, claim in enumerate(claims):
                claim_url_pairs.append([claim, url])
                url_claim_mapping.append((url_idx, claim_idx))
        
        print(f"🎯 Created {len(claim_url_pairs)} claim-URL pairs for scoring")
        
        # Score all pairs using GPU parallel processing
        if self.available_gpus > 0:
            all_scores = self._score_with_gpu_parallel_processing(claim_url_pairs)
        else:
            # CPU fallback
            all_scores = []
            for pair in claim_url_pairs:
                try:
                    scores = self.reranker.compute_score([pair], normalize=True)
                    all_scores.append(float(scores[0]))
                except Exception as e:
                    print(f"❌ Error scoring URL: {e}")
                    all_scores.append(0.0)
        
        # Reconstruct results by URL
        url_scores = {}
        for pair_idx, score in enumerate(all_scores):
            url_idx, claim_idx = url_claim_mapping[pair_idx]
            url = urls[url_idx]
            claim = claims[claim_idx]
            
            if url not in url_scores:
                url_scores[url] = {}
            url_scores[url][claim] = score
        
        print(f"✅ Completed scoring URLs against claims")
        # Normalize the scores while preserving the claim-to-score mapping
        for url in url_scores:
            url_scores[url] = _normalize_scores_preserve_mapping(url_scores[url])
        return url_scores
    
    def score_titles_against_queries(self, web_content_cache: Dict[str, str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        """Score all title collections against all queries using GPU parallel processing."""
        if not web_content_cache or not queries:
            return {}
        
        print(f"📝 Scoring title collections against {len(queries)} queries...")
        
        # Extract titles from all web content
        url_titles = {}
        for url, content in web_content_cache.items():
            titles = self.extract_titles_from_content(content)
            if titles:
                title_catalog = self.combine_titles_into_catalog(titles)
                url_titles[url] = title_catalog
        
        if not url_titles:
            print("⚠️ No titles found in web content")
            return {}
        
        print(f"📋 Found title collections for {len(url_titles)} URLs")
        
        # Create all query-title collection pairs
        query_title_pairs = []
        title_query_mapping = []  # Maps pair index back to (url_idx, query_idx)
        
        urls = list(url_titles.keys())
        for url_idx, url in enumerate(urls):
            title_catalog = url_titles[url]
            for query_idx, query in enumerate(queries):
                query_title_pairs.append([query, title_catalog])
                title_query_mapping.append((url_idx, query_idx))
        
        print(f"🎯 Created {len(query_title_pairs)} query-title collection pairs for scoring")
        
        # Score all pairs using GPU parallel processing
        if self.available_gpus > 0:
            all_scores = self._score_with_gpu_parallel_processing(query_title_pairs)
        else:
            # CPU fallback
            all_scores = []
            for pair in query_title_pairs:
                try:
                    scores = self.reranker.compute_score([pair], normalize=True)
                    all_scores.append(float(scores[0]))
                except Exception as e:
                    print(f"❌ Error scoring titles: {e}")
                    all_scores.append(0.0)
        
        # Reconstruct results by URL
        title_scores = {}
        for pair_idx, score in enumerate(all_scores):
            url_idx, query_idx = title_query_mapping[pair_idx]
            url = urls[url_idx]
            query = queries[query_idx]
            
            if url not in title_scores:
                title_scores[url] = {}
            title_scores[url][query] = score
        
        print(f"✅ Completed scoring title collections against queries")
        # Normalize the scores while preserving the query-to-score mapping
        for url in title_scores:
            title_scores[url] = _normalize_scores_preserve_mapping(title_scores[url])
        return title_scores
    
    def score_titles_against_claims(self, web_content_cache: Dict[str, str], claims: List[str]) -> Dict[str, Dict[str, float]]:
        """Score all title collections against all claims using GPU parallel processing."""
        if not web_content_cache or not claims:
            return {}
        
        print(f"📝 Scoring title collections against {len(claims)} claims...")
        
        # Extract titles from all web content
        url_titles = {}
        for url, content in web_content_cache.items():
            titles = self.extract_titles_from_content(content)
            if titles:
                title_catalog = self.combine_titles_into_catalog(titles)
                url_titles[url] = title_catalog
        
        if not url_titles:
            print("⚠️ No titles found in web content")
            return {}
        
        print(f"📋 Found title collections for {len(url_titles)} URLs")
        
        # Create all claim-title collection pairs
        claim_title_pairs = []
        title_claim_mapping = []  # Maps pair index back to (url_idx, claim_idx)
        
        urls = list(url_titles.keys())
        for url_idx, url in enumerate(urls):
            title_catalog = url_titles[url]
            for claim_idx, claim in enumerate(claims):
                claim_title_pairs.append([claim, title_catalog])
                title_claim_mapping.append((url_idx, claim_idx))
        
        print(f"🎯 Created {len(claim_title_pairs)} claim-title collection pairs for scoring")
        
        # Score all pairs using GPU parallel processing
        if self.available_gpus > 0:
            all_scores = self._score_with_gpu_parallel_processing(claim_title_pairs)
        else:
            # CPU fallback
            all_scores = []
            for pair in claim_title_pairs:
                try:
                    scores = self.reranker.compute_score([pair], normalize=True)
                    all_scores.append(float(scores[0]))
                except Exception as e:
                    print(f"❌ Error scoring titles: {e}")
                    all_scores.append(0.0)
        
        # Reconstruct results by URL
        title_scores = {}
        for pair_idx, score in enumerate(all_scores):
            url_idx, claim_idx = title_claim_mapping[pair_idx]
            url = urls[url_idx]
            claim = claims[claim_idx]
            
            if url not in title_scores:
                title_scores[url] = {}
            title_scores[url][claim] = score
        
        print(f"✅ Completed scoring title collections against claims")
        # Normalize the scores while preserving the claim-to-score mapping
        for url in title_scores:
            title_scores[url] = _normalize_scores_preserve_mapping(title_scores[url])
        return title_scores
    
    def compute_all_scores_upfront(self, web_content_cache: Dict[str, str], queries: List[str], claims: List[str]) -> Dict[str, Any]:
        """Compute all URL and title scores against queries and claims upfront."""
        print(f"\n{'='*50}")
        print("COMPUTING ALL URL AND TITLE SCORES UPFRONT")
        print(f"{'='*50}")
        
        # Get URLs from web content cache
        urls = list(web_content_cache.keys())
        print(f"📊 Processing {len(urls)} URLs, {len(queries)} queries, {len(claims)} claims")
        
        # Score URLs against queries
        print(f"\n🌐 Scoring URLs against queries...")
        url_query_scores = self.score_urls_against_queries(urls, queries)
        
        # Score URLs against claims
        print(f"\n🌐 Scoring URLs against claims...")
        url_claim_scores = self.score_urls_against_claims(urls, claims)
        
        # Score title collections against queries
        print(f"\n📝 Scoring title collections against queries...")
        title_query_scores = self.score_titles_against_queries(web_content_cache, queries)
        
        # Score title collections against claims
        print(f"\n📝 Scoring title collections against claims...")
        title_claim_scores = self.score_titles_against_claims(web_content_cache, claims)
        
        # Compile all results
        all_scores = {
            'url_against_query': url_query_scores,
            'url_against_claim': url_claim_scores,
            'titles_against_query': title_query_scores,
            'titles_against_claim': title_claim_scores
        }
        
        print(f"\n✅ Completed all upfront scoring!")
        print(f"📊 Results summary:")
        print(f"  - URL vs Query scores: {len(url_query_scores)} URLs × {len(queries)} queries")
        print(f"  - URL vs Claim scores: {len(url_claim_scores)} URLs × {len(claims)} claims")
        print(f"  - Titles vs Query scores: {len(title_query_scores)} URLs × {len(queries)} queries")
        print(f"  - Titles vs Claim scores: {len(title_claim_scores)} URLs × {len(claims)} claims")
        
        return all_scores
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'rerankers'):
            for gpu_id, reranker in self.rerankers:
                try:
                    torch.cuda.set_device(gpu_id)
                    del reranker
                    torch.cuda.empty_cache()
                except:
                    pass
