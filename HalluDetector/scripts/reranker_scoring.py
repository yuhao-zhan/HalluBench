#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import asyncio
import aiohttp
import os
import json
import math
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
import torch
from FlagEmbedding import FlagReranker
import warnings
from datetime import datetime
from sentence_transformers import SentenceTransformer
import threading

from utils import (
    OptimizedContextLocator, 
    fetch_pages_async,
    SemanticNERClusterer,
    QueryProcessor,
    WeightComputer,
    NumpyEncoder,
    STOP_WORDS
)

warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# GPU locks to prevent multiple processes from using the same GPU
GPU_LOCKS = {}
GPU_LOCKS_LOCK = threading.Lock()

def get_gpu_lock(gpu_id: int) -> threading.Lock:
    """Get or create a lock for a specific GPU to prevent concurrent access."""
    with GPU_LOCKS_LOCK:
        if gpu_id not in GPU_LOCKS:
            GPU_LOCKS[gpu_id] = threading.Lock()
        return GPU_LOCKS[gpu_id]


# ===== MODIFIED reranker_scoring.py =====

class BGEScorer:
    """Handles BGE Reranker scoring operations with memory-efficient GPU processing."""
    
    def __init__(self, num_gpus: int = 4):
        """Initialize BGEScorer with memory-efficient GPU management."""
        self.available_gpus = 0
        self.rerankers = []
        self.reranker = None  # CPU fallback
        
        # CONSERVATIVE batch sizes to prevent OOM
        self.gpu_batch_size = 2000   # REDUCED from 10000 to 2000
        self.query_batch_size = 1000 # REDUCED from 5000 to 1000
        
        if num_gpus > 0 and torch.cuda.is_available():
            self.available_gpus = min(num_gpus, torch.cuda.device_count())
            print(f"🔧 Using {self.available_gpus} GPUs with conservative memory management")
            
            # Initialize reranker on each available GPU with memory checks
            for gpu_id in range(self.available_gpus):
                try:
                    # Check available memory before initialization
                    torch.cuda.set_device(gpu_id)
                    props = torch.cuda.get_device_properties(gpu_id)
                    allocated = torch.cuda.memory_allocated(gpu_id)
                    total = props.total_memory
                    available = total - allocated
                    
                    print(f"🔍 GPU {gpu_id}: {available/1024**3:.2f}GB available of {total/1024**3:.2f}GB total")
                    
                    # Only initialize if enough memory is available (at least 2GB)
                    if available < 2 * 1024**3:  # 2GB minimum
                        print(f"⚠️ Skipping GPU {gpu_id} due to insufficient memory ({available/1024**3:.2f}GB)")
                        continue
                    
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
    
    def set_batch_sizes(self, gpu_batch_size: int = None, query_batch_size: int = None):
        """Set custom batch sizes for memory-efficient processing."""
        if gpu_batch_size is not None:
            # Enforce maximum limits to prevent OOM
            self.gpu_batch_size = min(gpu_batch_size, 3000)  # Hard limit of 3000
            print(f"🎯 GPU batch size set to {self.gpu_batch_size} (limited for memory safety)")
        
        if query_batch_size is not None:
            # Enforce maximum limits to prevent OOM
            self.query_batch_size = min(query_batch_size, 1500)  # Hard limit of 1500
            print(f"🎯 Query batch size set to {self.query_batch_size} (limited for memory safety)")
    
    def _distribute_work_with_memory_awareness(self, num_tasks: int) -> List[List[int]]:
        """Distribute work across GPUs with memory awareness to prevent OOM."""
        if not self.rerankers:
            return []
        
        num_gpus = len(self.rerankers)
        
        # Check current GPU memory usage and distribute accordingly
        gpu_memory_info = []
        for gpu_idx, (gpu_id, reranker) in enumerate(self.rerankers):
            try:
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                available = total - allocated
                gpu_memory_info.append((gpu_idx, gpu_id, available, allocated))
            except Exception as e:
                print(f"⚠️ Could not check GPU {gpu_id} memory: {e}")
                gpu_memory_info.append((gpu_idx, gpu_id, 0, total))
        
        # Sort GPUs by available memory (descending)
        gpu_memory_info.sort(key=lambda x: x[2], reverse=True)
        
        print(f"🎯 GPU Memory Distribution:")
        for gpu_idx, gpu_id, available, allocated in gpu_memory_info:
            print(f"  GPU {gpu_id}: {available/1024**3:.2f}GB available, {allocated/1024**3:.2f}GB used")
        
        # Distribute work based on available memory
        distribution = [[] for _ in range(num_gpus)]
        
        # Calculate work per GPU based on available memory
        total_available = sum(available for _, _, available, _ in gpu_memory_info)
        if total_available > 0:
            for gpu_idx, gpu_id, available, _ in gpu_memory_info:
                # Allocate tasks proportionally to available memory
                gpu_tasks = int((available / total_available) * num_tasks)
                if gpu_tasks > 0:
                    start_idx = len([task for tasks in distribution[:gpu_idx] for task in tasks])
                    end_idx = min(start_idx + gpu_tasks, num_tasks)
                    distribution[gpu_idx] = list(range(start_idx, end_idx))
        else:
            # Fallback: equal distribution
            tasks_per_gpu = num_tasks // num_gpus
            remainder = num_tasks % num_gpus
            
            for gpu_idx in range(num_gpus):
                start_idx = gpu_idx * tasks_per_gpu + min(gpu_idx, remainder)
                end_idx = start_idx + tasks_per_gpu + (1 if gpu_idx < remainder else 0)
                distribution[gpu_idx] = list(range(start_idx, end_idx))
        
        # Print distribution
        print(f"📊 Work Distribution:")
        for gpu_idx, tasks in enumerate(distribution):
            gpu_id = gpu_memory_info[gpu_idx][1] if gpu_idx < len(gpu_memory_info) else "unknown"
            print(f"  GPU {gpu_id}: {len(tasks)} tasks")
        
        return distribution
    
    def _score_with_memory_efficient_gpu_processing(self, query_chunk_pairs: List[List[str]]) -> List[float]:
        """Score query-chunk pairs using memory-efficient GPU processing."""
        if not self.rerankers:
            return [0.0] * len(query_chunk_pairs)
        
        # Import torch at the beginning to avoid UnboundLocalError
        import torch
        
        total_pairs = len(query_chunk_pairs)
        num_gpus = len(self.rerankers)
        
        print(f"🚀 MEMORY-EFFICIENT PROCESSING: {total_pairs} pairs using {num_gpus} GPUs")
        
        # Use smaller batch sizes to prevent OOM
        max_batch_size = min(self.gpu_batch_size, total_pairs // num_gpus + 1)
        print(f"🎯 Using batch size: {max_batch_size}")
        
        # Distribute work with memory awareness
        gpu_task_distribution = self._distribute_work_with_memory_awareness(total_pairs)
        
        all_scores = [0.0] * total_pairs
        
        def process_gpu_batch_with_chunking(gpu_idx: int, task_indices: List[int]) -> List[Tuple[int, float]]:
            """Process a GPU's batch with chunking to prevent OOM."""
            if not task_indices:
                return []
            
            gpu_id, reranker = self.rerankers[gpu_idx]
            
            # Get GPU lock to prevent concurrent access
            gpu_lock = get_gpu_lock(gpu_id)
            
            try:
                with gpu_lock:  # Ensure only one process uses this GPU at a time
                    torch.cuda.set_device(gpu_id)
                    
                    # Process in smaller chunks to prevent OOM
                    chunk_size = min(max_batch_size, 1000)  # Process max 1000 pairs at a time
                    results = []
                    
                    for start_idx in range(0, len(task_indices), chunk_size):
                        end_idx = min(start_idx + chunk_size, len(task_indices))
                        chunk_indices = task_indices[start_idx:end_idx]
                        
                        # Extract pairs for this chunk
                        chunk_pairs = [query_chunk_pairs[i] for i in chunk_indices]
                        
                        print(f"🎯 GPU {gpu_id} processing chunk {start_idx//chunk_size + 1} ({len(chunk_pairs)} pairs)")
                        
                        # Process this chunk
                        chunk_scores = reranker.compute_score(chunk_pairs, normalize=True)
                        
                        # Store results
                        for local_idx, score in enumerate(chunk_scores):
                            global_idx = chunk_indices[local_idx]
                            results.append((global_idx, float(score)))
                        
                        # Memory cleanup after each chunk
                        torch.cuda.empty_cache()
                    
                    print(f"✅ GPU {gpu_id} completed {len(task_indices)} pairs in {len(range(0, len(task_indices), chunk_size))} chunks")
                    return results
                    
            except Exception as e:
                print(f"❌ Error processing on GPU {gpu_id}: {e}")
                return [(idx, 0.0) for idx in task_indices]
        
        # Process all GPUs in parallel
        import concurrent.futures
        
        # Limit the number of workers to prevent too many processes
        max_workers = min(num_gpus, 2)  # Limit to max 2 processes to prevent OOM
        print(f"🔒 Using {max_workers} workers to prevent memory conflicts")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_gpu = {
                executor.submit(process_gpu_batch_with_chunking, gpu_idx, task_indices): gpu_idx
                for gpu_idx, task_indices in enumerate(gpu_task_distribution)
                if task_indices
            }
            
            for future in concurrent.futures.as_completed(future_to_gpu):
                gpu_idx = future_to_gpu[future]
                try:
                    results = future.result()
                    for global_idx, score in results:
                        if global_idx < len(all_scores):
                            all_scores[global_idx] = score
                except Exception as e:
                    print(f"❌ Error collecting results from GPU {gpu_idx}: {e}")
        
        print(f"🎉 Memory-efficient processing completed for {total_pairs} pairs")
        return all_scores
    
    def score_multiple_queries_parallel(self, queries: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Score multiple queries against chunks using memory-efficient streaming processing."""
        if not queries or not chunks:
            return []
        
        # Import torch at the beginning to avoid UnboundLocalError
        import torch
        
        total_queries = len(queries)
        total_chunks = len(chunks)
        total_pairs = total_queries * total_chunks
        
        print(f"🚀 MEMORY-EFFICIENT STREAMING MULTI-QUERY PROCESSING")
        print(f"📊 {total_queries} queries × {total_chunks} chunks = {total_pairs} total pairs")
        
        # Memory check before processing
        if torch.cuda.is_available() and total_pairs > 50000:
            print(f"⚠️ Large workload detected ({total_pairs} pairs). Using streaming processing.")
        
        if self.available_gpus == 0:
            # CPU fallback with streaming
            all_results = []
            for query in queries:
                query_scores = self.score_query_chunk_pairs(query, chunks)
                all_results.append(query_scores)
            return all_results
        
        # STREAMING APPROACH: Process queries in batches to prevent OOM
        batch_size = min(self.query_batch_size, max(1, total_queries // 8))  # Process max 25% of queries at once
        print(f"🎯 Using streaming batch size: {batch_size} queries per batch")
        
        all_results = []
        
        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch_queries = queries[batch_start:batch_end]
            
            print(f"🔄 Processing batch {batch_start//batch_size + 1}: queries {batch_start+1}-{batch_end}")
            
            # Process this batch of queries
            batch_results = self._process_query_batch_streaming(batch_queries, chunks)
            all_results.extend(batch_results)
            
            # Memory cleanup after each batch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"✅ Batch {batch_start//batch_size + 1} completed. Memory cleaned up.")
        
        print(f"🎉 Streaming multi-query processing completed for {total_queries} queries")
        return all_results
    
    def _process_query_batch_streaming(self, batch_queries: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Process a batch of queries using streaming to prevent OOM."""
        if not batch_queries or not chunks:
            return []
        
        batch_size = len(batch_queries)
        total_chunks = len(chunks)
        
        # Create query-chunk pairs for this batch only
        batch_query_chunk_pairs = []
        query_mapping = []
        
        for query_idx, query in enumerate(batch_queries):
            for chunk in chunks:
                batch_query_chunk_pairs.append([query, chunk['chunk_text']])
                query_mapping.append(query_idx)
        
        print(f"🎯 Processing {len(batch_query_chunk_pairs)} pairs for {batch_size} queries")
        
        # Process using memory-efficient GPU processing
        batch_scores = self._score_with_memory_efficient_gpu_processing(batch_query_chunk_pairs)
        
        # Reconstruct results by query for this batch
        batch_results = [{} for _ in range(batch_size)]
        
        for pair_idx, score in enumerate(batch_scores):
            query_idx = query_mapping[pair_idx]
            chunk_id = chunks[pair_idx % total_chunks]['chunk_id']
            batch_results[query_idx][chunk_id] = score
        
        return batch_results

'''
class BGEScorer:
    """Handles BGE Reranker scoring operations with TRUE single-load, single-pass GPU processing."""
    
    def __init__(self, num_gpus: int = 4):
        """Initialize BGEScorer with single-load GPU balancing support."""
        self.available_gpus = 0
        self.rerankers = []
        self.reranker = None  # CPU fallback
        
        # MASSIVE batch sizes for single-pass processing
        self.gpu_batch_size = 10000  # Process 10K pairs per GPU batch
        self.query_batch_size = 5000  # Process 5K queries per batch
        
        if num_gpus > 0 and torch.cuda.is_available():
            self.available_gpus = min(num_gpus, torch.cuda.device_count())
            print(f"Using {self.available_gpus} GPUs for single-pass processing")
            
            # Initialize reranker on each available GPU ONCE
            for gpu_id in range(self.available_gpus):
                try:
                    # Set device before initializing reranker
                    torch.cuda.set_device(gpu_id)
                    
                    # Initialize reranker on this specific GPU ONCE
                    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device=f'cuda:{gpu_id}')
                    
                    # Store (gpu_id, reranker) tuple
                    self.rerankers.append((gpu_id, reranker))
                    print(f"✅ Initialized reranker on GPU {gpu_id} (will be reused)")
                    
                except Exception as e:
                    print(f"❌ Failed to initialize reranker on GPU {gpu_id}: {e}")
                    continue
            
            if self.rerankers:
                print(f"🎯 Successfully initialized {len(self.rerankers)} GPU rerankers for single-pass processing")
            else:
                print("⚠️ Failed to initialize any GPU rerankers, falling back to CPU")
                self.available_gpus = 0
        
        # Initialize CPU fallback reranker
        if self.available_gpus == 0:
            print("🖥️ Initializing CPU reranker as fallback")
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False, device='cpu')
    
    def set_batch_sizes(self, gpu_batch_size: int = None, query_batch_size: int = None):
        """Set custom batch sizes for optimal performance."""
        if gpu_batch_size is not None:
            self.gpu_batch_size = gpu_batch_size
            print(f"🎯 GPU batch size set to {gpu_batch_size}")
        
        if query_batch_size is not None:
            self.query_batch_size = query_batch_size
            print(f"🎯 Query batch size set to {query_batch_size}")
    
    def get_optimal_batch_sizes(self) -> Dict[str, int]:
        """Get optimal batch sizes based on available GPU memory for single-pass processing."""
        if self.available_gpus == 0:
            return {
                'gpu_batch_size': 10000,
                'query_batch_size': 5000,
                'reason': 'CPU-only mode'
            }
        
        # Calculate optimal batch sizes based on GPU memory for single-pass
        total_gpu_memory = 0
        for gpu_id, _ in self.rerankers:
            props = torch.cuda.get_device_properties(gpu_id)
            total_gpu_memory += props.total_memory
        
        total_gpu_memory_gb = total_gpu_memory / (1024**3)
        
        if total_gpu_memory_gb >= 80:  # 4x 20GB+ GPUs
            optimal_gpu_batch = 20000
            optimal_query_batch = 10000
            reason = f"High memory setup ({total_gpu_memory_gb:.1f}GB total) - single-pass optimized"
        elif total_gpu_memory_gb >= 60:  # 4x 15GB+ GPUs
            optimal_gpu_batch = 15000
            optimal_query_batch = 7500
            reason = f"Medium-high memory setup ({total_gpu_memory_gb:.1f}GB total) - single-pass optimized"
        elif total_gpu_memory_gb >= 40:  # 4x 10GB+ GPUs
            optimal_gpu_batch = 10000
            optimal_query_batch = 5000
            reason = f"Medium memory setup ({total_gpu_memory_gb:.1f}GB total) - single-pass optimized"
        else:  # Lower memory GPUs
            optimal_gpu_batch = 5000
            optimal_query_batch = 2500
            reason = f"Lower memory setup ({total_gpu_memory_gb:.1f}GB total) - single-pass optimized"
        
        return {
            'gpu_batch_size': optimal_gpu_batch,
            'query_batch_size': optimal_query_batch,
            'reason': reason
        }
    
    def _print_gpu_memory_status(self):
        """Print current GPU memory status for all available GPUs."""
        if self.available_gpus == 0:
            return
        
        print("\n=== GPU Memory Status ===")
        for gpu_id in range(self.available_gpus):
            try:
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                available = total - allocated
                print(f"GPU {gpu_id}: {allocated/1024**3:.2f}GB used, {available/1024**3:.2f}GB available, {total/1024**3:.2f}GB total")
            except Exception as e:
                print(f"Error checking GPU {gpu_id} memory: {e}")
        print("========================\n")
    
    def _get_available_gpu_memory(self) -> List[Tuple[int, float]]:
        """Get available memory for each GPU in descending order."""
        if self.available_gpus == 0:
            return []
        
        gpu_memory = []
        for gpu_id in range(self.available_gpus):
            try:
                torch.cuda.set_device(gpu_id)
                allocated = torch.cuda.memory_allocated(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                available = total - allocated
                gpu_memory.append((gpu_id, available))
            except Exception as e:
                print(f"Error checking GPU {gpu_id} memory: {e}")
                continue
        
        # Sort by available memory (descending)
        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        return gpu_memory
    
    def _distribute_work_evenly_across_gpus(self, num_tasks: int) -> List[List[int]]:
        """Distribute tasks evenly across GPUs for balanced workload."""
        if self.available_gpus == 0:
            return [list(range(num_tasks))]
        
        # Simple round-robin distribution for even workload
        gpu_tasks = [[] for _ in range(self.available_gpus)]
        
        for task_id in range(num_tasks):
            gpu_idx = task_id % self.available_gpus
            gpu_tasks[gpu_idx].append(task_id)
        
        return gpu_tasks
    
    def score_query_chunk_pairs(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score all chunks against a query using BGE Reranker with single-pass GPU processing."""
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        
        # Create query-chunk pairs as list of lists
        query_chunk_pairs = [[query, chunk['chunk_text']] for chunk in chunks]
        
        if self.available_gpus == 0:
            # Use CPU reranker
            scores = self.reranker.compute_score(query_chunk_pairs, normalize=True)
        else:
            # Use GPU rerankers with single-pass processing
            scores = self._score_with_single_pass_gpu_processing(query_chunk_pairs)
        
        # Map scores back to chunk IDs
        chunk_scores = {}
        for i, score in enumerate(scores):
            chunk_scores[chunk_ids[i]] = float(score)
        
        return chunk_scores
    
    def _score_with_single_pass_gpu_processing(self, query_chunk_pairs: List[List[str]]) -> List[float]:
        """Score query-chunk pairs using multiple GPUs with TRUE single-pass processing."""
        if not self.rerankers:
            return [0.0] * len(query_chunk_pairs)
        
        total_pairs = len(query_chunk_pairs)
        num_gpus = len(self.rerankers)
        
        print(f"🚀 SINGLE-PASS PROCESSING: {total_pairs} pairs using {num_gpus} GPUs")
        print(f"🎯 Each GPU will process ~{total_pairs // num_gpus} pairs in ONE batch")
        
        # Distribute work evenly across GPUs
        gpu_task_distribution = self._distribute_work_evenly_across_gpus(total_pairs)
        
        all_scores = [0.0] * total_pairs
        
        # Process each GPU's workload in parallel (no device switching within GPU)
        import concurrent.futures
        
        def process_gpu_batch(gpu_idx: int, task_indices: List[int]) -> List[Tuple[int, float]]:
            """Process a GPU's batch of tasks."""
            if not task_indices:
                return []
            
            gpu_id, reranker = self.rerankers[gpu_idx]
            
            try:
                # Set device ONCE for this GPU
                torch.cuda.set_device(gpu_id)
                
                # Extract the pairs for this GPU
                gpu_pairs = [query_chunk_pairs[i] for i in task_indices]
                
                print(f"🎯 GPU {gpu_id} processing {len(gpu_pairs)} pairs (indices {task_indices[0]}-{task_indices[-1]})")
                
                # Process the entire batch on this GPU ONCE
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
                # Return zeros for failed processing
                return [(idx, 0.0) for idx in task_indices]
        
        # Process all GPUs in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            # Submit all GPU tasks
            future_to_gpu = {
                executor.submit(process_gpu_batch, gpu_idx, task_indices): gpu_idx
                for gpu_idx, task_indices in enumerate(gpu_task_distribution)
                if task_indices  # Only submit non-empty tasks
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
        
        print(f"🎉 Single-pass processing completed for {total_pairs} pairs")
        return all_scores
    
    def score_multiple_queries_parallel(self, queries: List[str], chunks: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Score multiple queries against chunks using TRUE single-pass GPU processing."""
        if not queries or not chunks:
            return []
        
        total_queries = len(queries)
        total_chunks = len(chunks)
        
        print(f"🚀 SINGLE-PASS MULTI-QUERY PROCESSING")
        print(f"📊 {total_queries} queries × {total_chunks} chunks = {total_queries * total_chunks} total pairs")
        
        if self.available_gpus == 0:
            # CPU fallback
            all_results = []
            for query in queries:
                query_scores = self.score_query_chunk_pairs(query, chunks)
                all_results.append(query_scores)
            return all_results
        
        # Create ALL query-chunk pairs at once for single-pass processing
        all_query_chunk_pairs = []
        query_mapping = []  # Maps pair index back to query index
        
        for query_idx, query in enumerate(queries):
            for chunk in chunks:
                all_query_chunk_pairs.append([query, chunk['chunk_text']])
                query_mapping.append(query_idx)
        
        print(f"🎯 Created {len(all_query_chunk_pairs)} total query-chunk pairs for single-pass processing")
        
        # Process ALL pairs using single-pass GPU processing
        all_scores = self._score_with_single_pass_gpu_processing(all_query_chunk_pairs)
        
        # Reconstruct results by query
        all_results = [{} for _ in range(total_queries)]
        
        for pair_idx, score in enumerate(all_scores):
            query_idx = query_mapping[pair_idx]
            chunk_id = chunks[pair_idx % total_chunks]['chunk_id']
            all_results[query_idx][chunk_id] = score
        
        print(f"🎉 Multi-query single-pass processing completed")
        return all_results
    
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
'''



class QueryExpander:
    """Handles query expansion using entity clusters."""
    
    def __init__(self, query_processor: QueryProcessor):
        self.query_processor = query_processor
    
    def expand_queries_with_entities(self, queries: List[str], query_clusters: Dict[int, Dict[str, Any]]) -> Tuple[List[str], Dict[int, int]]:
        """Expand queries using similar entities from clusters."""
        return self.query_processor.expand_queries(queries, query_clusters)


class BGEReranker:
    """Neural ranking system using BGE Reranker for scoring query-sentence pairs with query expansion."""
    
    def __init__(self, num_gpus: int = 4, sbert_model: str = 'all-MiniLM-L6-v2', ner_threshold: float = 0.5):
        print(f"Initializing BGE Reranker with query expansion...")
        
        # Initialize components
        self.query_processor = QueryProcessor(sbert_model, ner_threshold)
        self.weight_computer = WeightComputer()
        self.context_locator = OptimizedContextLocator()
        self.bge_scorer = BGEScorer(num_gpus)
        self.query_expander = QueryExpander(self.query_processor)
    
    async def fetch_and_rank(self, queries: List[str], urls: List[str]) -> Dict[str, Any]:
        """Main method: fetch URLs and rank sentences using BGE Reranker with query expansion."""
        
        # Fetch web content
        web_content = await fetch_pages_async(urls)
        documents = list(web_content.values())
        valid_urls = list(web_content.keys())
        
        # Compute collection statistics for weight computation
        self.weight_computer.compute_collection_stats(documents)
        
        # Extract entity clusters for query expansion
        query_clusters = self.query_processor.extract_entity_clusters(queries, documents)
        
        # Extract all entities and compute entity weights
        all_entities = self.query_processor.extract_all_entities(query_clusters)
        entity_weights = self.weight_computer.compute_entity_weights(all_entities, documents)
        
        # Expand queries using similar entities
        expanded_queries, query_mapping = self.query_expander.expand_queries_with_entities(queries, query_clusters)
        
        # Extract chunks from all documents
        all_chunks, url_chunk_mapping = self._extract_all_chunks(valid_urls, documents)
        
        # Score each chunk against each expanded query
        chunk_query_scores = self._score_all_chunks(expanded_queries, all_chunks)
        
        # Organize results by expanded query
        expanded_query_results = {}
        for chunk_id, query_scores in chunk_query_scores.items():
            chunk_info = all_chunks[chunk_id]
            chunk_url = url_chunk_mapping[chunk_id]
            
            for query_idx, score in query_scores.items():
                if query_idx not in expanded_query_results:
                    expanded_query_results[query_idx] = {
                        'query': expanded_queries[query_idx],
                        'original_query_idx': query_mapping[query_idx],
                        'chunks': []
                    }
                
                expanded_query_results[query_idx]['chunks'].append({
                    'chunk_id': chunk_id,
                    'chunk_text': chunk_info['chunk_text'],
                    'url': chunk_url,
                    'score': score,
                    'position': chunk_info['position'],
                    'length': chunk_info['length'],
                    'sentence_count': chunk_info['sentence_count'],
                    'sentence_indices': chunk_info['sentence_indices']
                })
        
        # Sort chunks within each expanded query by score
        for query_idx in expanded_query_results:
            expanded_query_results[query_idx]['chunks'].sort(key=lambda x: x['score'], reverse=True)
        
        # Prepare results
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_original_queries': len(queries),
                'num_expanded_queries': len(expanded_queries),
                'num_documents': len(documents),
                'num_chunks': len(all_chunks),
                'original_queries': queries,
                'expanded_queries': expanded_queries,
                'urls': valid_urls,
                'query_clusters': query_clusters,
                'entity_weights': entity_weights,
                'collection_stats': self.weight_computer.collection_stats,
                'model_info': {
                    'bge_reranker_model': 'BAAI/bge-reranker-v2-m3',
                    'sbert_model': self.query_processor.sbert_model_name,
                    'ner_threshold': self.query_processor.ner_threshold,
                    'device': 'GPU' if self.bge_scorer.available_gpus > 0 else 'CPU',
                    'weight_computation': {
                        'c_parameter': self.weight_computer.c,
                        'log_logistic_model': True
                    }
                }
            },
            'expanded_query_results': expanded_query_results
        }
        
        return results
    
    def _extract_all_chunks(self, valid_urls: List[str], documents: List[str]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """Extract chunks from all documents."""
        all_chunks = {}
        url_chunk_mapping = {}
        
        for url_idx, (url, document) in enumerate(zip(valid_urls, documents)):
            chunks = self.context_locator.extract_sentences(document)
            
            # Store chunks with URL information
            for chunk in chunks:
                chunk_id = chunk['chunk_id']
                all_chunks[chunk_id] = chunk
                url_chunk_mapping[chunk_id] = url
        
        return all_chunks, url_chunk_mapping
    
    def _score_all_chunks(self, expanded_queries: List[str], all_chunks: Dict[str, Dict]) -> Dict[str, Dict[int, float]]:
        """Score all chunks against all expanded queries."""
        chunk_query_scores = {}
        
        for query_idx, query in enumerate(expanded_queries):
            # Get all chunks
            chunks_list = list(all_chunks.values())
            
            # Score chunks against this query
            query_scores = self.bge_scorer.score_query_chunk_pairs(query, chunks_list)
            
            # Store scores
            for chunk_id, score in query_scores.items():
                if chunk_id not in chunk_query_scores:
                    chunk_query_scores[chunk_id] = {}
                chunk_query_scores[chunk_id][query_idx] = score
        
        return chunk_query_scores
