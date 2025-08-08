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


class BGEScorer:
    """Handles BGE Reranker scoring operations."""
    
    def __init__(self, num_gpus: int = 4):
        self.num_gpus = num_gpus
        self.available_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
        
        if self.available_gpus == 0:
            print("No GPUs available, using CPU")
            self.available_gpus = 0
        
        print(f"Using {'GPU' if self.available_gpus > 0 else 'CPU'} for processing")
        
        # Initialize reranker on main GPU if available
        if self.available_gpus > 0:
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        else:
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)
    
    def score_query_chunk_pairs(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score all chunks against a query using BGE Reranker."""
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        
        # Create query-chunk pairs as list of lists
        query_chunk_pairs = [[query, chunk['chunk_text']] for chunk in chunks]
        
        # Score all pairs using BGE Reranker
        scores = self.reranker.compute_score(query_chunk_pairs, normalize=True)
        
        # Map scores back to chunk IDs
        chunk_scores = {}
        for i, score in enumerate(scores):
            chunk_scores[chunk_ids[i]] = float(score)
        
        return chunk_scores


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

    def _compute_relevance_scores_per_original_query(self) -> Dict[int, float]:
        """Compute relevance scores per original query using shared data."""
        # Score each chunk against each expanded query
        chunk_query_scores = {}
        for query_idx, query in enumerate(self.expanded_queries):
            chunks_list = list(self.all_chunks.values())
            query_scores = self.bge_scorer.score_query_chunk_pairs(query, chunks_list)
            
            for chunk_id, score in query_scores.items():
                if chunk_id not in chunk_query_scores:
                    chunk_query_scores[chunk_id] = {}
                chunk_query_scores[chunk_id][query_idx] = score
        
        # Aggregate scores per original query
        original_query_scores = {}
        for chunk_id, query_scores in chunk_query_scores.items():
            # Group scores by original query
            original_query_grouped_scores = {}
            for query_idx, score in query_scores.items():
                original_query_idx = self.query_mapping[query_idx]
                if original_query_idx not in original_query_grouped_scores:
                    original_query_grouped_scores[original_query_idx] = []
                original_query_grouped_scores[original_query_idx].append(score)
            
            # Compute average score per original query for this chunk
            for original_query_idx, scores in original_query_grouped_scores.items():
                if original_query_idx not in original_query_scores:
                    original_query_scores[original_query_idx] = []
                original_query_scores[original_query_idx].append(sum(scores) / len(scores))
        
        # Compute final average score per original query
        final_scores = {}
        for original_query_idx, scores in original_query_scores.items():
            if scores:
                final_scores[original_query_idx] = sum(scores) / len(scores)
            else:
                final_scores[original_query_idx] = 0.0
        
        return final_scores

    def _compute_relevance_scores_with_details(self) -> Tuple[Dict[int, float], List[Dict]]:
        """Compute relevance scores with detailed chunk information."""
        # Score each chunk against each expanded query
        chunk_query_scores = {}
        for query_idx, query in enumerate(self.expanded_queries):
            chunks_list = list(self.all_chunks.values())
            query_scores = self.bge_scorer.score_query_chunk_pairs(query, chunks_list)
            
            for chunk_id, score in query_scores.items():
                if chunk_id not in chunk_query_scores:
                    chunk_query_scores[chunk_id] = {}
                chunk_query_scores[chunk_id][query_idx] = score
        
        # Create detailed chunk scores
        detailed_chunk_scores = []
        for chunk_id, query_scores in chunk_query_scores.items():
            chunk_info = self.all_chunks[chunk_id]
            url = self.url_chunk_mapping[chunk_id]
            
            # Group scores by original query
            original_query_grouped_scores = {}
            for query_idx, score in query_scores.items():
                original_query_idx = self.query_mapping[query_idx]
                if original_query_idx not in original_query_grouped_scores:
                    original_query_grouped_scores[original_query_idx] = []
                original_query_grouped_scores[original_query_idx].append(score)
            
            # Compute average score per original query for this chunk
            chunk_original_query_scores = {}
            for original_query_idx, scores in original_query_grouped_scores.items():
                chunk_original_query_scores[original_query_idx] = sum(scores) / len(scores)
            
            detailed_chunk_scores.append({
                'chunk_id': chunk_id,
                'chunk_text': chunk_info['chunk_text'],
                'url': url,
                'position': chunk_info.get('position', 0),
                'original_query_scores': {k: float(v) for k, v in chunk_original_query_scores.items()},
                'expanded_query_scores': {k: float(v) for k, v in query_scores.items()},
                'avg_score': float(sum(chunk_original_query_scores.values()) / len(chunk_original_query_scores)) if chunk_original_query_scores else 0.0
            })
        
        # Aggregate scores per original query
        original_query_scores = {}
        for chunk_id, query_scores in chunk_query_scores.items():
            # Group scores by original query
            original_query_grouped_scores = {}
            for query_idx, score in query_scores.items():
                original_query_idx = self.query_mapping[query_idx]
                if original_query_idx not in original_query_grouped_scores:
                    original_query_grouped_scores[original_query_idx] = []
                original_query_grouped_scores[original_query_idx].append(score)
            
            # Compute average score per original query for this chunk
            for original_query_idx, scores in original_query_grouped_scores.items():
                if original_query_idx not in original_query_scores:
                    original_query_scores[original_query_idx] = []
                original_query_scores[original_query_idx].append(sum(scores) / len(scores))
        
        # Compute final average score per original query
        final_scores = {}
        for original_query_idx, scores in original_query_scores.items():
            if scores:
                final_scores[original_query_idx] = sum(scores) / len(scores)
            else:
                final_scores[original_query_idx] = 0.0
        
        return final_scores, detailed_chunk_scores
