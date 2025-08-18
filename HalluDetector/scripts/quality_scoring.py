#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import asyncio
import aiohttp
import os
import logging
import json
import math
from typing import List, Dict, Tuple, Set, Any
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from utils import (
    SingleSentenceContextLocator, 
    fetch_pages_async, 
    SemanticNERClusterer,
    QueryProcessor,
    WeightComputer,
    NumpyEncoder,
    STOP_WORDS
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContextCollector:
    """Handles collecting scores for identical single-sentence contexts."""
    
    def collect_context_scores(self, contexts: List[Dict]) -> List[Dict]:
        """Collect all scores for identical contexts without text merging."""
        if len(contexts) <= 1:
            return contexts
        
        # Group contexts by their text content (since they're single sentences)
        context_groups = {}
        
        for context in contexts:
            context_text = context['context_text'].strip().lower()
            context_id = context['context_id']
            
            # Use normalized text as key for grouping identical contexts
            if context_text not in context_groups:
                context_groups[context_text] = {
                    'base_context': context,
                    'all_scores': [],
                    'all_entities': set(),
                    'entity_cluster_query_mappings': {}
                }
            
            # Collect score information
            entity = context['entity']
            query_idx = context['query_idx']
            cluster_key = context.get('cluster_key', 'unknown')
            score = context['score']
            
            context_groups[context_text]['all_scores'].append({
                'entity': entity,
                'query_idx': query_idx,
                'cluster_key': cluster_key,
                'score': score
            })
            
            context_groups[context_text]['all_entities'].add(entity)
            
            # Store entity mapping (take max score for same entity)
            if entity not in context_groups[context_text]['entity_cluster_query_mappings']:
                context_groups[context_text]['entity_cluster_query_mappings'][entity] = (cluster_key, query_idx, score)
            else:
                existing_score = context_groups[context_text]['entity_cluster_query_mappings'][entity][2]
                if score > existing_score:
                    context_groups[context_text]['entity_cluster_query_mappings'][entity] = (cluster_key, query_idx, score)
        
        # Create collected contexts
        collected_contexts = []
        for context_text, group_info in context_groups.items():
            base_context = group_info['base_context'].copy()
            
            # If multiple scores were collected for this context, mark it as collected
            if len(group_info['all_scores']) > 1:
                base_context.update({
                    'context_id': f"collected_{hash(context_text)}",
                    'collected_from': len(group_info['all_scores']),
                    'all_entities': list(group_info['all_entities']),
                    'entity_cluster_query_mappings': group_info['entity_cluster_query_mappings']
                })
            
            collected_contexts.append(base_context)
        
        return collected_contexts


class ContextScorer:
    """Handles scoring of contexts based on query-aware ranking algorithm."""
    
    def __init__(self, query_processor: QueryProcessor, weight_computer: WeightComputer, 
                 theta: float = 0.5, sigma: float = 10.0):
        self.query_processor = query_processor
        self.weight_computer = weight_computer
        self.theta = theta
        self.sigma = sigma
    
    def score_context(self, context: Dict, query_clusters: Dict[int, Dict[str, Any]], 
                     entity_weights: Dict[str, float], documents: List[str]) -> float:
        """Score a single context against all clusters in its query."""
        context_words = self.query_processor._preprocess_text(context['context_text'])
        center_entity = context['entity']
        query_idx = context['query_idx']
        
        # Find all clusters in the same query as the center entity
        current_query_clusters = query_clusters[query_idx]['clusters']
        total_score = 0.0
        
        # For each cluster in the same query, compute max score
        for cluster_key, cluster_data in current_query_clusters.items():
            cluster_score = self._compute_cluster_score(
                cluster_data, context_words, center_entity, documents
            )
            total_score += cluster_score
        
        return total_score
    
    def _compute_cluster_score(self, cluster_data: Dict, context_words: List[str], 
                              center_entity: str, documents: List[str]) -> float:
        """Compute max score for a cluster."""
        # Include both key_query_term and similar_entities
        cluster_entity_texts = [cluster_data['key_query_term']] + [
            e['text'] for e in cluster_data['similar_entities']
        ]
        key_query_term = cluster_data['key_query_term']
        
        cluster_max_score = 0.0
        for entity in cluster_entity_texts:
            entity_score = self._compute_entity_score(
                entity, context_words, center_entity, documents
            )
            
            # Multiply by similarity to key query term
            if entity != key_query_term:
                # For similar entities, multiply by their similarity to the key query term
                entity_similarity_to_key = self.query_processor._compute_similarity(entity, key_query_term)
                entity_score *= entity_similarity_to_key
            
            # Take maximum score for this cluster
            cluster_max_score = max(cluster_max_score, entity_score)
        
        return cluster_max_score
    
    def _compute_entity_score(self, entity: str, context_words: List[str], 
                             center_entity: str, documents: List[str]) -> float:
        """Compute score for a single entity (Equation 5)."""
        # Compute context similarity (Equation 4)
        sim_context = 0.0
        for word in context_words:
            sim = self.query_processor._compute_similarity(entity, word)
            if sim > self.theta:  # Threshold filter
                sim_context += sim
        
        # Compute lambda
        lambda_qj = self.weight_computer.compute_lambda(entity, documents)
        if lambda_qj <= 0:
            return 0.0
        
        # Compute dissimilarity (Equation 7)
        dis = 2.0 - self.query_processor._compute_similarity(center_entity, entity)
        
        # Compute score for this entity (Equation 5)
        entity_score = math.log((sim_context + lambda_qj) / lambda_qj) * dis
        
        return entity_score


class OptimizedQueryAwareRanking:
    """Streamlined implementation of query-aware context ranking."""
    
    def __init__(self, 
                 theta: float = 0.5,
                 sigma: float = 10.0,
                 c: float = 6.0,
                 sbert_model: str = 'all-MiniLM-L6-v2',
                 ner_threshold: float = 0.5):
        
        self.theta = theta
        self.sigma = sigma
        
        # Initialize components
        self.query_processor = QueryProcessor(sbert_model, ner_threshold)
        self.weight_computer = WeightComputer(c)
        self.context_locator = SingleSentenceContextLocator()
        self.context_collector = ContextCollector()
        self.context_scorer = ContextScorer(self.query_processor, self.weight_computer, theta, sigma)
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score (Equation 11)."""
        return score / (score + self.sigma)
    
    async def fetch_and_rank(self, queries: List[str], urls: List[str]) -> Dict[str, Any]:
        """Main method: fetch URLs and rank contexts."""
        print(f"Processing {len(queries)} queries and {len(urls)} URLs")
        
        # Fetch web content
        web_content = await fetch_pages_async(urls)
        documents = list(web_content.values())
        valid_urls = list(web_content.keys())
        
        # Compute collection stats
        self.weight_computer.compute_collection_stats(documents)
        
        # Extract entity clusters
        query_clusters = self.query_processor.extract_entity_clusters(queries, documents)
        
        # Extract all entities and compute entity weights
        print("Computing entity weights...")
        all_entities = self.query_processor.extract_all_entities(query_clusters)
        entity_weights = self.weight_computer.compute_entity_weights(all_entities, documents)
        print(f"Computed weights for {len(entity_weights)} entities")
        
        # Process contexts for each query
        all_contexts = self._process_all_contexts(
            query_clusters, valid_urls, documents, entity_weights
        )
        
        # Collect scores for identical contexts
        print("Collecting scores for identical contexts...")
        collected_contexts = self.context_collector.collect_context_scores(all_contexts)
        
        # Organize results by query
        query_results = {}
        for context in collected_contexts:
            query_idx = context['query_idx']
            if query_idx not in query_results:
                query_results[query_idx] = {
                    'query': queries[query_idx],
                    'contexts': []
                }
            query_results[query_idx]['contexts'].append({
                'context_id': context['context_id'],
                'context_text': context['context_text'],
                'url': context['url'],
                'score': context['score'],
                'entity': context['entity'],
                'cluster_key': context.get('cluster_key', 'unknown')
            })
        
        # Sort contexts within each query by score
        for query_idx in query_results:
            query_results[query_idx]['contexts'].sort(key=lambda x: x['score'], reverse=True)
        
        # Prepare results
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_queries': len(queries),
                'num_documents': len(documents),
                'num_contexts': len(collected_contexts),
                'queries': queries,
                'urls': valid_urls,
                'entity_weights': entity_weights,
                'query_clusters': query_clusters
            },
            'query_results': query_results
        }
        
        return results
    
    def _process_all_contexts(self, query_clusters: Dict, valid_urls: List[str], 
                             documents: List[str], entity_weights: Dict[str, float]) -> List[Dict]:
        """Process contexts for all queries and documents."""
        all_contexts = []
        all_contexts_dict = {}  # Dictionary to store contexts with their URL, query_idx, and processed status
        
        for query_idx, cluster_info in query_clusters.items():
            query = cluster_info['query']
            print(f"Processing query {query_idx}: '{query}'")
            
            # Process each cluster separately
            for cluster_key, cluster_data in cluster_info['clusters'].items():
                key_query_term = cluster_data['key_query_term']
                similar_entities = cluster_data['similar_entities']
                
                # All entities in this cluster (key term + similar entities)
                cluster_entities = [key_query_term] + [e['text'] for e in similar_entities]
                print(f"  Processing cluster '{cluster_key}' with {len(cluster_entities)} entities")
                
                # Locate contexts for each document using any entity from this cluster
                for url_idx, (url, document) in enumerate(zip(valid_urls, documents)):
                    contexts = self.context_locator.locate_contexts(document, cluster_entities)
                    
                    # Store contexts for later processing (to avoid duplicates)
                    for context in contexts:
                        context_id = context['context_id']
                        if context_id not in all_contexts_dict:
                            all_contexts_dict[context_id] = {
                                'context': context,
                                'url': url,
                                'query_idx': query_idx,
                                'cluster_key': cluster_key,
                                'processed': False
                            }
        
        # Now process each unique context with all clusters from its query
        for context_id, context_info in all_contexts_dict.items():
            if context_info['processed']:
                continue
                
            context = context_info['context']
            url = context_info['url']
            query_idx = context_info['query_idx']
            stored_cluster_key = context_info['cluster_key']
            
            center_entity = context['entity']
            
            # Score this context against all clusters in its query
            final_score = self.context_scorer.score_context(
                {'context_text': context['context_text'], 'entity': center_entity, 'query_idx': query_idx},
                query_clusters, entity_weights, documents
            )
            
            # Add to all_contexts
            context_info_final = {
                'context_id': context['context_id'],
                'context_text': context['context_text'],
                'url': url,
                'query_idx': query_idx,
                'cluster_key': stored_cluster_key,
                'entity': center_entity,
                'score': final_score,
                'entity_weight': entity_weights.get(center_entity, 0.0)
            }
            all_contexts.append(context_info_final)
            
            # Mark as processed
            all_contexts_dict[context_id]['processed'] = True
        
        return all_contexts

    def _compute_quality_scores_per_original_query(self) -> Dict[int, float]:
        """Compute quality scores per original query using shared data."""
        # Score each context
        scored_contexts = []
        for context in self.all_contexts:
            score = self.context_scorer.score_context(
                context, 
                self.query_clusters,
                self.entity_weights,
                self.documents
            )
            
            context_with_score = context.copy()
            context_with_score['score'] = score
            context_with_score['entity_weight'] = self.entity_weights.get(context['entity'], 0.0)
            scored_contexts.append(context_with_score)
        
        # Collect scores for identical contexts
        collected_contexts = self.context_collector.collect_context_scores(scored_contexts)
        
        # Aggregate scores per original query
        query_scores = {}
        for context in collected_contexts:
            query_idx = context['query_idx']
            score = context['score']
            
            if query_idx not in query_scores:
                query_scores[query_idx] = []
            query_scores[query_idx].append(score)
        
        # Compute average score per original query
        original_query_scores = {}
        for query_idx, scores in query_scores.items():
            if scores:
                original_query_scores[query_idx] = sum(scores) / len(scores)
            else:
                original_query_scores[query_idx] = 0.0
        
        return original_query_scores

    def _compute_quality_scores_with_details(self) -> Tuple[Dict[int, float], List[Dict]]:
        """Compute quality scores with detailed context information."""
        # Score each context
        scored_contexts = []
        for context in self.all_contexts:
            score = self.context_scorer.score_context(
                context, 
                self.query_clusters,
                self.entity_weights,
                self.documents
            )
            
            context_with_score = context.copy()
            context_with_score['score'] = score
            context_with_score['entity_weight'] = self.entity_weights.get(context['entity'], 0.0)
            scored_contexts.append(context_with_score)
        
        # Collect scores for identical contexts
        collected_contexts = self.context_collector.collect_context_scores(scored_contexts)
        
        # Convert numpy values to Python types for JSON serialization
        detailed_context_scores = []
        for context in collected_contexts:
            detailed_context_scores.append({
                'context_id': context['context_id'],
                'context_text': context['context_text'],
                'url': context['url'],
                'query_idx': context['query_idx'],
                'cluster_key': context['cluster_key'],
                'entity': context['entity'],
                'score': float(context['score']),
                'entity_weight': float(context['entity_weight']),
                'position': context.get('position', 0)
            })
        
        # Aggregate scores per original query
        query_scores = {}
        for context in collected_contexts:
            query_idx = context['query_idx']
            score = context['score']
            
            if query_idx not in query_scores:
                query_scores[query_idx] = []
            query_scores[query_idx].append(score)
        
        # Compute average score per original query
        original_query_scores = {}
        for query_idx, scores in query_scores.items():
            if scores:
                original_query_scores[query_idx] = sum(scores) / len(scores)
            else:
                original_query_scores[query_idx] = 0.0
        
        return original_query_scores, detailed_context_scores
