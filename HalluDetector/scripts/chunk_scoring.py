#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import asyncio
import os
import logging
import json
from typing import List, Dict, Tuple, Set, Any
from datetime import datetime

from utils import (
    OptimizedContextLocator,
    SingleSentenceContextLocator,
    fetch_pages_async,
    QueryProcessor,
    WeightComputer,
    NumpyEncoder,
    STOP_WORDS
)
from decomposition import get_question_via_LLM

from quality_scoring import OptimizedQueryAwareRanking
from reranker_scoring import BGEReranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntegratedChunkScorer:
    """Top program that orchestrates quality and relevance scoring."""
    
    def __init__(self, 
                 sbert_model: str = 'all-MiniLM-L6-v2',
                 ner_threshold: float = 0.5,
                 c: float = 6.0,
                 num_gpus: int = 4):
        
        self.sbert_model = sbert_model
        self.ner_threshold = ner_threshold
        self.c = c
        self.num_gpus = num_gpus
        
        # Initialize shared components
        self.query_processor = QueryProcessor(sbert_model, ner_threshold)
        self.weight_computer = WeightComputer(c)
        self.sentence_locator = SingleSentenceContextLocator()
        self.chunk_locator = OptimizedContextLocator()
    
    async def score_chunks(self, queries: List[str], urls: List[str], web_content_cache: Dict[str, str], cache_file: str = None, url_mapping: Dict[str, int] = None) -> Dict[str, Any]:
        """Main method to score chunks using both quality and relevance metrics."""
        
        # Convert queries to question format using LLM
        # question_queries = get_question_via_LLM(queries)
        question_queries = queries
        
        # Fetch web content
        web_content = web_content_cache
        documents = list(web_content.values())
        valid_urls = list(web_content.keys())
        
        # Compute collection statistics
        self.weight_computer.compute_collection_stats(documents)
        
        # Extract entity clusters using question queries
        query_clusters = self.query_processor.extract_entity_clusters(question_queries, documents)
        
        # Convert numpy values in query_clusters to Python types for JSON serialization
        for query_idx, cluster_info in query_clusters.items():
            for cluster_key, cluster_data in cluster_info['clusters'].items():
                for similar_entity in cluster_data['similar_entities']:
                    if 'similarity' in similar_entity:
                        similar_entity['similarity'] = float(similar_entity['similarity'])
        
        # Extract all entities and compute entity weights
        all_entities = self.query_processor.extract_all_entities(query_clusters)
        entity_weights = self.weight_computer.compute_entity_weights(all_entities, documents)
        
        # Convert numpy values to Python types for JSON serialization
        entity_weights = {k: float(v) for k, v in entity_weights.items()}
        
        # Expand queries using question queries
        expanded_queries, query_mapping = self.query_processor.expand_queries(question_queries, query_clusters)
        
        # Extract contexts and chunks
        all_contexts = self._extract_contexts_for_quality(query_clusters, valid_urls, documents)
        all_chunks, url_chunk_mapping = self._extract_chunks_for_relevance(valid_urls, documents)
        
        # Prepare shared data for bottom programs
        shared_data = {
            'documents': documents,
            'valid_urls': valid_urls,
            'queries': queries,  # Keep original queries for reference
            'question_queries': question_queries,  # Add question queries
            'expanded_queries': expanded_queries,
            'query_mapping': query_mapping,
            'query_clusters': query_clusters,
            'entity_weights': entity_weights,
            'all_contexts': all_contexts,
            'all_chunks': all_chunks,
            'url_chunk_mapping': url_chunk_mapping
        }
        
        # Call bottom programs
        quality_scores, detailed_context_scores = self._call_quality_scoring(shared_data)
        relevance_scores, detailed_chunk_scores = self._call_relevance_scoring(shared_data)
        
        # Convert scores to Python types for JSON serialization
        quality_scores = {k: float(v) for k, v in quality_scores.items()}
        relevance_scores = {k: float(v) for k, v in relevance_scores.items()}
        
        # Apply aggregation rule: if 1-sentence context appears in 4-sentence chunk, 
        # add quality_score * 1/100 to the chunk's relevance score
        aggregated_chunk_scores = self._apply_aggregation_rule(detailed_context_scores, detailed_chunk_scores)
    
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_queries': len(queries),
                'num_question_queries': len(question_queries),
                'num_expanded_queries': len(expanded_queries),
                'num_documents': len(documents),
                'queries': queries,
                'question_queries': question_queries,
                'expanded_queries': expanded_queries,
                'query_mapping': query_mapping,
                'urls': valid_urls,
                'query_clusters': query_clusters,
                'entity_weights': entity_weights,
                'collection_stats': self.weight_computer.collection_stats,
                'model_info': {
                    'sbert_model': self.query_processor.sbert_model_name,
                    'ner_threshold': self.query_processor.ner_threshold,
                    'c_parameter': self.weight_computer.c,
                    'log_logistic_model': True
                },
                'query_weights': {} # Removed as per edit hint
            },
            'quality_scores': quality_scores,
            'relevance_scores': relevance_scores,
            # 'combined_scores': combined_scores,
            'detailed_context_scores': detailed_context_scores,
            'detailed_chunk_scores': detailed_chunk_scores,
            'aggregated_chunk_scores': aggregated_chunk_scores
        }
    
    def _extract_contexts_for_quality(self, query_clusters: Dict, valid_urls: List[str], documents: List[str]) -> List[Dict]:
        """Extract single-sentence contexts for quality scoring."""
        all_contexts = []
        all_contexts_dict = {}
        
        for query_idx, cluster_info in query_clusters.items():
            for cluster_key, cluster_data in cluster_info['clusters'].items():
                key_query_term = cluster_data['key_query_term']
                similar_entities = cluster_data['similar_entities']
                cluster_entities = [key_query_term] + [e['text'] for e in similar_entities]
                
                for url_idx, (url, document) in enumerate(zip(valid_urls, documents)):
                    contexts = self.sentence_locator.locate_contexts(document, cluster_entities)
                    
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
        
        for context_id, context_info in all_contexts_dict.items():
            if not context_info['processed']:
                context = context_info['context']
                all_contexts.append({
                    'context_id': context['context_id'],
                    'context_text': context['context_text'],
                    'url': context_info['url'],
                    'query_idx': context_info['query_idx'],
                    'cluster_key': context_info['cluster_key'],
                    'entity': context['entity'],
                    'position': context['position']
                })
                context_info['processed'] = True
        
        return all_contexts
    
    def _extract_chunks_for_relevance(self, valid_urls: List[str], documents: List[str]) -> Tuple[Dict[str, Dict], Dict[str, str]]:
        """Extract 4-sentence chunks for relevance scoring."""
        all_chunks = {}
        url_chunk_mapping = {}
        
        for url_idx, (url, document) in enumerate(zip(valid_urls, documents)):
            chunks = self.chunk_locator.extract_sentences(document)
            
            for chunk in chunks:
                chunk_id = chunk['chunk_id']
                all_chunks[chunk_id] = chunk
                url_chunk_mapping[chunk_id] = url
        
        return all_chunks, url_chunk_mapping
    
    def _call_quality_scoring(self, shared_data: Dict) -> Tuple[Dict[int, float], List[Dict]]:
        """Call quality scoring and return scores per original query and detailed context scores."""
        quality_scorer = OptimizedQueryAwareRanking(
            theta=0.5,
            sigma=10.0,
            c=self.c,
            sbert_model=self.sbert_model,
            ner_threshold=self.ner_threshold
        )
        
        # Pass shared data to quality scoring
        quality_scorer.documents = shared_data['documents']
        quality_scorer.query_clusters = shared_data['query_clusters']
        quality_scorer.entity_weights = shared_data['entity_weights']
        quality_scorer.all_contexts = shared_data['all_contexts']
        
        # Get detailed context scores and aggregated scores
        aggregated_scores, detailed_context_scores = quality_scorer._compute_quality_scores_with_details()
        return aggregated_scores, detailed_context_scores
    
    def _call_relevance_scoring(self, shared_data: Dict) -> Tuple[Dict[int, float], List[Dict]]:
        """Call relevance scoring and return scores per original query and detailed chunk scores."""
        relevance_scorer = BGEReranker(
            num_gpus=self.num_gpus,
            sbert_model=self.sbert_model,
            ner_threshold=self.ner_threshold
        )
        
        # Pass shared data to relevance scoring
        relevance_scorer.documents = shared_data['documents']
        relevance_scorer.expanded_queries = shared_data['expanded_queries']
        relevance_scorer.query_mapping = shared_data['query_mapping']
        relevance_scorer.all_chunks = shared_data['all_chunks']
        relevance_scorer.url_chunk_mapping = shared_data['url_chunk_mapping']
        
        # Get detailed chunk scores and aggregated scores
        aggregated_scores, detailed_chunk_scores = relevance_scorer._compute_relevance_scores_with_details()
        return aggregated_scores, detailed_chunk_scores

    def _apply_aggregation_rule(self, detailed_context_scores: List[Dict], detailed_chunk_scores: List[Dict]) -> List[Dict]:
        """Apply aggregation rule: if 1-sentence context appears in 4-sentence chunk, 
        add quality_score * 1/100 to the chunk's relevance score."""
        
        aggregated_chunks = []
        
        for chunk in detailed_chunk_scores:
            chunk_id = chunk['chunk_id']
            chunk_text = chunk['chunk_text']
            url = chunk['url']
            
            # Find the best relevance score across all original queries for this chunk
            best_relevance_score = 0.0
            best_query_idx = 0
            for query_idx, score in chunk['original_query_scores'].items():
                if score > best_relevance_score:
                    best_relevance_score = score
                    best_query_idx = query_idx
            
            # Check if any 1-sentence contexts from quality scoring appear in this 4-sentence chunk
            quality_contribution = 0.0
            matching_contexts = []
            
            for context in detailed_context_scores:
                context_text = context['context_text'].strip()
                context_url = context['url']
                context_query_idx = context['query_idx']
                
                # Check if this context appears in the chunk and is from the same URL
                if (context_url == url and 
                    context_text in chunk_text):
                    
                    quality_score = context['score']
                    quality_contribution += quality_score * 0.01  # 1/100 as per aggregation rule
                    matching_contexts.append({
                        'context_text': context_text,
                        'quality_score': quality_score,
                        'contribution': quality_score * 0.01
                    })
            
            # Calculate combined score: relevance + quality contribution
            combined_score = best_relevance_score + quality_contribution
            
            aggregated_chunks.append({
                'chunk_id': chunk_id,
                'chunk_text': chunk_text,
                'url': url,
                'query_idx': best_query_idx,
                'relevance_score': best_relevance_score,
                'quality_contribution': quality_contribution,
                'combined_score': combined_score,
                'position': chunk.get('position', 0),
                'matching_contexts': matching_contexts,
                'original_query_scores': chunk['original_query_scores'],
                'expanded_query_scores': chunk['expanded_query_scores']
            })
        
        # Sort by combined score in descending order
        aggregated_chunks.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return aggregated_chunks



