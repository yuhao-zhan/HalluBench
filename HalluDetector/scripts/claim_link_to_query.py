#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import os
import logging
import torch
from FlagEmbedding import FlagReranker
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Any

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable progress bars and verbose output
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ClaimQueryReranker:
    """Reranker for computing scores between claims and queries."""
    
    def __init__(self, num_gpus: int = 4):
        print(f"Initializing Claim-Query Reranker...")
        self.num_gpus = num_gpus
        self.available_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
        
        if self.available_gpus == 0:
            print("No GPUs available, using CPU")
            self.available_gpus = 0
        
        print(f"Using {'GPU' if self.available_gpus > 0 else 'CPU'} for processing")
        
        # Initialize reranker
        if self.available_gpus > 0:
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        else:
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags and extra whitespace."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove noise characters and patterns
        text = re.sub(r'=+\s*', ' ', text)  # Remove repeated equals signs
        text = re.sub(r'-+\s*', ' ', text)   # Remove repeated dashes
        text = re.sub(r'_+\s*', ' ', text)   # Remove repeated underscores
        text = re.sub(r'\*+\s*', ' ', text)  # Remove repeated asterisks
        text = re.sub(r'\++\s*', ' ', text)  # Remove repeated plus signs
        text = re.sub(r'~+\s*', ' ', text)   # Remove repeated tildes
        
        # Clean up whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _score_claim_query_pairs(self, claims: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        """Score all claim-query pairs using BGE Reranker."""
        print(f"Scoring {len(claims)} claims against {len(queries)} queries...")
        
        # Handle empty lists
        if not claims or not queries:
            logger.warning(f"Empty claims ({len(claims)}) or queries ({len(queries)}) provided")
            # Return empty structure
            claim_query_scores = {}
            for claim_idx in range(len(claims)):
                claim_query_scores[claim_idx] = {}
                for query_idx in range(len(queries)):
                    claim_query_scores[claim_idx][query_idx] = 0.0
            return claim_query_scores
        
        # Clean all texts
        cleaned_claims = [self._clean_text(claim) for claim in claims]
        cleaned_queries = [self._clean_text(query) for query in queries]
        
        # Create all claim-query pairs
        claim_query_pairs = []
        pair_mapping = []  # To track which claim-query pair corresponds to which score
        
        for i, claim in enumerate(cleaned_claims):
            for j, query in enumerate(cleaned_queries):
                claim_query_pairs.append([claim, query])
                pair_mapping.append((i, j))
        
        print(f"Created {len(claim_query_pairs)} claim-query pairs for scoring")
        
        # Check if we have any pairs to score
        if not claim_query_pairs:
            logger.warning("No claim-query pairs created")
            # Return empty structure
            claim_query_scores = {}
            for claim_idx in range(len(claims)):
                claim_query_scores[claim_idx] = {}
                for query_idx in range(len(queries)):
                    claim_query_scores[claim_idx][query_idx] = 0.0
            return claim_query_scores
        
        # Score all pairs using BGE Reranker
        scores = self.reranker.compute_score(claim_query_pairs, normalize=True)
        
        # Map scores back to claim-query pairs
        claim_query_scores = {}
        for claim_idx in range(len(claims)):
            claim_query_scores[claim_idx] = {}
            for query_idx in range(len(queries)):
                claim_query_scores[claim_idx][query_idx] = 0.0
        
        for pair_idx, (claim_idx, query_idx) in enumerate(pair_mapping):
            claim_query_scores[claim_idx][query_idx] = float(scores[pair_idx])
        
        return claim_query_scores
    
    def _detect_claim_specific_thresholds(self, claim_query_scores: Dict[int, Dict[int, float]], 
                                        percentiles: List[int] = [20, 40, 60, 80, 90, 95, 98]) -> Dict[int, Dict[str, Any]]:
        """Detect individual thresholds for each claim based on percentile gaps."""
        print(f"Detecting claim-specific thresholds using percentile gap method with percentiles: {percentiles}")
        
        claim_thresholds = {}
        
        for claim_idx in range(len(claim_query_scores)):
            # Extract scores for this specific claim
            claim_scores = [claim_query_scores[claim_idx][query_idx] for query_idx in range(len(claim_query_scores[claim_idx]))]
            claim_scores_array = np.array(claim_scores)
            
            # Detect threshold for this specific claim using percentile gap
            threshold_info = self._percentile_gap_threshold_single_claim(claim_scores_array, claim_idx, percentiles)
            claim_thresholds[claim_idx] = threshold_info
        
        return claim_thresholds
    
    def _percentile_gap_threshold_single_claim(self, claim_scores: np.ndarray, claim_idx: int, 
                                             percentiles: List[int] = [20, 40, 60, 80, 90, 95, 98]) -> Dict[str, Any]:
        """Detect threshold for a single claim based on percentile gaps."""
        # Sort scores to find natural breaks
        sorted_scores = np.sort(claim_scores)
        
        # Calculate percentiles
        percentile_values = np.percentile(sorted_scores, percentiles)
        
        # Find the largest gap between consecutive percentiles
        gaps = np.diff(percentile_values)
        max_gap_idx = np.argmax(gaps)
        threshold = percentile_values[max_gap_idx]
        
        # Alternative: use 95th percentile as threshold
        threshold_95 = np.percentile(sorted_scores, 95)
        
        # Calculate statistics
        mean_score = np.mean(claim_scores)
        std_score = np.std(claim_scores)
        
        return {
            'claim_idx': claim_idx,
            'threshold': float(threshold),
            'threshold_95': float(threshold_95),
            'method': 'percentile_gap',
            'percentiles_used': percentiles,
            'statistics': {
                'mean': float(mean_score),
                'std': float(std_score),
                'min': float(np.min(claim_scores)),
                'max': float(np.max(claim_scores)),
                'percentiles': dict(zip(percentiles, [float(v) for v in percentile_values])),
                'gaps': dict(zip(range(len(gaps)), [float(g) for g in gaps])),
                'max_gap_idx': int(max_gap_idx),
                'max_gap_value': float(gaps[max_gap_idx])
            },
            'claim_scores': claim_scores.tolist()
        }
    
    def find_relevant_queries_for_claims(self, claims: List[str], queries: List[str], 
                                       percentiles: List[int] = [20, 40, 60, 80, 90, 95, 98]) -> Dict[str, Any]:
        """Find queries that are relevant to each claim based on claim-specific percentile gap thresholds."""
        print("Starting claim-query relevance analysis with percentile gap thresholding...")
        
        # Handle empty inputs
        if not claims or not queries:
            logger.warning(f"Empty claims ({len(claims)}) or queries ({len(queries)}) provided")
            return {}
        
        # Score all claim-query pairs
        claim_query_scores = self._score_claim_query_pairs(claims, queries)
        
        # Detect claim-specific thresholds using percentile gap method
        claim_thresholds = self._detect_claim_specific_thresholds(claim_query_scores, percentiles)
        
        print(f"Claim-specific thresholds detected using percentiles {percentiles}:")
        for claim_idx, threshold_info in claim_thresholds.items():
            print(f"  Claim {claim_idx + 1}: {threshold_info['threshold']:.4f} (max gap at percentile {threshold_info['statistics']['max_gap_idx']})")
        

        
        # Find relevant queries for each claim using claim-specific thresholds
        claim_relevant_queries = {}
        query_relevant_claims = {}
        
        for claim_idx, claim in enumerate(claims):
            claim_threshold = claim_thresholds[claim_idx]['threshold']
            
            claim_relevant_queries[claim_idx] = {
                'claim': claim,
                'relevant_queries': [],
                'all_scores': {},
                'threshold_used': claim_threshold,
                'threshold_method': 'percentile_gap',
                'percentiles_used': percentiles
            }
            
            for query_idx, query in enumerate(queries):
                score = claim_query_scores[claim_idx][query_idx]
                claim_relevant_queries[claim_idx]['all_scores'][query_idx] = {
                    'query': query,
                    'score': score,
                    'is_relevant': score >= claim_threshold
                }
                
                if score >= claim_threshold:
                    claim_relevant_queries[claim_idx]['relevant_queries'].append({
                        'query_idx': query_idx,
                        'query': query,
                        'score': score
                    })
            
            # Sort relevant queries by score (highest first)
            claim_relevant_queries[claim_idx]['relevant_queries'].sort(
                key=lambda x: x['score'], reverse=True
            )
        
        # Find claims relevant to each query (using the claim's own threshold)
        for query_idx, query in enumerate(queries):
            query_relevant_claims[query_idx] = {
                'query': query,
                'relevant_claims': [],
                'all_scores': {}
            }
            
            for claim_idx, claim in enumerate(claims):
                score = claim_query_scores[claim_idx][query_idx]
                claim_threshold = claim_thresholds[claim_idx]['threshold']
                
                query_relevant_claims[query_idx]['all_scores'][claim_idx] = {
                    'claim': claim,
                    'score': score,
                    'is_relevant': score >= claim_threshold,
                    'claim_threshold': claim_threshold
                }
                
                if score >= claim_threshold:
                    query_relevant_claims[query_idx]['relevant_claims'].append({
                        'claim_idx': claim_idx,
                        'claim': claim,
                        'score': score,
                        'claim_threshold': claim_threshold
                    })
            
            # Sort relevant claims by score (highest first)
            query_relevant_claims[query_idx]['relevant_claims'].sort(
                key=lambda x: x['score'], reverse=True
            )
        
        # Return only the selected queries for each claim
        selected_queries = {}
        for claim_idx, claim_info in claim_relevant_queries.items():
            selected_queries[claim_idx] = {
                'claim': claim_info['claim'],
                'relevant_queries': claim_info['relevant_queries']
            }
        
        return selected_queries

# Standalone function for easy import
def find_relevant_queries_for_claims(claims: List[str], queries: List[str], 
                                   percentiles: List[int] = [20, 40, 60, 80, 90, 95, 98]) -> Dict[str, Any]:
    """
    Standalone function to find relevant queries for claims using ClaimQueryReranker.
    
    Args:
        claims: List of claims
        queries: List of queries
        percentiles: List of percentiles for threshold detection
        
    Returns:
        Dictionary containing relevant queries for each claim
    """
    reranker = ClaimQueryReranker(num_gpus=4)
    return reranker.find_relevant_queries_for_claims(claims, queries, percentiles)


# Usage example:
# 
# from claim_link_to_query import ClaimQueryReranker
# 
# # Initialize reranker
# reranker = ClaimQueryReranker(num_gpus=4)
# 
# # Define your claims and queries
# claims = ["Claim 1", "Claim 2", ...]
# queries = ["Query 1", "Query 2", ...]
# 
# # Find relevant queries for claims
# selected_queries = reranker.find_relevant_queries_for_claims(claims, queries)
# 
# # Access results - selected_queries is a dict with claim_idx as keys
# for claim_idx, claim_info in selected_queries.items():
#     print(f"Claim {claim_idx}: {claim_info['claim']}")
#     print("Relevant queries:")
#     for query_info in claim_info['relevant_queries']:
#         print(f"  - {query_info['query']} (Score: {query_info['score']:.4f})")