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
from collections import defaultdict, Counter
import warnings
from datetime import datetime
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import OptimizedContextLocator, is_url
from BM25 import BM25Retriever


# Add the parent directory to sys.path to import from reproduce.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLI model
# LAZY LOADING: Don't load models until they're actually needed
_model = None
_tokenizer = None
_device = None

def _get_model_and_tokenizer():
    """Lazy load the NLI model and tokenizer only when needed."""
    global _model, _tokenizer, _device
    
    if _model is None:
        print("🔧 Lazy loading NLI model for claim checking...")
        
        # Check available GPU memory before loading
        if torch.cuda.is_available():
            try:
                # Find GPU with most available memory
                best_gpu = 0
                best_memory = 0
                
                for gpu_id in range(torch.cuda.device_count()):
                    torch.cuda.set_device(gpu_id)
                    allocated = torch.cuda.memory_allocated(gpu_id)
                    total = torch.cuda.get_device_properties(gpu_id).total_memory
                    available = total - allocated
                    
                    if available > best_memory:
                        best_memory = available
                        best_gpu = gpu_id
                
                # Only use GPU if we have at least 4GB available
                if best_memory > 4 * 1024**3:  # 4GB minimum
                    _device = torch.device(f"cuda:{best_gpu}")
                    print(f"🎮 Using GPU {best_gpu} with {best_memory/1024**3:.2f}GB available")
                else:
                    _device = torch.device("cpu")
                    print(f"⚠️ Insufficient GPU memory ({best_memory/1024**3:.2f}GB), using CPU")
            except Exception as e:
                print(f"⚠️ GPU setup failed: {e}, using CPU")
                _device = torch.device("cpu")
        else:
            _device = torch.device("cpu")
            print("🖥️ CUDA not available, using CPU")
        
        # Load model and tokenizer
        model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _model = _model.to(_device)
        
        print(f"✅ NLI model loaded on {_device}")
    
    return _model, _tokenizer, _device

def nli_score(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Score the relationship between premise and hypothesis using NLI model.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        
    Returns:
        Dictionary with entailment, neutral, and contradiction scores
    """
    model, tokenizer, device = _get_model_and_tokenizer()
    
    input_text = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    input_text = {k: v.to(device) for k, v in input_text.items()}
    
    with torch.no_grad():
        output = model(**input_text)
        prediction = torch.softmax(output["logits"][0], -1).tolist()
    
    label_names = ["entailment", "neutral", "contradiction"]
    scores = {name: float(pred) for pred, name in zip(prediction, label_names)}
    
    return scores


class NLIClaimChecker:
    """NLI-based claim checking system that scores claims against document chunks with iterative refinement."""
    
    def __init__(self):
        """Initialize the NLI claim checker."""
        self.context_locator = OptimizedContextLocator()
        
    def load_chunks_from_cache(self, cache_file: str, urls: List[str] = None) -> Tuple[List[Dict], List[Dict], Dict[str, List[int]]]:
        """
        Load pre-computed chunk information from cache to ensure chunk ID consistency.
        Optionally filter chunks to only include those from specified URLs.
        
        Args:
            cache_file: Path to the cache file containing chunk information
            urls: Optional list of URLs to filter chunks by. If None, loads all chunks.
            
        Returns:
            Tuple of (all_chunks, chunk_metadata, url_to_chunks)
        """
        all_chunks = []
        chunk_metadata = []
        url_to_chunks = {}
        
        if not cache_file or not os.path.exists(cache_file):
            return all_chunks, chunk_metadata, url_to_chunks
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            chunk_scores = cache_data.get('chunk_score', {})
            if chunk_scores:
                print(f"✅ Loading pre-computed chunks from cache to ensure chunk ID consistency")
                
                # Reconstruct chunks from cache to maintain chunk ID consistency
                for chunk_id, chunk_data in chunk_scores.items():
                    if isinstance(chunk_data, dict) and 'chunk_text' in chunk_data:
                        # Extract URL from chunk data
                        url = chunk_data.get('url', '')
                        
                        # Filter by URLs if specified
                        if urls is not None and url not in urls:
                            continue
                            
                        chunk_idx = len(all_chunks)
                        all_chunks.append({'chunk_text': chunk_data['chunk_text'], 'url': url})
                        
                        if url:
                            chunk_metadata.append({
                                'chunk_id': chunk_id,
                                'source_url': url,
                                'chunk_index': chunk_idx,
                                'chunk_length': len(chunk_data['chunk_text']),
                                'sentence_count': chunk_data.get('sentence_count', 0),
                                'sentence_indices': chunk_data.get('sentence_indices', [])
                            })
                            
                            # Track chunk indices for this URL
                            if url not in url_to_chunks:
                                url_to_chunks[url] = []
                            url_to_chunks[url].append(chunk_idx)
                
                if urls is not None:
                    print(f"✅ Successfully loaded {len(all_chunks)} chunks from cache for {len(urls)} specified URLs with consistent chunk IDs")
                else:
                    print(f"✅ Successfully loaded {len(all_chunks)} chunks from cache with consistent chunk IDs")
                
        except Exception as e:
            print(f"⚠️ Error loading chunk cache: {e}")
            
        return all_chunks, chunk_metadata, url_to_chunks
    
    def find_relevant_chunks(self, claim_scores: List[Dict]) -> List[Dict]:
        """
        Find chunks where entailment_score or contradiction_score is the highest among the three scores.
        
        Args:
            claim_scores: List of chunk scores for a claim
            
        Returns:
            List of relevant chunks (those with highest entailment or contradiction scores)
        """
        relevant_chunks = []
        
        for chunk_score in claim_scores:
            entailment = chunk_score['entailment_score']
            neutral = chunk_score['neutral_score']
            contradiction = chunk_score['contradiction_score']
            
            # Check if entailment or contradiction is the highest score
            max_score = max(entailment, neutral, contradiction)
            if entailment == max_score or contradiction == max_score:
                relevant_chunks.append(chunk_score)

        # Sort the relevant chunks by the max score (entailment or contradiction) in descending order
        relevant_chunks.sort(key=lambda x: max(x['entailment_score'], x['neutral_score'], x['contradiction_score']), reverse=True)
        return relevant_chunks
    
    def concatenate_chunks(self, relevant_chunks: List[Dict]) -> str:
        """
        Concatenate the text of relevant chunks.
        
        Args:
            relevant_chunks: List of relevant chunk dictionaries
            
        Returns:
            Concatenated text from all relevant chunks
        """
        # Get the full chunk text for each relevant chunk
        chunk_texts = []
        for chunk in relevant_chunks:
            # We need to get the full chunk text, not the truncated version
            # The chunk_text in the score dict is truncated, so we need to get it from the original chunks
            chunk_texts.append(chunk['full_chunk_text'])
        
        # Concatenate with a separator
        concatenated_text = " [SEP] ".join(chunk_texts)
        return concatenated_text
    

    def _score_chunks_with_nli(self, claim: str, chunks: List[str], metadatas: List[Dict]) -> List[Dict]:
        """
        Score a batch of chunks against a claim using the NLI model.

        Returns a list of dictionaries with scores and metadata per chunk.
        """
        claim_scores: List[Dict] = []
        for j, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
            try:
                scores = nli_score(chunk, claim)
                claim_scores.append({
                    'chunk_id': metadata['chunk_id'],
                    'source_url': metadata['source_url'],
                    'chunk_index': metadata['chunk_index'],
                    'chunk_text': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    'full_chunk_text': chunk,
                    'entailment_score': scores['entailment'],
                    'neutral_score': scores['neutral'],
                    'contradiction_score': scores['contradiction'],
                    'chunk_length': metadata['chunk_length']
                })
            except Exception as e:
                logger.error(f"Error scoring chunk {j} for claim '{claim}': {e}")
                continue
        return claim_scores

    def _bm25_iterative_loop(self,
                              claim: str,
                              all_chunks: List[Dict],
                              chunk_metadata: List[Dict],
                              cache_data: Dict,
                              top_k: int = 5,
                              max_batches: int = 5) -> Tuple[str, Dict[str, float], List[Dict]]:
        """
        Iteratively retrieve top-k chunks via BM25 in batches and score with NLI.

        Stops once a batch contains at least one entailment or contradiction chunk;
        if both exist in that batch, concatenate those relevant chunks and run a second-round NLI
        to decide the final judgment. If only one type exists, use that as final.

        If after max_batches no relevant chunks found (all neutral), return neutral.
        """
        # print(f"  Starting BM25→NLI iterative loop (top {top_k} per batch, up to {max_batches} batches)")

        # Load url_against_claim, titles_against_claim from cache_file
        url_against_claim = cache_data.get('url_against_claim', {})
        titles_against_claim = cache_data.get('titles_against_claim', [])

        # Get the scores of current claim against all urls and titles
        claim_scores_url = {}
        claim_scores_title = {}
        for url in url_against_claim:
            claim_scores_url[url] = url_against_claim[url].get(claim, 0.0)
        for title in titles_against_claim:
            claim_scores_title[title] = titles_against_claim[title].get(claim, 0.0)
        

        bm25_retriever = BM25Retriever([chunk['chunk_text'] for chunk in all_chunks])
        scores = bm25_retriever.score(claim)
        scores_list = []
        # Average the BM25 scores, claim_scores and claim_scores for current claim
        for idx, chunk in enumerate(all_chunks):
            # Get the url of the chunk
            url = chunk['url']
            # Get the BM25 score of the chunk
            bm25_score = scores[idx]
            # Get the claim_scores_url of the chunk
            claim_scores_url_score = claim_scores_url.get(url, 0.0)
            # Get the claim_scores_title of the chunk
            claim_scores_title_score = claim_scores_title.get(url, 0.0)
            # Average the scores
            scores_list.append((0.7 * bm25_score + 0.15 * claim_scores_url_score + 0.15 * claim_scores_title_score))

        sorted_indices = np.argsort(scores_list)[::-1]

        total_chunks = len(all_chunks)
        batch_counter = 0

        while batch_counter < max_batches:
            start = batch_counter * top_k
            end = min(start + top_k, total_chunks)
            if start >= end:
                break

            batch_indices = sorted_indices[start:end]
            batch_chunks = [all_chunks[idx]['chunk_text'] for idx in batch_indices]
            batch_meta = [chunk_metadata[idx] for idx in batch_indices]
            # print(f"  BM25 batch {batch_counter + 1}: scoring {len(batch_chunks)} chunks (indices {start}..{end-1})")
            print("="*50)
            print(f" BM25 {batch_counter+1}th batch:")
            for idx, chunk in enumerate(batch_chunks):
                print(f" BM25 Retrieved Chunk {idx+1}: {chunk[:200]}...")

            batch_scores = self._score_chunks_with_nli(claim, batch_chunks, batch_meta)
            relevant_chunks = self.find_relevant_chunks(batch_scores)

            if not relevant_chunks:
                print("    Batch result: all neutral; moving to next batch")
                batch_counter += 1
                continue

            entailment_chunks = [c for c in relevant_chunks if c['entailment_score'] == max(c['entailment_score'], c['neutral_score'], c['contradiction_score'])]
            contradiction_chunks = [c for c in relevant_chunks if c['contradiction_score'] == max(c['entailment_score'], c['neutral_score'], c['contradiction_score'])]

            has_both = len(entailment_chunks) > 0 and len(contradiction_chunks) > 0
            print(f"    Batch result: {len(entailment_chunks)} entailment, {len(contradiction_chunks)} contradiction")

            if has_both:
                concatenated_text = self.concatenate_chunks(relevant_chunks)
                # print("    Both present; running second-round NLI on concatenated relevant chunks")
                try:
                    final_scores = nli_score(concatenated_text, claim)
                except Exception as e:
                    logger.error(f"Error scoring concatenated text for claim '{claim}' in BM25 loop: {e}")
                    final_scores = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}
                final_judgment = max(final_scores, key=final_scores.get)
                return final_judgment, final_scores, relevant_chunks
            else:
                # Only one type present in this batch; use that directly
                if len(entailment_chunks) > 0:
                    final_judgment = 'entailment'
                    final_scores = {
                        'entailment': max(c['entailment_score'] for c in entailment_chunks),
                        'neutral': max(c['neutral_score'] for c in relevant_chunks),
                        'contradiction': max(c['contradiction_score'] for c in relevant_chunks),
                    }
                else:
                    final_judgment = 'contradiction'
                    final_scores = {
                        'entailment': max(c['entailment_score'] for c in relevant_chunks),
                        'neutral': max(c['neutral_score'] for c in relevant_chunks),
                        'contradiction': max(c['contradiction_score'] for c in contradiction_chunks),
                    }
                return final_judgment, final_scores, relevant_chunks

        # Exhausted batches: all neutral
        print("  BM25→NLI loop exhausted: all checked chunks neutral")
        return 'neutral', {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}, []

    async def process_claims_and_urls(self, claims: List[str], urls: List[str], web_content_cache: Dict[str, str], observation_memory: str, cache_file: str = None) -> tuple[List[Dict[str, Any]], str]:
        """
        Process claims against URLs using NLI scoring with iterative refinement.
        Prioritizes scoring against specific URLs when claims are URLs themselves.
        Only processes chunks from the specified URLs parameter, not all cached URLs.
        
        Args:
            claims: List of claims to evaluate
            urls: List of URLs to fetch and analyze (only chunks from these URLs will be processed)
            web_content_cache: Cached web content
            observation_memory: Observation memory for iterative checking
            cache_file: Optional cache file path to load pre-computed chunk information
            
        Returns:
            Tuple of (results, updated_observation_memory) where results is a list of claim results
            and updated_observation_memory is the observation memory updated with entailment claims
        """
        print(f"Fetching content from {len(urls)} URLs...")
        
        # Fetch web content
        web_content = web_content_cache
        
        # Try to load pre-computed chunk information from cache to avoid regenerating chunks
        all_chunks, chunk_metadata, url_to_chunks = self.load_chunks_from_cache(cache_file, urls)
        
        # Debug: Print the URLs in url_to_chunks to see what's stored
        # print(f"🔍 Debug: URLs stored in url_to_chunks: {list(url_to_chunks.keys())}")
        
        # # If no cache or failed to load, fall back to regenerating chunks (for backward compatibility)
        # if not all_chunks:
        #     print("🔄 No chunk cache available, regenerating chunks (this may cause chunk ID mismatch)")
            
        #     # Only process URLs from the urls parameter, not all URLs in web_content
        #     urls_to_process = urls if urls else list(web_content.keys())
            
        #     for url in urls_to_process:
        #         if url not in web_content:
        #             logger.warning(f"URL {url} not found in web_content, skipping")
        #             continue
                    
        #         content = web_content[url]
        #         if content.startswith("[Error]"):
        #             logger.warning(f"Skipping {url} due to fetch error: {content}")
        #             continue
                    
        #         # Use OptimizedContextLocator to create chunks
        #         chunks = self.context_locator.extract_sentences(content)
                
        #         # Track chunk indices for this URL
        #         url_chunk_indices = []
                
        #         # Add chunks with metadata
        #         for i, chunk in enumerate(chunks):
        #             chunk_idx = len(all_chunks)
        #             all_chunks.append(chunk['chunk_text'])
        #             chunk_metadata.append({
        #                 'chunk_id': chunk['chunk_id'],
        #                 'source_url': url,
        #                 'chunk_index': i,
        #                 'chunk_length': chunk['length'],
        #                 'sentence_count': chunk['sentence_count'],
        #                 'sentence_indices': chunk['sentence_indices']
        #             })
        #             url_chunk_indices.append(chunk_idx)
                
        #         url_to_chunks[url] = url_chunk_indices
        
        # Debug: Print the final URLs in url_to_chunks after all processing
        print(f"🔍 Debug: Final URLs in url_to_chunks: {list(url_to_chunks.keys())}")
        
        # Normalize all URLs in url_to_chunks to remove trailing slashes for consistent comparison
        normalized_url_to_chunks = {}
        for url, chunk_indices in url_to_chunks.items():
            normalized_url = url.rstrip('/') if url.endswith('/') else url
            normalized_url_to_chunks[normalized_url] = chunk_indices
        
        # Replace the original url_to_chunks with normalized version
        url_to_chunks = normalized_url_to_chunks
        print(f"🔍 Debug: Normalized URLs in url_to_chunks: {list(url_to_chunks.keys())}")
        
        # Process each claim
        results = []
        current_target_url = None  # Track the current target URL for URL-based claims
        skip_until_next_url = False  # Flag to skip claims until next URL is found

        # Load url_against_query, url_against_claim, titles_against_query, titles_against_claim from cache_file
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
       
        for i, claim in enumerate(claims):
            # print(f"Processing claim {i+1}/{len(claims)}: {claim}")
            
            # Check if this claim is a URL
            if is_url(claim):
                # Extract the URL from the claim
                                 # Extract URL from markdown link href (parentheses)
                match = re.search(r'\[.*?\]\((https?://[^\s\)]+)\)', claim)
                if match:
                    extracted_url = match.group(1)
                else:
                    # Fallback to original pattern for non-markdown URLs
                    match = re.search(r'(https?://[^\s]+)', claim)
                    if match:
                        extracted_url = match.group(1)
                
                if 'extracted_url' in locals():
                    # If there is '/' at the last of the url, remove it
                    if extracted_url.endswith('/'):
                        extracted_url = extracted_url[:-1]
                    current_target_url = extracted_url
                    # print(f"🔍 Debug: Extracted URL: '{extracted_url}'")
                    # print(f"🔍 Debug: Current target URL set to: '{current_target_url}'")
                
                # Check if the URL content contains [ERROR]
                if extracted_url in web_content:
                    content = web_content[extracted_url]
                    if "[ERROR]" in content:
                        print(f"  ⚠️ URL contains error: {extracted_url}")
                        skip_until_next_url = True
                    else:
                        # URL is valid, stop skipping
                        skip_until_next_url = False
                        print(f"  Claim is a valid URL: {extracted_url}")
                    continue
                else:
                    # URL not found in web_content, treat as error
                    print(f"  ⚠️ URL not found in web_content: {extracted_url}")
                    skip_until_next_url = True
                    result = {
                        'claim': claim,
                        'final_judgment': 'neutral',
                        'relevant_chunks': [],
                        'error_note': f"URL not found in web_content, skipping until next URL"
                    }
                    results.append(result)
                    continue
            else:
                # If we're skipping until next URL, skip this claim
                if skip_until_next_url:
                    print(f"  ⏭️ Skipping claim due to previous URL error: {claim}")
                    continue
                
            
            # Initialize defaults for final decision and relevant chunks used
            final_scores: Dict[str, float] = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}
            final_judgment: str = 'neutral'
            relevant_chunks: List[Dict] = []

            # Check the claim against the observation memory first
            # Split observation_memory by \n\n and form a list
            observation_memory_list = observation_memory.split('\n\n')
            # final_judgment, final_scores, relevant_chunks = self._bm25_iterative_loop(
            #     claim, observation_memory_list, chunk_metadata, top_k=5, max_batches=5
            # )
            # Use BM25 to retrieve the top 5 chunks
            bm25_retriever = BM25Retriever(observation_memory_list)
            scores = bm25_retriever.score(claim)
            sorted_indices = np.argsort(scores)[::-1]
            top_5_chunks = [observation_memory_list[idx] for idx in sorted_indices[:3]]
            # Concatenate the top 5 chunks and score with NLI
            concatenated_text = " [SEP] ".join(top_5_chunks)
            final_scores = nli_score(concatenated_text, claim)
            final_judgment = max(final_scores, key=final_scores.get)

            print(f"Observation memory entailment score: {final_judgment} {final_scores['entailment']}")

            # If strong entailment, add the observation memory to the results
            if final_judgment == 'entailment' and final_scores['entailment'] > 0.5:
                print(f" ✅  Observation memory entailment: {claim}")
                final_relevant_chunks = [
                    {
                        'chunk_id': observation_memory,
                        'source_url': observation_memory,
                        'chunk_text': observation_memory,
                        'score': -1.0
                    }
                ]
                result = {
                    'claim': claim,
                    'final_judgment': final_judgment,
                    'relevant_chunks': final_relevant_chunks
                }
                results.append(result)
                # Update observation memory with this claim
                observation_memory += claim + '\n\n'
                print('-'*50)
                continue

            # print(f"Current target URL: '{current_target_url}'")
            # print(f"🔍 Debug: Checking if '{current_target_url}' in url_to_chunks keys: {list(url_to_chunks.keys())}")
            # print(f"🔍 Debug: URL in url_to_chunks check result: {current_target_url in url_to_chunks}")
            if current_target_url and current_target_url in url_to_chunks:
                # Case 2: With URL ahead — score all chunks of that URL
                target_indices = url_to_chunks[current_target_url]
                url_chunks = [all_chunks[idx] for idx in target_indices]
                url_metas = [chunk_metadata[idx] for idx in target_indices]
                print(f"  Scoring against {len(url_chunks)} chunks from target URL: {current_target_url}")

                # claim_scores = self._score_chunks_with_nli(claim, url_chunks, url_metas)
                final_judgment, final_scores, relevant_chunks = self._bm25_iterative_loop(
                    claim, url_chunks, url_metas, cache_data, top_k=5, max_batches=5
                )
                # relevant_chunks = self.find_relevant_chunks(claim_scores)
                print(f"  URL scoring found {len(relevant_chunks)} relevant chunks out of {len(url_chunks)}")

                
            else:
                # Case 1: No URL ahead — use BM25→NLI iterative loop
                # Only process chunks from the specified URLs, not all cached chunks
                print(f"  No target URL for claim; using BM25→NLI iterative loop on {len(all_chunks)} chunks from specified URLs")
                final_judgment, final_scores, relevant_chunks = self._bm25_iterative_loop(
                    claim, all_chunks, chunk_metadata, cache_data, top_k=5, max_batches=5
                )
            
            # Find relevant chunks based on final judgment for the simplified result
            final_relevant_chunks = []
            if final_judgment in ['entailment', 'contradiction']:
                for chunk in relevant_chunks:
                    entailment = chunk['entailment_score']
                    neutral = chunk['neutral_score']
                    contradiction = chunk['contradiction_score']
                    
                    # Find the highest score among the three
                    max_score = max(entailment, neutral, contradiction)
                    
                    # Check if the final judgment score is the highest for this chunk
                    if final_judgment == 'entailment' and entailment == max_score:
                        final_relevant_chunks.append({
                            'chunk_id': chunk['chunk_id'],
                            'source_url': chunk['source_url'],
                            'chunk_text': chunk['chunk_text'],
                            'score': entailment
                        })
                    elif final_judgment == 'contradiction' and contradiction == max_score:
                        final_relevant_chunks.append({
                            'chunk_id': chunk['chunk_id'],
                            'source_url': chunk['source_url'],
                            'chunk_text': chunk['chunk_text'],
                            'score': contradiction
                        })
                
            # Store simplified results
            result = {
                'claim': claim,
                'final_judgment': final_judgment,
                'relevant_chunks': final_relevant_chunks
            }
            
            results.append(result)
            if final_judgment == 'entailment':
                print(f"✅ Fact Entailment [Before Noise Domination]: {claim}")
                print("Relevant Chunks:")
                for chunk in final_relevant_chunks:
                    print(f"  - {chunk['chunk_text'][:200]}...")

            if final_judgment == 'contradiction':
                print(f"❌ [H4] Fact Contradiction: {claim}")
                print("Relevant Chunks:")
                for chunk in final_relevant_chunks:
                    print(f"  - {chunk['chunk_text'][:200]}...")

            if final_judgment == 'neutral':
                print(f"❌ [H3] Fact Fabrication: {claim}")

            print('-'*50)

           
        return results, observation_memory

# Standalone function for easy import
async def process_claims_and_urls(claims: List[str], urls: List[str], web_content_cache: Dict[str, str], observation_memory: str = "", cache_file: str = None) -> tuple[List[Dict[str, Any]], str]:
    """
    Standalone function to process claims and URLs using NLIClaimChecker.
    
    Args:
        claims: List of claims to check
        urls: List of URLs to search
        web_content_cache: Cached web content
        observation_memory: Observation memory for iterative checking
        cache_file: Optional cache file path to load pre-computed chunk information
        
    Returns:
        Tuple of (results, updated_observation_memory) where results is a list of claim results
        and updated_observation_memory is the observation memory updated with entailment claims
    """
    checker = NLIClaimChecker()
    return await checker.process_claims_and_urls(claims, urls, web_content_cache, observation_memory, cache_file)


# Usage example:
# 
# from claim_checking import NLIClaimChecker
# import asyncio
# 
# async def main():
#     # Initialize checker
#     checker = NLIClaimChecker()
#     
#     # Define claims and URLs
#     claims = ["Claim 1", "Claim 2", ...]
#     urls = ["https://example1.com", "https://example2.com", ...]
#     
#     # Process claims and URLs
#     results = await checker.process_claims_and_urls(claims, urls)
#     
#     # Access results
#     for result in results:
#         print(f"Claim: {result['claim']}")
#         print(f"Final judgment: {result['final_judgment']}")
#         print(f"Relevant chunks: {len(result['relevant_chunks'])}")
#         for chunk in result['relevant_chunks']:
#             print(f"  - {chunk['source_url']} (Score: {chunk['score']:.3f})")
# 
# # Run the example
# # asyncio.run(main())
