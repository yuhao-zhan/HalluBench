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
from utils import OptimizedContextLocator, fetch_pages_async
from urllib.parse import urlparse

# Add the parent directory to sys.path to import from reproduce.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLI model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)

def nli_score(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Score the relationship between premise and hypothesis using NLI model.
    
    Args:
        premise: The premise text
        hypothesis: The hypothesis text
        
    Returns:
        Dictionary with entailment, neutral, and contradiction scores
    """
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
    
    def is_url(self, text: str) -> bool:
        """
        Check if a text string is a URL.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text appears to be a URL, False otherwise
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Check if it starts with common URL schemes
        url_schemes = ['http://', 'https://', 'www.']
        if any(text.lower().startswith(scheme) for scheme in url_schemes):
            return True
        
        # Try to parse as URL
        try:
            result = urlparse(text)
            return all([result.scheme, result.netloc])
        except:
            return False

    async def process_claims_and_urls(self, claims: List[str], urls: List[str], web_content_cache: Dict[str, str]) -> Dict[str, Any]:
        """
        Process claims against URLs using NLI scoring with iterative refinement.
        Prioritizes scoring against specific URLs when claims are URLs themselves.
        
        Args:
            claims: List of claims to evaluate
            urls: List of URLs to fetch and analyze
            
        Returns:
            Dictionary containing results for each claim
        """
        logger.info(f"Fetching content from {len(urls)} URLs...")
        
        # Fetch web content
        web_content = web_content_cache
        
        # Extract chunks from all documents with URL mapping
        all_chunks = []
        chunk_metadata = []
        url_to_chunks = {}  # Map URL to its chunk indices
        
        for url, content in web_content.items():
            if content.startswith("[Error]"):
                logger.warning(f"Skipping {url} due to fetch error: {content}")
                continue
                
            logger.info(f"Processing content from {url} ({len(content)} chars)")
            
            # Use OptimizedContextLocator to create chunks
            chunks = self.context_locator.extract_sentences(content)
            
            logger.info(f"Created {len(chunks)} chunks from {url}")
            
            # Track chunk indices for this URL
            url_chunk_indices = []
            
            # Add chunks with metadata
            for i, chunk in enumerate(chunks):
                chunk_idx = len(all_chunks)
                all_chunks.append(chunk['chunk_text'])
                chunk_metadata.append({
                    'chunk_id': chunk['chunk_id'],
                    'source_url': url,
                    'chunk_index': i,
                    'chunk_length': chunk['length'],
                    'sentence_count': chunk['sentence_count'],
                    'sentence_indices': chunk['sentence_indices']
                })
                url_chunk_indices.append(chunk_idx)
            
            url_to_chunks[url] = url_chunk_indices

        # Process each claim
        results = []
        current_target_url = None  # Track the current target URL for URL-based claims
        
        for i, claim in enumerate(claims):
            logger.info(f"Processing claim {i+1}/{len(claims)}: {claim}")
            
            # Check if this claim is a URL
            if self.is_url(claim):
                current_target_url = claim
                logger.info(f"  Claim is a URL: {claim}")
            
            # Determine which chunks to score against
            chunks_to_score = []
            metadata_to_score = []
            
            if current_target_url and current_target_url in url_to_chunks:
                # Score against specific URL chunks first
                target_chunk_indices = url_to_chunks[current_target_url]
                chunks_to_score = [all_chunks[idx] for idx in target_chunk_indices]
                metadata_to_score = [chunk_metadata[idx] for idx in target_chunk_indices]
                logger.info(f"  Scoring against {len(chunks_to_score)} chunks from target URL: {current_target_url}")
            else:
                # Score against all chunks
                chunks_to_score = all_chunks
                metadata_to_score = chunk_metadata
                logger.info(f"  Scoring against all {len(chunks_to_score)} chunks")
            
            # First round: Score claim against selected chunks
            claim_scores = []
            
            for j, (chunk, metadata) in enumerate(zip(chunks_to_score, metadata_to_score)):
                try:
                    # Use chunk as premise and claim as hypothesis
                    scores = nli_score(chunk, claim)
                    
                    claim_scores.append({
                        'chunk_id': metadata['chunk_id'],
                        'source_url': metadata['source_url'],
                        'chunk_index': metadata['chunk_index'],
                        'chunk_text': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        'full_chunk_text': chunk,  # Store full text for concatenation
                        'entailment_score': scores['entailment'],
                        'neutral_score': scores['neutral'],
                        'contradiction_score': scores['contradiction'],
                        'chunk_length': metadata['chunk_length']
                    })
                    
                    if j % 20 == 0:  # Log progress every 20 chunks
                        logger.info(f"  Scored {j+1}/{len(chunks_to_score)} chunks for claim {i+1}")
                        
                except Exception as e:
                    logger.error(f"Error scoring chunk {j} for claim '{claim}': {e}")
                    continue
            
            # Find relevant chunks (those with highest entailment or contradiction scores)
            relevant_chunks = self.find_relevant_chunks(claim_scores)
            
            logger.info(f"  Found {len(relevant_chunks)} relevant chunks out of {len(claim_scores)} total chunks")
            
            # Check if both entailment and contradiction judgments exist in first round
            entailment_chunks = [chunk for chunk in relevant_chunks if chunk['entailment_score'] == max(chunk['entailment_score'], chunk['neutral_score'], chunk['contradiction_score'])]
            contradiction_chunks = [chunk for chunk in relevant_chunks if chunk['contradiction_score'] == max(chunk['entailment_score'], chunk['neutral_score'], chunk['contradiction_score'])]
            
            has_both_judgments = len(entailment_chunks) > 0 and len(contradiction_chunks) > 0
            
            logger.info(f"  First round: {len(entailment_chunks)} entailment chunks, {len(contradiction_chunks)} contradiction chunks")
            
            # Second round: Score against concatenated relevant chunks ONLY if both judgments exist
            final_scores = None
            concatenated_text = ""
            
            if relevant_chunks and has_both_judgments:
                # Concatenate relevant chunks
                concatenated_text = self.concatenate_chunks(relevant_chunks)
                
                logger.info(f"  Both entailment and contradiction detected - performing second round scoring")
                logger.info(f"  Concatenated text length: {len(concatenated_text)} characters")
                
                # Score the concatenated text against the claim
                try:
                    final_scores = nli_score(concatenated_text, claim)
                    logger.info(f"  Final scores - Entail: {final_scores['entailment']:.3f}, "
                              f"Neutral: {final_scores['neutral']:.3f}, "
                              f"Contra: {final_scores['contradiction']:.3f}")
                except Exception as e:
                    logger.error(f"Error scoring concatenated text for claim '{claim}': {e}")
                    final_scores = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}
            else:
                if relevant_chunks:
                    # Directly use the first round scores for the final judgment
                    final_scores = {
                        'entailment': max(chunk['entailment_score'] for chunk in relevant_chunks),
                        'neutral': max(chunk['neutral_score'] for chunk in relevant_chunks),
                        'contradiction': max(chunk['contradiction_score'] for chunk in relevant_chunks)
                    }
                else:
                    final_scores = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}

            # Determine final judgment from second round scores
            final_judgment = max(final_scores, key=final_scores.get)
            
            # If we scored against a specific URL and got neutral, fall back to scoring against all chunks
            if (current_target_url and final_judgment == 'neutral' and 
                len(chunks_to_score) < len(all_chunks)):
                
                logger.info(f"  Got neutral judgment from target URL, falling back to all chunks")
                
                # Score against all chunks
                all_claim_scores = []
                for j, (chunk, metadata) in enumerate(zip(all_chunks, chunk_metadata)):
                    try:
                        scores = nli_score(chunk, claim)
                        all_claim_scores.append({
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
                        logger.error(f"Error scoring chunk {j} for claim '{claim}' in fallback: {e}")
                        continue
                
                # Find relevant chunks from all chunks
                all_relevant_chunks = self.find_relevant_chunks(all_claim_scores)
                
                # Check if both entailment and contradiction judgments exist
                all_entailment_chunks = [chunk for chunk in all_relevant_chunks if chunk['entailment_score'] == max(chunk['entailment_score'], chunk['neutral_score'], chunk['contradiction_score'])]
                all_contradiction_chunks = [chunk for chunk in all_relevant_chunks if chunk['contradiction_score'] == max(chunk['entailment_score'], chunk['neutral_score'], chunk['contradiction_score'])]
                
                all_has_both_judgments = len(all_entailment_chunks) > 0 and len(all_contradiction_chunks) > 0
                
                # Second round scoring with all chunks if needed
                if all_relevant_chunks and all_has_both_judgments:
                    all_concatenated_text = self.concatenate_chunks(all_relevant_chunks)
                    try:
                        final_scores = nli_score(all_concatenated_text, claim)
                    except Exception as e:
                        logger.error(f"Error scoring concatenated text for claim '{claim}' in fallback: {e}")
                        final_scores = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}
                else:
                    if all_relevant_chunks:
                        final_scores = {
                            'entailment': max(chunk['entailment_score'] for chunk in all_relevant_chunks),
                            'neutral': max(chunk['neutral_score'] for chunk in all_relevant_chunks),
                            'contradiction': max(chunk['contradiction_score'] for chunk in all_relevant_chunks)
                        }
                    else:
                        final_scores = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}
                
                # Update final judgment and relevant chunks
                final_judgment = max(final_scores, key=final_scores.get)
                relevant_chunks = all_relevant_chunks
            
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
           
        return results

# Standalone function for easy import
async def process_claims_and_urls(claims: List[str], urls: List[str], web_content_cache: Dict[str, str]) -> Dict[str, Any]:
    """
    Standalone function to process claims and URLs using NLIClaimChecker.
    
    Args:
        claims: List of claims to check
        urls: List of URLs to search
        
    Returns:
        Dictionary containing claim checking results
    """
    checker = NLIClaimChecker()
    return await checker.process_claims_and_urls(claims, urls, web_content_cache)


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
