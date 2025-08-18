import re
import os
import logging
import aiohttp
import asyncio
from typing import List, Dict, Set, Any, Tuple
import logging
import numpy as np
import math
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelWithLMHead, AutoTokenizer
import spacy
from urllib.parse import urljoin
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

# Stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 
    'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 
    'the', 'to', 'was', 'were', 'will', 'with'
}

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class QueryProcessor:
    """Handles query expansion and entity clustering operations."""
    
    def __init__(self, sbert_model: str = 'all-MiniLM-L6-v2', ner_threshold: float = 0.5):
        self.sbert_model_name = sbert_model
        self.ner_threshold = ner_threshold
        self.sbert = SentenceTransformer(sbert_model)
        self.ner_clusterer = SemanticNERClusterer(sbert_model, ner_threshold)
        
        # Cache for embeddings
        self.word_embeddings = {}
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get cached embedding for text."""
        if text not in self.word_embeddings:
            embedding = self.sbert.encode([text], convert_to_numpy=True)[0]
            self.word_embeddings[text] = embedding / (np.linalg.norm(embedding) + 1e-8)
        return self.word_embeddings[text]
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between texts."""
        vec1 = self._get_embedding(text1)
        vec2 = self._get_embedding(text2)
        return np.dot(vec1, vec2)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Fast text preprocessing."""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        return [w for w in words if w not in STOP_WORDS]
    
    def extract_entity_clusters(self, queries: List[str], documents: List[str]) -> Dict[int, Dict[str, Any]]:
        """Extract and cluster entities from queries - shared implementation."""
        if not self.ner_clusterer:
            # Fallback to simple term extraction
            query_clusters = {}
            for i, query in enumerate(queries):
                terms = self._preprocess_text(query)[:5]  # Limit to top 5 terms
                clusters = {}
                for term in terms:
                    clusters[term] = {
                        'key_query_term': term,
                        'similar_entities': []  # No similarity search in fallback
                    }
                query_clusters[i] = {
                    'query': query,
                    'key_query_terms': terms,
                    'clusters': clusters
                }
            return query_clusters
        
        # Use NER clustering
        all_docs = " ".join(documents)
        doc_entities = self.ner_clusterer.extract_entities(all_docs)
        query_clusters = {}
        
        for i, query in enumerate(queries):
            query_entities = self.ner_clusterer.extract_entities(query)
            if not query_entities:
                # Fallback to terms
                terms = self._preprocess_text(query)[:3]
                clusters = {}
                for term in terms:
                    clusters[term] = {
                        'key_query_term': term,
                        'similar_entities': []
                    }
                query_clusters[i] = {
                    'query': query,
                    'key_query_terms': terms,
                    'clusters': clusters
                }
                continue
            
            # Build clusters for each key query term
            clusters = {}
            key_query_terms = []
            
            for query_entity in query_entities:
                key_term = query_entity['text']
                key_query_terms.append(key_term)
                
                # Find similar entities from documents
                similar_entities = []
                for doc_entity in doc_entities:
                    if doc_entity['text'] != key_term:
                        # Compute similarity
                        similarity = self._compute_similarity(key_term, doc_entity['text'])
                        if similarity > self.ner_threshold:
                            similar_entities.append({
                                'text': doc_entity['text'],
                                'label': doc_entity['label'],
                                'similarity': float(similarity)  # Convert numpy float to Python float
                            })
                
                # Sort by similarity
                similar_entities.sort(key=lambda x: x['similarity'], reverse=True)
                
                clusters[key_term] = {
                    'key_query_term': key_term,
                    'similar_entities': similar_entities[:10]  # Top 10 similar entities
                }
            
            query_clusters[i] = {
                'query': query,
                'key_query_terms': key_query_terms,
                'clusters': clusters
            }
        
        return query_clusters
    
    def extract_all_entities(self, query_clusters: Dict[int, Dict[str, Any]]) -> List[str]:
        """Extract all entities (key and similar) from query clusters."""
        all_entities = set()
        
        for query_idx, cluster_info in query_clusters.items():
            if 'clusters' in cluster_info:
                for cluster_key, cluster_data in cluster_info['clusters'].items():
                    # Add key entity
                    all_entities.add(cluster_data['key_query_term'])
                    
                    # Add similar entities
                    for similar_entity in cluster_data['similar_entities']:
                        all_entities.add(similar_entity['text'])
        
        return list(all_entities)
    
    def expand_queries(self, queries: List[str], query_clusters: Dict[int, Dict[str, Any]]) -> tuple[List[str], Dict[int, int]]:
        """Expand queries using similar entities from clusters."""
        expanded_queries = []
        query_mapping = {}  # Maps expanded query index to original query index
        
        for i, query in enumerate(queries):
            if i not in query_clusters:
                expanded_queries.append(query)
                query_mapping[len(expanded_queries) - 1] = i
                continue
            
            cluster_info = query_clusters[i]
            original_query = cluster_info['query']
            clusters = cluster_info['clusters']
            
            # Create expanded queries by replacing key entities with similar entities
            expanded_versions = [original_query]  # Keep original query
            
            for cluster_key, cluster_data in clusters.items():
                key_term = cluster_data['key_query_term']
                similar_entities = cluster_data['similar_entities']
                
                # Create expanded versions for each similar entity
                for similar_entity in similar_entities:
                    expanded_query = original_query.replace(key_term, similar_entity['text'])
                    if expanded_query != original_query:
                        expanded_versions.append(expanded_query)
            
            # Add all expanded versions and map them to original query index
            for expanded_query in expanded_versions:
                expanded_queries.append(expanded_query)
                query_mapping[len(expanded_queries) - 1] = i
        
        return expanded_queries, query_mapping

class WeightComputer:
    """Handles weight computation for entities and queries."""
    
    def __init__(self, c: float = 6.0):
        self.c = c  # Parameter for log-logistic model
        self.collection_stats = {}
    
    def compute_collection_stats(self, documents: List[str]) -> None:
        """Compute collection statistics for weight computation."""
        total_length = 0
        total_docs = len(documents)
        
        for doc in documents:
            doc_words = doc.split()
            total_length += len(doc_words)
        
        avdl = total_length / total_docs if total_docs > 0 else 0
        
        self.collection_stats = {
            'avdl': avdl,
            'total_docs': total_docs,
            'total_length': total_length
        }
        
        print(f"Collection stats: avg doc length={avdl:.2f}, total docs={total_docs}")
    
    def compute_lambda(self, entity: str, documents: List[str]) -> float:
        """Compute lambda (document frequency) for entity."""
        entity_lower = entity.lower()
        doc_freq = sum(1 for doc in documents if entity_lower in doc.lower())
        return doc_freq / len(documents) if len(documents) > 0 else 0.0
    
    def compute_entity_weights(self, entities: List[str], documents: List[str]) -> Dict[str, float]:
        """Compute query entity weights using Log-logistic model."""
        weights = {}
        all_text = " ".join(documents).lower()
        
        for entity in entities:
            entity_lower = entity.lower()
            tf = all_text.count(entity_lower)
            if tf == 0:
                weights[entity] = 1.0  # Default weight for entities not found in documents
                continue
            
            # Log-logistic weight calculation
            avdl = self.collection_stats['avdl']
            doc_length = len(all_text.split()) / len(documents)
            normalized_tf = tf * math.log(1 + self.c * avdl / doc_length)
            lambda_w = self.compute_lambda(entity, documents)
            
            try:
                weight = math.log((normalized_tf + lambda_w) / lambda_w) if lambda_w > 0 else 1.0
                # Ensure weight is not negative or zero
                weight = max(weight, 1.0)
            except (ValueError, ZeroDivisionError):
                # Fallback to simple TF-based weight
                weight = math.log(tf + 1) if tf > 0 else 1.0
                weight = max(weight, 1.0)
            
            weights[entity] = weight
        
        return weights
    
    def compute_query_weight(self, query: str, query_clusters: Dict[str, Any], entity_weights: Dict[str, float]) -> float:
        """Compute query weight based on entities contained in the query."""
        query_lower = query.lower()
        query_entities = []
        
        # Extract key entities from clusters
        if 'clusters' in query_clusters:
            for cluster_key, cluster_data in query_clusters['clusters'].items():
                key_term = cluster_data['key_query_term']
                if key_term.lower() in query_lower:
                    query_entities.append(key_term)
                
                # Check similar entities
                for similar_entity in cluster_data['similar_entities']:
                    if similar_entity['text'].lower() in query_lower:
                        query_entities.append(similar_entity['text'])
        
        # If no entities found in clusters, extract simple terms
        if not query_entities:
            terms = re.findall(r'\b[a-zA-Z]{3,}\b', query_lower)
            terms = [w for w in terms if w not in STOP_WORDS]
            query_entities = terms[:3]  # Limit to top 3 terms
        
        # Compute average weight of entities in the query
        if query_entities:
            entity_weights_in_query = [entity_weights.get(entity, 1.0) for entity in query_entities]
            query_weight = sum(entity_weights_in_query) / len(entity_weights_in_query)
            # Ensure minimum weight of 1.0
            query_weight = max(query_weight, 1.0)
        else:
            query_weight = 1.0  # Default weight if no entities found
        
        return query_weight

# Context locator for single sentences (used by quality_scoring.py)
class SingleSentenceContextLocator:
    """Context locator that extracts individual sentences."""
    
    def __init__(self):
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self.markdown_pattern = re.compile(r'[#*\[\](){}|`_~]+')
    
    def _clean_text(self, text: str) -> str:
        """Simplified text cleaning with URL preservation."""
        # First, protect URLs by temporarily replacing them with placeholders
        urls = self.url_pattern.findall(text)
        url_placeholders = {}
        
        for i, url in enumerate(urls):
            # Use a unique placeholder that won't be affected by any cleaning operations
            placeholder = f"URLSAFE{i}URLSAFE"
            url_placeholders[placeholder] = url
            text = text.replace(url, placeholder)
        
        # Remove markdown characters but preserve newlines for sentence boundaries
        # First, replace newlines with a special marker
        text = text.replace('\n', ' @NEWLINE@ ')
        
        # Remove other markdown characters
        text = self.markdown_pattern.sub(' ', text)
        
        # Restore newlines and clean up whitespace
        text = text.replace(' @NEWLINE@ ', '\n')
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Replace multiple newlines with single newline
        
        return text.strip()
    
    def _find_complete_sentence(self, text: str, entity_pos: int, entity_length: int) -> tuple[int, int]:
        """Find complete sentence boundaries around entity position."""
        # Define sentence ending characters - only true sentence endings
        sentence_end_chars = {'.', '!', '?', '\n'}
        
        # Find sentence start (backward from entity position)
        start = entity_pos
        while start > 0:
            char = text[start - 1]
            if char in sentence_end_chars:
                # Found sentence boundary, move to next character
                start = start
                break
            start -= 1
        
        # Find sentence end (forward from entity position)
        end = entity_pos + entity_length
        while end < len(text):
            char = text[end]
            if char in sentence_end_chars:
                # Found sentence boundary, include this character
                end = end + 1
                break
            end += 1
        
        # Ensure we don't cut words in the middle
        # Move start to word boundary
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        
        # Move end to word boundary
        while end < len(text) and text[end].isalnum():
            end += 1
        
        return start, min(end, len(text))
    
    def locate_contexts(self, document: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Locate contexts around entities in document."""
        document = self._clean_text(document)
        doc_lower = document.lower()
        contexts = []
        
        for entity in entities:
            if not entity or entity.lower() in STOP_WORDS:
                continue
                
            entity_lower = entity.lower()
            pos = 0
            
            while True:
                pos = doc_lower.find(entity_lower, pos)
                if pos == -1:
                    break
                
                # Check if this is a whole word match
                is_whole_word = True
                # Check start boundary
                if pos > 0 and document[pos-1].isalnum():
                    is_whole_word = False
                # Check end boundary
                end_pos = pos + len(entity)
                if end_pos < len(document) and document[end_pos].isalnum():
                    is_whole_word = False
                
                if not is_whole_word:
                    pos += 1
                    continue
                
                # Find complete sentence containing the entity
                sent_start, sent_end = self._find_complete_sentence(document, pos, len(entity))
                context_text = document[sent_start:sent_end].strip()
                
                if len(context_text) > 20 and entity_lower in context_text.lower():
                    # Limit context length to prevent overly long contexts
                    if len(context_text) > 500:
                        # Find a good breaking point
                        words = context_text.split()
                        if len(words) > 50:
                            # Try to break at sentence boundaries or word boundaries
                            truncated = ' '.join(words[:50])
                            # Find the last complete sentence or word
                            last_period = truncated.rfind('.')
                            last_exclamation = truncated.rfind('!')
                            last_question = truncated.rfind('?')
                            break_point = max(last_period, last_exclamation, last_question)
                            if break_point > 0:
                                context_text = truncated[:break_point + 1]
                            else:
                                context_text = truncated
                    
                    contexts.append({
                        'entity': entity,
                        'context_text': context_text,
                        'position': pos,
                        'context_id': f"{entity}_{pos}_{hash(context_text[:100])}"
                    })
                
                pos += 1
        
        return contexts

# OptimizedContextLocator is used to extract sentences and chunks from a document
class OptimizedContextLocator:
    """Fast context locator with 4-sentence chunks with 1-sentence overlap."""
    
    def __init__(self):
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self.markdown_pattern = re.compile(r'[#*\[\](){}|`_~]+')
    
    def _clean_text(self, text: str) -> str:
        """Simplified text cleaning with URL preservation."""
        # First, protect URLs by temporarily replacing them with placeholders
        urls = self.url_pattern.findall(text)
        url_placeholders = {}
        
        for i, url in enumerate(urls):
            # Use a unique placeholder that won't be affected by any cleaning operations
            placeholder = f"URLSAFE{i}URLSAFE"
            url_placeholders[placeholder] = url
            text = text.replace(url, placeholder)
        
        # Remove markdown characters but preserve newlines for sentence boundaries
        # First, replace newlines with a special marker
        text = text.replace('\n', ' @NEWLINE@ ')
        
        # Remove other markdown characters
        text = self.markdown_pattern.sub(' ', text)
        
        # Remove noise characters and patterns
        # Remove repeated equals signs (markdown headers)
        text = re.sub(r'=+\s*', ' ', text)
        # Remove repeated dashes (markdown separators)
        text = re.sub(r'-+\s*', ' ', text)
        # Remove repeated underscores (markdown separators)
        text = re.sub(r'_+\s*', ' ', text)
        # Remove repeated asterisks (markdown separators)
        text = re.sub(r'\*+\s*', ' ', text)
        # Remove repeated plus signs
        text = re.sub(r'\++\s*', ' ', text)
        # Remove repeated tildes
        text = re.sub(r'~+\s*', ' ', text)
        
        # Restore newlines and clean up whitespace
        text = text.replace(' @NEWLINE@ ', '\n')
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Replace multiple newlines with single newline
        
        # Restore the original URLs
        for placeholder, original_url in url_placeholders.items():
            text = text.replace(placeholder, original_url)
        
        return text.strip()
    
    def _extract_all_sentences(self, document: str) -> List[Dict[str, Any]]:
        """Extract all complete sentences from document."""
        document = self._clean_text(document)
        sentences = []
        
        # First, protect URLs in the document to prevent splitting at dots within URLs
        urls = self.url_pattern.findall(document)
        url_placeholders = {}
        
        for i, url in enumerate(urls):
            placeholder = f"URLSAFE{i}URLSAFE"
            url_placeholders[placeholder] = url
            document = document.replace(url, placeholder)
        
        # Split by sentence endings (but not dots within URL placeholders)
        sentence_pattern = re.compile(r'[^.!?\n]+[.!?\n]+')
        matches = sentence_pattern.finditer(document)
        
        for i, match in enumerate(matches):
            sentence_text = match.group().strip()
            
            # Restore URLs in this sentence
            for placeholder, original_url in url_placeholders.items():
                sentence_text = sentence_text.replace(placeholder, original_url)
            
            # Filter out very short sentences and sentences that are just whitespace
            if len(sentence_text) > 1 and not sentence_text.isspace():
                # Limit sentence length to prevent overly long sentences
                if len(sentence_text) > 500:
                    # Find a good breaking point
                    words = sentence_text.split()
                    if len(words) > 50:
                        # Try to break at sentence boundaries or word boundaries
                        truncated = ' '.join(words[:50])
                        # Find the last complete sentence or word
                        last_period = truncated.rfind('.')
                        last_exclamation = truncated.rfind('!')
                        last_question = truncated.rfind('?')
                        break_point = max(last_period, last_exclamation, last_question)
                        if break_point > 0:
                            sentence_text = truncated[:break_point + 1]
                        else:
                            sentence_text = truncated
                
                sentences.append({
                    'sentence_id': f"sentence_{i}_{hash(sentence_text[:100])}",
                    'sentence_text': sentence_text,
                    'position': match.start(),
                    'length': len(sentence_text),
                    'sentence_index': i
                })
        
        return sentences
    
    def _create_sentence_chunks_with_overlap(self, sentences: List[Dict[str, Any]], chunk_size: int = 4, overlap_size: int = 1) -> List[Dict[str, Any]]:
        """Create chunks of 4 sentences with 1 sentence overlap."""
        if len(sentences) == 0:
            return []
        
        if len(sentences) <= chunk_size:
            # If we have fewer sentences than chunk size, create single chunk
            chunk_text = ' '.join([sent['sentence_text'] for sent in sentences])
            return [{
                'chunk_id': f"chunk_0_{hash(chunk_text[:100])}",
                'chunk_text': chunk_text,
                'position': sentences[0]['position'] if sentences else 0,
                'length': len(chunk_text),
                'sentence_count': len(sentences),
                'sentence_indices': [sent['sentence_index'] for sent in sentences]
            }]
        
        chunks = []
        step_size = chunk_size - overlap_size  # How many new sentences to add per chunk
        
        for i in range(0, len(sentences) - chunk_size + 1, step_size):
            # Get sentences for this chunk
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = ' '.join([sent['sentence_text'] for sent in chunk_sentences])
            
            # Calculate position and length
            start_position = chunk_sentences[0]['position']
            end_position = chunk_sentences[-1]['position'] + chunk_sentences[-1]['length']
            total_length = end_position - start_position
            
            chunks.append({
                'chunk_id': f"chunk_{i//step_size}_{hash(chunk_text[:100])}",
                'chunk_text': chunk_text,
                'position': start_position,
                'length': total_length,
                'sentence_count': len(chunk_sentences),
                'sentence_indices': [sent['sentence_index'] for sent in chunk_sentences]
            })
        
        # Handle the last chunk if there are remaining sentences
        if len(sentences) % step_size != 0:
            remaining_start = len(sentences) - chunk_size
            if remaining_start >= 0:
                last_chunk_sentences = sentences[remaining_start:]
                last_chunk_text = ' '.join([sent['sentence_text'] for sent in last_chunk_sentences])
                
                start_position = last_chunk_sentences[0]['position']
                end_position = last_chunk_sentences[-1]['position'] + last_chunk_sentences[-1]['length']
                total_length = end_position - start_position
                
                chunks.append({
                    'chunk_id': f"chunk_final_{hash(last_chunk_text[:100])}",
                    'chunk_text': last_chunk_text,
                    'position': start_position,
                    'length': total_length,
                    'sentence_count': len(last_chunk_sentences),
                    'sentence_indices': [sent['sentence_index'] for sent in last_chunk_sentences]
                })
        
        return chunks
    
    def extract_sentences(self, document: str) -> List[Dict[str, Any]]:
        """Extract chunks of 4 sentences with 1 sentence overlap."""
        # First extract all sentences
        all_sentences = self._extract_all_sentences(document)
        
        # Then create chunks with overlap
        return self._create_sentence_chunks_with_overlap(all_sentences, chunk_size=4, overlap_size=1)

async def fetch_pages_async(urls: List[str]) -> Dict[str, str]:
    """Fast async web fetching."""
    results = {}
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                jina_url = f"https://r.jina.ai/{url}"
                async with session.get(jina_url, timeout=aiohttp.ClientTimeout(total=20)) as response:
                    if response.status == 200:
                        content = await response.text()
                        results[url] = content
                        print(f"Fetched {url} ({len(content)} chars)")
                    else:
                        results[url] = f"[Error] HTTP {response.status}"
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                results[url] = f"[Error] {e}"
    
    return results


DESKTOP_UA   = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
)

async def another_candidate_fetch_pages_async(urls: List[str]) -> Dict[str, str]:
    """
    Asynchronously fetches web pages, parses them to make links absolute,
    and returns the text content.
    """
    results: Dict[str, str] = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=DESKTOP_UA,
            locale="en-US",
            viewport={"width": 1280, "height": 800}
        )
        page = await context.new_page()
        for url in urls:
            try:
                await page.goto(url, timeout=60000)
                html = await page.content()
                soup = BeautifulSoup(html, "lxml")

                # Find all hyperlinks and replace them with "text (absolute_url)"
                for a in soup.find_all("a", href=True):
                    text = a.get_text(strip=True) or "[No Text]"
                    raw_href = a["href"]
                    full_href = urljoin(url, raw_href)
                    a.replace_with(f"{text} ({full_href})")
                
                # Extract text from the modified body content
                if soup.body:
                    content_with_links = soup.body.get_text(separator="\n", strip=True)
                else:
                    content_with_links = soup.get_text(separator="\n", strip=True)
                
                results[url] = content_with_links

            except PlaywrightTimeoutError:
                results[url] = "[Error] Timeout"
            except Exception as e:
                results[url] = f"[Error] {e!r}"

        await context.close()
        await browser.close()
    return results


# def get_cached_web_content(cache_file: str) -> Dict[str, str]:
#     """
#     Get web content from cache file.
    
#     Args:
#         cache_file: Path to cache file
        
#     Returns:
#         Dictionary mapping URLs to their content
#     """
#     if not os.path.exists(cache_file):
#         logger.error(f"❌ Cache file not found: {cache_file}")
#         return {}
    
#     try:
#         with open(cache_file, 'r', encoding='utf-8') as f:
#             cached_content = json.load(f)
#         print(f"📖 Loaded {len(cached_content)} URLs from cache: {cache_file}")
#         return cached_content
#     except Exception as e:
#         logger.error(f"❌ Error reading cache file: {e}")
#         return {}



async def fetch_url_with_retry(session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> str:
    """
    Fetch a single URL with retry logic.
    
    Args:
        session: aiohttp session
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        
    Returns:
        Content of the URL or error message
    """
    for attempt in range(max_retries):
        try:
            jina_url = f"https://r.jina.ai/{url}"
            async with session.get(jina_url, timeout=aiohttp.ClientTimeout(total=200)) as response:
                if response.status == 200:
                    content = await response.text()
                    if "ERROR" not in content:
                        print(f"✅ Successfully fetched {url} ({len(content)} chars) on attempt {attempt + 1}")
                        return content
                else:
                    print(f"❌ Failed to fetch {url} on attempt {attempt + 1} with iniital method!")
                    try:
                        web_content = await another_candidate_fetch_pages_async([url])
                        return web_content[url]
                    except Exception as e:
                        logger.warning(f"[Error] Playwright fetch failed for {url}: {e}")
                        error_msg = f"[Error] HTTP {response.status} (Playwright fallback failed: {e})"
                        return error_msg
        except Exception as e:
            logger.warning(f"[Error] Initial fetch failed for {url}: {e}")
            error_msg = f"[Error] Initial fetch failed: {e}"
            return error_msg
                    
    
    return f"[Error] Failed after {max_retries} attempts"


async def fetch_all_urls_and_cache(all_urls: List[str], cache_file: str) -> Dict[str, str]:
    """
    Fetch all URLs and store them in a cache file.
    
    Args:
        all_urls: List of URLs to fetch
        cache_file: Path to cache file
        
    Returns:
        Dictionary mapping URLs to their content
    """
    # Check if cache file exists
    if os.path.exists(cache_file):
        print(f"📁 Found existing cache file: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_content = json.load(f)
            print(f"📖 Loaded {len(cached_content)} URLs from cache")
            return cached_content
        except Exception as e:
            logger.warning(f"⚠️ Error reading cache file: {e}. Will fetch URLs again.")
    
    print(f"🌐 Fetching {len(all_urls)} URLs and caching to: {cache_file}")
    
    # Fetch all URLs
    web_content = {}
    async with aiohttp.ClientSession() as session:
        for i, url in enumerate(all_urls, 1):
            print(f"Fetching URL {i}/{len(all_urls)}: {url}")
            content = await fetch_url_with_retry(session, url)
            web_content[url] = content
    
    # Save to cache file
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(web_content, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {len(web_content)} URLs to cache: {cache_file}")
    except Exception as e:
        logger.error(f"❌ Error saving cache file: {e}")
    
    return web_content



def remove_titles_and_split_sentences(text):
    """
    Remove paragraph titles and split text into sentences using punctuation marks.
    """
    # Remove titles by filtering out lines that are likely titles
    lines = text.strip().split('\n')
    content_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and lines that look like titles (short lines without ending punctuation)
        if line and not (len(line) < 30 and not line.endswith(('.', '!', '?', '。', '！', '？'))):
            content_lines.append(line)
    
    full_text = ' '.join(content_lines)
    
    # Split by sentence endings (both English and Chinese)
    sentences = re.split(r'[.!?。！？]', full_text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Filter out very short fragments
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


# Name Entity Recognition
class SemanticNERClusterer:
    """
    Semantic NER clustering focused on semantic similarity regardless of entity labels.
    Allows cross-label clustering based on semantic relevance and context.
    """
    
    def __init__(self, 
                 sbert_model_name: str = 'all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.5,  # Lower threshold for more flexible matching
                 entity_types: Set[str] = None):
        
        self.similarity_threshold = similarity_threshold
        # More inclusive entity types - we'll rely on semantic similarity rather than label constraints
        self.entity_types = entity_types or {
            'PERSON', 'ORG', 'GPE', 'PRODUCT', 'WORK_OF_ART', 'EVENT', 'FAC', 
            'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'PERCENT', 'QUANTITY'
        }
        
        # Initialize SBERT model quietly
        self.sbert_model = SentenceTransformer(sbert_model_name)
        
        self.nlp = spacy.load('en_core_web_sm')
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities with more flexible filtering"""
        if not text or not text.strip():
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # More lenient filtering - include more entity types and shorter entities
            if ent.label_ in self.entity_types and len(ent.text.strip()) >= 1:
                entity_info = {
                    'text': ent.text.strip(),
                    'label': ent.label_,
                }
                entities.append(entity_info)
        
        # Remove duplicates while preserving different labels for same text
        seen = set()
        unique_entities = []
        for entity in entities:
            # Only dedupe exact matches (same text AND label)
            entity_key = (entity['text'].lower(), entity['label'])
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def compute_entity_embeddings(self, entities: List[Dict[str, Any]]) -> np.ndarray:
        if not entities:
            return np.array([])
        
        entity_texts = [entity['text'] for entity in entities]
        embeddings = self.sbert_model.encode(entity_texts, show_progress_bar=False)
        return embeddings
    
    def semantic_cluster_entities(self, 
                                query_entities: List[Dict[str, Any]], 
                                doc_entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Semantic clustering that ignores label constraints and focuses on semantic similarity.
        """
        
        if not query_entities or not doc_entities:
            return {}
        
        # Compute embeddings
        query_embeddings = self.compute_entity_embeddings(query_entities)
        doc_embeddings = self.compute_entity_embeddings(doc_entities)
        
        if query_embeddings.size == 0 or doc_embeddings.size == 0:
            return {}
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)
        
        # Semantic clustering - ignore label constraints
        clusters = {}
        
        for q_idx, query_entity in enumerate(query_entities):
            similar_entities = []
            
            for d_idx, doc_entity in enumerate(doc_entities):
                similarity = similarity_matrix[q_idx, d_idx]
                
                # Use semantic similarity regardless of labels
                if similarity >= self.similarity_threshold:
                    similar_entities.append({
                        'text': doc_entity['text'],
                        'label': doc_entity['label'],
                        'similarity': round(float(similarity), 3)
                    })
            
            if similar_entities:
                # Sort by similarity
                similar_entities.sort(key=lambda x: x['similarity'], reverse=True)
                # Include both text and label in cluster key for clarity
                cluster_key = f"{query_entity['text']} ({query_entity['label']})"
                clusters[cluster_key] = similar_entities
        
        return clusters
    
    def enhanced_semantic_clustering(self, 
                                   query_entities: List[Dict[str, Any]], 
                                   doc_entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enhanced clustering that also considers contextual relationships and variations.
        """
        
        basic_clusters = self.semantic_cluster_entities(query_entities, doc_entities)
        
        # For entities that didn't find matches, try with lower threshold
        enhanced_clusters = {}
        lower_threshold = max(0.25, self.similarity_threshold - 0.15)  # More lenient threshold
        
        if not query_entities or not doc_entities:
            return basic_clusters
        
        # Compute embeddings
        query_embeddings = self.compute_entity_embeddings(query_entities)
        doc_embeddings = self.compute_entity_embeddings(doc_entities)
        
        if query_embeddings.size == 0 or doc_embeddings.size == 0:
            return basic_clusters
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(query_embeddings, doc_embeddings)
        
        for q_idx, query_entity in enumerate(query_entities):
            cluster_key = f"{query_entity['text']} ({query_entity['label']})"
            
            # Start with basic clusters if they exist
            if cluster_key in basic_clusters:
                enhanced_clusters[cluster_key] = basic_clusters[cluster_key].copy()
            else:
                enhanced_clusters[cluster_key] = []
            
            # Add more entities with lower threshold if we don't have many matches
            if len(enhanced_clusters[cluster_key]) < 3:
                for d_idx, doc_entity in enumerate(doc_entities):
                    similarity = similarity_matrix[q_idx, d_idx]
                    
                    # Check if this entity is already in our cluster
                    already_included = any(
                        existing['text'] == doc_entity['text'] and existing['label'] == doc_entity['label']
                        for existing in enhanced_clusters[cluster_key]
                    )
                    
                    if not already_included and similarity >= lower_threshold:
                        enhanced_clusters[cluster_key].append({
                            'text': doc_entity['text'],
                            'label': doc_entity['label'],
                            'similarity': round(float(similarity), 3)
                        })
                
                # Sort by similarity
                enhanced_clusters[cluster_key].sort(key=lambda x: x['similarity'], reverse=True)
        
        # Remove empty clusters
        enhanced_clusters = {k: v for k, v in enhanced_clusters.items() if v}
        
        return enhanced_clusters
    
    async def process_queries_and_urls(self, 
                                     queries: List[str], 
                                     urls: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        
        # Fetch web content
        web_content = await fetch_pages_async(urls)
        valid_content = {url: content for url, content in web_content.items() 
                        if content and not content.startswith("[Error]")}
        
        if not valid_content:
            print("ERROR: No valid web content could be fetched")
            return {}
        
        # Combine web content
        combined_document = "\n\n".join([content for content in valid_content.values()])
        combined_query = " ".join(queries)
        
        # Extract entities
        query_entities = self.extract_entities(combined_query)
        doc_entities = self.extract_entities(combined_document)
        
        print(f"Extracted {len(query_entities)} query entities and {len(doc_entities)} document entities")
        
        if not query_entities or not doc_entities:
            return {}
        
        # Use enhanced semantic clustering
        clusters = self.enhanced_semantic_clustering(query_entities, doc_entities)
        return clusters
    
    def print_clusters(self, clusters: Dict[str, List[Dict[str, Any]]]) -> None:
        if not clusters:
            print("No entity clusters found above similarity threshold.")
            return
        
        print(f"\nSEMANTIC ENTITY CLUSTERS (threshold: {self.similarity_threshold})")
        print("=" * 70)
        
        for query_entity, similar_entities in clusters.items():
            print(f"\nQuery Entity: {query_entity}")
            print(f"Semantically Similar Entities ({len(similar_entities)}):")
            for entity in similar_entities:
                print(f"  • {entity['text']} ({entity['label']}) - {entity['similarity']}")



def get_question(answer="Yes", context="", max_length=64):
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = tokenizer([input_text], return_tensors='pt')

    output = model.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length)

    return tokenizer.decode(output[0])

def is_url(text: str) -> bool:
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
    if any(scheme in text.lower() for scheme in url_schemes):
        # Extract the first URL from the text and return it
        return True
    else:
        return False

def _min_max_norm(values: List[float]) -> Tuple[List[float], float, float]:
    """Normalize values using min-max normalization."""
    if not values:
        return [], 0.0, 0.0
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [0.0 for _ in values], vmin, vmax
    return [float((v - vmin) / (vmax - vmin)) for v in values], vmin, vmax
