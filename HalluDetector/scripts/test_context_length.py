#!/usr/bin/env python3
"""
Script to chunk web content from cache into word-sized chunks with overlap.
Each chunk is 250 words with 50 words overlap between consecutive chunks.
Chunks are saved in a separate output file to preserve the original cache.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

def count_words(text: str) -> int:
    """
    Count words in text by splitting on whitespace.
    """
    # Split on whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)

def create_chunks(text: str, chunk_size: int = 250, overlap: int = 50) -> List[str]:
    """
    Create overlapping chunks of text based on word count.
    
    Args:
        text: Input text to chunk
        chunk_size: Target size of each chunk in words
        overlap: Number of overlapping words between consecutive chunks
    
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
    
    # Split text into words
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(words):
        # Find the end position for this chunk
        end_pos = current_pos + chunk_size
        
        if end_pos >= len(words):
            # Last chunk - take remaining words
            chunk_words = words[current_pos:]
            if chunk_words:
                chunk_text = ' '.join(chunk_words)
                chunks.append(chunk_text)
            break
        
        # Get words for this chunk
        chunk_words = words[current_pos:end_pos]
        chunk_text = ' '.join(chunk_words)
        
        # Try to find a good breaking point (end of sentence)
        if end_pos < len(words):
            # Look for sentence endings in the last few words
            last_words = chunk_words[-10:]  # Check last 10 words
            for i, word in enumerate(reversed(last_words)):
                if word.endswith('.') or word.endswith('!') or word.endswith('?'):
                    # Found sentence end, adjust chunk
                    actual_end = end_pos - i
                    chunk_words = words[current_pos:actual_end]
                    chunk_text = ' '.join(chunk_words)
                    end_pos = actual_end
                    break
        
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        
        # Move to next position with overlap
        current_pos = max(current_pos + 1, end_pos - overlap)
    
    return chunks

def process_cache_file(cache_file_path: str, output_file_path: str, chunk_size: int = 250, overlap: int = 50):
    """
    Process the cache file and create chunks for each URL, saving them in a separate output file.
    
    Args:
        cache_file_path: Path to the input cache JSON file
        output_file_path: Path to the output file with chunks
        chunk_size: Target size of each chunk in words
        overlap: Number of overlapping words between consecutive chunks
    """
    # Read the cache file
    print(f"Reading cache file: {cache_file_path}")
    with open(cache_file_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
    
    print(f"Found {len(cache_data)} entries in cache")
    
    # Filter out entries that are already chunks or metadata
    original_urls = {}
    for key, value in cache_data.items():
        # Skip entries that are already dictionaries (chunks or metadata)
        if isinstance(value, dict):
            continue
        # Only process string content (original URLs)
        if isinstance(value, str):
            original_urls[key] = value
    
    print(f"Found {len(original_urls)} original URLs to process")
    
    # Create new data structure with chunks
    chunked_data = {}
    total_chunks = 0
    
    # First, add all existing entries to preserve them
    chunked_data.update(cache_data)
    
    for url, content in original_urls.items():
        print(f"\nProcessing: {url}")
        
        # Create chunks
        chunks = create_chunks(content, chunk_size, overlap)
        
        if not chunks:
            print(f"  No content to chunk for {url}")
            continue
        
        print(f"  Created {len(chunks)} chunks")
        
        # Add each chunk with a descriptive name
        for i, chunk in enumerate(chunks):
            chunk_name = f"{url}_chunk_{i+1:03d}"
            chunked_data[chunk_name] = {
                "original_url": url,
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "word_count": count_words(chunk),
                "chunk_size": chunk_size,
                "overlap": overlap,
                "content": chunk
            }
            total_chunks += 1
        
        # Add metadata for this URL
        metadata_name = f"{url}_metadata"
        chunked_data[metadata_name] = {
            "original_url": url,
            "total_chunks": len(chunks),
            "total_words": count_words(content),
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_names": [f"{url}_chunk_{i+1:03d}" for i in range(len(chunks))]
        }
    
    # Save the chunked data to the output file
    print(f"\nSaving {total_chunks} new chunks to {output_file_path}")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete!")
    print(f"New chunks created: {total_chunks}")
    print(f"Output file: {output_file_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Original URLs processed: {len(original_urls)}")
    print(f"New chunks created: {total_chunks}")
    print(f"Total entries in output file: {len(chunked_data)}")
    print(f"Original cache file preserved: {cache_file_path}")

def main():
    parser = argparse.ArgumentParser(description='Chunk web content from cache into word-sized chunks')
    parser.add_argument('--cache-file', 
                       default='../web_content_cache/cache_gemini_PhD_jobs.json',
                       help='Path to the input cache JSON file')
    parser.add_argument('--output-file', 
                       default='../web_content_cache/cache_gemini_PhD_jobs_chunked.json',
                       help='Path to the output file with chunks')
    parser.add_argument('--chunk-size', 
                       type=int, 
                       default=200,
                       help='Target size of each chunk in words (default: 250)')
    parser.add_argument('--overlap', 
                       type=int, 
                       default=25,
                       help='Number of overlapping words between consecutive chunks (default: 50)')
    
    args = parser.parse_args()
    
    # Check if cache file exists
    if not os.path.exists(args.cache_file):
        print(f"Error: Cache file not found: {args.cache_file}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        process_cache_file(args.cache_file, args.output_file, args.chunk_size, args.overlap)
    except Exception as e:
        print(f"Error processing cache file: {e}")
        raise

if __name__ == "__main__":
    main()
