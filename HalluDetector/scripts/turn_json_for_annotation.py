#!/usr/bin/env python3
"""
Script to convert JSON file with specific format into CSV for annotation.
Dynamically extracts attributes: query_list, all action_list_x, all claim_list_x, and all non-empty atomic_claims from report.
Each attribute becomes a column, with binary hallucination values in the next column.
All non-empty atomic_claims from report are combined into a single column.
Filters out any items containing URLs and maintains original JSON attribute order.
"""

import json
import csv
import sys
import re
from typing import Dict, List, Any


def contains_url(text: str) -> bool:
    """Check if text contains a URL."""
    if not isinstance(text, str):
        return False
    # Simple URL pattern matching
    url_pattern = r'https?://|www\.|\.com|\.org|\.net|\.edu|\.gov'
    return bool(re.search(url_pattern, text, re.IGNORECASE))


def get_specific_attributes(data: Dict[str, Any]) -> List[str]:
    """Dynamically identify the specific attributes to extract from the JSON, maintaining order."""
    attributes = []
    
    # Add query_list first if it exists
    if "query_list" in data:
        attributes.append("query_list")
    
    # Find all action_list_x and claim_list_x attributes in order of appearance
    if "iterations" in data:
        for iteration in data["iterations"]:
            for key in iteration.keys():
                if re.match(r'^action_list_\d+$', key) or re.match(r'^claim_list_\d+$', key):
                    if key not in attributes:
                        attributes.append(key)
    
    # Add atomic_claims last
    attributes.append("atomic_claims")
    
    return attributes


def get_max_length(data: Dict[str, Any], attributes: List[str]) -> int:
    """Get the maximum length of any attribute list to determine CSV rows."""
    max_len = 0
    
    for attr in attributes:
        if attr == "atomic_claims":
            # Count all non-empty atomic_claims from report (filtering URLs)
            if "report" in data:
                count = sum(len([claim for claim in item["atomic_claims"] 
                               if item.get("atomic_claims") and not contains_url(claim)])
                           for item in data["report"] 
                           if item.get("atomic_claims") and len(item["atomic_claims"]) > 0)
                max_len = max(max_len, count)
        elif attr == "query_list":
            # Handle query_list directly (filtering URLs)
            if "query_list" in data and isinstance(data["query_list"], list):
                filtered_count = len([item for item in data["query_list"] if not contains_url(item)])
                max_len = max(max_len, filtered_count)
        else:
            # Handle action_list_x and claim_list_x from iterations (filtering URLs)
            if "iterations" in data:
                for iteration in data["iterations"]:
                    if attr in iteration and isinstance(iteration[attr], list):
                        filtered_count = len([item for item in iteration[attr] if not contains_url(item)])
                        max_len = max(max_len, filtered_count)
    
    return max_len


def get_all_atomic_claims(data: Dict[str, Any]) -> List[str]:
    """Get all non-empty atomic_claims from report as a flat list, filtering out URLs."""
    all_claims = []
    if "report" in data:
        for item in data["report"]:
            if item.get("atomic_claims") and len(item["atomic_claims"]) > 0:
                # Filter out claims containing URLs
                filtered_claims = [claim for claim in item["atomic_claims"] if not contains_url(claim)]
                all_claims.extend(filtered_claims)
    return all_claims


def get_filtered_list_items(data: Dict[str, Any], attr: str, index: int) -> str:
    """Get filtered list items (without URLs) for a specific attribute and index."""
    if attr == "query_list":
        if "query_list" in data and isinstance(data["query_list"], list):
            filtered_items = [item for item in data["query_list"] if not contains_url(item)]
            if index < len(filtered_items):
                return filtered_items[index]
    elif attr == "atomic_claims":
        # Handle atomic_claims specially - get all non-empty claims (filtered)
        all_claims = get_all_atomic_claims(data)
        if index < len(all_claims):
            return all_claims[index]
    else:
        # Handle action_list_x and claim_list_x from iterations (filtered)
        if "iterations" in data:
            for iteration in data["iterations"]:
                if attr in iteration and isinstance(iteration[attr], list):
                    filtered_items = [item for item in iteration[attr] if not contains_url(item)]
                    if index < len(filtered_items):
                        return filtered_items[index]
    return ""


def convert_json_to_csv(json_file_path: str, csv_file_path: str):
    """Convert JSON file to CSV format for annotation."""
    
    # Read JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{json_file_path}': {e}")
        return
    
    # Get the specific attributes to extract (dynamically, maintaining order)
    attributes = get_specific_attributes(data)
    
    # Get maximum length for CSV rows
    max_len = get_max_length(data, attributes)
    
    # Prepare CSV data
    csv_data = []
    
    for i in range(max_len):
        row = {}
        
        for attr in attributes:
            # Get the filtered attribute value
            value = get_filtered_list_items(data, attr, i)
            row[f"{attr}"] = value
            row[f"{attr}_hallucination"] = ""  # To be filled by annotators
        
        csv_data.append(row)
    
    # Write CSV file
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                fieldnames = []
                for attr in attributes:
                    fieldnames.append(attr)
                    fieldnames.append(f"{attr}_hallucination")
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
                
                print(f"Successfully converted '{json_file_path}' to '{csv_file_path}'")
                print(f"CSV contains {len(csv_data)} rows and {len(fieldnames)} columns")
                print(f"Attributes extracted (in order): {', '.join(attributes)}")
                
                # Print some statistics
                if "report" in data:
                    total_claims = len(get_all_atomic_claims(data))
                    print(f"Total atomic claims extracted (URLs filtered): {total_claims}")
                
            else:
                print("No data to write to CSV.")
                
    except Exception as e:
        print(f"Error writing CSV file: {e}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) != 3:
        print("Usage: python turn_json_for_annotation.py <input_json_file> <output_csv_file>")
        print("Example: python turn_json_for_annotation.py input.json output.csv")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    csv_file_path = sys.argv[2]
    
    convert_json_to_csv(json_file_path, csv_file_path)


if __name__ == "__main__":
    main()
