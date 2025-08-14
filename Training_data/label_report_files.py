#!/usr/bin/env python3
"""
Script to extract filenames from Report folder and label matching rows in overview.csv
"""

import os
import csv
from pathlib import Path

def extract_filenames_from_report(report_path):
    """
    Extract all filenames (without extensions) from the Report folder
    
    Args:
        report_path (str): Path to the Report folder
        
    Returns:
        set: Set of filenames without extensions
    """
    filenames = set()
    
    if not os.path.exists(report_path):
        print(f"Warning: Report folder not found at {report_path}")
        return filenames
    
    for filename in os.listdir(report_path):
        # Skip hidden files and directories
        if filename.startswith('.') or os.path.isdir(os.path.join(report_path, filename)):
            continue
            
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        filenames.add(name_without_ext)
    
    return filenames

def label_matching_rows(csv_path, report_filenames):
    """
    Read overview.csv and add a column to label rows where md5 matches report filenames
    
    Args:
        csv_path (str): Path to overview.csv
        report_filenames (set): Set of filenames from Report folder
        
    Returns:
        list: List of rows with new column
    """
    rows = []
    header = None
    
    # Read the CSV file
    with open(csv_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Get header row
        
        # Add new column header
        header.append('in_report')
        rows.append(header)
        
        # Process data rows
        for row in csv_reader:
            if len(row) > 0:  # Skip empty rows
                # Check if md5 column (index 1) matches any report filename
                md5_value = row[1] if len(row) > 1 else ""
                in_report = "1" if md5_value in report_filenames else "0"
                row.append(in_report)
                rows.append(row)
    
    return rows

def main():
    # Define paths
    current_dir = Path(__file__).parent
    report_path = current_dir / "gemini" / "Report"
    csv_path = current_dir / "overview.csv"
    
    print(f"Report folder path: {report_path}")
    print(f"CSV file path: {csv_path}")
    
    # Extract filenames from Report folder
    print("\nExtracting filenames from Report folder...")
    report_filenames = extract_filenames_from_report(report_path)
    print(f"Found {len(report_filenames)} files in Report folder")
    
    # Print first few filenames for verification
    if report_filenames:
        print("Sample filenames:", list(report_filenames)[:5])
    
    # Get all md5 values from overview.csv for comparison
    print("\nExtracting md5 values from overview.csv...")
    csv_md5_values = set()
    with open(csv_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            if len(row) > 1 and row[1]:  # Check if md5 column exists and is not empty
                csv_md5_values.add(row[1])
    
    print(f"Found {len(csv_md5_values)} md5 values in overview.csv")
    
    # Find mismatched files
    matched_files = report_filenames.intersection(csv_md5_values)
    unmatched_files = report_filenames - csv_md5_values
    missing_in_csv = csv_md5_values - report_filenames
    
    print(f"\nMatching analysis:")
    print(f"Files in Report folder: {len(report_filenames)}")
    print(f"MD5 values in CSV: {len(csv_md5_values)}")
    print(f"Matched files: {len(matched_files)}")
    print(f"Unmatched files (in Report but not in CSV): {len(unmatched_files)}")
    print(f"Missing files (in CSV but not in Report): {len(missing_in_csv)}")
    
    # Show unmatched files
    if unmatched_files:
        print(f"\nUnmatched files (in Report but not in CSV):")
        for filename in sorted(unmatched_files):
            print(f"  {filename}")
    
    # Show missing files
    if missing_in_csv:
        print(f"\nMissing files (in CSV but not in Report):")
        for md5 in sorted(missing_in_csv)[:10]:  # Show first 10
            print(f"  {md5}")
        if len(missing_in_csv) > 10:
            print(f"  ... and {len(missing_in_csv) - 10} more")
    
    # Label matching rows in overview.csv
    print("\nProcessing overview.csv...")
    updated_rows = label_matching_rows(csv_path, report_filenames)
    
    # Save updated CSV
    output_path = current_dir / "overview_labeled.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(updated_rows)
    
    print(f"\nUpdated CSV saved to: {output_path}")
    
    # Show summary of labeling
    print("\nSummary of labeling:")
    total_rows = len(updated_rows) - 1  # Exclude header
    labeled_count = sum(1 for row in updated_rows[1:] if row[-1] == "1")
    print(f"Total rows: {total_rows}")
    print(f"Labeled rows: {labeled_count}")
    print(f"Unlabeled rows: {total_rows - labeled_count}")
    
    # Show some examples of labeled rows
    print("\nSample of labeled rows:")
    labeled_rows = [row for row in updated_rows[1:] if row[-1] == "1"]
    if labeled_rows:
        for i, row in enumerate(labeled_rows[:3]):  # Show first 3 labeled rows
            print(f"Row {i+1}: md5={row[1]}, in_report={row[-1]}")
            # Truncate long queries for display
            query = row[0][:100] + "..." if len(row[0]) > 100 else row[0]
            print(f"  Query: {query}")
    else:
        print("No matching rows found")

if __name__ == "__main__":
    main()
