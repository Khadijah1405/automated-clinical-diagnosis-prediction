#!/usr/bin/env python3
"""
Comprehensive diagnostic to analyze both input CSV and mapping CSV
to understand the code mismatch issue
"""

import pandas as pd
import json
from collections import Counter

def analyze_input_csv():
    """Analyze the input CSV structure and codes"""
    print("ANALYZING INPUT CSV: textfinal_hcup_categories_FIXED.csv")
    print("=" * 60)
    
    try:
        df = pd.read_csv("textfinal_hcup_categories_FIXED.csv", low_memory=False)
        
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        
        # Analyze icd_code column
        if 'icd_code' in df.columns:
            print(f"\nICD_CODE column analysis:")
            icd_codes = df['icd_code'].dropna()
            print(f"  Non-null values: {len(icd_codes)}")
            print(f"  Sample values: {list(icd_codes.head(10))}")
            
            # Parse comma-separated codes
            all_codes = set()
            for codes_str in icd_codes:
                if pd.notna(codes_str):
                    codes = [c.strip() for c in str(codes_str).split(',')]
                    all_codes.update(codes)
            
            print(f"  Unique codes after splitting: {len(all_codes)}")
            print(f"  Sample individual codes: {list(list(all_codes)[:10])}")
        
        # Analyze icd_version column
        if 'icd_version' in df.columns:
            print(f"\nICD_VERSION column analysis:")
            versions = df['icd_version'].value_counts()
            print(f"  Value counts: {dict(versions)}")
            print(f"  Sample values: {list(df['icd_version'].dropna().head(10))}")
        
        # Analyze hcup columns
        hcup_columns = [col for col in df.columns if 'hcup' in col.lower()]
        print(f"\nHCUP-related columns: {hcup_columns}")
        
        for col in hcup_columns:
            if col in df.columns:
                non_null = df[col].dropna()
                print(f"  {col}: {len(non_null)} non-null values")
                if len(non_null) > 0:
                    print(f"    Sample: {list(non_null.head(3))}")
        
        return df
        
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return None

def analyze_mapping_csv():
    """Analyze the mapping CSV structure and codes"""
    print("\nANALYZING MAPPING CSV: final_hcup_mappings.csv")
    print("=" * 60)
    
    try:
        df = pd.read_csv("final_hcup_mappings.csv")
        
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Column names: {list(df.columns)}")
        
        # Analyze icd_code column
        if 'icd_code' in df.columns:
            print(f"\nICD_CODE column analysis:")
            icd_codes = df['icd_code'].dropna()
            print(f"  Non-null values: {len(icd_codes)}")
            print(f"  Sample raw values: {list(icd_codes.head(10))}")
            
            # Clean codes (remove quotes)
            clean_codes = [str(code).strip().strip("'\"") for code in icd_codes]
            clean_codes = [code for code in clean_codes if code]
            
            print(f"  Sample cleaned codes: {clean_codes[:10]}")
            print(f"  Unique cleaned codes: {len(set(clean_codes))}")
        
        # Analyze source column
        if 'source' in df.columns:
            print(f"\nSOURCE column analysis:")
            sources = df['source'].value_counts()
            print(f"  Value counts: {dict(sources)}")
        
        # Show sample rows
        print(f"\nSample rows:")
        for i, row in df.head(5).iterrows():
            icd_code = row.get('icd_code', 'N/A')
            source = row.get('source', 'N/A')
            print(f"  Row {i}: icd_code='{icd_code}', source='{source}'")
        
        return df
        
    except Exception as e:
        print(f"Error reading mapping CSV: {e}")
        return None

def analyze_code_relationship(input_df, mapping_df):
    """Analyze the relationship between codes in both files"""
    print("\nANALYZING CODE RELATIONSHIPS")
    print("=" * 60)
    
    if input_df is None or mapping_df is None:
        print("Cannot analyze - one or both files failed to load")
        return
    
    # Extract codes from input CSV
    input_codes = set()
    if 'icd_code' in input_df.columns:
        for codes_str in input_df['icd_code'].dropna():
            if pd.notna(codes_str):
                codes = [c.strip() for c in str(codes_str).split(',')]
                input_codes.update(codes)
    
    # Extract codes from mapping CSV
    mapping_codes = set()
    if 'icd_code' in mapping_df.columns:
        for code in mapping_df['icd_code'].dropna():
            clean_code = str(code).strip().strip("'\"")
            if clean_code:
                mapping_codes.add(clean_code)
    
    print(f"Input CSV codes: {len(input_codes)}")
    print(f"Mapping CSV codes: {len(mapping_codes)}")
    
    # Find overlaps
    overlap = input_codes.intersection(mapping_codes)
    input_only = input_codes - mapping_codes
    mapping_only = mapping_codes - input_codes
    
    print(f"\nCode overlap analysis:")
    print(f"  Codes in both: {len(overlap)} ({len(overlap)/len(input_codes)*100:.1f}% of input codes)")
    print(f"  Only in input: {len(input_only)}")
    print(f"  Only in mapping: {len(mapping_only)}")
    
    print(f"\nSample overlapping codes: {list(overlap)[:10]}")
    print(f"Sample input-only codes: {list(input_only)[:10]}")
    print(f"Sample mapping-only codes: {list(mapping_only)[:10]}")
    
    # Analyze format differences
    def analyze_code_formats(codes, name):
        formats = {
            'quoted': sum(1 for c in list(codes)[:100] if c.startswith("'") or c.startswith('"')),
            'numeric': sum(1 for c in list(codes)[:100] if c.replace('.', '').replace("'", '').replace('"', '').isdigit()),
            'alphanumeric': sum(1 for c in list(codes)[:100] if not c.replace('.', '').replace("'", '').replace('"', '').isdigit()),
            'with_dots': sum(1 for c in list(codes)[:100] if '.' in c),
            'length_3_5': sum(1 for c in list(codes)[:100] if 3 <= len(c.strip("'\"")) <= 5),
            'length_6_plus': sum(1 for c in list(codes)[:100] if len(c.strip("'\"")) > 5)
        }
        
        print(f"\n{name} format analysis (sample of 100):")
        for fmt, count in formats.items():
            print(f"  {fmt}: {count}")
    
    analyze_code_formats(input_codes, "Input codes")
    analyze_code_formats(mapping_codes, "Mapping codes")

def check_dataset_creation():
    """Check how the QA dataset was created"""
    print("\nANALYZING DATASET CREATION")
    print("=" * 60)
    
    try:
        with open("simple_qaicd.jsonl", "r") as f:
            qa_items = [json.loads(line) for line in f]
        
        print(f"QA dataset contains {len(qa_items)} items")
        
        # Analyze target versions
        target_versions = []
        all_targets = set()
        
        for item in qa_items:
            targets = item.get("valid_targets", [])
            version = item.get("target_version", "unknown")
            target_versions.append(version)
            all_targets.update(targets)
        
        version_counts = Counter(target_versions)
        print(f"Target version distribution: {dict(version_counts)}")
        print(f"Total unique target codes: {len(all_targets)}")
        print(f"Sample target codes: {list(all_targets)[:10]}")
        
        # Check first few items in detail
        print(f"\nFirst 3 QA items:")
        for i, item in enumerate(qa_items[:3]):
            targets = item.get("valid_targets", [])
            version = item.get("target_version", "unknown")
            print(f"  Item {i}: targets={targets}, version='{version}'")
        
    except Exception as e:
        print(f"Error analyzing QA dataset: {e}")

def main():
    """Run comprehensive diagnostic"""
    print("COMPREHENSIVE CSV DIAGNOSTIC ANALYSIS")
    print("=" * 80)
    
    # Analyze both CSV files
    input_df = analyze_input_csv()
    mapping_df = analyze_mapping_csv()
    
    # Analyze relationships
    analyze_code_relationship(input_df, mapping_df)
    
    # Check dataset creation
    check_dataset_creation()
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("This should help identify why codes aren't matching properly.")

if __name__ == "__main__":
    main()
