#!/usr/bin/env python3
"""
Final Fixed HCUP Script - Handles the exact ICD-9 file format from your screenshots
"""

import pandas as pd
import os
from pathlib import Path

def process_icd10_ccsr_file():
    """Process ICD-10 CCSR file"""
    print("📊 Processing ICD-10 CCSR files...")
    
    extract_path = Path('/home/khsh060f/rp/downloads/icd10_extract')
    target_file = extract_path / "DXCCSR_v2025-1.csv"
    
    icd10_mappings = {}
    
    if target_file.exists():
        try:
            df = pd.read_csv(target_file, dtype=str, low_memory=False)
            df.columns = [col.strip().strip("'\"") for col in df.columns]
            
            icd_col = 'ICD-10-CM CODE'
            desc_col = 'ICD-10-CM CODE DESCRIPTION' 
            ccsr_code_col = 'Default CCSR CATEGORY IP'
            ccsr_desc_col = 'Default CCSR CATEGORY DESCRIPTION IP'
            
            processed_count = 0
            for _, row in df.iterrows():
                try:
                    icd_code = str(row[icd_col]).strip().replace('.', '')
                    
                    if pd.isna(icd_code) or icd_code == 'nan' or not icd_code:
                        continue
                    
                    icd_description = str(row[desc_col]).strip()
                    ccsr_code = str(row[ccsr_code_col]).strip()
                    ccsr_description = str(row[ccsr_desc_col]).strip()
                    
                    if pd.isna(ccsr_code) or ccsr_code == 'nan' or not ccsr_code:
                        continue
                    
                    icd10_mappings[icd_code] = {
                        'icd_code': icd_code,
                        'icd_description': icd_description,
                        'category_code': ccsr_code,
                        'category_description': ccsr_description,
                        'source': 'ICD10_CCSR'
                    }
                    processed_count += 1
                    
                except Exception as e:
                    continue
            
            print(f"  ✅ Processed {processed_count} ICD-10 mappings")
            
        except Exception as e:
            print(f"  ❌ Error processing ICD-10 file: {e}")
    
    return icd10_mappings

def load_ccs_category_labels():
    """Load CCS category labels from dxlabel 2015.csv"""
    print("📊 Loading CCS category labels...")
    
    extract_path = Path('/home/khsh060f/rp/downloads/icd9_extract')
    label_file = extract_path / "dxlabel 2015.csv"
    
    ccs_labels = {}
    
    if label_file.exists():
        try:
            # Read the label file
            df = pd.read_csv(label_file, dtype=str)
            print(f"  📁 Found label file: {label_file.name}")
            print(f"  📊 Shape: {df.shape}")
            print(f"  📋 Columns: {list(df.columns)}")
            
            # Clean column names
            df.columns = [col.strip() for col in df.columns]
            
            # Expected columns: 'CCS DIAGNOSIS CATEGORIES', 'CCS DIAGNOSIS CATEGORIES LABELS'
            category_col = df.columns[0]  # CCS codes
            label_col = df.columns[1]     # CCS descriptions
            
            processed_count = 0
            for _, row in df.iterrows():
                try:
                    ccs_code = str(row[category_col]).strip()
                    ccs_description = str(row[label_col]).strip()
                    
                    if ccs_code and ccs_description and ccs_code != 'nan':
                        ccs_labels[ccs_code] = ccs_description
                        processed_count += 1
                
                except Exception as e:
                    continue
            
            print(f"  ✅ Loaded {processed_count} CCS category labels")
            
            # Show sample labels
            print(f"  📝 Sample CCS labels:")
            sample_keys = list(ccs_labels.keys())[:5]
            for key in sample_keys:
                print(f"    {key}: {ccs_labels[key]}")
            
        except Exception as e:
            print(f"  ❌ Error loading CCS labels: {e}")
    else:
        print(f"  ❌ Label file not found: {label_file}")
    
    return ccs_labels

def process_icd9_dxref_file():
    """Process the specific $dxref 2015.csv file with CCS label lookup"""
    print("📊 Processing ICD-9 $dxref file...")
    
    # First, load the CCS category labels
    ccs_labels = load_ccs_category_labels()
    
    extract_path = Path('/home/khsh060f/rp/downloads/icd9_extract')
    target_file = extract_path / "$dxref 2015.csv"
    
    icd9_mappings = {}
    
    if target_file.exists():
        print(f"  📁 Found target file: {target_file.name}")
        
        try:
            # Read the raw file content first
            with open(target_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Split into lines
            lines = raw_content.strip().split('\n')
            print(f"  📊 Total lines in file: {len(lines)}")
            
            # Show first few lines for debugging
            print(f"  📝 First 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"    {i}: {line[:100]}...")
            
            # Find the header line (should be line 1 based on your screenshot)
            header_line = None
            data_start = 0
            
            for i, line in enumerate(lines):
                if 'ICD-9-CM CODE' in line and 'CCS CATEGORY' in line:
                    header_line = line
                    data_start = i + 1
                    print(f"  📋 Found header at line {i}: {header_line}")
                    break
            
            if header_line is None:
                print(f"  ❌ Could not find header line with expected columns")
                return {}
            
            # Parse the header to understand column structure
            # Remove quotes and split by comma
            header_parts = [col.strip().strip("'\"") for col in header_line.split(',')]
            print(f"  📋 Header columns: {header_parts}")
            
            # Process data lines
            processed_count = 0
            error_count = 0
            labels_found = 0
            labels_missing = 0
            
            for line_num, line in enumerate(lines[data_start:], start=data_start):
                try:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Split by comma and remove quotes
                    parts = [part.strip().strip("'\"") for part in line.split(',')]
                    
                    # Skip if not enough parts
                    if len(parts) < 4:
                        continue
                    
                    icd_code = parts[0].strip()
                    ccs_code = parts[1].strip()
                    ccs_description_from_file = parts[2].strip()
                    icd_description = parts[3].strip() if len(parts) > 3 else ""
                    
                    # Skip invalid entries
                    if not icd_code or not ccs_code or icd_code in ['', 'ICD-9-CM CODE']:
                        continue
                    
                    # Skip the '0' category (invalid codes)
                    if ccs_code == '0':
                        continue
                    
                    # Clean the ICD code
                    icd_code = icd_code.replace('.', '')
                    
                    # Use CCS label from lookup table if available, otherwise use file description
                    if ccs_code in ccs_labels:
                        ccs_description = ccs_labels[ccs_code]
                        labels_found += 1
                    else:
                        ccs_description = ccs_description_from_file
                        labels_missing += 1
                    
                    # Use ICD description from file, or fallback to CCS description
                    if not icd_description:
                        icd_description = f"ICD-9: {ccs_description}"
                    
                    # Store the mapping
                    icd9_mappings[icd_code] = {
                        'icd_code': icd_code,
                        'icd_description': icd_description,
                        'category_code': ccs_code,
                        'category_description': ccs_description,
                        'source': 'ICD9_CCS'
                    }
                    processed_count += 1
                    
                    # Show progress
                    if processed_count % 1000 == 0:
                        print(f"    📊 Processed {processed_count} mappings...")
                    
                except Exception as e:
                    error_count += 1
                    if error_count < 5:  # Show first few errors
                        print(f"    ⚠️  Error on line {line_num}: {e}")
                    continue
            
            print(f"  ✅ Successfully processed {processed_count} ICD-9 mappings")
            print(f"  📊 CCS labels found in lookup: {labels_found}")
            print(f"  📊 CCS labels from file: {labels_missing}")
            print(f"  📊 Errors encountered: {error_count}")
            
            # Show sample mappings
            if icd9_mappings:
                print(f"  📝 Sample ICD-9 mappings with CCS labels:")
                sample_keys = list(icd9_mappings.keys())[:5]
                for key in sample_keys:
                    data = icd9_mappings[key]
                    print(f"    {key} → CCS:{data['category_code']} ({data['category_description'][:50]}...)")
            
        except Exception as e:
            print(f"  ❌ Error processing $dxref file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ❌ File not found: {target_file}")
    
    return icd9_mappings

def save_final_mappings(icd10_mappings, icd9_mappings):
    """Save the final processed mappings"""
    print(f"\n💾 Saving Final HCUP Mappings...")
    
    # Combine mappings
    all_mappings = {}
    all_mappings.update(icd10_mappings)
    
    # Add ICD-9 mappings (don't overwrite ICD-10)
    icd9_added = 0
    for icd_code, data in icd9_mappings.items():
        if icd_code not in all_mappings:
            all_mappings[icd_code] = data
            icd9_added += 1
    
    print(f"  📊 ICD-9 mappings added: {icd9_added}")
    
    # Convert to DataFrame
    mapping_data = []
    for icd_code, data in all_mappings.items():
        mapping_data.append({
            'icd_code': data['icd_code'],
            'icd_description': data['icd_description'],
            'category_code': data['category_code'],
            'category_description': data['category_description'],
            'source': data['source']
        })
    
    df = pd.DataFrame(mapping_data)
    
    # Save main file
    output_file = 'final_hcup_mappings.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df):,} mappings to '{output_file}'")
    
    # Generate comprehensive statistics
    source_counts = df['source'].value_counts()
    category_counts = df['category_code'].value_counts()
    
    print(f"\n📊 Final Statistics:")
    for source, count in source_counts.items():
        print(f"   {source}: {count:,} mappings")
    
    print(f"   Total unique categories: {len(category_counts):,}")
    
    # Analyze category distribution by source
    icd10_categories = len(df[df['source'] == 'ICD10_CCSR']['category_code'].unique())
    icd9_categories = len(df[df['source'] == 'ICD9_CCS']['category_code'].unique())
    
    print(f"   ICD-10 unique categories: {icd10_categories:,}")
    print(f"   ICD-9 unique categories: {icd9_categories:,}")
    
    # Show most common categories
    print(f"\n📊 Top 15 Categories:")
    for i, (cat, count) in enumerate(category_counts.head(15).items(), 1):
        sample_desc = df[df['category_code'] == cat]['category_description'].iloc[0]
        source = df[df['category_code'] == cat]['source'].iloc[0]
        print(f"   {i:2d}. {cat} ({source}): {count:,} codes - {sample_desc[:45]}...")
    
    # Create comprehensive summary
    with open('final_processing_summary.txt', 'w') as f:
        f.write("FINAL HCUP MAPPING PROCESSING RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"TOTAL MAPPINGS PROCESSED: {len(df):,}\n\n")
        
        f.write("BY SOURCE:\n")
        for source, count in source_counts.items():
            f.write(f"  {source}: {count:,} mappings\n")
        f.write(f"\nUNIQUE CATEGORIES: {len(category_counts):,}\n")
        f.write(f"  ICD-10 categories: {icd10_categories:,}\n")
        f.write(f"  ICD-9 categories: {icd9_categories:,}\n\n")
        
        f.write("SOLUTION TO YOUR 96% UNIQUE ICD PROBLEM:\n")
        f.write(f"- Original problem: ICD codes were 96% unique\n")
        f.write(f"- Solution: Group by 'category_code' column\n")
        f.write(f"- Result: {len(category_counts):,} categories instead of unique ICDs\n")
        f.write(f"- Reduction: Massive improvement in data grouping capability\n\n")
        
        f.write("USAGE INSTRUCTIONS:\n")
        f.write("1. Use 'category_code' column for statistical analysis\n")
        f.write("2. Use 'category_description' for human-readable labels\n")
        f.write("3. Filter by 'source' if you want only ICD-9 or ICD-10\n\n")
        
        f.write("TOP CATEGORIES:\n")
        for i, (cat, count) in enumerate(category_counts.head(20).items(), 1):
            desc = df[df['category_code'] == cat]['category_description'].iloc[0]
            source = df[df['category_code'] == cat]['source'].iloc[0]
            f.write(f"  {i:2d}. {cat} ({source}): {count:,} codes - {desc}\n")
    
    print(f"✅ Saved comprehensive summary to 'final_processing_summary.txt'")
    
    return df

def main():
    """Main function with targeted ICD-9 file processing"""
    print("🏥 FINAL FIXED HCUP MAPPING PROCESSOR")
    print("="*55)
    print("🎯 Targets the exact $dxref 2015.csv format")
    print("📊 Handles the specific file structure from your screenshots")
    print("="*55)
    
    try:
        # Process ICD-10
        print("\n🔄 Processing ICD-10 files...")
        icd10_mappings = process_icd10_ccsr_file()
        
        # Process ICD-9 with targeted approach
        print("\n🔄 Processing ICD-9 files with targeted parsing...")
        icd9_mappings = process_icd9_dxref_file()
        
        # Save final results
        final_df = save_final_mappings(icd10_mappings, icd9_mappings)
        
        print(f"\n🎉 FINAL PROCESSING COMPLETE!")
        print(f"📊 Results:")
        print(f"   • ICD-10: {len(icd10_mappings):,} mappings")
        print(f"   • ICD-9: {len(icd9_mappings):,} mappings")
        print(f"   • Total: {len(final_df):,} mappings")
        print(f"   • Unique categories: {len(final_df['category_code'].unique()):,}")
        
        print(f"\n📁 Output files:")
        print(f"   • final_hcup_mappings.csv")
        print(f"   • final_processing_summary.txt")
        
        if len(icd9_mappings) > 0:
            print(f"\n✅ SUCCESS! Both ICD-9 and ICD-10 files processed successfully!")
            print(f"🎯 Your 96% unique ICD problem is now solved!")
            print(f"📊 Use the 'category_code' column for grouping your data")
        else:
            print(f"\n⚠️  ICD-9 processing still failed")
            print(f"💡 But you have {len(icd10_mappings):,} ICD-10 mappings to work with")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
