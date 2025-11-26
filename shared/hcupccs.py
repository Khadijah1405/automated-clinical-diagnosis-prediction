#!/usr/bin/env python3
"""
Complete HCUP CCS Download and Integration Script
Downloads official HCUP mappings and combines with your custom mappings
"""

import pandas as pd
import requests
import zipfile
import os
from pathlib import Path
import urllib.request

def download_hcup_ccs_files():
    """
    Download official HCUP CCS mapping files
    """
    print("🔄 Downloading Official HCUP CCS Files...")
    
    # Create directories
    os.makedirs("official_data", exist_ok=True)
    os.makedirs("downloads", exist_ok=True)
    
    # HCUP CCS file URLs (as of 2024)
    urls = {
        # ICD-10-CM Single Level CCS
        "icd10_ccs": "https://hcup-us.ahrq.gov/toolssoftware/ccsr/DXCCSR_v2025-1.zip",
        
        # ICD-9-CM Single Level CCS  
        "icd9_ccs": "https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Single_Level_CCS_2015.zip",
        
        # CCS Category Labels
        "ccs_labels": "https://hcup-us.ahrq.gov/toolssoftware/ccsr/DXCCSR-Reference-File-v2025-1.xlsx"
    }
    
    downloaded_files = {}
    
    for file_type, url in urls.items():
        try:
            print(f"  📥 Downloading {file_type}...")
            local_filename = f"downloads/{file_type}.zip"
            
            # Download with headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded_files[file_type] = local_filename
            print(f"    ✅ Downloaded {file_type}")
            
        except Exception as e:
            print(f"    ❌ Failed to download {file_type}: {e}")
            # Try alternative approach for this file
            downloaded_files[file_type] = None
    
    return downloaded_files


def process_icd10_diagnosis_file():
    """
    Process ONLY the ICD-10 diagnosis mapping file (DXCCSR_v2025-1.csv)
    Extract EXACTLY what's in the file - nothing more, nothing less
    """
    print("\n🔄 Processing ICD-10 Diagnosis File...")
    
    diagnosis_file = "downloads/icd10_extract/DXCCSR_v2025-1.csv"
    
    if not os.path.exists(diagnosis_file):
        print(f"❌ File not found: {diagnosis_file}")
        return {}
    
    try:
        # Read first few rows to see structure
        sample_df = pd.read_csv(diagnosis_file, nrows=5)
        print(f"📋 File columns: {list(sample_df.columns)}")
        print(f"📝 Sample data:")
        print(sample_df.to_string(index=False))
        
        # Read full file
        df = pd.read_csv(diagnosis_file)
        print(f"📊 Total rows in file: {len(df):,}")
        
        # Extract mappings - use EXACTLY what's in the file
        mappings = {}
        processed = 0
        
        for _, row in df.iterrows():
            try:
                # Extract columns exactly as they appear in file
                col_values = [str(val).strip() for val in row.values]
                
                # Skip empty rows
                if not any(col_values) or all(val in ['', 'nan', 'NaN'] for val in col_values):
                    continue
                
                # Use the data exactly as provided in the file
                # Assuming structure: ICD_CODE, DESCRIPTION, CCSR_CODE (adjust based on actual file)
                icd_code = col_values[0].replace("'", "").replace('"', '').replace('.', '')
                
                if len(col_values) >= 2:
                    # Store exactly what's in the file
                    mappings[icd_code] = tuple(col_values[1:])  # All other columns as-is
                    processed += 1
                
            except Exception as e:
                continue  # Skip problematic rows
        
        print(f"✅ Processed {processed:,} mappings from diagnosis file")
        
        # Show sample of what was extracted
        sample_keys = list(mappings.keys())[:5]
        print(f"📝 Sample extracted data:")
        for key in sample_keys:
            print(f"   {key}: {mappings[key]}")
        
        return mappings
        
    except Exception as e:
        print(f"❌ Error processing diagnosis file: {e}")
        return {}

def process_icd9_diagnosis_files():
    """
    Process ICD-9 diagnosis files - extract exactly what's there
    """
    print("\n🔄 Processing ICD-9 Diagnosis Files...")
    
    icd9_dir = Path("downloads/icd9_extract")
    if not icd9_dir.exists():
        print("❌ ICD-9 extract directory not found")
        return {}
    
    # Look for diagnosis-related files (not procedure files)
    diagnosis_files = []
    for file in icd9_dir.glob("*.csv"):
        filename = file.name.lower()
        # Skip procedure files, keep diagnosis files
        if 'dx' in filename or 'diag' in filename or 'label' in filename:
            diagnosis_files.append(file)
    
    print(f"📁 Found {len(diagnosis_files)} diagnosis files:")
    for file in diagnosis_files:
        print(f"   - {file.name}")
    
    if not diagnosis_files:
        print("❌ No diagnosis files found")
        return {}
    
    mappings = {}
    
    for file in diagnosis_files:
        try:
            print(f"📊 Processing {file.name}...")
            
            # Read sample
            sample_df = pd.read_csv(file, nrows=3)
            print(f"   Columns: {list(sample_df.columns)}")
            
            # Read full file
            df = pd.read_csv(file)
            
            processed = 0
            for _, row in df.iterrows():
                try:
                    col_values = [str(val).strip() for val in row.values]
                    
                    if not any(col_values) or all(val in ['', 'nan', 'NaN'] for val in col_values):
                        continue
                    
                    icd_code = col_values[0].replace("'", "").replace('"', '').replace('.', '')
                    
                    if len(col_values) >= 2 and icd_code not in mappings:
                        mappings[icd_code] = tuple(col_values[1:])
                        processed += 1
                        
                except:
                    continue
            
            print(f"   ✅ Extracted {processed} mappings from {file.name}")
            
        except Exception as e:
            print(f"   ❌ Error processing {file.name}: {e}")
    
    return mappings

def extract_raw_mappings(downloaded_files):
    """
    Extract raw mappings from downloaded files - NO PROCESSING, NO ADDITIONS
    """
    print("\n🔄 Extracting Raw Mappings from Downloaded Files...")
    
    all_mappings = {}
    
    # Extract ICD-10 files
    if downloaded_files.get("icd10_ccsr"):
        try:
            with zipfile.ZipFile(downloaded_files["icd10_ccsr"], 'r') as zip_ref:
                zip_ref.extractall("downloads/icd10_extract")
            
            icd10_mappings = process_icd10_diagnosis_file()
            all_mappings.update(icd10_mappings)
            
        except Exception as e:
            print(f"❌ Error extracting ICD-10: {e}")
    
    # Extract ICD-9 files
    if downloaded_files.get("icd9_ccs"):
        try:
            with zipfile.ZipFile(downloaded_files["icd9_ccs"], 'r') as zip_ref:
                zip_ref.extractall("downloads/icd9_extract")
            
            icd9_mappings = process_icd9_diagnosis_files()
            
            # Add ICD-9 mappings only if ICD code not already present
            for icd_code, mapping in icd9_mappings.items():
                if icd_code not in all_mappings:
                    all_mappings[icd_code] = mapping
            
        except Exception as e:
            print(f"❌ Error extracting ICD-9: {e}")
    
    return all_mappings

def save_raw_mappings(mappings):
    """
    Save exactly what was extracted - no modifications
    """
    print(f"\n💾 Saving Raw Extracted Mappings...")
    
    if not mappings:
        print("❌ No mappings to save")
        return
    
    # Determine column structure from first mapping
    first_mapping = list(mappings.values())[0]
    num_columns = len(first_mapping)
    
    print(f"📊 Detected {num_columns} data columns per mapping")
    
    # Create DataFrame with dynamic columns
    mapping_data = []
    for icd_code, mapping_tuple in mappings.items():
        row_data = {'icd_code': icd_code}
        
        # Add columns based on what's actually in the data
        for i, value in enumerate(mapping_tuple):
            row_data[f'column_{i+1}'] = value
        
        mapping_data.append(row_data)
    
    df = pd.DataFrame(mapping_data)
    
    # Save raw extraction
    output_file = 'raw_hcup_extraction.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df):,} raw mappings to '{output_file}'")
    
    # Show structure
    print(f"\n📋 Extracted data structure:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Sample rows:")
    print(df.head(3).to_string(index=False))
    
    # Save extraction log
    with open('extraction_log.txt', 'w') as f:
        f.write("RAW HCUP DATA EXTRACTION LOG\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total mappings extracted: {len(df):,}\n")
        f.write(f"Data columns: {num_columns}\n")
        f.write(f"Column names: {list(df.columns)}\n\n")
        f.write("EXTRACTION METHOD:\n")
        f.write("- Downloaded official HCUP files\n")
        f.write("- Extracted data exactly as provided\n")
        f.write("- No processing, categorization, or additions\n")
        f.write("- Raw data only\n")
    
    print(f"✅ Saved extraction log to 'extraction_log.txt'")
    
    return df

def main():
    """
    Main function - download and extract raw data only
    """
    
    print("🏥 RAW HCUP DATA EXTRACTOR")
    print("="*50)
    print("📋 Extracts ONLY what's in the downloaded files")
    print("🚫 No processing, no categories, no additions")
    print("✅ Pure raw data extraction")
    print("="*50)
    
    try:
        # 1. Download files
        downloaded_files = download_hcup_ccs_files()
        
        # 2. Extract raw mappings
        raw_mappings = extract_raw_mappings(downloaded_files)
        
        if len(raw_mappings) == 0:
            print("\n❌ No data extracted from files")
            return
        
        # 3. Save raw data
        mapping_df = save_raw_mappings(raw_mappings)
        
        print(f"\n🎉 RAW EXTRACTION COMPLETE!")
        print(f"📊 Extracted {len(raw_mappings):,} mappings")
        print(f"📁 Output files:")
        print(f"   • raw_hcup_extraction.csv")
        print(f"   • extraction_log.txt")
        print(f"\n✅ Use this raw data as needed for your research!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
