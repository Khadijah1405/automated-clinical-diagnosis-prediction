#!/usr/bin/env python3
"""
STEP 1: Complete Data Preparation for Medical Diagnosis Prediction
- Removes information leakage sections
- Intelligently trims text to manage token limits
- Creates clean train/val/test splits
- Optimized for diagnosis prediction from admission data only
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def estimate_tokens(text):
    """Rough token estimation (1 token ≈ 4 characters)"""
    if pd.isna(text) or text == '':
        return 0
    return len(str(text)) // 4

def remove_discharge_sections(text):
    """Remove discharge-related sections that cause label leakage"""
    if pd.isna(text) or text == '':
        return text
    
    text = str(text)
    
    # Define patterns to remove (case-insensitive, multiline)
    discharge_patterns = [
        r'Discharge Medications?:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
        r'Discharge Diagnosis:.*?(?=\n\n|\n[A-Z][a-z]+:|$)', 
        r'Discharge Instructions?:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
        r'Follow-?up Instructions?:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
        r'Discharge Disposition:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
        r'Discharge Condition:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
        r'Discharge Summary:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
        # Also remove assessment/plan which often contains diagnosis
        r'Assessment and Plan:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
        r'Assessment/Plan:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
        r'Plan:.*?(?=\n\n|\n[A-Z][a-z]+:|$)',
    ]
    
    cleaned_text = text
    for pattern in discharge_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

def smart_trim_section(text, max_tokens, section_name):
    """Intelligently trim text while preserving key diagnostic information"""
    if pd.isna(text) or text == '':
        return text
    
    text = str(text)
    max_chars = max_tokens * 4  # Rough conversion
    
    if len(text) <= max_chars:
        return text
    
    # Section-specific trimming strategies
    if section_name == 'History of Present Illness':
        # Most important: keep beginning (presenting symptoms) 
        # and try to preserve timeline
        sentences = text.split('. ')
        if len(sentences) <= 3:
            return text[:max_chars] + "..." if len(text) > max_chars else text
        
        # Keep first 70% for presenting symptoms, last 30% for recent events
        first_part_chars = int(max_chars * 0.7)
        last_part_chars = max_chars - first_part_chars
        
        beginning = text[:first_part_chars]
        # Find last sentence boundary in beginning
        last_period = beginning.rfind('. ')
        if last_period > first_part_chars * 0.8:
            beginning = beginning[:last_period + 1]
        
        # Get ending part
        ending_start = max(first_part_chars, len(text) - last_part_chars)
        ending = text[ending_start:]
        
        return f"{beginning} ... {ending}"
    
    elif section_name == 'Physical Exam':
        # Prioritize vital signs (usually at beginning) and abnormal findings
        # Keep beginning which usually has vital signs
        return text[:max_chars] + "..." if len(text) > max_chars else text
    
    elif section_name == 'Past Medical History':
        # Keep most recent/relevant conditions (often at beginning)
        return text[:max_chars] + "..." if len(text) > max_chars else text
    
    elif section_name == 'Pertinent Results':
        # Keep all if possible, otherwise prioritize abnormal values
        # Usually already concise, but if long, keep beginning
        return text[:max_chars] + "..." if len(text) > max_chars else text
    
    else:
        # Default: keep beginning
        return text[:max_chars] + "..." if len(text) > max_chars else text

def clean_basic_text(text):
    """Basic text cleaning"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.,;:()\-/]', ' ', text)
    
    return text.strip()

def main():
    print("🏥 COMPLETE MEDICAL DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Configuration
    input_file = "textfinal_hcup_categories_FIXED.csv"  # Change to your file
    target_total_tokens = 2500  # Target token limit per sample
    
    # Token limits per section (prioritized by diagnostic value)
    section_token_limits = {
        'Chief Complaint': 150,           # Keep most - usually short
        'History of Present Illness': 1000,  # Most important for diagnosis
        'Physical Exam': 400,             # High diagnostic value
        'Past Medical History': 300,      # Important context
        'Pertinent Results': 400,         # Key labs/imaging
        'Allergies': 100,                # Short, keep all
        'Medications on Admission': 200   # Some diagnostic clues
    }
    
    print(f"📁 Loading dataset from: {input_file}")
    
    # Load dataset
    try:
        df = pd.read_csv(input_file, low_memory=False)
        print(f"✓ Loaded dataset with shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found!")
        return
    
    print(f"Original columns ({len(df.columns)}): {list(df.columns)}")
    
    # STEP 1: Filter columns (remove information leakage)
    print(f"\n📋 STEP 1: Filtering columns to prevent information leakage...")
    
    # Keep only admission-time information (no discharge info)
    input_columns = [
        'subject_id',                    # Patient ID
        'hadm_id',                      # Hospital admission ID
        'Sex',                          # Patient demographics
        'Service',                      # Admitting service
        'Allergies',                    # Known allergies
        'Chief Complaint',              # Main complaint
        'History of Present Illness',   # Current illness history
        'Past Medical History',         # Previous medical history
        'Physical Exam',                # Physical examination findings
        'Pertinent Results',            # Lab/imaging results
        'Medications on Admission'      # Admission medications
        # REMOVED: Brief Hospital Course (major source of leakage)
        # REMOVED: All discharge-related columns
    ]
    
    # Target columns (what we want to predict)
    target_columns = []
    # Check which target columns exist
    possible_targets = ['hcup_primary_description', 'hcup_primary_category',]
    for col in possible_targets:
        if col in df.columns:
            target_columns.append(col)
            print(f"✓ Found target column: {col}")
            break
    
    if not target_columns:
        print("❌ No target diagnosis column found!")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Combine columns to keep
    columns_to_keep = input_columns + target_columns
    available_columns = [col for col in columns_to_keep if col in df.columns]
    missing_columns = set(columns_to_keep) - set(available_columns)
    
    if missing_columns:
        print(f"⚠️  Warning: Missing columns: {missing_columns}")
    
    # Filter dataframe
    df_filtered = df[available_columns].copy()
    print(f"✓ Filtered to {len(df_filtered.columns)} columns")
    print(f"Kept columns: {list(df_filtered.columns)}")
    
    # STEP 2: Remove rows without target
    print(f"\n🎯 STEP 2: Filtering valid samples...")
    print(f"Initial dataset size: {len(df_filtered):,}")
    
    target_col = target_columns[0]
    df_clean = df_filtered.dropna(subset=[target_col]).copy()
    df_clean = df_clean[df_clean[target_col].str.strip() != ''].copy()
    print(f"After removing missing/empty targets: {len(df_clean):,}")
    
    # STEP 3: Text cleaning and leakage removal
    print(f"\n🧹 STEP 3: Cleaning text and removing information leakage...")
    
    text_columns = ['Chief Complaint', 'History of Present Illness', 
                   'Past Medical History', 'Physical Exam', 'Pertinent Results', 
                   'Allergies', 'Medications on Admission']
    
    for col in text_columns:
        if col in df_clean.columns:
            print(f"  Processing {col}...")
            # Fill missing values
            df_clean[col] = df_clean[col].fillna('')
            
            # Remove discharge sections (information leakage)
            df_clean[col] = df_clean[col].apply(remove_discharge_sections)
            
            # Basic text cleaning
            df_clean[col] = df_clean[col].apply(clean_basic_text)
    
    # STEP 4: Intelligent text trimming
    print(f"\n✂️  STEP 4: Trimming text to manage token limits...")
    
    # Apply section-specific trimming
    for col in text_columns:
        if col in df_clean.columns and col in section_token_limits:
            max_tokens = section_token_limits[col]
            print(f"  Trimming {col} to max {max_tokens} tokens...")
            df_clean[col] = df_clean[col].apply(
                lambda x: smart_trim_section(x, max_tokens, col)
            )
    
    # Calculate token estimates
    print(f"  Calculating token estimates...")
    df_clean['estimated_tokens'] = df_clean[text_columns].apply(
        lambda row: sum(estimate_tokens(row[col]) for col in text_columns if col in row.index), 
        axis=1
    )
    
    # Show token distribution
    print(f"\n📊 Token distribution after trimming:")
    print(df_clean['estimated_tokens'].describe())
    over_limit = (df_clean['estimated_tokens'] > target_total_tokens).sum()
    print(f"Samples over {target_total_tokens} tokens: {over_limit} ({over_limit/len(df_clean)*100:.1f}%)")
    
    # STEP 5: Final adaptive trimming for outliers
    if over_limit > 0:
        print(f"\n🎛️  STEP 5: Adaptive trimming for {over_limit} samples over token limit...")
        
        def adaptive_trim_row(row):
            current_total = sum(estimate_tokens(row[col]) for col in text_columns if col in row.index)
            if current_total <= target_total_tokens:
                return row
            
            # Calculate reduction factor
            reduction_factor = target_total_tokens / current_total
            
            # Apply proportional reduction
            for col in text_columns:
                if col in row.index and pd.notna(row[col]) and row[col] != '':
                    current_chars = len(row[col])
                    new_max_chars = int(current_chars * reduction_factor)
                    if new_max_chars < current_chars:
                        row[col] = row[col][:new_max_chars] + "..."
            
            return row
        
        # Apply adaptive trimming to samples over limit
        mask = df_clean['estimated_tokens'] > target_total_tokens
        df_clean.loc[mask] = df_clean.loc[mask].apply(adaptive_trim_row, axis=1)
        
        # Recalculate tokens
        df_clean['estimated_tokens'] = df_clean[text_columns].apply(
            lambda row: sum(estimate_tokens(row[col]) for col in text_columns if col in row.index), 
            axis=1
        )
        
        print(f"✓ After adaptive trimming:")
        print(df_clean['estimated_tokens'].describe())
    
    # STEP 6: Show target distribution
    print(f"\n📈 STEP 6: Target diagnosis distribution...")
    target_counts = df_clean[target_col].value_counts()
    print(f"Total unique diagnoses: {len(target_counts)}")
    print(f"Top 10 diagnoses:")
    for i, (diagnosis, count) in enumerate(target_counts.head(10).items(), 1):
        print(f"  {i:2d}. {diagnosis} ({count:,} cases)")
    
    # STEP 7: Create train/validation/test splits
    print(f"\n🔄 STEP 7: Creating dataset splits...")
    
    # Split by patients to avoid data leakage
    unique_patients = df_clean['subject_id'].unique()
    print(f"Total unique patients: {len(unique_patients):,}")
    
    # First split: train+val vs test (85% vs 15%)
    train_val_patients, test_patients = train_test_split(
        unique_patients, test_size=0.15, random_state=42, 
        stratify=None  # Can't stratify on patient level easily
    )
    
    # Second split: train vs val (70% vs 15% of total)
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.176, random_state=42  # 0.15/0.85 ≈ 0.176
    )
    
    # Create dataset splits
    train_df = df_clean[df_clean['subject_id'].isin(train_patients)].copy()
    val_df = df_clean[df_clean['subject_id'].isin(val_patients)].copy()
    test_df = df_clean[df_clean['subject_id'].isin(test_patients)].copy()
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_df):,} samples ({len(train_patients):,} patients)")
    print(f"  Val:   {len(val_df):,} samples ({len(val_patients):,} patients)")
    print(f"  Test:  {len(test_df):,} samples ({len(test_patients):,} patients)")
    
    # Show token distribution per split
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        avg_tokens = split_df['estimated_tokens'].mean()
        max_tokens = split_df['estimated_tokens'].max()
        print(f"  {name} tokens - Avg: {avg_tokens:.0f}, Max: {max_tokens:.0f}")
    
    # STEP 8: Save processed datasets
    print(f"\n💾 STEP 8: Saving processed datasets...")
    
    # Remove token estimate column before saving
    columns_to_save = [col for col in df_clean.columns if col != 'estimated_tokens']
    
    train_df[columns_to_save].to_csv('train_dataset_clean.csv', index=False)
    val_df[columns_to_save].to_csv('val_dataset_clean.csv', index=False)
    test_df[columns_to_save].to_csv('test_dataset_clean.csv', index=False)
    
    print(f"✓ Saved train_dataset_clean.csv ({len(train_df):,} samples)")
    print(f"✓ Saved val_dataset_clean.csv ({len(val_df):,} samples)")
    print(f"✓ Saved test_dataset_clean.csv ({len(test_df):,} samples)")
    
    # STEP 9: Generate summary report
    print(f"\n📊 STEP 9: Summary Report")
    print("=" * 70)
    print(f"✓ Information leakage removed (discharge sections)")
    print(f"✓ Text trimmed to manageable token limits")
    print(f"✓ Target column: {target_col}")
    print(f"✓ Unique diagnoses: {len(target_counts)}")
    print(f"✓ Average tokens per sample: {df_clean['estimated_tokens'].mean():.0f}")
    print(f"✓ Max tokens per sample: {df_clean['estimated_tokens'].max():.0f}")
    print(f"✓ Samples over {target_total_tokens} tokens: {(df_clean['estimated_tokens'] > target_total_tokens).sum()}")
    
    # Save configuration for next steps
    config = {
        'target_column': target_col,
        'text_columns': text_columns,
        'total_samples': len(df_clean),
        'unique_diagnoses': len(target_counts),
        'avg_tokens': float(df_clean['estimated_tokens'].mean()),
        'max_tokens': int(df_clean['estimated_tokens'].max())
    }
    
    import json
    with open('preprocessing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Saved preprocessing_config.json")
    print(f"\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
    print(f"Next step: Create QA format for model training")

if __name__ == "__main__":
    main()
