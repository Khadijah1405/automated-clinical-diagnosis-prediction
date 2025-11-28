#!/usr/bin/env python3
"""
Medical CCS Dataset Creator - Final Version

Clean, focused dataset creation for medical diagnosis training.
- Handles missing data properly using pandas settings
- Removes diagnosis-leaking sections 
- Creates only train/val splits (no redundant eval set)
- Token-aware text processing for LLaMA models
"""

import os
import re
import json
import argparse
from typing import List, Dict, Tuple
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------- Text Processing -------------------------------

def remove_discharge_sections(text: str) -> str:
    """Remove sections that leak diagnosis information."""
    if not isinstance(text, str) or not text:
        return text

    patterns = [
        # Discharge sections
        r'(?i)discharge medications?:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
        r'(?i)discharge diagnosis(?:es)?:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)', 
        r'(?i)discharge instructions?:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
        r'(?i)discharge disposition:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
        r'(?i)discharge condition:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
        
        # Assessment/diagnosis sections
        r'(?i)assessment(?: and)?(?:/)?\s*plan:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
        r'(?i)impression:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
        r'(?i)plan:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
        r'(?i)brief hospital course:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
        r'(?i)hospital course:.*?(?=\n[a-zA-Z][^:\n]*:|\n\n|\Z)',
    ]
    
    text_cleaned = text
    for pattern in patterns:
        text_cleaned = re.sub(pattern, '', text_cleaned, flags=re.DOTALL)

    # Normalize whitespace
    text_cleaned = re.sub(r'\r\n?', '\n', text_cleaned)
    text_cleaned = re.sub(r'\n{3,}', '\n\n', text_cleaned)
    text_cleaned = re.sub(r'[ \t]{2,}', ' ', text_cleaned)
    
    return text_cleaned.strip()

def clean_clinical_text(text: str) -> str:
    """Clean and normalize clinical text."""
    if not isinstance(text, str):
        return ""
    
    # Remove common data artifacts
    text = re.sub(r'_+', ' ', text)
    text = re.sub(r'\*+', ' ', text) 
    text = re.sub(r'#+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------------------------------- QA Formatting -------------------------------

SYSTEM_PROMPTS = [
    "You are an experienced physician specializing in internal medicine. Analyze the clinical presentation and provide a precise primary diagnosis using standard medical terminology.",
    "You are a diagnostic expert with extensive clinical experience. Based on the patient data, determine the primary diagnosis using accurate medical terminology.", 
    "You are a board-certified internist. Review the clinical information and provide your primary diagnosis using precise medical language.",
    "You are a skilled diagnostician. Analyze the clinical presentation and determine the most likely primary diagnosis using proper medical terminology.",
]

QUESTION_TEMPLATES = [
    "Based on this clinical presentation, what is the primary diagnosis?",
    "What is the most likely diagnosis for this patient?",
    "Given the clinical findings, what diagnosis best explains this case?", 
    "What primary diagnosis would you assign based on this clinical data?",
]

def build_chat_prompt(system_msg: str, user_msg: str, answer: str) -> str:
    """Build Llama chat format."""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{answer}<|eot_id|>"
    )

def create_qa_format(
    clinical_text: str,
    target_diagnosis: str,
    prompt_variant: int = 0,
    include_clinical_text_in_qa: bool = False,
) -> str:
    """Create QA format for training."""
    sys = SYSTEM_PROMPTS[prompt_variant % len(SYSTEM_PROMPTS)]
    q = QUESTION_TEMPLATES[prompt_variant % len(QUESTION_TEMPLATES)]
    
    if include_clinical_text_in_qa:
        user_block = f"""{q}

Patient Record:
{clinical_text}"""
    else:
        user_block = q
    
    return build_chat_prompt(sys, user_block, target_diagnosis)

# ------------------------------- Token Utilities -------------------------------

def get_tokenizer(model_name: str):
    """Load tokenizer if available."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"[WARN] Could not load tokenizer for '{model_name}': {e}")
        return None

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens with fallback to character-based estimation."""
    if tokenizer is None or not isinstance(text, str):
        return max(1, len(str(text)) // 4)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def trim_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Trim text to fit token limit."""
    if not isinstance(text, str):
        return text
    if tokenizer is None:
        return text[:max_tokens * 4]  # Rough character limit
    
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)

# ------------------------------- Dataset Creator -------------------------------

class MedicalDatasetCreator:
    def __init__(
        self,
        csv_path: str,
        base_model: str = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_len: int = 8192,
        token_reserve: int = 900,
        min_text_len: int = 100,
        include_clinical_text_in_qa: bool = False,
        val_split: float = 0.15,
        output_dir: str = "./medical_datasets",
    ):
        self.csv_path = csv_path
        self.base_model = base_model
        self.max_seq_len = max_seq_len
        self.token_reserve = token_reserve
        self.min_text_len = min_text_len
        self.include_clinical_text_in_qa = include_clinical_text_in_qa
        self.val_split = val_split
        self.output_dir = output_dir
        self.df = None
        self.tokenizer = get_tokenizer(base_model)

    def load_data(self):
        """Load CSV with proper handling of missing data."""
        print(f"Loading data from: {self.csv_path}")
        
        # Load with settings that handle missing data appropriately
        self.df = pd.read_csv(
            self.csv_path, 
            low_memory=False,
            keep_default_na=False,  # Don't auto-convert to NaN
            na_values=['', 'NULL', 'null', 'None', 'none', 'NaN', 'nan'],
        )
        
        print(f"Loaded {len(self.df)} records")
        print(f"Available columns: {list(self.df.columns)}")
        
        # Check missing data in clinical columns
        clinical_columns = ['Chief Complaint', 'History of Present Illness', 'Past Medical History', 'Physical Exam', 'Pertinent Results']
        missing_summary = {}
        for col in clinical_columns:
            if col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                missing_summary[col] = missing_count
        
        print(f"Missing data summary: {missing_summary}")

        # Convert ID columns to numeric
        for col in ('subject_id', 'hadm_id'):
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def extract_clinical_text_from_row(self, row: pd.Series) -> str:
        """Extract and combine clinical text from CSV row."""
        clinical_sections = []
        
        # Clinical columns in order of importance
        clinical_columns = [
            'Chief Complaint',
            'History of Present Illness',  
            'Past Medical History',
            'Physical Exam',
            'Pertinent Results'
        ]
        
        for column_name in clinical_columns:
            if column_name in self.df.columns and pd.notna(row.get(column_name)):
                raw_content = str(row[column_name]).strip()
                if raw_content and raw_content not in ('', 'nan', 'none'):
                    # Clean and process content
                    content = clean_clinical_text(raw_content)
                    content = remove_discharge_sections(content)
                    
                    if len(content) > 10:  # Only substantial content
                        # Add section header if not present
                        if not content.startswith(column_name + ':'):
                            content = f"{column_name}: {content}"
                        clinical_sections.append(content)
        
        if not clinical_sections:
            return ""
        
        # Combine sections
        combined_text = " | ".join(clinical_sections)
        combined_text = re.sub(r'\s+', ' ', combined_text)
        combined_text = re.sub(r'\|+', '|', combined_text)
        
        return combined_text.strip()

    def process_records(self) -> List[Dict]:
        """Process CSV records into training examples."""
        print("Processing records...")
        processed = []
        skipped = {'no_target': 0, 'no_text': 0, 'too_short': 0}

        target_col = 'hcup_primary_description'
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV")

        max_clinical_tokens = max(256, self.max_seq_len - self.token_reserve)

        for idx, row in self.df.iterrows():
            # Get target diagnosis
            target = row.get(target_col, '')
            if pd.isna(target) or not str(target).strip():
                skipped['no_target'] += 1
                continue
            target = str(target).strip()

            # Extract clinical text
            clinical_text = self.extract_clinical_text_from_row(row)
            if not clinical_text:
                skipped['no_text'] += 1
                continue
            if len(clinical_text) < self.min_text_len:
                skipped['too_short'] += 1
                continue

            # Token-aware trimming
            clinical_text = trim_to_tokens(clinical_text, self.tokenizer, max_clinical_tokens)

            # Create QA format
            prompt_variant = idx % 4
            qa_text = create_qa_format(
                clinical_text,
                target,
                prompt_variant=prompt_variant,
                include_clinical_text_in_qa=self.include_clinical_text_in_qa,
            )

            record = {
                'subject_id': int(row['subject_id']) if ('subject_id' in row and pd.notna(row['subject_id'])) else None,
                'hadm_id': int(row['hadm_id']) if ('hadm_id' in row and pd.notna(row['hadm_id'])) else None,
                'clinical_text': clinical_text,
                'target': target,
                'qa_format': qa_text,
                'text_length': len(clinical_text),
                'prompt_variant': prompt_variant,
            }
            processed.append(record)

            if (idx + 1) % 5000 == 0:
                print(f"Processed {idx + 1}/{len(self.df)} records...")

        print("Processing complete:")
        print(f"  Valid records: {len(processed)}")
        print(f"  Skipped - no target: {skipped['no_target']}")
        print(f"  Skipped - no text: {skipped['no_text']}")
        print(f"  Skipped - too short: {skipped['too_short']}")
        
        return processed

    def create_train_val_test_split(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/validation/test split with stratification when possible."""
        print(f"Creating train/validation/test split...")
        
        y = [d['target'] for d in data]
        cnt = Counter(y)
        can_stratify = all(c >= 2 for c in cnt.values())
        
        # First split: separate train from val+test
        train_data, temp_data = train_test_split(
            data,
            test_size=0.3,  # 30% for val+test
            random_state=42,
            shuffle=True,
            stratify=y if can_stratify else None
        )
        
        # Second split: separate val from test
        y_temp = [d['target'] for d in temp_data]
        cnt_temp = Counter(y_temp)
        can_stratify_temp = all(c >= 2 for c in cnt_temp.values())
        
        val_data, test_data = train_test_split(
            temp_data,
            test_size=0.5,  # Split remaining 30% into 15% val, 15% test
            random_state=42,
            shuffle=True,
            stratify=y_temp if can_stratify_temp else None
        )
        
        print(f"Split complete:")
        print(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
        print(f"  Val:   {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
        print(f"  Test:  {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")
        
        # Show target distribution
        top10 = pd.Series([d['target'] for d in train_data]).value_counts().head(10)
        print("\nTop 10 targets in training set:")
        for target, count in top10.items():
            print(f"  '{target}': {count}")
        
        return train_data, val_data, test_data

    def save_datasets(self, train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]) -> Tuple[str, str, str, str]:
        """Save train/val/test datasets and summary."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save all three datasets
        train_path = os.path.join(self.output_dir, "train_dataset.json")
        val_path = os.path.join(self.output_dir, "val_dataset.json")
        test_path = os.path.join(self.output_dir, "test_dataset.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        # Create summary statistics
        all_data = train_data + val_data + test_data
        text_lengths = [d['text_length'] for d in all_data]
        
        # Sample token statistics
        sample_size = min(1000, len(train_data))
        sample_qa_tokens = [count_tokens(train_data[i]["qa_format"], self.tokenizer) for i in range(sample_size)]
        
        summary = {
            "dataset_info": {
                "total_records": len(all_data),
                "train_records": len(train_data),
                "val_records": len(val_data),
                "test_records": len(test_data),
                "unique_targets": len(set(d['target'] for d in all_data)),
            },
            "text_statistics": {
                "avg_text_length": round(sum(text_lengths) / len(text_lengths), 1),
                "min_text_length": min(text_lengths),
                "max_text_length": max(text_lengths),
            },
            "token_statistics": {
                "sample_size": sample_size,
                "avg_qa_tokens": round(sum(sample_qa_tokens) / len(sample_qa_tokens), 1),
                "max_qa_tokens": max(sample_qa_tokens),
                "pct_over_4k": round(100 * sum(t > 4096 for t in sample_qa_tokens) / len(sample_qa_tokens), 2),
            },
            "target_distribution": {
                "train": pd.Series([d['target'] for d in train_data]).value_counts().head(20).to_dict(),
                "val": pd.Series([d['target'] for d in val_data]).value_counts().head(10).to_dict(),
                "test": pd.Series([d['target'] for d in test_data]).value_counts().head(10).to_dict(),
            },
            "config": {
                "base_model": self.base_model,
                "max_seq_len": self.max_seq_len,
                "token_reserve": self.token_reserve,
                "min_text_len": self.min_text_len,
            }
        }
        
        summary_path = os.path.join(self.output_dir, "dataset_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\nDatasets saved:")
        print(f"  Training:   {train_path}")
        print(f"  Validation: {val_path}")
        print(f"  Test:       {test_path}")
        print(f"  Summary:    {summary_path}")

        return train_path, val_path, test_path, summary_path

    def run(self) -> bool:
        """Run the complete dataset creation pipeline."""
        print("MEDICAL CCS DATASET CREATION PIPELINE")
        print("=" * 50)
        
        if not os.path.exists(self.csv_path):
            print(f"Error: Input file '{self.csv_path}' not found.")
            return False

        try:
            self.load_data()
            data = self.process_records()
            
            if not data:
                print("No valid records after processing.")
                return False

            train, val, test = self.create_train_val_test_split(data)
            self.save_datasets(train, val, test)
            
            print(f"\nDataset creation complete!")
            print(f"Training examples: {len(train)}")
            print(f"Validation examples: {len(val)}")
            print(f"Test examples: {len(test)}")
            return True
            
        except Exception as e:
            print(f"Error during processing: {e}")
            return False

# ------------------------------- Main -------------------------------

def main():
    # Configuration
    csv_file = "textfinal_hcup_categories_FIXED.csv"
    output_dir = "./medical_datasetsfull"
    base_model = "unsloth/llama-3-8b-bnb-4bit"
    max_seq_len = 8192
    token_reserve = 900
    include_clinical_text_in_qa = False
    val_split = 0.15
    min_text_len = 100

    print("MEDICAL CCS DATASET CREATOR - FINAL VERSION")
    print("=" * 50)

    creator = MedicalDatasetCreator(
        csv_path=csv_file,
        base_model=base_model,
        max_seq_len=max_seq_len,
        token_reserve=token_reserve,
        include_clinical_text_in_qa=include_clinical_text_in_qa,
        val_split=val_split,
        output_dir=output_dir,
        min_text_len=min_text_len,
    )
    
    success = creator.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
