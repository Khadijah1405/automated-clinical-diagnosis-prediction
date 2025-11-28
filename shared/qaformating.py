#!/usr/bin/env python3
"""
STEP 2: QA Formatting for Medical Diagnosis Inference
- Creates diverse question-answer pairs for inference
- Formats data for Unsloth Llama 3 8B
- Target: hcup_primary_description
"""

import pandas as pd
import json
import random
import os
from typing import List, Dict

def format_patient_context(row: pd.Series) -> str:
    """Format patient information for the prompt"""
    
    context_parts = []
    
    # Define the order and display names
    fields = [
        ('Sex', 'Patient Gender'),
        ('Allergies', 'Known Allergies'),
        ('Service', 'Admitting Service'),
        ('Chief Complaint', 'Chief Complaint'),
        ('History of Present Illness', 'History of Present Illness'),
        ('Past Medical History', 'Past Medical History'),
        ('Physical Exam', 'Physical Examination'),
        ('Pertinent Results', 'Laboratory and Imaging Results'),
        ('Brief Hospital Course', 'Brief Hospital Course'),
        ('Medications on Admission', 'Medications on Admission')
    ]
    
    for field, display_name in fields:
        if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
            value = str(row[field]).strip()
            
            # Skip if value is too short or meaningless
            if len(value) > 3 and value.lower() not in ['none', 'n/a', 'unknown', '']:
                context_parts.append(f"**{display_name}:** {value}")
    
    return "\n".join(context_parts)

def create_inference_prompt(row: pd.Series, prompt_variant: int = 0) -> Dict:
    """Create a single inference prompt for Llama 3"""
    
    # Simple and direct question templates (20 variations)
    question_templates = [
        "What is the diagnosis for this patient?",
        "What diagnosis would you assign to this patient?", 
        "What is your diagnosis based on this case?",
        "What diagnosis fits this patient's presentation?",
        "What is the diagnosis?",
        "What diagnosis explains this patient's condition?",
        "What would you diagnose this patient with?",
        "What is your diagnostic impression?",
        "What diagnosis does this case represent?",
        "What is the correct diagnosis?",
        "What diagnosis best fits this case?",
        "What is the diagnosis for this case?",
        "What diagnosis would you make?",
        "What is your diagnosis?",
        "What diagnosis accounts for these findings?",
        "What diagnosis do you assign?",
        "What is the appropriate diagnosis?",
        "What diagnosis matches this presentation?",
        "What would be your diagnosis?",
        "What diagnosis explains these symptoms?"
    ]
    
    # System prompts encouraging natural phrasing
    system_prompts = [
        "You are a medical AI assistant. Based on the patient information, provide your diagnosis. You may phrase it naturally, such as 'The patient has acute myocardial infarction' or 'This case represents congestive heart failure', but ensure the diagnosis name is accurate.",
        "You are an experienced physician. Analyze the clinical data and provide your diagnostic impression. Phrase your response naturally but include the specific diagnosis name.",
        "You are a diagnostic expert. Based on the clinical presentation, provide your diagnosis. You can phrase it conversationally but ensure the medical diagnosis is precise and clear.",
        "You are a medical consultant. Review the patient data and provide your diagnosis. Express it in a natural way while maintaining medical accuracy.",
        "You are a clinical AI system. Determine the diagnosis from the clinical information. Phrase your response naturally, such as 'This patient presents with...' or 'The diagnosis is...'",
        "You are a healthcare professional. Analyze the patient presentation and provide your diagnostic assessment. Use natural phrasing while ensuring the diagnosis is medically accurate.",
        "You are a physician assistant. Based on the clinical evidence, state your diagnosis. You may phrase it naturally but include the specific medical condition name.",
        "You are a medical expert. Review the case and provide your diagnosis. Express it conversationally while maintaining precision about the medical condition."
    ]
    
    # Select prompts based on variant for diversity
    system_prompt = system_prompts[prompt_variant % len(system_prompts)]
    question = question_templates[prompt_variant % len(question_templates)]
    
    # Format patient context
    patient_context = format_patient_context(row)
    
    # Create Llama 3 chat format
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"{question}\n\n## Patient Information:\n{patient_context}"
        }
    ]
    
    # Get target (hcup_primary_description)
    target_diagnosis = str(row['hcup_primary_description']).strip()
    
    return {
        'messages': messages,
        'target_diagnosis': target_diagnosis,
        'subject_id': int(row['subject_id']),
        'hadm_id': int(row['hadm_id']),
        'system_prompt': system_prompt,
        'question': question
    }

def format_dataset_for_inference(df: pd.DataFrame, output_file: str) -> List[Dict]:
    """Format entire dataset for inference"""
    
    print(f"Formatting {len(df)} samples for inference...")
    
    inference_data = []
    
    for idx, row in df.iterrows():
        try:
            # Use different prompt variants for diversity
            prompt_data = create_inference_prompt(row, prompt_variant=idx)
            inference_data.append(prompt_data)
            
            if (len(inference_data)) % 1000 == 0:
                print(f"  Processed {len(inference_data):,} samples...")
                
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Save to JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in inference_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"✓ Saved {len(inference_data):,} inference prompts to {output_file}")
    
    return inference_data

def analyze_prompts(inference_data: List[Dict], dataset_name: str):
    """Analyze the diversity of generated prompts"""
    
    print(f"\n📊 {dataset_name} Analysis:")
    print("-" * 30)
    
    # Count unique system prompts and questions
    system_prompts = [data['system_prompt'] for data in inference_data]
    questions = [data['question'] for data in inference_data]
    
    unique_systems = len(set(system_prompts))
    unique_questions = len(set(questions))
    
    print(f"Total prompts: {len(inference_data):,}")
    print(f"Unique system prompts: {unique_systems}")
    print(f"Unique questions: {unique_questions}")
    
    # Show sample questions
    unique_question_list = list(set(questions))
    print(f"Sample questions:")
    for i, q in enumerate(unique_question_list[:5], 1):
        print(f"  {i}. {q}")
    
    # Show target distribution
    targets = [data['target_diagnosis'] for data in inference_data]
    target_counts = pd.Series(targets).value_counts()
    
    print(f"Unique target diagnoses: {len(target_counts)}")
    print(f"Top 3 targets:")
    for i, (target, count) in enumerate(target_counts.head(3).items(), 1):
        print(f"  {i}. {target} ({count:,} cases)")

def main():
    print("🤖 STEP 2: QA FORMATTING FOR MEDICAL DIAGNOSIS")
    print("=" * 60)
    
    # Check if processed datasets exist
    required_files = ['train_dataset.csv', 'val_dataset.csv', 'test_dataset.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Error: Missing required files: {missing_files}")
        print("Please run 'python step1_data_prep.py' first!")
        return
    
    # Load processed datasets
    print("📂 Loading processed datasets...")
    
    train_df = pd.read_csv('train_dataset.csv')
    val_df = pd.read_csv('val_dataset.csv')
    test_df = pd.read_csv('test_dataset.csv')
    
    print(f"✓ Train: {len(train_df):,} samples")
    print(f"✓ Val:   {len(val_df):,} samples")
    print(f"✓ Test:  {len(test_df):,} samples")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Format datasets for inference
    print(f"\n🔄 STEP 2.1: Formatting validation set...")
    val_inference = format_dataset_for_inference(val_df, 'val_inference.jsonl')
    
    print(f"\n🔄 STEP 2.2: Formatting test set...")
    test_inference = format_dataset_for_inference(test_df, 'test_inference.jsonl')
    
    # Analyze generated prompts
    print(f"\n📊 STEP 2.3: Analyzing generated prompts...")
    analyze_prompts(val_inference, "Validation Set")
    analyze_prompts(test_inference, "Test Set")
    
    # Show a sample prompt
    print(f"\n📄 STEP 2.4: Sample inference prompt:")
    print("=" * 50)
    
    sample = test_inference[0]
    print(f"QUESTION: {sample['question']}")
    print(f"SYSTEM: {sample['messages'][0]['content'][:100]}...")
    print(f"USER: {sample['messages'][1]['content'][:200]}...")
    print(f"TARGET: {sample['target_diagnosis']}")
    
    # Summary
    print(f"\n🎉 STEP 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✓ Val inference prompts: val_inference.jsonl ({len(val_inference):,} samples)")
    print(f"✓ Test inference prompts: test_inference.jsonl ({len(test_inference):,} samples)")
    print(f"✓ Target: hcup_primary_description")
    print(f"✓ Questions: 20 simple variations (e.g., 'What is the diagnosis?')")
    print(f"✓ Natural phrasing encouraged for responses")
    print(f"✓ Ready for Step 3: Inference")
    print(f"\nNext: Run 'python step3_inference.py'")

if __name__ == "__main__":
    main()
