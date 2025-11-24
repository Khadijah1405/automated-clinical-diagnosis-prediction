import argparse
import pandas as pd
import gc
import torch
from misclassification_extractor import run_misclassification_extraction
from synonym_replacer import update_dataframe_texts
from data_augmentation import balance_dataset
from classification_pipeline import run_classification

def main():
    parser = argparse.ArgumentParser(
        description="DeBERTa Text Classification with Memory Optimization"
    )
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to the CSV dataset file")
    parser.add_argument("--synonyms", type=str, default="false",
                        help="Set to 'true' to perform synonym replacement")
    parser.add_argument("--balance", type=str, default="false",
                        help="Set to 'true' to perform data augmentation")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (default: 16, reduce if OOM)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max sequence length (default: 128, reduce if OOM)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample only N rows for testing (None = use all)")
    args = parser.parse_args()
    
    # Load the original dataset
    print(f"Loading dataset from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    print(f"Original dataset size: {len(df)} rows, {df['class_label'].nunique()} classes")
    
    # Optional: Sample for faster testing
    if args.sample_size:
        print(f"Sampling {args.sample_size} rows for testing...")
        df = df.sample(n=min(args.sample_size, len(df)), random_state=42)
        print(f"Sampled dataset size: {len(df)} rows")
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run misclassification-based synonym replacement if enabled
    if args.synonyms.lower() == "true":
        print("\n" + "="*80)
        print("ITERATION 1: Synonym replacement enabled")
        print("="*80)
        
        print("\nRunning first misclassification analysis...")
        common_words = run_misclassification_extraction(
            df, 
            threshold=0.2, 
            epochs=10
        )
        
        # Clear memory after training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nUpdating texts with synonym replacements...")
        df = update_dataframe_texts(df, common_words)
        print("First iteration complete.")
        
        # Save intermediate checkpoint
        checkpoint_path = "checkpoint_iter1.csv"
        df.to_csv(checkpoint_path, index=False)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Clear memory before second iteration
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\n" + "="*80)
        print("ITERATION 2: Running second misclassification analysis")
        print("="*80)
        
        common_words = run_misclassification_extraction(
            df, 
            threshold=0.2, 
            epochs=10
        )
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nUpdating texts with synonym replacements...")
        df = update_dataframe_texts(df, common_words)
        print("Second iteration complete.")
        
        # Save final checkpoint
        checkpoint_path = "checkpoint_iter2.csv"
        df.to_csv(checkpoint_path, index=False)
        print(f"Checkpoint saved to {checkpoint_path}")
    else:
        print("Synonym replacement disabled. Proceeding with original dataset.")
    
    # Clear memory before balancing
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run data augmentation for balancing if enabled
    if args.balance.lower() == "true":
        print("\n" + "="*80)
        print("DATA BALANCING: Augmenting dataset")
        print("="*80)
        df = balance_dataset(df)
        print(f"Dataset balancing complete. New size: {len(df)} rows")
        
        # Save balanced dataset
        balanced_path = "dataset_balanced.csv"
        df.to_csv(balanced_path, index=False)
        print(f"Balanced dataset saved to {balanced_path}")
    else:
        print("Dataset balancing disabled.")
    
    # Clear memory before final classification
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run the main classification pipeline
    print("\n" + "="*80)
    print("FINAL CLASSIFICATION: Running main pipeline")
    print("="*80)
    run_classification(df)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
