import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

print("Starting data preprocessing...")

# ============================================================================
# STEP 1: DATA PREPROCESSING
# ============================================================================

def preprocess_dataset(input_file, output_file):
    """
    Process the original dataset to create text-target pairs
    """
    print("Loading original dataset...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Define the columns to merge for text
    text_columns = [
        'Chief Complaint',
        'History of Present Illness', 
        'Past Medical History',
        'Physical Exam',
        'Pertinent Results'
    ]
    
    # Define the target column - using hcup_primary_description
    target_column = 'hcup_primary_description'
    
    # Check if required columns exist
    missing_text_cols = [col for col in text_columns if col not in df.columns]
    missing_target_col = target_column not in df.columns
    
    if missing_text_cols:
        print(f"Warning: Missing text columns: {missing_text_cols}")
    if missing_target_col:
        print(f"Error: Missing target column: {target_column}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Show available columns if some are missing
    if missing_text_cols:
        print(f"Available columns in dataset: {list(df.columns)}")
    
    # Create the text column by merging specified columns WITH COLUMN NAMES
    def merge_text_columns(row):
        text_parts = []
        for col in text_columns:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                content = str(row[col]).strip()
                # Include column name as label
                text_parts.append(f"{col}: {content}")
        return ", ".join(text_parts) if text_parts else ""
    
    # Get target column
    def get_target(row):
        if pd.notna(row[target_column]) and str(row[target_column]).strip():
            return str(row[target_column]).strip()
        return ""
    
    print("Creating text and target columns...")
    print("Text format will include column names (e.g., 'Chief Complaint: pain, History of Present Illness: ...')")
    df['text'] = df.apply(merge_text_columns, axis=1)
    df['target'] = df.apply(get_target, axis=1)
    
    # Remove rows where text or target is empty
    initial_count = len(df)
    df_filtered = df[(df['text'] != '') & (df['target'] != '')].copy()
    empty_removed = initial_count - len(df_filtered)
    print(f"Removed {empty_removed} rows with empty text or target")
    
    # Create the final dataset with only text and target columns
    final_df = df_filtered[['text', 'target']].copy()
    
    # Rename target to class_label to match training scripts
    final_df.rename(columns={'target': 'class_label'}, inplace=True)
    
    # Print statistics before filtering
    print(f"\nDataset Statistics (before filtering):")
    print(f"Total samples: {len(final_df)}")
    print(f"Number of unique targets: {final_df['class_label'].nunique()}")
    
    # Show all target distribution
    print(f"\nAll target distribution:")
    target_counts = final_df['class_label'].value_counts()
    print(target_counts)
    
    # Remove "Expired" entries if they exist
    expired_count = (final_df['class_label'] == 'Expired').sum()
    if expired_count > 0:
        print(f"\nRemoving 'Expired' entries...")
        initial_count_expired = len(final_df)
        final_df = final_df[final_df['class_label'] != 'Expired'].copy()
        expired_removed = initial_count_expired - len(final_df)
        print(f"Removed {expired_removed} 'Expired' entries")
    else:
        print(f"\nNo 'Expired' entries found to remove.")
    
    # Remove classes with less than 5 samples
    print(f"\nRemoving classes with less than 5 samples...")
    target_counts_updated = final_df['class_label'].value_counts()
    classes_to_remove = target_counts_updated[target_counts_updated < 5].index.tolist()
    
    if len(classes_to_remove) > 0:
        print(f"Classes with less than 5 samples to be removed:")
        for i, class_name in enumerate(classes_to_remove[:10]):  # Show first 10
            count = target_counts_updated[class_name]
            print(f"  - {class_name}: {count} samples")
        if len(classes_to_remove) > 10:
            print(f"  ... and {len(classes_to_remove) - 10} more classes")
        
        initial_count_small = len(final_df)
        final_df = final_df[~final_df['class_label'].isin(classes_to_remove)].copy()
        small_classes_removed = initial_count_small - len(final_df)
        print(f"Removed {small_classes_removed} samples from {len(classes_to_remove)} small classes")
    else:
        print("No classes with <5 samples found.")
    
    # Double-check: verify no classes with <5 samples remain
    remaining_counts = final_df['class_label'].value_counts()
    problematic_classes = remaining_counts[remaining_counts < 5]
    if len(problematic_classes) > 0:
        print(f"WARNING: Still found {len(problematic_classes)} classes with <5 samples:")
        print(problematic_classes)
        # Remove them
        final_df = final_df[~final_df['class_label'].isin(problematic_classes.index)].copy()
        print(f"Removed additional {len(problematic_classes)} problematic classes")
    else:
        print("✓ Verification passed: All remaining classes have ≥5 samples")
    
    # Final statistics
    print(f"\nFinal Dataset Statistics:")
    print(f"Total samples: {len(final_df)}")
    print(f"Number of unique targets: {final_df['class_label'].nunique()}")
    
    # Show final target distribution
    print(f"\nFinal target distribution (top 20):")
    final_target_counts = final_df['class_label'].value_counts()
    print(final_target_counts.head(20))
    
    # Summary of changes
    print(f"\nSummary of filtering:")
    print(f"Original samples: {initial_count}")
    print(f"After removing empty text/target: {len(df_filtered)}")
    if expired_count > 0:
        print(f"After removing 'Expired': {len(final_df) + small_classes_removed}")
    print(f"After removing classes <5 samples: {len(final_df)}")
    print(f"Total samples removed: {initial_count - len(final_df)}")
    print(f"Retention rate: {len(final_df)/initial_count*100:.1f}%")
    
    # Show some sample data WITH COLUMN NAMES
    print(f"\nSample data (showing text format with column names):")
    for i in range(min(3, len(final_df))):
        print(f"\nSample {i+1}:")
        print(f"Text: {final_df.iloc[i]['text'][:300]}...")
        print(f"Target: {final_df.iloc[i]['class_label']}")
    
    # Save the processed dataset
    final_df.to_csv(output_file, index=False)
    print(f"\nProcessed dataset saved to: {output_file}")
    
    return final_df

def split_dataset(df, output_prefix=""):
    """
    Split dataset into 70-15-15 train-validation-test splits
    """
    print("\nSplitting dataset into train/validation/test sets...")
    
    # Check if we have enough samples for stratification
    min_class_count = df['class_label'].value_counts().min()
    use_stratify = min_class_count >= 2
    
    print(f"Minimum class count: {min_class_count}")
    if not use_stratify:
        print("Warning: Some classes have less than 2 samples. Stratification will be disabled.")
    else:
        print("All classes have at least 2 samples. Using stratified sampling.")
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['text'], 
        df['class_label'], 
        test_size=0.3, 
        random_state=42,
        stratify=df['class_label'] if use_stratify else None
    )
    
    # Check stratification for second split
    temp_df = pd.DataFrame({'text': X_temp, 'class_label': y_temp})
    min_temp_class_count = temp_df['class_label'].value_counts().min()
    use_stratify_temp = min_temp_class_count >= 2
    
    # Second split: 15% validation, 15% test from the 30% temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, 
        y_temp, 
        test_size=0.5, 
        random_state=42,
        stratify=y_temp if use_stratify_temp else None
    )
    
    # Create dataframes
    train_df = pd.DataFrame({'text': X_train, 'class_label': y_train})
    val_df = pd.DataFrame({'text': X_val, 'class_label': y_val})
    test_df = pd.DataFrame({'text': X_test, 'class_label': y_test})
    
    # Save splits
    train_file = f"{output_prefix}train_split.csv"
    val_file = f"{output_prefix}val_split.csv"
    test_file = f"{output_prefix}test_split.csv"
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nDataset split completed:")
    print(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%) -> {train_file}")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%) -> {val_file}")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%) -> {test_file}")
    
    print(f"\nClass distribution in each split:")
    print(f"Train set - unique classes: {train_df['class_label'].nunique()}")
    print(f"Validation set - unique classes: {val_df['class_label'].nunique()}")
    print(f"Test set - unique classes: {test_df['class_label'].nunique()}")
    
    # Verify all classes have adequate samples in splits
    train_class_counts = train_df['class_label'].value_counts()
    val_class_counts = val_df['class_label'].value_counts()
    test_class_counts = test_df['class_label'].value_counts()
    
    min_train = train_class_counts.min() if len(train_class_counts) > 0 else 0
    min_val = val_class_counts.min() if len(val_class_counts) > 0 else 0
    min_test = test_class_counts.min() if len(test_class_counts) > 0 else 0
    
    print(f"Minimum samples per class - Train: {min_train}, Val: {min_val}, Test: {min_test}")
    
    return train_df, val_df, test_df

# ============================================================================
# STEP 2: PROCESS YOUR DATASET
# ============================================================================

# Process your dataset
input_dataset = "textfinal_hcup_categories_FIXEDdischargesplit.csv"  # Your original dataset
processed_dataset = "hcup_processed_medical_dataset_with_labels.csv"

# Check if the processed dataset already exists
if not os.path.exists(processed_dataset):
    try:
        print(f"Processing dataset: {input_dataset}")
        df = preprocess_dataset(input_dataset, processed_dataset)
        if df is None:
            print("Failed to process dataset due to missing columns.")
            exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find the input file '{input_dataset}'")
        print("Please make sure the file exists in the current directory.")
        print(f"Current directory contents: {os.listdir('.')}")
        exit(1)
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        exit(1)
else:
    print(f"Loading existing processed dataset: {processed_dataset}")
    df = pd.read_csv(processed_dataset)
    print(f"Loaded dataset shape: {df.shape}")
    
    # Check if filtering is needed
    target_counts = df['class_label'].value_counts()
    expired_count = (df['class_label'] == 'Expired').sum()
    small_classes = target_counts[target_counts < 5]
    
    if expired_count > 0 or len(small_classes) > 0:
        print(f"\nFiltering needed on loaded dataset:")
        print(f"'Expired' samples found: {expired_count}")
        print(f"Classes with <5 samples: {len(small_classes)}")
        
        # Apply filtering to loaded dataset
        print(f"Applying filtering...")
        initial_count = len(df)
        
        # Remove "Expired" entries
        if expired_count > 0:
            df = df[df['class_label'] != 'Expired'].copy()
            print(f"Removed {expired_count} 'Expired' entries")
        
        # Remove classes with less than 5 samples
        if len(small_classes) > 0:
            print(f"Removing {len(small_classes)} classes with <5 samples...")
            classes_to_remove = small_classes.index.tolist()
            samples_to_remove = small_classes.sum()
            
            print(f"Classes being removed (showing first 10):")
            for i, class_name in enumerate(classes_to_remove[:10]):
                count = small_classes[class_name]
                print(f"  - {class_name}: {count} samples")
            if len(classes_to_remove) > 10:
                print(f"  ... and {len(classes_to_remove) - 10} more classes")
            
            df = df[~df['class_label'].isin(classes_to_remove)].copy()
            print(f"Removed {samples_to_remove} samples from small classes")
        
        # Save the filtered dataset (and use it for splitting)
        filtered_dataset = "hcup_filtered_medical_dataset_with_labels.csv"
        df.to_csv(filtered_dataset, index=False)
        print(f"Filtered dataset saved to: {filtered_dataset}")
        
        # Update the processed dataset to point to filtered version
        processed_dataset = filtered_dataset
        
        # Final statistics
        print(f"\nFiltering summary:")
        print(f"Original samples: {initial_count}")
        print(f"After filtering: {len(df)}")
        print(f"Samples removed: {initial_count - len(df)}")
        print(f"Retention rate: {len(df)/initial_count*100:.1f}%")
        
        final_target_counts = df['class_label'].value_counts()
        print(f"Final unique classes: {len(final_target_counts)}")
        print(f"Minimum class size: {final_target_counts.min()}")
        
        # Show top 10 remaining classes
        print(f"\nTop 10 remaining classes:")
        print(final_target_counts.head(10))
        
    else:
        print("No filtering needed - dataset already clean.")

# Split the dataset
try:
    train_df, val_df, test_df = split_dataset(df, output_prefix="hcup_labeled_medical_")
    
    print(f"\n" + "="*60)
    print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Files created:")
    print(f"  - {processed_dataset} (final dataset for training)")
    print(f"  - hcup_labeled_medical_train_split.csv (training set)")
    print(f"  - hcup_labeled_medical_val_split.csv (validation set)")
    print(f"  - hcup_labeled_medical_test_split.csv (test set)")
    print(f"\nYou can now use these files with your DeBERTa training script.")
    print(f"The main dataset file to use for training is: {processed_dataset}")
    
    # Final summary
    print(f"\nFinal Dataset Summary:")
    print(f"  - Total samples: {len(df):,}")
    print(f"  - Unique classes: {df['class_label'].nunique():,}")
    print(f"  - Minimum class size: {df['class_label'].value_counts().min()}")
    print(f"  - Training samples: {len(train_df):,}")
    print(f"  - Validation samples: {len(val_df):,}")
    print(f"  - Test samples: {len(test_df):,}")
    
    print(f"\n" + "="*60)
    print("TEXT FORMAT INCLUDES COLUMN LABELS:")
    print("Example: 'Chief Complaint: pain, History of Present Illness: ...'")
    print("="*60)
    
except Exception as e:
    print(f"Error during dataset splitting: {str(e)}")
    exit(1)
