"""
Optimized Clinical BERT Text Classification Pipeline
Memory-efficient version with aggressive optimizations
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
import json
from collections import defaultdict
import argparse
import numpy as np
import gc
import os

# Model path configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, "Bio_ClinicalBERT_local")
HUGGINGFACE_MODEL = 'emilyalsentzer/Bio_ClinicalBERT'

def get_model_path():
    """Get model path - prefer local if available."""
    if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        print(f"✓ Using local model: {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH
    else:
        print(f"⚠ Local model not found, will try to download from HuggingFace")
        print(f"  Expected path: {LOCAL_MODEL_PATH}")
        print(f"  Run download_clinical_bert.py first to avoid internet issues")
        return HUGGINGFACE_MODEL

# ============================================================================
# PREPROCESSING
# ============================================================================

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess_text(self, text):
        """Simple preprocessing without caching to save memory."""
        text_str = str(text)
        text_clean = re.sub(r'[^a-zA-Z\s]', '', text_str)
        words = text_clean.split()
        processed_words = [
            self.stemmer.stem(word.lower()) 
            for word in words 
            if word.lower() not in self.stop_words
        ]
        return ' '.join(processed_words)

preprocessor = TextPreprocessor()

# ============================================================================
# DATASET CLASS
# ============================================================================

class ClinicalTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# SMART SAMPLING - FIXED VERSION
# ============================================================================

def sample_for_misclassification_analysis(df, samples_per_class=100, min_samples=50, seed=42):
    """
    Aggressively sample dataset for faster analysis.
    FIXED: Handles classes with fewer samples than requested.
    """
    sampled_dfs = []
    
    for class_label in df['class_label'].unique():
        class_df = df[df['class_label'] == class_label]
        class_size = len(class_df)
        
        # Take minimum of: available samples, requested samples, or min_samples
        if class_size < min_samples:
            # If class is too small, take all and oversample with replacement
            n_samples = min_samples
            sampled = class_df.sample(n=n_samples, random_state=seed, replace=True)
        else:
            # Normal sampling without replacement
            n_samples = min(class_size, samples_per_class)
            sampled = class_df.sample(n=n_samples, random_state=seed, replace=False)
        
        sampled_dfs.append(sampled)
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"Sampled {len(result)} samples ({len(result)/len(df)*100:.1f}% of original)")
    print(f"  Classes sampled: {result['class_label'].nunique()}")
    return result

# ============================================================================
# TRAINING WITH MEMORY OPTIMIZATION
# ============================================================================

def train_clinical_bert_optimized(df, num_labels, epochs=3, batch_size=16, 
                                  max_length=128, seed=42):
    """
    Memory-optimized training for analysis phase.
    """
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print(f"Training on {len(df)} samples...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['class_label'])
    
    # Preprocess text in chunks to save memory
    print("Preprocessing texts...")
    processed_texts = []
    chunk_size = 5000
    for i in range(0, len(df), chunk_size):
        chunk = df['text'].iloc[i:i+chunk_size]
        processed_chunk = [preprocessor.preprocess_text(t) for t in chunk]
        processed_texts.extend(processed_chunk)
        if (i // chunk_size + 1) % 5 == 0:
            print(f"  Processed {i+len(chunk)}/{len(df)} texts")
    
    df['processed_text'] = processed_texts
    
    # Load tokenizer and model (check local first, then HuggingFace)
    local_model_path = "./Bio_ClinicalBERT_local"
    if os.path.exists(local_model_path):
        model_name = local_model_path
        print(f"Loading tokenizer from local path: {model_name}")
    else:
        model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        print(f"Loading tokenizer from HuggingFace: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    texts = df['processed_text'].tolist()
    labels = df['label_encoded'].tolist()
    
    dataset = ClinicalTextDataset(texts, labels, tokenizer, max_length=max_length)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           pin_memory=True, num_workers=0)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    if os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"Loading from local path: {model_path}")
    else:
        model_path = 'emilyalsentzer/Bio_ClinicalBERT'
        print(f"Loading from HuggingFace: {model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # Reinitialize classification head for new num_labels
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training loop with validation
    print(f"\nStarting training for {epochs} epochs...")
    best_val_loss = float('inf')
    patience = 2
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            # Progress update every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = train_loss / batch_count
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        torch.cuda.empty_cache()
        gc.collect()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Evaluated {batch_idx+1}/{len(test_loader)} batches")
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    print(f"Confusion matrix computed: {cm.shape}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return cm, label_encoder, df

# ============================================================================
# MISCLASSIFICATION ANALYSIS
# ============================================================================

def compute_misclassification_rates(cm, label_encoder):
    """Compute misclassification rates from confusion matrix."""
    num_classes = cm.shape[0]
    misclassification_rates = {}
    
    for i in range(num_classes):
        total = cm[i, :].sum()
        if total == 0:
            continue
            
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                try:
                    class_i = label_encoder.inverse_transform([i])[0]
                    class_j = label_encoder.inverse_transform([j])[0]
                    key = f"{class_i} - {class_j}"
                    rate = cm[i, j] / total
                    misclassification_rates[key] = rate
                except:
                    continue
    
    print(f"Found {len(misclassification_rates)} misclassification pairs")
    return misclassification_rates

def extract_significant_pairs(misclassification_rates, threshold=0.15):
    """Extract class pairs with significant misclassification rates."""
    significant_pairs = {}
    for key, rate in misclassification_rates.items():
        if rate >= threshold:
            parts = key.split(' - ', 1)
            if len(parts) == 2:
                class1, class2 = parts
                if class1 not in significant_pairs:
                    significant_pairs[class1] = []
                significant_pairs[class1].append(class2)
    
    print(f"Found {len(significant_pairs)} classes with significant misclassifications")
    return significant_pairs

def extract_common_words(df, significant_pairs, max_texts_per_class=200):
    """Extract common words efficiently with limited sampling."""
    def tokenize(text):
        text_clean = re.sub(r'[^a-zA-Z\s]', '', str(text))
        words = text_clean.lower().split()
        return [w for w in words if w not in preprocessor.stop_words]
    
    print("Extracting common words...")
    # Sample texts for efficiency
    texts_by_class = defaultdict(list)
    for class_label in df['class_label'].unique():
        class_texts = df[df['class_label'] == class_label]['text'].head(max_texts_per_class)
        for text in class_texts:
            texts_by_class[class_label].extend(tokenize(text))
    
    # Convert to sets for fast intersection
    word_sets = {k: set(v) for k, v in texts_by_class.items()}
    
    common_words_by_pair = {}
    for main_class, misclasses in significant_pairs.items():
        if main_class not in word_sets:
            continue
        main_words = word_sets[main_class]
        for misclass in misclasses:
            if misclass in word_sets:
                common_words = main_words.intersection(word_sets[misclass])
                # Only keep pairs with substantial overlap
                if len(common_words) >= 5:
                    common_words_by_pair[f"{main_class} - {misclass}"] = common_words
    
    print(f"Extracted common words for {len(common_words_by_pair)} pairs")
    return common_words_by_pair

# ============================================================================
# SYNONYM REPLACEMENT
# ============================================================================

def get_synonym(word):
    """Returns a synonym for the given word."""
    synonyms = set()
    for syn in wordnet.synsets(word)[:3]:  # Limit to first 3 synsets
        for lemma in syn.lemmas()[:2]:  # Limit to first 2 lemmas
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace('_', ' '))
    return next(iter(synonyms), word)

def update_dataframe_texts(df, common_words_by_pair, text_column='text'):
    """Update texts with synonym replacements efficiently."""
    df = df.copy()
    
    print(f"Applying synonym replacement to {len(common_words_by_pair)} class pairs...")
    
    for idx, (key, common_words) in enumerate(common_words_by_pair.items()):
        parts = key.split(' - ', 1)
        if len(parts) != 2:
            continue
        
        class1, class2 = [s.strip() for s in parts]
        # Limit to top 20 most common words to replace
        lower_common_words = set(list(common_words)[:20])
        
        mask = df['class_label'] == class2
        if mask.sum() == 0:
            continue
        
        def replace_words(text):
            words = str(text).split()
            replaced = [get_synonym(w) if w.lower() in lower_common_words else w for w in words]
            return ' '.join(replaced)
        
        df.loc[mask, text_column] = df.loc[mask, text_column].apply(replace_words)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(common_words_by_pair)} pairs")
    
    return df

# ============================================================================
# SMART DATA BALANCING
# ============================================================================

def balance_dataset_capped(df, label_column='class_label', max_samples_per_class=1500, seed=42):
    """Balance dataset with a strict cap."""
    class_counts = df[label_column].value_counts()
    target_count = min(class_counts.max(), max_samples_per_class)
    
    print(f"\nBalancing to {target_count} samples per class...")
    print(f"Original size: {len(df)} samples")
    
    balanced_dfs = []
    for class_name in df[label_column].unique():
        class_df = df[df[label_column] == class_name]
        current_count = len(class_df)
        
        if current_count < target_count:
            # Oversample
            additional = class_df.sample(n=target_count - current_count, replace=True, random_state=seed)
            balanced_dfs.append(pd.concat([class_df, additional]))
        elif current_count > target_count:
            # Downsample
            balanced_dfs.append(class_df.sample(n=target_count, random_state=seed))
        else:
            balanced_dfs.append(class_df)
    
    result = pd.concat(balanced_dfs, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"Balanced size: {len(result)} samples ({len(result)/len(df)*100:.1f}% of original)")
    print(f"Classes: {result[label_column].nunique()}")
    
    return result

# ============================================================================
# FINAL CLASSIFICATION
# ============================================================================

def run_final_classification(df, epochs=5, batch_size=16, max_length=128):
    """Run final classification with full metrics."""
    print("\n" + "="*80)
    print("FINAL CLASSIFICATION WITH CLINICAL BERT")
    print("="*80)
    
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['class_label'])
    num_labels = len(label_encoder.classes_)
    
    print(f"\nDataset: {len(df)} samples, {num_labels} classes")
    
    print("Preprocessing texts...")
    processed_texts = []
    chunk_size = 5000
    for i in range(0, len(df), chunk_size):
        chunk = df['text'].iloc[i:i+chunk_size]
        processed_chunk = [preprocessor.preprocess_text(t) for t in chunk]
        processed_texts.extend(processed_chunk)
        if (i // chunk_size + 1) % 10 == 0:
            print(f"  Processed {i+len(chunk)}/{len(df)} texts")
    
    df['processed_text'] = processed_texts
    
    # Use local model if available
    local_model_path = "./Bio_ClinicalBERT_local"
    if os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"Using local model: {model_path}")
    else:
        model_path = 'emilyalsentzer/Bio_ClinicalBERT'
        print(f"Using HuggingFace model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    texts = df['processed_text'].tolist()
    labels = df['label_encoded'].tolist()
    
    dataset = ClinicalTextDataset(texts, labels, tokenizer, max_length=max_length)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           pin_memory=True, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    
    if os.path.exists(local_model_path):
        model_path = local_model_path
    else:
        model_path = 'emilyalsentzer/Bio_ClinicalBERT'
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True  # Reinitialize classification head for new num_labels
    )
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Training
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.4f}")
                torch.cuda.empty_cache()
        
        avg_epoch_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} complete - Avg Loss: {avg_epoch_loss:.4f}")
        torch.cuda.empty_cache()
        gc.collect()
    
    # Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Evaluated {batch_idx+1}/{len(test_loader)} batches")
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    with open('clinical_bert_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to clinical_bert_results.json")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimized Clinical BERT Pipeline")
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--synonyms", type=str, default="false")
    parser.add_argument("--balance", type=str, default="false")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_samples_per_class", type=int, default=1500)
    parser.add_argument("--samples_for_analysis", type=int, default=100)
    
    args = parser.parse_args()
    
    print("="*80)
    print("CLINICAL BERT CLASSIFICATION PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Synonyms: {args.synonyms}")
    print(f"  Balance: {args.balance}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples per class: {args.max_samples_per_class}")
    print(f"  Samples for analysis: {args.samples_for_analysis}")
    
    print("\nLoading dataset...")
    df = pd.read_csv(args.csv_file)
    print(f"Original dataset: {len(df)} rows, {df['class_label'].nunique()} classes")
    
    # Print class distribution stats
    class_counts = df['class_label'].value_counts()
    print(f"\nClass size distribution:")
    print(f"  Min samples per class: {class_counts.min()}")
    print(f"  Max samples per class: {class_counts.max()}")
    print(f"  Median samples per class: {class_counts.median():.0f}")
    print(f"  Mean samples per class: {class_counts.mean():.1f}")
    
    if args.synonyms.lower() == "true":
        print(f"\n{'='*80}")
        print("SYNONYM REPLACEMENT")
        print("="*80)
        
        # Sample for analysis
        df_sample = sample_for_misclassification_analysis(
            df, 
            samples_per_class=args.samples_for_analysis
        )
        
        # Train on sample to find misclassifications
        cm, label_encoder, _ = train_clinical_bert_optimized(
            df_sample, 
            num_labels=df_sample['class_label'].nunique(),
            epochs=3,
            batch_size=args.batch_size
        )
        
        # Extract and apply replacements
        mis_rates = compute_misclassification_rates(cm, label_encoder)
        significant_pairs = extract_significant_pairs(mis_rates, threshold=0.15)
        
        if significant_pairs:
            common_words = extract_common_words(df, significant_pairs)
            if common_words:
                df = update_dataframe_texts(df, common_words)
                df.to_csv("checkpoint_synonyms.csv", index=False)
                print("✅ Synonym replacement complete")
            else:
                print("⚠️  No common words found, skipping replacement")
        else:
            print("⚠️  No significant misclassification pairs found")
    
    if args.balance.lower() == "true":
        print(f"\n{'='*80}")
        print("DATA BALANCING")
        print("="*80)
        df = balance_dataset_capped(df, max_samples_per_class=args.max_samples_per_class)
        df.to_csv("checkpoint_balanced.csv", index=False)
    
    # Final classification
    results = run_final_classification(
        df, 
        epochs=5, 
        batch_size=args.batch_size, 
        max_length=args.max_length
    )
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
