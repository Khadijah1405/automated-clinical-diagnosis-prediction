import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from transformers import DebertaTokenizer, DebertaForSequenceClassification
from torch.optim import AdamW
import torch.nn as nn
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import nltk
import os

# ============================================================================
# NLTK DATA INITIALIZATION - Run once at module load
# ============================================================================
def _ensure_nltk_data():
    """Ensure NLTK data is downloaded before use."""
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        stemmer.stem('test')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)

# Initialize NLTK data
_ensure_nltk_data()

# ============================================================================
# MODULE-LEVEL CACHES FOR PERFORMANCE
# ============================================================================
_STOP_WORDS_CACHE = None
_STEMMER_CACHE = None

def get_stop_words():
    """Get cached stopwords set."""
    global _STOP_WORDS_CACHE
    if _STOP_WORDS_CACHE is None:
        _STOP_WORDS_CACHE = set(stopwords.words('english'))
    return _STOP_WORDS_CACHE

def get_stemmer():
    """Get cached Porter stemmer."""
    global _STEMMER_CACHE
    if _STEMMER_CACHE is None:
        _STEMMER_CACHE = PorterStemmer()
    return _STEMMER_CACHE

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
def preprocess_text(text):
    """Remove special characters, lowercase, remove stopwords, and stem."""
    stop_words = get_stop_words()
    stemmer = get_stemmer()
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    words = text.split()
    processed_words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(processed_words)

# ============================================================================
# DATASET CLASS
# ============================================================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
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
# CUSTOM DEBERTA CLASSIFIER
# ============================================================================
class CustomDebertaClassifier(nn.Module):
    def __init__(self, base_model_name='microsoft/deberta-base', num_labels=2, dropout_rate=0.3):
        super(CustomDebertaClassifier, self).__init__()
        self.model = DebertaForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss, outputs.logits

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_model(df, num_labels, epochs=10, batch_size=16, seed=42):
    """Train DeBERTa model and return confusion matrix."""
    print(f"Encoding labels for {len(df)} samples...")
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['class_label'])
    
    # Add progress bar for preprocessing
    print(f"Preprocessing {len(df)} texts...")
    tqdm.pandas(desc="Preprocessing texts")
    df['processed_text'] = df['text'].progress_apply(preprocess_text)
    
    print("Loading tokenizer...")
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    
    texts = df['processed_text'].tolist()
    labels = df['label_encoded'].tolist()
    
    print("Creating dataset...")
    dataset = TextDataset(texts, labels, tokenizer, max_length=64)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CustomDebertaClassifier(num_labels=num_labels)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    # Training loop with gradient accumulation
    accumulation_steps = 2
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            loss, _ = model(input_ids, attention_mask, labels_batch)
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1} training complete.")
        torch.cuda.empty_cache()
    
    # Evaluate on test set
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            _, logits = model(input_ids, attention_mask, labels_batch)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
            del input_ids, attention_mask, labels_batch, logits, preds
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    del model
    torch.cuda.empty_cache()
    
    return cm, label_encoder, df

# ============================================================================
# MISCLASSIFICATION ANALYSIS
# ============================================================================
def compute_misclassification_rates(cm, label_encoder):
    """Compute misclassification rates from confusion matrix."""
    num_classes = len(label_encoder.classes_)
    
    if cm.shape[0] != num_classes or cm.shape[1] != num_classes:
        print(f"Warning: Confusion matrix shape {cm.shape} doesn't match num_classes {num_classes}")
        actual_size = min(cm.shape[0], cm.shape[1], num_classes)
        cm = cm[:actual_size, :actual_size]
        num_classes = actual_size
        print(f"Adjusted to size: {actual_size}")
    
    misclassification_rates = {}
    self_classification_rates = {}
    
    for i in range(num_classes):
        total = cm[i, :].sum()
        self_rate = cm[i, i] / total if total > 0 else 0
        class_name = label_encoder.inverse_transform([i])[0]
        self_classification_rates[class_name] = self_rate
        
        for j in range(num_classes):
            if i != j:
                true_class = label_encoder.inverse_transform([i])[0]
                pred_class = label_encoder.inverse_transform([j])[0]
                key = f"{true_class} - {pred_class}"
                rate = cm[i, j] / total if total > 0 else 0
                misclassification_rates[key] = rate
    
    return misclassification_rates, self_classification_rates

def extract_significant_pairs(misclassification_rates, threshold=0.2):
    """Extract class pairs with significant misclassification rates."""
    significant_pairs = {}
    for key, rate in misclassification_rates.items():
        if rate >= threshold:
            # Split on ' - ' with maxsplit=1 to handle hyphens in class names
            parts = key.split(' - ', 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed key: {key}")
                continue
            class1, class2 = parts
            if class1 not in significant_pairs:
                significant_pairs[class1] = []
            significant_pairs[class1].append(class2)
    return significant_pairs

def extract_common_words(df, significant_pairs):
    """Extract common words between misclassified class pairs."""
    # Use cached stopwords and stemmer
    stop_words = get_stop_words()
    stemmer = get_stemmer()
    
    def tokenize(text):
        """Tokenize text using cached resources."""
        text = re.sub(r'[^a-zA-Z\s]', '', str(text))
        words = text.split()
        return [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    
    from collections import defaultdict
    texts_by_class = defaultdict(list)
    
    print(f"Tokenizing texts for {len(df)} samples...")
    for idx, row in enumerate(df.iterrows()):
        _, row_data = row
        tokens = tokenize(row_data['text'])
        texts_by_class[row_data['class_label']].extend(tokens)
        
        # Progress indicator every 10000 rows
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} rows...")
    
    print("Computing common words for class pairs...")
    common_words_by_pair = {}
    for main_class, misclasses in significant_pairs.items():
        main_words = set(texts_by_class[main_class])
        for misclass in misclasses:
            misclass_words = set(texts_by_class[misclass])
            common_words = main_words.intersection(misclass_words)
            common_words_by_pair[f"{main_class} - {misclass}"] = common_words
            print(f"  {main_class} - {misclass}: {len(common_words)} common words")
    
    return common_words_by_pair

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_misclassification_extraction(input_data, threshold=0.2, epochs=3):
    """Run the complete misclassification extraction pipeline."""
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()
    
    num_labels = df['class_label'].nunique()
    print(f"Training model with {num_labels} classes for {epochs} epochs...")
    
    cm, label_encoder, df_out = train_model(df, num_labels=num_labels, epochs=epochs)
    print(f"Confusion matrix shape: {cm.shape}")
    print(f"Number of classes in label encoder: {len(label_encoder.classes_)}")
    
    mis_rates, self_rates = compute_misclassification_rates(cm, label_encoder)
    significant_pairs = extract_significant_pairs(mis_rates, threshold=threshold)
    common_words_by_pair = extract_common_words(df_out, significant_pairs)
    
    print(f"Found {len(significant_pairs)} classes with significant misclassifications")
    print(f"Extracted common words for {len(common_words_by_pair)} class pairs")
    
    return common_words_by_pair
