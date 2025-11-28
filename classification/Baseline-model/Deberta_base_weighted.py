import os
import re
import torch
import pandas as pd
import gc  # Added for memory management
import psutil  # Added for memory monitoring
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import DebertaTokenizer, Trainer, TrainingArguments
from transformers import DebertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss

# GPU utilization monitoring function
def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(0) / 1024**3      # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB, Total: {total:.1f}GB")
        print(f"GPU Utilization: {(allocated/total)*100:.1f}%")
    
    # System RAM check
    ram = psutil.virtual_memory()
    print(f"System RAM - Used: {ram.used/1024**3:.1f}GB, Available: {ram.available/1024**3:.1f}GB, Total: {ram.total/1024**3:.1f}GB")
    print(f"RAM Usage: {ram.percent:.1f}%")

# Memory cleanup function
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Initial GPU check
check_gpu_memory()

# Step 1: Load Dataset
csv_file = "hcup_processed_medical_dataset_with_labels.csv"
print(f"Loading dataset: {csv_file}")
df = pd.read_csv(csv_file)
print(f"Dataset loaded: {len(df)} samples")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# CRITICAL: Set GPU memory fraction to prevent OOM
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
    print("Set GPU memory fraction to 80%")

# Step 2: Device Setup
device = torch.device("cuda")

# Memory check after loading data
print("After loading dataset:")
check_gpu_memory()

# Step 4: Train-Test-Validation Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['class_label'], test_size=0.2
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.1
)

print(f"Data split - Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")

# Step 5: Label Mapping
label_mapping = {label: idx for idx, label in enumerate(df['class_label'].unique())}
y_train_numeric = [label_mapping[label] for label in y_train]
y_valid_numeric = [label_mapping[label] for label in y_valid]
y_test_numeric = [label_mapping[label] for label in y_test]

print(f"Number of unique labels: {len(label_mapping)}")

# Step 6: Compute Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(list(label_mapping.values())),
    y=y_train_numeric
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Clean up dataframe from memory
del df
cleanup_memory()
print("Cleaned up original dataframe from memory")

# Step 7: Tokenizer and Tokenization (MEMORY-EFFICIENT VERSION)
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

def tokenize_in_batches(texts, batch_size=500):
    """Tokenize in smaller batches to prevent memory overload"""
    all_input_ids = []
    all_attention_masks = []
    
    print(f"Tokenizing {len(texts)} samples in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        batch_encoded = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=256,  # CRITICAL: Reduced from 256 to 128 for stability
            return_tensors="pt"
        )
        
        all_input_ids.append(batch_encoded['input_ids'])
        all_attention_masks.append(batch_encoded['attention_mask'])
        
        # Progress and memory cleanup
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i + len(batch_texts)}/{len(texts)} samples")
            cleanup_memory()
    
    print(f"Concatenating {len(all_input_ids)} batches...")
    # Concatenate all batches
    return {
        'input_ids': torch.cat(all_input_ids, dim=0),
        'attention_mask': torch.cat(all_attention_masks, dim=0)
    }

print("Tokenizing data in batches...")
train_encodings = tokenize_in_batches(X_train.tolist(), batch_size=500)
print("Train tokenization complete")
check_gpu_memory()

valid_encodings = tokenize_in_batches(X_valid.tolist(), batch_size=500)
print("Validation tokenization complete")
check_gpu_memory()

test_encodings = tokenize_in_batches(X_test.tolist(), batch_size=500)
print("Test tokenization complete")
check_gpu_memory()

# Step 8: Create Hugging Face Datasets
# CRITICAL: Don't move everything to GPU immediately - let Trainer handle it
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': torch.tensor(y_train_numeric, dtype=torch.long)
})

valid_dataset = Dataset.from_dict({
    'input_ids': valid_encodings['input_ids'],
    'attention_mask': valid_encodings['attention_mask'],
    'labels': torch.tensor(y_valid_numeric, dtype=torch.long)
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': torch.tensor(y_test_numeric, dtype=torch.long)
})

# Clean up tokenized data from CPU memory
del train_encodings, valid_encodings, test_encodings
cleanup_memory()
print("Cleaned up tokenization data from memory")
check_gpu_memory()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(self.model.device)
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        outputs = model(**inputs)

        logits = outputs['logits']
        loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
    # CRITICAL: Fix method signature for newer transformers version
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        
        # Clean GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return loss.detach()

# Step 9: Custom Model with Weighted Loss
class DebertaForWeightedClassification(DebertaForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **kwargs
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

# Initialize the Model
print("Loading DeBERTa model...")
# Load the base model without gradient_checkpointing parameter
base_model = DebertaForSequenceClassification.from_pretrained(
    'microsoft/deberta-base',
    num_labels=len(label_mapping)
)

# Create our custom model using the base model's config
model = DebertaForWeightedClassification(base_model.config, class_weights)

# Copy the pretrained weights
model.load_state_dict(base_model.state_dict(), strict=False)

# Enable gradient checkpointing after model creation
model.gradient_checkpointing_enable()

# Move to device
model = model.to(device)

print("Model loaded")
check_gpu_memory()

# Step 10: Metrics Function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Step 11: Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    gradient_accumulation_steps=8,  # CRITICAL: Increased from 4 to 8
    num_train_epochs=5,  # CRITICAL: Reduced from 10 to 5 epochs
    per_device_train_batch_size=2,  # CRITICAL: Reduced from 8 to 2
    per_device_eval_batch_size=4,   # CRITICAL: Reduced from 8 to 4
    eval_strategy='epoch',
    save_total_limit=1,  # CRITICAL: Reduced from 2 to 1 to save disk space
    logging_dir='./logs',
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=50,  # CRITICAL: Reduced logging frequency
    save_steps=1000,   # CRITICAL: Reduced save frequency
    report_to="none",
    dataloader_pin_memory=False,  # CRITICAL: Disable pinned memory
    dataloader_num_workers=0,     # CRITICAL: No multiprocessing
    fp16=False,  # CRITICAL: Disabled fp16 to prevent overflow errors
)

# Step 12: Custom Callback for Metric Logging
from transformers import TrainerCallback

class LogMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"Epoch {state.epoch:.1f}: {metrics}")
        # Check memory after each evaluation
        check_gpu_memory()
        cleanup_memory()

# Step 13: Initialize Trainer
print("Initializing trainer...")
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[LogMetricsCallback()]
)

print("Trainer initialized")
check_gpu_memory()

# Step 14: Train Model
print("Starting training...")
print(f"Training parameters:")
print(f"  - Batch size: {training_args.per_device_train_batch_size}")
print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  - Epochs: {training_args.num_train_epochs}")
print(f"  - Mixed precision (fp16): {training_args.fp16}")

check_gpu_memory()
trainer.train()
trainer.save_model('./saved_model')
print("Training completed and model saved")

# Step 15: Test Evaluation
print("Evaluating the model on the test set...")
check_gpu_memory()
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test Results: {test_results}")

# Step 16: Get Predictions on Test Data
print("Generating predictions...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# Classification Report
print("\nClassification Report:")
# === SAFE CLASSIFICATION REPORT & COMPACT CONFUSION MATRIX ===
import os, json, csv
import numpy as np
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)

# Ensure output dir
os.makedirs("results", exist_ok=True)

# Convert to numpy ints
y_true = np.asarray(y_test_numeric, dtype=int)
y_pred = np.asarray(y_pred, dtype=int)

# Build id->name mapping because your label_mapping is {name -> id}
id2name = {v: k for k, v in label_mapping.items()}

# Only use labels that actually appear in y_true or y_pred
present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
target_names = [id2name.get(i, f"CLS_{i}") for i in present_labels]

# 1) Text report (robust: never crashes on missing classes)
report_text = classification_report(
    y_true, y_pred,
    labels=present_labels,
    target_names=target_names,
    zero_division=0,
    digits=4
)

# 2) Dict form for saving structured outputs
report_dict = classification_report(
    y_true, y_pred,
    labels=present_labels,
    target_names=target_names,
    zero_division=0,
    output_dict=True
)

# 3) Overall summary metrics
summary = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
    "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    "n_classes_in_mapping": int(len(id2name)),
    "n_classes_present_in_eval": int(len(present_labels)),
    "n_unique_in_y_true": int(len(set(y_true))),
    "n_unique_in_y_pred": int(len(set(y_pred))),
}

# 4) Save to disk
with open("results/test_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
with open("results/test_report.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# 5) Per-class CSV (easy to sort/filter later)
rows = []
for lab, name in zip(present_labels, target_names):
    entry = report_dict.get(name, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
    rows.append({
        "class_id": int(lab),
        "class_name": name,
        "precision": float(entry.get("precision", 0.0)),
        "recall": float(entry.get("recall", 0.0)),
        "f1": float(entry.get("f1-score", 0.0)),
        "support": int(entry.get("support", 0)),
    })
with open("results/test_per_class.csv", "w", newline="", encoding="utf-8") as cf:
    writer = csv.DictWriter(cf, fieldnames=["class_id", "class_name", "precision", "recall", "f1", "support"])
    writer.writeheader()
    writer.writerows(rows)

# 6) Console output
print("=== SUMMARY METRICS ===")
for k, v in summary.items():
    print(f"{k}: {v}")
print("\n=== PER-CLASS REPORT ===")
print(report_text)

# 7) Compact confusion matrix for TOP-K classes by support (so it’s readable)
top_k = 20
# compute supports on y_true
support = {lab: 0 for lab in present_labels}
for lab in y_true:
    support[int(lab)] = support.get(int(lab), 0) + 1

labels_sorted = sorted(present_labels, key=lambda l: support.get(l, 0), reverse=True)
top_labels = labels_sorted[:top_k]

# Filter to top-K classes
mask = np.isin(y_true, top_labels)
y_true_top = y_true[mask]
y_pred_top = y_pred[mask]

if y_true_top.size > 0:
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_labels)
    top_names = [id2name.get(i, f"CLS_{i}") for i in top_labels]

    plt.rcParams['figure.figsize'] = [max(8, top_k * 0.6), max(6, top_k * 0.5)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=top_names)
    disp.plot(cmap='Blues', xticks_rotation=90)
    plt.title(f"Confusion Matrix (Top {top_k} by support)")
    plt.tight_layout()
    plt.savefig("results/test_confusion_top20datset.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix: results/test_confusion_top20dataset.png")
else:
    print("No samples among top-K labels to plot confusion matrix.")
# === END SAFE BLOCK ===
# Final memory check
print("\nFinal memory usage:")
check_gpu_memory()
