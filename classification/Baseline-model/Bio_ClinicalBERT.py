import os
import torch
import pandas as pd
import gc
import psutil
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from transformers import TrainerCallback
import warnings
warnings.filterwarnings('ignore')

# ==================== MEMORY MONITORING ====================
def check_gpu_memory():
    """Monitor GPU and system RAM usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB / {total:.2f}GB ({(allocated/total)*100:.1f}%)")
    
    ram = psutil.virtual_memory()
    print(f"System RAM - Used: {ram.used/1024**3:.2f}GB / {ram.total/1024**3:.2f}GB ({ram.percent:.1f}%)")

def cleanup_memory():
    """Force garbage collection and clear GPU cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==================== SETUP ====================
print("=" * 60)
print("MEDICAL DIAGNOSIS CLASSIFICATION - CLINICAL BERT OPTIMIZED")
print("=" * 60)

# Check CUDA availability
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA compiled version: {torch.version.cuda}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available! This script requires GPU. Exiting.")

print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Configure memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.cuda.set_per_process_memory_fraction(0.95)
device = torch.device("cuda")

print(f"\n✓ Device: {device}")
print("\nInitial Memory Status:")
check_gpu_memory()

# ==================== LOAD DATA ====================
print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

csv_file = "hcup_processed_medical_dataset_with_labels.csv"
df = pd.read_csv(csv_file)
print(f"\n✓ Dataset loaded: {len(df):,} samples")
print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Using FULL dataset for training")

# Check class distribution
print(f"\n✓ Unique classes: {df['class_label'].nunique()}")
print("\nClass distribution (top 10):")
class_counts = df['class_label'].value_counts()
print(class_counts.head(10))

# Filter rare classes
MIN_SAMPLES_PER_CLASS = 7
class_counts = df['class_label'].value_counts()
classes_to_keep = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
df_filtered = df[df['class_label'].isin(classes_to_keep)].copy()

if len(df_filtered) < len(df):
    removed = len(df) - len(df_filtered)
    removed_classes = len(class_counts) - len(classes_to_keep)
    print(f"⚠ Filtered out {removed:,} samples from {removed_classes} classes with <{MIN_SAMPLES_PER_CLASS} samples")
    print(f"✓ Remaining: {len(df_filtered):,} samples, {len(classes_to_keep)} classes")
    df = df_filtered

# ==================== DATA SPLITTING ====================
print("\n" + "=" * 60)
print("DATA SPLITTING")
print("=" * 60)

# First split: 75% train, 25% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    df['text'], 
    df['class_label'], 
    test_size=0.25,
    stratify=df['class_label'],
    random_state=42
)

# Second split: 60% of temp for test, 40% for validation
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, 
    y_temp, 
    test_size=0.6,
    stratify=y_temp,
    random_state=42
)

print("✓ Stratified split completed successfully")
print(f"\n✓ Train set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"✓ Validation set: {len(X_valid):,} samples ({len(X_valid)/len(df)*100:.1f}%)")
print(f"✓ Test set: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")

# ==================== LABEL ENCODING ====================
label_mapping = {label: idx for idx, label in enumerate(sorted(df['class_label'].unique()))}
inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

y_train_numeric = [label_mapping[label] for label in y_train]
y_valid_numeric = [label_mapping[label] for label in y_valid]
y_test_numeric = [label_mapping[label] for label in y_test]

print(f"\n✓ Number of classes: {len(label_mapping)}")

# ==================== CLASS WEIGHTS ====================
print("\n" + "=" * 60)
print("COMPUTING CLASS WEIGHTS")
print("=" * 60)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(list(label_mapping.values())),
    y=y_train_numeric
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\n✓ Class weights computed (range: {class_weights.min():.2f} - {class_weights.max():.2f})")

# Clean up original dataframe
del df, X_temp, y_temp
cleanup_memory()

# ==================== TOKENIZATION ====================
print("\n" + "=" * 60)
print("TOKENIZATION WITH CLINICAL BERT")
print("=" * 60)

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
print(f"\n✓ Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_in_batches(texts, labels, batch_size=1000, max_length=512):
    """Efficient batch tokenization with progress tracking"""
    print(f"\nTokenizing {len(texts):,} samples (batch size: {batch_size}, max_length: {max_length})...")
    
    all_input_ids = []
    all_attention_masks = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = list(texts[i:i+batch_size])
        
        encoded = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        all_input_ids.append(encoded['input_ids'])
        all_attention_masks.append(encoded['attention_mask'])
        
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Progress: {min(i + batch_size, len(texts)):,}/{len(texts):,} samples")
            cleanup_memory()
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': torch.cat(all_input_ids, dim=0),
        'attention_mask': torch.cat(all_attention_masks, dim=0),
        'labels': torch.tensor(labels, dtype=torch.long)
    })
    
    # Clean up intermediate tensors
    del all_input_ids, all_attention_masks
    cleanup_memory()
    
    return dataset

# Tokenize all splits
train_dataset = tokenize_in_batches(X_train.tolist(), y_train_numeric, batch_size=1000, max_length=512)
print("✓ Train tokenization complete")

valid_dataset = tokenize_in_batches(X_valid.tolist(), y_valid_numeric, batch_size=1000, max_length=512)
print("✓ Validation tokenization complete")

test_dataset = tokenize_in_batches(X_test.tolist(), y_test_numeric, batch_size=1000, max_length=512)
print("✓ Test tokenization complete")

print("\nMemory after tokenization:")
check_gpu_memory()

# ==================== CUSTOM TRAINER ====================
class WeightedLossTrainer(Trainer):
    """Custom trainer with weighted cross-entropy loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weighted cross-entropy loss
        loss_fct = CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ==================== MODEL INITIALIZATION ====================
print("\n" + "=" * 60)
print("LOADING CLINICAL BERT MODEL")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_mapping),
    problem_type="single_label_classification"
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model = model.to(device)

print(f"\n✓ Model loaded: {MODEL_NAME}")
print(f"✓ Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

print("\nMemory after model loading:")
check_gpu_memory()

# ==================== METRICS ====================
def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ==================== TRAINING CONFIGURATION - OPTIMIZED ====================
print("\n" + "=" * 60)
print("TRAINING CONFIGURATION - OPTIMIZED FOR ACCURACY")
print("=" * 60)

training_args = TrainingArguments(
    output_dir='./clinical_bert_results',
    
    # Training parameters - OPTIMIZED FOR ACCURACY
    num_train_epochs=10,             # 🔥 INCREASED from 2 to 10
    per_device_train_batch_size=64,  # Large batch for H100
    per_device_eval_batch_size=128,  # Even larger for eval
    gradient_accumulation_steps=1,
    
    # Optimization - TUNED FOR BETTER CONVERGENCE
    learning_rate=5e-5,              # 🔥 INCREASED from 3e-5 to 5e-5
    weight_decay=0.01,
    warmup_ratio=0.15,               # 🔥 INCREASED from 0.1 to 0.15
    lr_scheduler_type="cosine",      # 🔥 ADDED: Cosine decay
    max_grad_norm=1.0,
    
    # Evaluation and saving
    eval_strategy='steps',
    eval_steps=1000,                 # Evaluate every 1000 steps
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=2,              # Keep 2 best checkpoints
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    
    # Logging
    logging_dir='./clinical_bert_logs',
    logging_steps=100,
    logging_first_step=True,
    report_to="none",
    
    # Memory optimization - H100
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    fp16=False,
    bf16=True,                       # BF16 for H100
    bf16_full_eval=True,
    dataloader_prefetch_factor=2,
    
    # Speed optimizations
    gradient_checkpointing=False,
    optim="adamw_torch_fused",
    tf32=True,                       # TF32 for H100
    
    # Other
    seed=42,
    remove_unused_columns=False,
    use_cpu=False,
)

effective_batch_size = (
    training_args.per_device_train_batch_size * 
    training_args.gradient_accumulation_steps * 
    (torch.cuda.device_count() if torch.cuda.is_available() else 1)
)

total_steps = (len(train_dataset) // effective_batch_size) * training_args.num_train_epochs
estimated_hours = total_steps * 0.5 / 3600  # Assume 0.5 sec/step

print(f"\n✓ Epochs: {training_args.num_train_epochs} (was 2)")
print(f"✓ Batch size per device: {training_args.per_device_train_batch_size}")
print(f"✓ Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
print(f"✓ Effective batch size: {effective_batch_size}")
print(f"✓ Learning rate: {training_args.learning_rate} (was 3e-5)")
print(f"✓ LR scheduler: {training_args.lr_scheduler_type} (was linear)")
print(f"✓ Warmup ratio: {training_args.warmup_ratio} (was 0.1)")
print(f"✓ Mixed precision: BF16")
print(f"✓ Optimizer: {training_args.optim}")
print(f"\n📊 Training Estimates:")
print(f"✓ Total training steps: ~{total_steps:,}")
print(f"✓ Estimated training time: ~{estimated_hours:.1f} hours")
print(f"✓ Steps per epoch: ~{total_steps // training_args.num_train_epochs:,}")
print(f"\n🎯 Expected accuracy improvement: 32% → 50-65%")

# ==================== CALLBACKS ====================
class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory usage during training"""
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n--- End of Epoch {int(state.epoch)} ---")
        check_gpu_memory()
        cleanup_memory()

# ==================== TRAINING ====================
print("\n" + "=" * 60)
print("STARTING OPTIMIZED TRAINING")
print("=" * 60)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[MemoryMonitorCallback()]
)

print("\n🚀 Training started with optimized settings...\n")
train_result = trainer.train()

print("\n✓ Training completed!")
print(f"✓ Best checkpoint: {trainer.state.best_model_checkpoint}")

# Save the model
model_save_path = './clinical_bert_final_model_optimized'
trainer.save_model(model_save_path)
print(f"✓ Model saved to: {model_save_path}")

# ==================== EVALUATION ====================
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

test_results = trainer.evaluate(eval_dataset=test_dataset)
print("\n📊 Test Results:")
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")

# ==================== PREDICTIONS ====================
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# Classification report
print("\n📋 Classification Report:\n")
print(classification_report(
    y_test_numeric, 
    y_pred, 
    target_names=[inverse_label_mapping[i] for i in range(len(label_mapping))],
    digits=4
))

# ==================== CONFUSION MATRIX ====================
print("\n" + "=" * 60)
print("CONFUSION MATRIX VISUALIZATION")
print("=" * 60)

cm = confusion_matrix(y_test_numeric, y_pred)

# Limit to top classes if too many
max_display_classes = 20
if len(label_mapping) > max_display_classes:
    print(f"\n⚠ Too many classes ({len(label_mapping)}). Showing top {max_display_classes} most frequent classes.")
    
    # Get top classes by frequency in test set
    class_counts_test = pd.Series(y_test_numeric).value_counts().head(max_display_classes)
    top_indices = class_counts_test.index.tolist()
    
    # Filter confusion matrix
    cm_subset = cm[np.ix_(top_indices, top_indices)]
    labels_subset = [inverse_label_mapping[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_subset, display_labels=labels_subset)
else:
    fig, ax = plt.subplots(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[inverse_label_mapping[i] for i in range(len(label_mapping))]
    )

disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='d')
plt.title(f"Medical Diagnosis Classification - Clinical BERT (Optimized)\nTest Accuracy: {test_results['eval_accuracy']:.2%}", 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save figure
output_file = "clinical_bert_confusion_matrix_optimized.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Confusion matrix saved to: {output_file}")
plt.show()

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 60)
print("TRAINING SUMMARY - OPTIMIZED VERSION")
print("=" * 60)

print(f"""
Model: {MODEL_NAME}
Total Samples: {len(X_train) + len(X_valid) + len(X_test):,}
Classes: {len(label_mapping)}

Optimization Changes:
  • Epochs: 2 → 10 (5x more training)
  • Learning Rate: 3e-5 → 5e-5 (67% higher)
  • LR Scheduler: linear → cosine (better decay)
  • Warmup: 0.1 → 0.15 (50% more warmup)

Final Metrics:
  • Accuracy:  {test_results['eval_accuracy']:.4f}
  • Precision: {test_results['eval_precision']:.4f}
  • Recall:    {test_results['eval_recall']:.4f}
  • F1 Score:  {test_results['eval_f1']:.4f}

Improvement vs Baseline (2 epochs):
  • Previous Accuracy: 0.3224 (32.2%)
  • Current Accuracy:  {test_results['eval_accuracy']:.4f} ({test_results['eval_accuracy']*100:.1f}%)
  • Improvement: {(test_results['eval_accuracy'] - 0.3224)*100:+.1f} percentage points

Model saved to: {model_save_path}
""")

print("\nFinal Memory Status:")
check_gpu_memory()

print("\n✅ Optimized training pipeline completed successfully!")
print("=" * 60)
