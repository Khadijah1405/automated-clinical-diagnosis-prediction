import os
import sys
import torch
import pandas as pd
import gc
import psutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss

# ============================================================================
# CRITICAL: CUDA AVAILABILITY CHECK - FAIL FAST IF NO CUDA
# ============================================================================

print("="*60)
print("CHECKING CUDA AVAILABILITY")
print("="*60)

if not torch.cuda.is_available():
    print("\n❌ ERROR: CUDA is not available!")
    print("\nPossible reasons:")
    print("1. PyTorch was installed without CUDA support")
    print("2. CUDA drivers are not properly installed")
    print("3. GPU is not accessible in your environment")
    print("\nTo fix:")
    print("- Check CUDA installation: nvidia-smi")
    print("- Reinstall PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("- Check SLURM GPU allocation: squeue -u $USER")
    sys.exit(1)

print(f"✓ CUDA is available!")
print(f"✓ CUDA version: {torch.version.cuda}")
print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
print(f"✓ Current GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ GPU Compute Capability: {torch.cuda.get_device_capability(0)}")

# Force all operations to use CUDA
device = torch.device("cuda:0")
torch.cuda.set_device(0)
print(f"✓ Device set to: {device}")
print("="*60 + "\n")

# ============================================================================
# MEMORY MONITORING UTILITIES
# ============================================================================

def check_gpu_memory():
    """Monitor GPU and RAM usage"""
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB, Total: {total:.1f}GB")
    print(f"GPU Utilization: {(allocated/total)*100:.1f}%")
    
    ram = psutil.virtual_memory()
    print(f"System RAM - Used: {ram.used/1024**3:.1f}GB, Available: {ram.available/1024**3:.1f}GB, Total: {ram.total/1024**3:.1f}GB")
    print(f"RAM Usage: {ram.percent:.1f}%")

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ============================================================================
# CHECKPOINT UTILITIES
# ============================================================================

def find_latest_checkpoint(output_dir):
    """Find the most recent checkpoint in output directory"""
    checkpoint_dir = Path(output_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob('checkpoint-*'))
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split('-')[1]))
    return str(latest_checkpoint)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Environment setup for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set GPU memory fraction - FIXED: Reduced to safer value
torch.cuda.set_per_process_memory_fraction(0.85)  # Reduced from 0.90
print("Set GPU memory fraction to 85%\n")

check_gpu_memory()

# ============================================================================
# MODEL SELECTION
# ============================================================================

# Choose your model (uncomment one):
MODEL_NAME = "dmis-lab/biobert-v1.1"  # BioBERT - Good for general biomedical text
# MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"  # PubMedBERT
# MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"  # BiomedBERT

print(f"Using model: {MODEL_NAME}\n")

# ============================================================================
# TRAINING CONFIGURATION (OPTIMIZED FOR CAPELLA GPUs)
# ============================================================================

OUTPUT_DIR = './biobert_results_capella'
NUM_EPOCHS = 15
# Capella GPUs typically have more memory - try larger batch sizes
TRAIN_BATCH_SIZE = 16  # Increased for Capella (adjust if OOM)
EVAL_BATCH_SIZE = 32  # Increased for faster evaluation
GRADIENT_ACCUM_STEPS = 1  # Effective batch = 16 (reduced since batch size increased)
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 256
SAVE_STEPS = 1000  # Save more frequently
EVAL_STEPS = 1000  # Evaluate more frequently
USE_GRADIENT_CHECKPOINTING = False  # Disabled by default, enable only if OOM

# Check for FP16 support
gpu_capability = torch.cuda.get_device_capability()[0]
USE_FP16 = gpu_capability >= 7
print(f"GPU Compute Capability: {gpu_capability}.x")
print(f"FP16 training: {'ENABLED ✓' if USE_FP16 else 'DISABLED (GPU too old)'}")
print(f"Gradient Checkpointing: {'ENABLED' if USE_GRADIENT_CHECKPOINTING else 'DISABLED'}\n")

# ============================================================================
# CHECK FOR EXISTING CHECKPOINT
# ============================================================================

checkpoint_to_resume = find_latest_checkpoint(OUTPUT_DIR)

if checkpoint_to_resume:
    print("="*60)
    print("CHECKPOINT FOUND!")
    print("="*60)
    print(f"Found checkpoint: {checkpoint_to_resume}")
    print("Training will resume from this checkpoint")
    print("="*60 + "\n")
    import torch.serialization
    import numpy
    torch.serialization.add_safe_globals([
        numpy.ndarray, 
        numpy.dtype, 
        numpy.core.multiarray._reconstruct
    ])
    print("✓ Added safe globals for checkpoint compatibility")
else:
    print("="*60)
    print("NO CHECKPOINT FOUND")
    print("="*60)
    print("Starting fresh training from scratch")
    print("="*60 + "\n")


# ============================================================================
# DATA LOADING
# ============================================================================

csv_file = "hcup_processed_medical_dataset_with_labels.csv"
print(f"Loading dataset: {csv_file}")
df = pd.read_csv(csv_file)
print(f"✓ Dataset loaded: {len(df)} samples")
print(f"✓ Number of unique classes: {df['class_label'].nunique()}")

# Display class distribution
print("\nClass distribution:")
class_dist = df['class_label'].value_counts()
print(f"Most common class: {class_dist.index[0]} ({class_dist.iloc[0]} samples)")
print(f"Least common class: {class_dist.index[-1]} ({class_dist.iloc[-1]} samples)\n")

check_gpu_memory()

# ============================================================================
# DATA SPLITTING
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['class_label'], test_size=0.2, random_state=42, stratify=df['class_label']
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
)

print("Data split:")
print(f"  Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Valid: {len(X_valid)} ({len(X_valid)/len(df)*100:.1f}%)")
print(f"  Test:  {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)\n")

# ============================================================================
# LABEL ENCODING
# ============================================================================

label_mapping = {label: idx for idx, label in enumerate(sorted(df['class_label'].unique()))}
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

y_train_numeric = [label_mapping[label] for label in y_train]
y_valid_numeric = [label_mapping[label] for label in y_valid]
y_test_numeric = [label_mapping[label] for label in y_test]

print(f"Number of unique labels: {len(label_mapping)}")

# ============================================================================
# CLASS WEIGHTS COMPUTATION
# ============================================================================

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(list(label_mapping.values())),
    y=y_train_numeric
)
# FIXED: Keep as tensor but don't force to CUDA yet
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print(f"✓ Class weights computed (min: {class_weights.min():.3f}, max: {class_weights.max():.3f})\n")

# Clean up original dataframe
del df
cleanup_memory()

# ============================================================================
# TOKENIZATION (USING HUGGING FACE DATASETS MAP)
# ============================================================================

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✓ Tokenizer loaded\n")

# FIXED: Use HuggingFace datasets.map for efficient tokenization
def tokenize_function(examples):
    """Tokenization function for dataset mapping"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

print("Creating and tokenizing train dataset...")
train_dataset = Dataset.from_dict({
    'text': X_train.tolist(),
    'labels': y_train_numeric
})
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print(f"✓ Train dataset ready: {len(train_dataset)} samples")
check_gpu_memory()

print("Creating and tokenizing validation dataset...")
valid_dataset = Dataset.from_dict({
    'text': X_valid.tolist(),
    'labels': y_valid_numeric
})
valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print(f"✓ Validation dataset ready: {len(valid_dataset)} samples")
check_gpu_memory()

print("Creating and tokenizing test dataset...")
test_dataset = Dataset.from_dict({
    'text': X_test.tolist(),
    'labels': y_test_numeric
})
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
print(f"✓ Test dataset ready: {len(test_dataset)} samples\n")
check_gpu_memory()

cleanup_memory()

# ============================================================================
# CUSTOM TRAINER WITH WEIGHTED LOSS (SIMPLIFIED)
# ============================================================================

class WeightedLossTrainer(Trainer):
    """Custom trainer with class-weighted loss"""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Loss with class weights - will be moved to correct device automatically
        loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_mapping),
    problem_type="single_label_classification"
)

# FIXED: Conditionally enable gradient checkpointing
if USE_GRADIENT_CHECKPOINTING:
    print("Enabling gradient checkpointing for memory efficiency...")
    model.gradient_checkpointing_enable()

# Move model to CUDA
model = model.to(device)

# Verify model is on CUDA
print(f"✓ Model on device: {next(model.parameters()).device}")
assert next(model.parameters()).device.type == 'cuda', "Model is not on CUDA!"

print("✓ Model loaded successfully on CUDA")
check_gpu_memory()
print()

# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(pred):
    """Calculate evaluation metrics"""
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

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
    
    # Learning rate optimization
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    
    # Evaluation and saving - STEP-BASED for better recovery
    evaluation_strategy='steps',
    eval_steps=EVAL_STEPS,
    save_strategy='steps',
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    
    # Logging
    logging_dir=f'{OUTPUT_DIR}/logs',
    logging_steps=50,
    logging_first_step=True,
    report_to="none",
    
    # CUDA settings
    dataloader_pin_memory=True,
    dataloader_num_workers=2,
    fp16=USE_FP16,
    
    # Optimization
    optim="adamw_torch",
    max_grad_norm=1.0,
    
    # Resume settings
    ignore_data_skip=False,
)

# ============================================================================
# CALLBACKS
# ============================================================================

class MemoryMonitorCallback(TrainerCallback):
    """Monitor memory and metrics during training"""
    
    def on_step_end(self, args, state, control, **kwargs):
        # Show progress every 500 steps
        if state.global_step % 500 == 0:
            print(f"\n{'='*60}")
            print(f"Step {state.global_step} | Epoch {state.epoch:.2f}")
            print(f"{'='*60}")
            check_gpu_memory()
    
    def on_save(self, args, state, control, **kwargs):
        print(f"\n✓ Checkpoint saved at step {state.global_step}")
        cleanup_memory()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n{'='*60}")
            print(f"Validation at step {state.global_step} | Epoch {state.epoch:.2f}")
            print(f"{'='*60}")
            for key, value in metrics.items():
                if key.startswith('eval_'):
                    print(f"  {key}: {value:.4f}")
            print(f"{'='*60}")

# ============================================================================
# TRAINER INITIALIZATION
# ============================================================================

print("Initializing trainer...")
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[MemoryMonitorCallback()]
)

print("\n" + "="*60)
print("TRAINING CONFIGURATION")
print("="*60)
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")
print(f"Batch size (per device): {TRAIN_BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUM_STEPS}")
print(f"Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUM_STEPS}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")
print(f"Number of classes: {len(label_mapping)}")
print(f"FP16 training: {USE_FP16}")
print(f"Gradient checkpointing: {USE_GRADIENT_CHECKPOINTING}")
print(f"Save checkpoint every: {SAVE_STEPS} steps")
print(f"Evaluate every: {EVAL_STEPS} steps")
print("="*60 + "\n")

# Calculate expected training time
total_steps = len(train_dataset) // (TRAIN_BATCH_SIZE * GRADIENT_ACCUM_STEPS) * NUM_EPOCHS
print(f"Total training steps: {total_steps}")
print(f"Number of checkpoints: ~{total_steps // SAVE_STEPS}")
print()

# ============================================================================
# TRAINING (WITH AUTO-RESUME)
# ============================================================================

print("="*60)
if checkpoint_to_resume:
    print("RESUMING TRAINING FROM CHECKPOINT")
else:
    print("STARTING FRESH TRAINING")
print("="*60 + "\n")

check_gpu_memory()

try:
    if checkpoint_to_resume:
        trainer.train(resume_from_checkpoint=checkpoint_to_resume)
    else:
        trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")
    
except KeyboardInterrupt:
    print("\n" + "="*60)
    print("TRAINING INTERRUPTED BY USER")
    print("="*60)
    print("You can resume training by running this script again.")
    print("The latest checkpoint will be automatically loaded.")
    raise

except Exception as e:
    print("\n" + "="*60)
    print("TRAINING FAILED WITH ERROR")
    print("="*60)
    print(f"Error: {e}")
    print("\nYou can try resuming by running this script again.")
    raise

# Save final model
print("\nSaving final model...")
trainer.save_model('./biobert_saved_model')
tokenizer.save_pretrained('./biobert_saved_model')
print("✓ Model and tokenizer saved to './biobert_saved_model'")

# ============================================================================
# TEST EVALUATION
# ============================================================================

print("\n" + "="*60)
print("EVALUATING ON TEST SET")
print("="*60 + "\n")

check_gpu_memory()
test_results = trainer.evaluate(eval_dataset=test_dataset)

print("\nTest Results:")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# ============================================================================
# DETAILED PREDICTIONS AND ANALYSIS
# ============================================================================

print("\nGenerating predictions on test set...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60 + "\n")
print(classification_report(
    y_test_numeric, 
    y_pred, 
    target_names=list(label_mapping.keys()),
    digits=4
))

# Per-class accuracy
print("\nPer-class accuracy:")
for label_idx in sorted(label_mapping.values()):
    label_name = reverse_label_mapping[label_idx]
    mask = np.array(y_test_numeric) == label_idx
    if mask.sum() > 0:
        class_acc = (np.array(y_pred)[mask] == label_idx).mean()
        print(f"  {label_name}: {class_acc:.4f} ({mask.sum()} samples)")

# ============================================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================================

print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test_numeric, y_pred)

num_classes = len(label_mapping)
fig_size = min(16, max(10, num_classes * 0.6))
plt.rcParams['figure.figsize'] = [fig_size, fig_size]

max_classes_to_show = 25
if num_classes > max_classes_to_show:
    print(f"Too many classes ({num_classes}), showing top {max_classes_to_show} by frequency")
    class_counts = pd.Series(y_test_numeric).value_counts().head(max_classes_to_show)
    top_indices = class_counts.index
    
    cm_subset = cm[np.ix_(top_indices, top_indices)]
    labels_subset = [reverse_label_mapping[i] for i in top_indices]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_subset, display_labels=labels_subset)
else:
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=list(label_mapping.keys())
    )

disp.plot(cmap='Blues', xticks_rotation=45)
plt.title(f"BioBERT Medical Classification\nTest Accuracy: {test_results['eval_accuracy']:.4f}", 
          fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("biobert_confusion_matrix.png", dpi=150, bbox_inches='tight')
print("✓ Confusion matrix saved to 'biobert_confusion_matrix.png'")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test F1 Score: {test_results['eval_f1']:.4f}")
print(f"Test Precision: {test_results['eval_precision']:.4f}")
print(f"Test Recall: {test_results['eval_recall']:.4f}")
print("="*60)

print("\nFinal memory usage:")
check_gpu_memory()

cleanup_memory()
print("\n✓ All done! 🎉")
print("\nCheckpoints saved in:", OUTPUT_DIR)
print("Final model saved in: ./biobert_saved_model")
