import os
import torch
import pandas as pd
import numpy as np
from transformers import DebertaTokenizer, Trainer, TrainingArguments
from transformers import DebertaForSequenceClassification
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import json
import csv

print("Loading dataset for test split...")
df = pd.read_csv("hcup_processed_medical_dataset_with_labels.csv")

# Recreate the exact same splits (same random state to get same test set)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['class_label'], test_size=0.2, random_state=42  # Add random_state for reproducibility
)

# Label mapping
label_mapping = {label: idx for idx, label in enumerate(df['class_label'].unique())}
y_test_numeric = [label_mapping[label] for label in y_test]

print(f"Test set size: {len(X_test)}")
print(f"Number of classes: {len(label_mapping)}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

# Tokenize test data
print("Tokenizing test data...")
test_encodings = tokenizer(
    X_test.tolist(),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': torch.tensor(y_test_numeric, dtype=torch.long)
})

# Load the saved model
print("Loading saved model from ./saved_model...")
model = DebertaForSequenceClassification.from_pretrained('./saved_model')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Setup trainer for inference
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=4,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)

# Get predictions
print("Generating predictions...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

# === SAFE CLASSIFICATION REPORT ===
os.makedirs("results", exist_ok=True)

y_true = np.asarray(y_test_numeric, dtype=int)
y_pred = np.asarray(y_pred, dtype=int)

# Build id->name mapping
id2name = {v: k for k, v in label_mapping.items()}

# Only use labels that actually appear
present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
target_names = [id2name.get(i, f"CLS_{i}") for i in present_labels]

# Text report
report_text = classification_report(
    y_true, y_pred,
    labels=present_labels,
    target_names=target_names,
    zero_division=0,
    digits=4
)

# Dict form
report_dict = classification_report(
    y_true, y_pred,
    labels=present_labels,
    target_names=target_names,
    zero_division=0,
    output_dict=True
)

# Summary metrics
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

# Save outputs
with open("results/test_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
with open("results/test_report.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# Per-class CSV
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

# Console output
print("\n=== SUMMARY METRICS ===")
for k, v in summary.items():
    print(f"{k}: {v}")
print("\n=== PER-CLASS REPORT (First 50 lines) ===")
print('\n'.join(report_text.split('\n')[:50]))

# Confusion matrix for top 20 classes
top_k = 20
support = {lab: 0 for lab in present_labels}
for lab in y_true:
    support[int(lab)] = support.get(int(lab), 0) + 1

labels_sorted = sorted(present_labels, key=lambda l: support.get(l, 0), reverse=True)
top_labels = labels_sorted[:top_k]

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
    plt.savefig("results/test_confusion_top20dataset.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved confusion matrix: results/test_confusion_top20dataset.png")
else:
    print("No samples among top-K labels to plot confusion matrix.")

print("\n✓ All reports saved in ./results/")
print("  - test_report.txt")
print("  - test_report.json") 
print("  - test_per_class.csv")
