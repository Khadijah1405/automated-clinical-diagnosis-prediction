import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ImprovedMedicalClassifier:
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_mapping = {}
        self.class_weights = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_and_preprocess_data(self, csv_file, min_samples_per_class=10):
        """Load data with better preprocessing and class filtering"""
        print(f"Loading dataset: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"Original dataset: {len(df)} samples, {df['class_label'].nunique()} classes")
        
        # 1. Filter out classes with too few samples
        class_counts = df['class_label'].value_counts()
        valid_classes = class_counts[class_counts >= min_samples_per_class].index
        df_filtered = df[df['class_label'].isin(valid_classes)]
        
        print(f"After filtering (min {min_samples_per_class} samples/class): {len(df_filtered)} samples, {len(valid_classes)} classes")
        
        # 2. Text preprocessing
        def clean_text(text):
            if pd.isna(text):
                return ""
            # Convert to string and basic cleaning
            text = str(text).strip()
            # Remove excessive whitespace
            text = ' '.join(text.split())
            return text
        
        df_filtered['text'] = df_filtered['text'].apply(clean_text)
        
        # 3. Remove empty texts
        df_filtered = df_filtered[df_filtered['text'].str.len() > 10]
        print(f"After text cleaning: {len(df_filtered)} samples")
        
        return df_filtered
    
    def create_balanced_split(self, df, test_size=0.2, val_size=0.1):
        """Create stratified splits maintaining class distribution"""
        # Use stratified split to maintain class distribution
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(sss.split(df['text'], df['class_label']))
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Further split train into train/val
        if val_size > 0:
            sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=42)
            train_idx2, val_idx = next(sss_val.split(train_df['text'], train_df['class_label']))
            val_df = train_df.iloc[val_idx]
            train_df = train_df.iloc[train_idx2]
            
            print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df
        else:
            print(f"Data split - Train: {len(train_df)}, Test: {len(test_df)}")
            return train_df, None, test_df
    
    def setup_model_and_tokenizer(self, num_labels):
        """Initialize tokenizer and model with better configuration and fallbacks"""
        models_to_try = [self.model_name]
        
        for model_name in models_to_try:
            try:
                print(f"Trying to load {model_name}...")
                
                # Use AutoTokenizer for better compatibility
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True,
                    trust_remote_code=False
                )
                
                # Add padding token if missing
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                
                # Load model with better configuration
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels,
                    problem_type="single_label_classification",
                    hidden_dropout_prob=0.3,  # Increased dropout for regularization
                    attention_probs_dropout_prob=0.3,
                    output_attentions=False,
                    output_hidden_states=False,
                    trust_remote_code=False
                )
                
                # Resize token embeddings if needed
                self.model.resize_token_embeddings(len(self.tokenizer))
                
                # Enable gradient checkpointing for memory efficiency
                #self.model.gradient_checkpointing_enable()
                
                print(f"✓ Successfully loaded {model_name}")
                self.model_name = model_name  # Update to the successful model
                break
                
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {str(e)[:100]}...")
                if model_name == models_to_try[-1]:  # Last model failed
                    raise RuntimeError(f"Failed to load any of the models: {models_to_try}")
                continue
        
        print("✓ Model and tokenizer loaded successfully")
    
    def create_datasets(self, train_df, val_df=None, test_df=None, max_length=512):
        """Create HuggingFace datasets with improved tokenization"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
        
        # Create label mapping
        all_labels = list(train_df['class_label'].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        def create_dataset(df):
            labels = [self.label_mapping[label] for label in df['class_label']]
            dataset = Dataset.from_dict({
                'text': df['text'].tolist(),
                'labels': labels
            })
            return dataset.map(tokenize_function, batched=True, remove_columns=['text'])
        
        datasets = {}
        datasets['train'] = create_dataset(train_df)
        
        if val_df is not None:
            datasets['val'] = create_dataset(val_df)
        
        if test_df is not None:
            datasets['test'] = create_dataset(test_df)
        
        return datasets
    
    def compute_class_weights(self, train_labels):
        """Compute class weights with smoothing"""
        unique_labels = np.unique(train_labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_labels,
            y=train_labels
        )
        
        # Apply smoothing to prevent extreme weights
        max_weight = np.percentile(class_weights, 95)  # Cap at 95th percentile
        class_weights = np.minimum(class_weights, max_weight)
        
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        print(f"Class weights computed - Min: {self.class_weights.min():.3f}, Max: {self.class_weights.max():.3f}")
    
    def setup_training_args(self, output_dir='./medical_v2_training', num_epochs=10):
        """Setup improved training arguments"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,  # Increased from 2
            per_device_eval_batch_size=16,  # Increased from 4
            gradient_accumulation_steps=4,  # Reduced from 8
            learning_rate=2e-5,  # Reduced learning rate
            weight_decay=0.01,
            warmup_ratio=0.1,  # Add warmup
            lr_scheduler_type="cosine",  # Better LR scheduling
            
            # Evaluation and saving
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            
            # Optimization
            fp16=True,  # Enable mixed precision
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            group_by_length=True,  # Group similar lengths for efficiency
            
            # Logging
            logging_dir=f'{output_dir}/training_logs',
            logging_steps=100,
            report_to="none",
            
            # Memory optimization
            dataloader_drop_last=False,
            remove_unused_columns=True,
        )
    
    def compute_metrics(self, eval_pred):
        """Enhanced metrics computation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Weighted metrics for imbalanced data
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Macro metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }
    
    def train(self, train_dataset, val_dataset, training_args):
        """Train the model with improved trainer"""
        from torch.nn import CrossEntropyLoss
        
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Updated compute_loss with **kwargs to handle version differences"""
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get('logits')
                
                if labels is not None and self.class_weights is not None:
                    loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
                    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                else:
                    loss = outputs.loss
                
                return (loss, outputs) if return_outputs else loss
            
            def __init__(self, class_weights=None, **kwargs):
                super().__init__(**kwargs)
                self.class_weights = class_weights
        
        # Create early stopping callback
        try:
            early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
        except TypeError:
            # Fallback for older versions that might use different parameter names
            early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
        
        # Initialize trainer
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            class_weights=self.class_weights,
            callbacks=[early_stopping_callback]
        )
        
        # Train
        print("Starting training...")
        print(f"  - Model: {self.model_name}")
        print(f"  - Output directory: {training_args.output_dir}")
        print(f"  - Number of classes: {self.model.config.num_labels}")
        print(f"  - Training samples: {len(train_dataset)}")
        print(f"  - Validation samples: {len(val_dataset) if val_dataset else 0}")
        
        try:
            trainer.train()
            print("✓ Training completed!")
        except Exception as e:
            print(f"✗ Training failed: {e}")
            # Try training without early stopping if it fails
            print("Retrying without early stopping callback...")
            trainer = WeightedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                class_weights=self.class_weights,
                callbacks=[]  # No callbacks
            )
            trainer.train()
            print("✓ Training completed (without early stopping)!")
        
        # Save best model with V2 naming
        model_save_path = f"{training_args.output_dir}/final_model_v2"
        trainer.save_model(model_save_path)
        trainer.tokenizer.save_pretrained(model_save_path)
        print(f"✓ Model and tokenizer saved to: {model_save_path}")
        
        return trainer
    
    def evaluate_model(self, trainer, test_dataset, save_results=True):
        """Comprehensive model evaluation"""
        print("Evaluating on test set...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        if save_results:
            # Save detailed results
            reverse_mapping = {v: k for k, v in self.label_mapping.items()}
            target_names = [reverse_mapping[i] for i in range(len(self.label_mapping))]
            
            report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
            
            # Save classification report with V2 naming
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv("medical_v2_classification_report.csv")
            print("✓ Saved detailed classification report: medical_v2_classification_report.csv")
            
            # Save prediction results
            results_df = pd.DataFrame({
                'true_label_id': y_true,
                'predicted_label_id': y_pred,
                'true_label_name': [target_names[i] for i in y_true],
                'predicted_label_name': [target_names[i] for i in y_pred],
                'correct': y_true == y_pred
            })
            results_df.to_csv("medical_v2_predictions.csv", index=False)
            print("✓ Saved predictions: medical_v2_predictions.csv")
            
            # Create and save confusion matrix for top classes
            self.plot_confusion_matrix(y_true, y_pred, target_names)
            print("✓ Saved confusion matrix: medical_v2_confusion_matrix.png")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_true
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, target_names, top_k=20):
        """Plot confusion matrix for top-k classes"""
        # Get top k classes by frequency
        class_counts = Counter(y_true)
        top_classes = [cls for cls, _ in class_counts.most_common(top_k)]
        
        # Filter data for top classes
        mask = np.isin(y_true, top_classes)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
        
        # Plot
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[target_names[i][:30] for i in top_classes],
                    yticklabels=[target_names[i][:30] for i in top_classes])
        plt.title(f'Medical DeBERTa V2 - Confusion Matrix (Top {top_k} Classes)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('medical_v2_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved for top {top_k} classes")
        plt.show()

# Usage example
def main():
    print("="*60)
    print("IMPROVED MEDICAL DeBERTa CLASSIFIER V2")
    print("="*60)
    
    # Create separate output directories
    os.makedirs('./medical_v2_training', exist_ok=True)
    os.makedirs('./medical_v2_training/training_logs', exist_ok=True)
    os.makedirs('./medical_v2_training/final_model_v2', exist_ok=True)
    os.makedirs('./cache', exist_ok=True)
    os.makedirs('./datasets_cache', exist_ok=True)
    
    # Set cache directories
    os.environ['HF_HOME'] = './cache'  # Updated from TRANSFORMERS_CACHE
    os.environ['HF_DATASETS_CACHE'] = './datasets_cache'
    
    print(f"✓ Created separate directories for V2 training")
    print(f"✓ Output directory: ./medical_v2_training")
    print(f"✓ Cache directories: ./cache, ./datasets_cache")
    
    # Initialize classifier with fallback models
    classifier = ImprovedMedicalClassifier(
        model_name='microsoft/deberta-v3-base',
       
    )
    
    # Load and preprocess data
    print("\n" + "="*40)
    print("DATA PREPROCESSING")
    print("="*40)
    df = classifier.load_and_preprocess_data('hcup_processed_medical_dataset_with_labels.csv', min_samples_per_class=20)
    
    # Create balanced splits
    train_df, val_df, test_df = classifier.create_balanced_split(df)
    
    print("\n" + "="*40)
    print("MODEL SETUP")
    print("="*40)
    # Setup model
    classifier.setup_model_and_tokenizer(num_labels=len(train_df['class_label'].unique()))
    
    # Create datasets
    print("Creating datasets...")
    datasets = classifier.create_datasets(train_df, val_df, test_df, max_length=512)
    
    # Compute class weights
    train_labels = [classifier.label_mapping[label] for label in train_df['class_label']]
    classifier.compute_class_weights(train_labels)
    
    print("\n" + "="*40)
    print("TRAINING CONFIGURATION")
    print("="*40)
    # Setup training arguments with V2 directories
    training_args = classifier.setup_training_args(
        output_dir='./medical_v2_training', 
        num_epochs=15
    )
    
    print(f"Output directory: {training_args.output_dir}")
    print(f"Logging directory: {training_args.logging_dir}")
    print(f"Number of epochs: {training_args.num_train_epochs}")
    print(f"Batch size (train/eval): {training_args.per_device_train_batch_size}/{training_args.per_device_eval_batch_size}")
    print(f"Mixed precision: {training_args.fp16}")
    
    print("\n" + "="*40)
    print("STARTING TRAINING")
    print("="*40)
    # Train model
    trainer = classifier.train(datasets['train'], datasets['val'], training_args)
    
    print("\n" + "="*40)
    print("MODEL EVALUATION")
    print("="*40)
    # Evaluate
    results = classifier.evaluate_model(trainer, datasets['test'])
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print(f"Model saved to: ./medical_v2_training/final_model_v2/")
    print(f"Classification report: medical_v2_classification_report.csv")
    print(f"Confusion matrix: medical_v2_confusion_matrix.png")
    print(f"Training logs: ./medical_v2_training/training_logs/")
    print("="*60)

if __name__ == "__main__":
    main()
