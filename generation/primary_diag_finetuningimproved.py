#!/usr/bin/env python3
"""
Generative Medical Diagnosis: Direct LLaMA Text Generation with Hugging Face
- Modified version with Early Stopping and 10 epochs
- Handles comma-separated primary and secondary diagnoses in target field
- Uses Hugging Face Transformers (NO TRL dependency)
- Direct text generation for medical diagnoses
- Enhanced evaluation for generative outputs
- Standard PyTorch training approach
"""

# ============================ EARLY ENV SETUP ============================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("GPU_ID", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ============================ STANDARD IMPORTS ===========================
import json
import re
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM

# ============================ COMPREHENSIVE EVALUATION IMPORTS ============================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Text generation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# BERT embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    try:
        from transformers import AutoTokenizer, AutoModel
        BERT_TRANSFORMERS_AVAILABLE = True
        SENTENCE_TRANSFORMERS_AVAILABLE = False
    except ImportError:
        BERT_TRANSFORMERS_AVAILABLE = False
        SENTENCE_TRANSFORMERS_AVAILABLE = False

# GPU Memory Monitoring
import subprocess

def log_gpu_memory(context=""):
    """Log current GPU memory usage"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    total_mb, used_mb, free_mb = map(int, lines[0].split(', '))
                    total_gb = total_mb / 1024
                    used_gb = used_mb / 1024
                    free_gb = free_mb / 1024
                    utilization = (used_gb / total_gb) * 100
                else:
                    total_gb = used_gb = free_gb = utilization = 0
            except:
                total_gb = used_gb = free_gb = utilization = 0
            
            print(f"\n=== GPU MEMORY [{context}] ===")
            print(f"PyTorch Allocated: {allocated:.2f} GB")
            print(f"PyTorch Reserved:  {reserved:.2f} GB") 
            print(f"PyTorch Max:       {max_allocated:.2f} GB")
            if total_gb > 0:
                print(f"GPU Total:         {total_gb:.2f} GB")
                print(f"GPU Used:          {used_gb:.2f} GB")
                print(f"GPU Free:          {free_gb:.2f} GB")
                print(f"GPU Utilization:   {utilization:.1f}%")
            print("=" * 40)
            
    except Exception as e:
        print(f"Error logging GPU memory: {e}")

# Medical text processing
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    print("[INFO] SpaCy not available for text processing")
    spacy = None
    nlp = None

# ============================ EARLY STOPPING CALLBACK ============================

class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback that stops training when eval_loss stops improving.
    
    Args:
        early_stopping_patience: Number of evaluations with no improvement after which training stops
        early_stopping_threshold: Minimum change in monitored metric to qualify as improvement
    """
    
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.patience_counter = 0
        
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """Check if we should stop training based on evaluation metrics"""
        metric_value = metrics.get("eval_loss")
        
        if metric_value is None:
            return control
            
        # Initialize best metric on first evaluation
        if self.best_metric is None:
            self.best_metric = metric_value
            print(f"Early Stopping: Initial best eval_loss = {self.best_metric:.4f}")
            return control
        
        # Check if metric improved (lower is better for loss)
        if metric_value < (self.best_metric - self.early_stopping_threshold):
            improvement = self.best_metric - metric_value
            self.best_metric = metric_value
            self.patience_counter = 0
            print(f"Early Stopping: New best eval_loss = {self.best_metric:.4f} (improved by {improvement:.4f})")
        else:
            self.patience_counter += 1
            print(f"Early Stopping: No improvement. Patience {self.patience_counter}/{self.early_stopping_patience}")
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early Stopping: Stopping training after {self.early_stopping_patience} evaluations without improvement!")
                control.should_training_stop = True
        
        return control

# =========================================================================
# Medical Text Preprocessor
# =========================================================================

class MedicalTextProcessor:
    """Processes and cleans medical text for better generation"""
    
    def __init__(self):
        self.medical_stopwords = {
            'patient', 'pt', 'year', 'old', 'y/o', 'yo', 'male', 'female',
            'presents', 'presented', 'presenting', 'c/o', 'complains', 'complaint'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize medical text"""
        if not text:
            return ""
        
        text = text.replace("**", "").replace("|", " ")
        text = re.sub(r'\s+', ' ', text)
        
        if len(text.strip()) < 20:
            return text
        
        text = re.sub(r'\b\d+\s*y/?o\b', 'years old', text, flags=re.IGNORECASE)
        text = re.sub(r'\bc/o\b', 'complains of', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpt\b', 'patient', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_key_features(self, text: str) -> str:
        """Extract key medical features from text"""
        return self.clean_text(text)

# =========================================================================
# Multi-Diagnosis Handler for PS Dataset
# =========================================================================

class MultiDiagnosisHandler:
    """Handles parsing and formatting of comma-separated diagnoses for PS dataset"""
    
    @staticmethod
    def parse_comma_separated_diagnoses(target_string: str) -> Dict[str, str]:
        """Parse comma-separated diagnoses from PS dataset target field"""
        if not target_string or not isinstance(target_string, str):
            return {'primary': '', 'secondary': ''}
        
        diagnoses = [d.strip() for d in target_string.split(',') if d.strip()]
        
        if not diagnoses:
            return {'primary': '', 'secondary': ''}
        
        primary = diagnoses[0].strip()
        secondary = diagnoses[1].strip() if len(diagnoses) > 1 else ''
        
        return {
            'primary': primary,
            'secondary': secondary
        }
    
    @staticmethod
    def format_multi_diagnosis_response(primary: str, secondary: str = '') -> str:
        """Format multi-diagnosis response for training"""
        if not primary:
            return ""
        
        if secondary:
            return f"Primary Diagnosis: {primary}\nSecondary Diagnosis: {secondary}"
        else:
            return f"Primary Diagnosis: {primary}"

# =========================================================================
# Enhanced Prompt Engineering for Multi-Diagnosis Generation
# =========================================================================

def get_llama_chat_template_multi(clinical_text: str, target_diagnoses: str = None) -> str:
    """Build chat template for Llama 3 with multi-diagnosis support"""
    
    sys_prompt = """You are an expert physician specializing in medical diagnosis. 
Given a clinical presentation, provide the primary diagnosis and, if applicable, a secondary diagnosis using standard medical terminology.
Focus on the most likely conditions based on the clinical findings presented.
Format your response as:
Primary Diagnosis: [main condition]
Secondary Diagnosis: [if applicable, secondary condition]

If there is no clear secondary diagnosis, only provide the primary diagnosis."""

    user_prompt = f"""Based on the following clinical presentation, what are the primary and secondary medical diagnoses?

Clinical Presentation:
{clinical_text}

Diagnoses:"""

    if target_diagnoses:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_diagnoses}<|eot_id|>"
    else:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

class GenerativePromptBuilderPS:
    """Builds prompts for generative medical diagnosis with PS dataset format"""
    
    @staticmethod
    def build_training_prompt(clinical_text: str, target_diagnoses: str) -> str:
        """Build training prompt for generative approach with multi-diagnosis"""
        diagnosis_dict = MultiDiagnosisHandler.parse_comma_separated_diagnoses(target_diagnoses)
        
        formatted_response = MultiDiagnosisHandler.format_multi_diagnosis_response(
            diagnosis_dict['primary'], 
            diagnosis_dict['secondary']
        )
        
        return get_llama_chat_template_multi(clinical_text, formatted_response)
    
    @staticmethod
    def build_inference_prompt(clinical_text: str) -> str:
        """Build inference prompt for generation"""
        return get_llama_chat_template_multi(clinical_text)

# =========================================================================
# Generative Training Dataset
# =========================================================================

class GenerativeMedicalDataset(Dataset):
    """Dataset for generative medical diagnosis training"""
    
    def __init__(self, data_items: List[Dict], tokenizer, max_length: int = 2048):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_processor = MedicalTextProcessor()
        
        print(f"Created generative dataset with {len(data_items)} items")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        prompt = item["text"]
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }

# =========================================================================
# Generative Fine-tuner with Early Stopping
# =========================================================================

class GenerativeMedicalFineTuner:
    """Fine-tuner for generative medical diagnosis using standard Hugging Face"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.text_processor = MedicalTextProcessor()
        print("Generative Medical Fine-tuner initialized | device:", self.device)

    def load_base_model(self) -> bool:
        """Load Llama 3 model using Hugging Face Transformers"""
        base_model = self.config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct")
        max_len = self.config.get("max_seq_length", 2048)
        
        print(f"Loading model: {base_model}")
        
        log_gpu_memory("Before Model Loading")
        
        try:
            token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if not token:
                print("HUGGING_FACE_HUB_TOKEN not found in environment variables")
                return False
            
            print("Found Hugging Face token")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
                use_fast=True,
                token=token
            )
            
            log_gpu_memory("After Tokenizer Loading")
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "right"
            self.tokenizer.model_max_length = max_len
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
                use_cache=False,
                token=token
            )
            
            log_gpu_memory("After Model Loading")
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.get("lora_r", 4),
                lora_alpha=self.config.get("lora_alpha", 32),
                lora_dropout=self.config.get("lora_dropout", 0.1),
                target_modules=self.config.get("lora_target_modules", [
                    "q_proj", "k_proj", "v_proj", "o_proj", "up_proj"
                ]),
                bias="none",
            )
            
            self.model = get_peft_model(self.model, peft_config)
            
            log_gpu_memory("After LoRA Application")
            
            trainable_params = 0
            all_param = 0
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            
            print(f"Model ready | trainable: {trainable_params:,} / {all_param:,} "
                f"({100 * trainable_params / all_param:.2f}%)")
            
            return True
            
        except Exception as e:
            print("Model loading failed:", e)
            log_gpu_memory("Model Loading Failed")
            import traceback
            traceback.print_exc()
            return False

    def load_datasets(self, train_file: str, val_file: str):
        """Load and prepare datasets for generative training with PS format"""
        print("Loading datasets for generative training (PS format)...")
        
        log_gpu_memory("Before Dataset Loading")
        
        try:
            with open(train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            with open(val_file, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            
            print(f"Datasets loaded: Train={len(train_data)}, Val={len(val_data)}")
            
            max_train_samples = self.config.get("max_train_samples", None)
            max_val_samples = self.config.get("max_val_samples", None)
            
            if max_train_samples and len(train_data) > max_train_samples:
                print(f"Limiting training data from {len(train_data)} to {max_train_samples} samples")
                train_data = train_data[:max_train_samples]
            
            if max_val_samples and len(val_data) > max_val_samples:
                print(f"Limiting validation data from {len(val_data)} to {max_val_samples} samples")
                val_data = val_data[:max_val_samples]
            
            train_prompts = []
            val_prompts = []
            
            for item in train_data:
                clinical_text = self.text_processor.clean_text(item["clinical_text"])
                target = item["target"]
                prompt = GenerativePromptBuilderPS.build_training_prompt(clinical_text, target)
                train_prompts.append({"text": prompt})
            
            for item in val_data:
                clinical_text = self.text_processor.clean_text(item["clinical_text"])
                target = item["target"]
                prompt = GenerativePromptBuilderPS.build_training_prompt(clinical_text, target)
                val_prompts.append({"text": prompt})
            
            max_len = self.config.get("max_seq_length", 2048)
            train_ds = GenerativeMedicalDataset(train_prompts, self.tokenizer, max_len)
            val_ds = GenerativeMedicalDataset(val_prompts, self.tokenizer, max_len)
            
            log_gpu_memory("After Dataset Creation")
            
            return train_ds, val_ds
            
        except Exception as e:
            print("Dataset loading failed:", e)
            log_gpu_memory("Dataset Loading Failed")
            return None, None

    def fine_tune(self, train_dataset, val_dataset):
        """Fine-tune the model for generative diagnosis with early stopping"""
        print("Starting generative fine-tuning with early stopping...")
        
        log_gpu_memory("Before Training Setup")
        
        out_dir = self.config.get('output_dir', './generative_medical_lora_p163')
        
        from inspect import signature
        run_name = self.config.get('run_name', 'gen-med-finetune-p')

        ta_kwargs = dict(
            output_dir=out_dir,
            num_train_epochs=self.config.get('num_epochs', 10),
            per_device_train_batch_size=self.config.get('batch_size', 1),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 1),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 16),
            learning_rate=self.config.get('learning_rate', 3e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            logging_steps=self.config.get('logging_steps', 10),
            eval_steps=self.config.get('eval_steps', 500),
            save_steps=self.config.get('save_steps', 1000),
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_ratio=0.03,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            lr_scheduler_type="linear",
            max_grad_norm=1.0,
            seed=self.config.get('seed', 42),
            report_to=["wandb"],
            run_name=run_name, 
            save_safetensors=True,
            save_strategy="steps",
            evaluation_strategy="steps" if "evaluation_strategy" in signature(TrainingArguments).parameters else None,
            eval_strategy="steps" if "eval_strategy" in signature(TrainingArguments).parameters else None,
        )
        
        ta_kwargs = {k: v for k, v in ta_kwargs.items() if v is not None}
        args = TrainingArguments(**ta_kwargs)

        # Create early stopping callback
        early_stopping_patience = self.config.get('early_stopping_patience', 3)
        early_stopping_threshold = self.config.get('early_stopping_threshold', 0.0)
        
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold
        )
        
        print(f"Early stopping enabled: patience={early_stopping_patience}, threshold={early_stopping_threshold}")

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[early_stopping]
        )
                
        log_gpu_memory("After Trainer Setup")

        try:
            print("Starting training with early stopping enabled...")
            log_gpu_memory("Training Start")
            
            resume_from = None
            if os.path.exists(out_dir):
                checkpoints = [d for d in os.listdir(out_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(out_dir, d))]
                if checkpoints:
                    checkpoint_nums = []
                    for cp in checkpoints:
                        try:
                            num = int(cp.split('-')[1])
                            checkpoint_nums.append((num, cp))
                        except (IndexError, ValueError):
                            continue
                    
                    if checkpoint_nums:
                        checkpoint_nums.sort(key=lambda x: x[0])
                        latest_checkpoint = checkpoint_nums[-1][1]
                        resume_path = os.path.join(out_dir, latest_checkpoint)
                        
                        # Check trainer files
                        checkpoint_files = ['trainer_state.json', 'training_args.bin']
                        has_trainer = all(os.path.exists(os.path.join(resume_path, f)) for f in checkpoint_files)
                        
                        # Check adapter in any format
                        has_adapter = any([
                            os.path.exists(os.path.join(resume_path, 'adapter_model.safetensors')),
                            os.path.exists(os.path.join(resume_path, 'adapter_model.bin')),
                            os.path.exists(os.path.join(resume_path, 'adapter_config.json'))
                        ])
                        
                        is_valid = has_trainer and has_adapter

                        if is_valid:
                            resume_from = resume_path
                            print(f"Found valid checkpoint to resume from: {resume_from}")
                        else:
                            print(f"Found checkpoint {latest_checkpoint} but it appears incomplete - starting fresh")
                    else:
                        print("Found checkpoint directories but none are valid - starting fresh")
                else:
                    print("No existing checkpoints found - starting fresh training")
            else:
                print("Output directory doesn't exist - starting fresh training")
            
            if resume_from:
                print(f"Resuming training from checkpoint: {resume_from}")
                train_result = trainer.train(resume_from_checkpoint=resume_from)
            else:
                print("Starting fresh training (no valid checkpoints to resume from)")
                train_result = trainer.train()
            
            log_gpu_memory("Training Complete")
            
            # Check if training was stopped early
            total_expected_steps = args.num_train_epochs * len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
            stopped_early = trainer.state.global_step < total_expected_steps
            
            if stopped_early:
                print(f"\nTraining stopped early at step {trainer.state.global_step}")
            else:
                print(f"\nTraining completed all {args.num_train_epochs} epochs")
            
            print(f"Final loss: {float(train_result.training_loss):.4f}")
            
            final_output_dir = os.path.join(out_dir, 'final_model')
            os.makedirs(final_output_dir, exist_ok=True)
            trainer.save_model(final_output_dir)
            self.tokenizer.save_pretrained(final_output_dir)
            
            config_to_save = self.config.copy()
            config_to_save['early_stopping_used'] = True
            config_to_save['early_stopping_patience'] = early_stopping_patience
            config_to_save['early_stopping_threshold'] = early_stopping_threshold
            config_to_save['training_stopped_early'] = stopped_early
            config_to_save['final_global_step'] = trainer.state.global_step
            
            with open(os.path.join(final_output_dir, 'training_config.json'), 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            print(f"Model saved to: {final_output_dir}")
            
            try:
                print("Running evaluation...")
                log_gpu_memory("Evaluation Start")
                
                metrics = trainer.evaluate()
                print(f"Eval metrics: {metrics}")
                
                with open(os.path.join(final_output_dir, 'eval_metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                log_gpu_memory("Evaluation Complete")
                    
            except Exception as e:
                print(f"Evaluation failed: {e}")
                log_gpu_memory("Evaluation Failed")
            
            return trainer
            
        except Exception as e:
            print(f"Training failed: {e}")
            log_gpu_memory("Training Failed")
            import traceback
            traceback.print_exc()
            return None

    def run_full_pipeline(self, train_file: str, val_file: str) -> bool:
        """Run the complete training pipeline"""
        print("GENERATIVE MEDICAL DIAGNOSIS FINE-TUNING (PS Dataset)")
        print("=" * 60)
        
        if not self.load_base_model():
            return False
            
        train_ds, val_ds = self.load_datasets(train_file, val_file)
        if train_ds is None:
            return False
            
        trainer = self.fine_tune(train_ds, val_ds)
        return trainer is not None

# =========================================================================
# Generative Inference for Multi-Diagnosis
# =========================================================================

class GenerativeMedicalInferencePS:
    """Inference engine for generative medical diagnosis with PS format"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.text_processor = MedicalTextProcessor()
        
        config_path = os.path.join(model_path, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def load_finetuned_model(self) -> bool:
        """Load the fine-tuned generative model using AutoPeft"""
        print("Loading fine-tuned generative model from:", self.model_path)
        try:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
                use_auth_token=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_auth_token=True
            )
            
            self.tokenizer.model_max_length = 2048
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            self.model.eval()
            torch.set_grad_enabled(False)
            
            print("Generative model loaded successfully!")
            return True
            
        except Exception as e:
            print("Failed to load generative model:", e)
            import traceback
            traceback.print_exc()
            return False

    def predict(self, clinical_text: str, max_new_tokens: int = 100) -> Tuple[str, Dict]:
        """Generate multi-diagnosis prediction"""
        if not self.model:
            return "", {"error": "Model not loaded"}
            
        cleaned_text = self.text_processor.clean_text(clinical_text)
        prompt = GenerativePromptBuilderPS.build_inference_prompt(cleaned_text)
        
        try:
            inp = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=3700
            )
            inp = {k: v.to(self.model.device) for k, v in inp.items()}
            
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_token_id is None:
                eot_token_id = self.tokenizer.eos_token_id
            
            with torch.no_grad():
                out = self.model.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eot_token_id,
                    repetition_penalty=1.1,
                )
            
            generated_ids = out[0][inp["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
            
            if response.endswith(('.', '!', '?')):
                pass
            else:
                last_sentence_end = max(
                    response.rfind('.'),
                    response.rfind('!'),
                    response.rfind('?')
                )
                if last_sentence_end > len(response) // 2:
                    response = response[:last_sentence_end + 1]
            
            metadata = {
                'cleaned_input': cleaned_text,
                'raw_response': response,
                'prompt_length': len(prompt),
            }
            
            return response, metadata
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "", {'error': str(e)}

# =========================================================================
# Comprehensive Evaluation with BERT, Text Generation Metrics
# =========================================================================

class ComprehensiveMedicalEvaluatorPS:
    """Enhanced evaluator with BERT embeddings, text generation metrics for PS dataset"""
    
    def __init__(self, inference_model):
        self.inference = inference_model
        
        self.bert_model = None
        self.bert_tokenizer = None
        self._init_bert_model()
        
        self.rouge_scorer = None
        self.smoothing_function = None
        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing_function = SmoothingFunction().method1
        
        self.semantic_thresholds = {
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        
        self.top_diagnoses = []
    def _init_bert_model(self):
    try:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.to('cuda')
            print("Loaded SentenceTransformer model for BERT embeddings")
        elif BERT_TRANSFORMERS_AVAILABLE:
            model_name = 'bert-base-uncased'
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.bert_model = self.bert_model.to('cuda')
            print("Loaded BERT model for embeddings")    
        else:
            print("BERT embeddings not available")
        except Exception as e:
            print(f"Failed to load BERT model: {e}")
            self.bert_model = None

    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get BERT embeddings for a list of texts"""
        if not self.bert_model:
            return None
            
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                embeddings = self.bert_model.encode(texts, convert_to_tensor=False)
                return np.array(embeddings)
            elif BERT_TRANSFORMERS_AVAILABLE:
                embeddings = []
                for text in texts:
                    inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        embeddings.append(embedding[0])
                return np.array(embeddings)
        except Exception as e:
            print(f"Error getting BERT embeddings: {e}")
            return None

    def calculate_bert_similarity(self, pred: str, target: str) -> float:
        """Calculate BERT-based semantic similarity"""
        if not self.bert_model or not pred or not target:
            return 0.0
            
        try:
            embeddings = self.get_bert_embeddings([pred, target])
            if embeddings is None or len(embeddings) != 2:
                return 0.0
            
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return max(0.0, float(similarity))
        except Exception as e:
            print(f"Error calculating BERT similarity: {e}")
            return 0.0

    def calculate_text_generation_metrics(self, predicted: str, target: str) -> Dict[str, float]:
        """Calculate BLEU, ROUGE, and other text generation metrics"""
        metrics = {
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
            'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0,
            'length_ratio': 0.0, 'exact_match': 0.0
        }
        
        if not predicted or not target:
            return metrics
        
        metrics['exact_match'] = 1.0 if predicted.lower().strip() == target.lower().strip() else 0.0
        
        pred_len = len(predicted.split())
        target_len = len(target.split())
        if target_len > 0:
            metrics['length_ratio'] = pred_len / target_len
        
        if not NLTK_AVAILABLE:
            return metrics
            
        try:
            pred_tokens = predicted.lower().split()
            target_tokens = target.lower().split()
            
            if len(pred_tokens) > 0 and len(target_tokens) > 0:
                for n in range(1, 5):
                    try:
                        bleu_score = sentence_bleu(
                            [target_tokens], pred_tokens, 
                            weights=tuple([1/n]*n + [0]*(4-n)),
                            smoothing_function=self.smoothing_function
                        )
                        metrics[f'bleu_{n}'] = bleu_score
                    except:
                        metrics[f'bleu_{n}'] = 0.0
            
            if self.rouge_scorer:
                rouge_scores = self.rouge_scorer.score(target, predicted)
                metrics['rouge_1_f'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge_2_f'] = rouge_scores['rouge2'].fmeasure
                metrics['rouge_l_f'] = rouge_scores['rougeL'].fmeasure
                
        except Exception as e:
            print(f"Error calculating text generation metrics: {e}")
            
        return metrics

    def parse_generated_response(self, response: str) -> Dict[str, str]:
        """Parse generated response to extract primary and secondary diagnoses"""
        primary = ""
        secondary = ""
        
        if not response:
            return {'primary': primary, 'secondary': secondary}
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith('primary diagnosis:'):
                primary = line[len('primary diagnosis:'):].strip()
            elif line.lower().startswith('secondary diagnosis:'):
                secondary = line[len('secondary diagnosis:'):].strip()
        
        if not primary and not secondary:
            clean_response = response.strip()
            if ',' in clean_response:
                parts = clean_response.split(',', 1)
                primary = parts[0].strip()
                secondary = parts[1].strip() if len(parts) > 1 else ""
            else:
                primary = clean_response
        
        return {'primary': primary, 'secondary': secondary}

    def extract_top_diagnoses(self, targets: List[str], top_k: int = 20) -> List[str]:
        """Extract top K most common diagnoses from targets for confusion matrix"""
        primary_diagnoses = []
        
        for target in targets:
            target_dict = MultiDiagnosisHandler.parse_comma_separated_diagnoses(target)
            if target_dict['primary']:
                diagnosis = target_dict['primary'].lower().strip()
                diagnosis = re.sub(r'^(acute|chronic|severe|mild)\s+', '', diagnosis)
                primary_diagnoses.append(diagnosis)
        
        diagnosis_counts = Counter(primary_diagnoses)
        top_diagnoses = [diag for diag, _ in diagnosis_counts.most_common(top_k)]
        
        return top_diagnoses

    def create_confusion_matrix_data(self, predictions: List[str], targets: List[str], 
                                   top_diagnoses: List[str]) -> Tuple[List[str], List[str]]:
        """Create confusion matrix data for top diagnoses"""
        pred_labels = []
        true_labels = []
        
        for pred, target in zip(predictions, targets):
            pred_dict = self.parse_generated_response(pred)
            target_dict = MultiDiagnosisHandler.parse_comma_separated_diagnoses(target)
            
            pred_primary = pred_dict['primary'].lower().strip() if pred_dict['primary'] else ""
            target_primary = target_dict['primary'].lower().strip() if target_dict['primary'] else ""
            
            pred_primary = re.sub(r'^(acute|chronic|severe|mild)\s+', '', pred_primary)
            target_primary = re.sub(r'^(acute|chronic|severe|mild)\s+', '', target_primary)
            
            pred_label = pred_primary if pred_primary in top_diagnoses else "Other"
            true_label = target_primary if target_primary in top_diagnoses else "Other"
            
            pred_labels.append(pred_label)
            true_labels.append(true_label)
        
        return pred_labels, true_labels

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            title: str = "Confusion Matrix - Top 20 Diagnoses (PS Dataset)"):
        """Create and save confusion matrix plot"""
        try:
            all_labels = sorted(list(set(y_true + y_pred)))
            
            cm = confusion_matrix(y_true, y_pred, labels=all_labels)
            
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=all_labels, yticklabels=all_labels)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Diagnosis', fontsize=12)
            plt.ylabel('True Diagnosis', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plt.savefig('confusion_matrix_top20.png', dpi=300, bbox_inches='tight')
            plt.savefig('confusion_matrix_top20.pdf', bbox_inches='tight')
            print("Confusion matrix saved as 'confusion_matrix_top20.png' and 'confusion_matrix_top20.pdf'")
            
            return cm, all_labels
            
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            return None, None

    def evaluate_multi_diagnosis_similarity(self, pred_dict: Dict[str, str], target_string: str) -> Dict[str, float]:
        """Calculate similarity scores for multi-diagnosis prediction"""
        target_dict = MultiDiagnosisHandler.parse_comma_separated_diagnoses(target_string)
        
        scores = {
            'primary_exact': 0.0, 'secondary_exact': 0.0,
            'primary_semantic': 0.0, 'secondary_semantic': 0.0,
            'overall_score': 0.0
        }
        
        if pred_dict['primary'] and target_dict['primary']:
            if pred_dict['primary'].lower().strip() == target_dict['primary'].lower().strip():
                scores['primary_exact'] = 1.0
            scores['primary_semantic'] = self.calculate_semantic_similarity(
                pred_dict['primary'], target_dict['primary']
            )
        
        if pred_dict['secondary'] and target_dict['secondary']:
            if pred_dict['secondary'].lower().strip() == target_dict['secondary'].lower().strip():
                scores['secondary_exact'] = 1.0
            scores['secondary_semantic'] = self.calculate_semantic_similarity(
                pred_dict['secondary'], target_dict['secondary']
            )
        elif not pred_dict['secondary'] and not target_dict['secondary']:
            scores['secondary_exact'] = 1.0
            scores['secondary_semantic'] = 1.0
        
        scores['overall_score'] = 0.7 * scores['primary_semantic'] + 0.3 * scores['secondary_semantic']
        
        return scores

    def calculate_semantic_similarity(self, pred: str, target: str) -> float:
        """Calculate semantic similarity"""
        if not pred or not target:
            return 0.0
            
        if pred.lower().strip() == target.lower().strip():
            return 1.0
        
        pred_words = set(pred.lower().split())
        target_words = set(target.lower().split())
        
        if len(pred_words | target_words) == 0:
            return 0.0
        
        jaccard = len(pred_words & target_words) / len(pred_words | target_words)
        
        medical_terms = {'disease', 'syndrome', 'disorder', 'infection', 'cancer', 
                        'failure', 'deficiency', 'injury', 'pain', 'inflammation',
                        'hypertension', 'diabetes', 'pneumonia', 'fracture', 'anemia'}
        
        pred_medical = pred_words & medical_terms
        target_medical = target_words & medical_terms
        key_medical_overlap = 0
        
        if pred_medical and target_medical:
            key_medical_overlap = len(pred_medical & target_medical) / max(len(pred_medical), len(target_medical))
        
        return 0.7 * jaccard + 0.3 * key_medical_overlap

    def comprehensive_evaluation(self, test_file: str, max_samples: int = None) -> Dict:
        """Run comprehensive evaluation with all metrics"""
        if max_samples is None:
            print(f"Starting comprehensive evaluation on FULL dataset...")
        else:
            print(f"Starting comprehensive evaluation on up to {max_samples} samples...")
        
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        if max_samples is not None:
            test_data = test_data[:max_samples]
            print(f"Evaluating on {len(test_data)} samples (limited)")
        else:
            print(f"Evaluating on full dataset: {len(test_data)} samples")
        
        all_predictions = []
        all_targets = []
        all_scores = []
        all_bert_scores = []
        all_text_gen_metrics = []
        failed_predictions = 0
        
        semantic_counters = {
            'primary_high': 0, 'primary_medium': 0, 'primary_low': 0,
            'secondary_high': 0, 'secondary_medium': 0, 'secondary_low': 0,
            'overall_high': 0, 'overall_medium': 0, 'overall_low': 0
        }
        
        print("Processing samples...")
        for i, item in enumerate(test_data, 1):
            clinical_text = item['clinical_text']
            target = item['target']
            
            pred_response, metadata = self.inference.predict(clinical_text, max_new_tokens=100)
            all_predictions.append(pred_response)
            all_targets.append(target)
            
            if not pred_response:
                failed_predictions += 1
                all_scores.append({
                    'primary_exact': 0.0, 'secondary_exact': 0.0,
                    'primary_semantic': 0.0, 'secondary_semantic': 0.0,
                    'overall_score': 0.0
                })
                all_bert_scores.append({'primary_bert': 0.0, 'secondary_bert': 0.0})
                all_text_gen_metrics.append({})
                continue
            
            pred_dict = self.parse_generated_response(pred_response)
            target_dict = MultiDiagnosisHandler.parse_comma_separated_diagnoses(target)
            
            scores = self.evaluate_multi_diagnosis_similarity(pred_dict, target)
            all_scores.append(scores)
            
            bert_scores = {
                'primary_bert': self.calculate_bert_similarity(
                    pred_dict['primary'], target_dict['primary']
                ),
                'secondary_bert': self.calculate_bert_similarity(
                    pred_dict['secondary'], target_dict['secondary']
                ) if pred_dict['secondary'] and target_dict['secondary'] else 0.0
            }
            all_bert_scores.append(bert_scores)
            
            text_gen_metrics = self.calculate_text_generation_metrics(pred_response, target)
            all_text_gen_metrics.append(text_gen_metrics)
            
            self._update_semantic_counters(scores, semantic_counters)
            
            if i % 100 == 0 or i == len(test_data):
                progress = i / len(test_data) * 100
                print(f"Progress: {i}/{len(test_data)} ({progress:.1f}%)")
        
        results = self._calculate_comprehensive_metrics(
            all_scores, all_bert_scores, all_text_gen_metrics, 
            semantic_counters, test_data, failed_predictions
        )
        
        results['predictions'] = all_predictions
        results['targets'] = all_targets
        results['all_scores'] = all_scores
        results['all_bert_scores'] = all_bert_scores
        results['all_text_gen_metrics'] = all_text_gen_metrics
        
        print("Generating confusion matrix for top 20 diagnoses...")
        self.top_diagnoses = self.extract_top_diagnoses(all_targets, top_k=20)
        pred_labels, true_labels = self.create_confusion_matrix_data(
            all_predictions, all_targets, self.top_diagnoses
        )
        
        cm, cm_labels = self.plot_confusion_matrix(true_labels, pred_labels)
        results['confusion_matrix'] = cm.tolist() if cm is not None else None
        results['confusion_matrix_labels'] = cm_labels
        results['top_diagnoses'] = self.top_diagnoses
        
        self._print_comprehensive_results(results)
        self._show_sample_examples(test_data, all_predictions, all_targets, all_scores, all_bert_scores)
        
        return results

    def _update_semantic_counters(self, scores: Dict[str, float], counters: Dict[str, int]):
        """Update semantic similarity counters based on thresholds"""
        if scores['primary_semantic'] >= self.semantic_thresholds['high']:
            counters['primary_high'] += 1
        elif scores['primary_semantic'] >= self.semantic_thresholds['medium']:
            counters['primary_medium'] += 1
        elif scores['primary_semantic'] >= self.semantic_thresholds['low']:
            counters['primary_low'] += 1
        
        if scores['secondary_semantic'] >= self.semantic_thresholds['high']:
            counters['secondary_high'] += 1
        elif scores['secondary_semantic'] >= self.semantic_thresholds['medium']:
            counters['secondary_medium'] += 1
        elif scores['secondary_semantic'] >= self.semantic_thresholds['low']:
            counters['secondary_low'] += 1
        
        if scores['overall_score'] >= self.semantic_thresholds['high']:
            counters['overall_high'] += 1
        elif scores['overall_score'] >= self.semantic_thresholds['medium']:
            counters['overall_medium'] += 1
        elif scores['overall_score'] >= self.semantic_thresholds['low']:
            counters['overall_low'] += 1

    def _calculate_comprehensive_metrics(self, all_scores: List[Dict], all_bert_scores: List[Dict],
                                       all_text_gen_metrics: List[Dict], semantic_counters: Dict[str, int],
                                       test_data: List[Dict], failed_predictions: int) -> Dict:
        """Calculate all comprehensive metrics"""
        n_samples = len(test_data)
        
        primary_exact_matches = sum(1 for s in all_scores if s['primary_exact'] == 1.0)
        secondary_exact_matches = sum(1 for s in all_scores if s['secondary_exact'] == 1.0)
        
        mean_primary_semantic = np.mean([s['primary_semantic'] for s in all_scores])
        mean_secondary_semantic = np.mean([s['secondary_semantic'] for s in all_scores])
        mean_overall_score = np.mean([s['overall_score'] for s in all_scores])
        
        mean_primary_bert = np.mean([s['primary_bert'] for s in all_bert_scores]) if all_bert_scores else 0.0
        mean_secondary_bert = np.mean([s['secondary_bert'] for s in all_bert_scores]) if all_bert_scores else 0.0
        
        if all_text_gen_metrics:
            mean_bleu_1 = np.mean([m.get('bleu_1', 0) for m in all_text_gen_metrics])
            mean_bleu_2 = np.mean([m.get('bleu_2', 0) for m in all_text_gen_metrics])
            mean_bleu_4 = np.mean([m.get('bleu_4', 0) for m in all_text_gen_metrics])
            mean_rouge_1 = np.mean([m.get('rouge_1_f', 0) for m in all_text_gen_metrics])
            mean_rouge_l = np.mean([m.get('rouge_l_f', 0) for m in all_text_gen_metrics])
            mean_length_ratio = np.mean([m.get('length_ratio', 0) for m in all_text_gen_metrics])
            exact_match_rate = np.mean([m.get('exact_match', 0) for m in all_text_gen_metrics])
        else:
            mean_bleu_1 = mean_bleu_2 = mean_bleu_4 = 0.0
            mean_rouge_1 = mean_rouge_l = mean_length_ratio = exact_match_rate = 0.0
        
        semantic_rates = {}
        for key, count in semantic_counters.items():
            semantic_rates[f'{key}_rate'] = count / n_samples
        
        return {
            'primary_exact_accuracy': primary_exact_matches / n_samples,
            'secondary_exact_accuracy': secondary_exact_matches / n_samples,
            'failed_prediction_rate': failed_predictions / n_samples,
            'mean_primary_semantic': mean_primary_semantic,
            'mean_secondary_semantic': mean_secondary_semantic,
            'mean_overall_score': mean_overall_score,
            'mean_primary_bert': mean_primary_bert,
            'mean_secondary_bert': mean_secondary_bert,
            'mean_bleu_1': mean_bleu_1,
            'mean_bleu_2': mean_bleu_2,
            'mean_bleu_4': mean_bleu_4,
            'mean_rouge_1': mean_rouge_1,
            'mean_rouge_l': mean_rouge_l,
            'exact_match_rate': exact_match_rate,
            'mean_length_ratio': mean_length_ratio,
            **semantic_rates,
            'n_samples': n_samples,
            'primary_exact_matches': primary_exact_matches,
            'secondary_exact_matches': secondary_exact_matches,
            'failed_predictions': failed_predictions,
        }

    def _print_comprehensive_results(self, results: Dict):
        """Print comprehensive evaluation results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MEDICAL DIAGNOSIS EVALUATION RESULTS (PS DATASET)")
        print("="*80)
        
        print("\nBASIC ACCURACY METRICS")
        print(f"Primary Exact Match Accuracy:      {results['primary_exact_accuracy']:.1%}")
        print(f"Secondary Exact Match Accuracy:    {results['secondary_exact_accuracy']:.1%}")
        print(f"Failed Prediction Rate:            {results['failed_prediction_rate']:.1%}")
        
        print("\nSEMANTIC SIMILARITY METRICS")
        print(f"Mean Primary Semantic Score:       {results['mean_primary_semantic']:.3f}")
        print(f"Mean Secondary Semantic Score:     {results['mean_secondary_semantic']:.3f}")
        print(f"Mean Overall Semantic Score:       {results['mean_overall_score']:.3f}")
        
        if results['mean_primary_bert'] > 0:
            print("\nBERT SIMILARITY METRICS")
            print(f"Mean Primary BERT Similarity:      {results['mean_primary_bert']:.3f}")
            print(f"Mean Secondary BERT Similarity:    {results['mean_secondary_bert']:.3f}")
        
        if results['mean_bleu_1'] > 0:
            print("\nTEXT GENERATION METRICS")
            print(f"BLEU-1 Score:                     {results['mean_bleu_1']:.3f}")
            print(f"BLEU-2 Score:                     {results['mean_bleu_2']:.3f}")
            print(f"BLEU-4 Score:                     {results['mean_bleu_4']:.3f}")
            print(f"ROUGE-1 F1 Score:                 {results['mean_rouge_1']:.3f}")
            print(f"ROUGE-L F1 Score:                 {results['mean_rouge_l']:.3f}")
            print(f"Exact Match Rate:                 {results['exact_match_rate']:.1%}")
            print(f"Mean Length Ratio:                {results['mean_length_ratio']:.2f}")
        
        print("\nSEMANTIC THRESHOLD ANALYSIS")
        print(f"Primary High Semantic (≥0.7):     {results['primary_high_rate']:.1%}")
        print(f"Primary Medium Semantic (≥0.5):   {results['primary_medium_rate']:.1%}")
        print(f"Secondary High Semantic (≥0.7):   {results['secondary_high_rate']:.1%}")
        print(f"Secondary Medium Semantic (≥0.5):  {results['secondary_medium_rate']:.1%}")
        print(f"Overall High Score (≥0.7):        {results['overall_high_rate']:.1%}")
        print(f"Overall Medium Score (≥0.5):      {results['overall_medium_rate']:.1%}")

    def _show_sample_examples(self, test_data: List[Dict], predictions: List[str], 
                             targets: List[str], all_scores: List[Dict], all_bert_scores: List[Dict]):
        """Show sample examples with all metrics"""
        print(f"\nSAMPLE PREDICTIONS (Top 5):")
        print("="*80)
        
        for i in range(min(5, len(predictions))):
            target_dict = MultiDiagnosisHandler.parse_comma_separated_diagnoses(targets[i])
            pred_dict = self.parse_generated_response(predictions[i])
            
            print(f"\nExample {i+1}:")
            print(f"Clinical Text: {test_data[i]['clinical_text'][:150]}...")
            print(f"\nPredicted Response: {predictions[i]}")
            print(f"  Primary: '{pred_dict['primary']}'")
            print(f"  Secondary: '{pred_dict['secondary']}'")
            print(f"\nTarget: {targets[i]}")
            print(f"  Primary: '{target_dict['primary']}'")
            print(f"  Secondary: '{target_dict['secondary']}'")
            
            print(f"\nScores:")
            scores = all_scores[i]
            bert_scores = all_bert_scores[i] if i < len(all_bert_scores) else {}
            
            print(f"  Semantic - Primary: {scores['primary_semantic']:.3f}, Secondary: {scores['secondary_semantic']:.3f}")
            print(f"  Overall Score: {scores['overall_score']:.3f}")
            if bert_scores:
                print(f"  BERT - Primary: {bert_scores['primary_bert']:.3f}, Secondary: {bert_scores['secondary_bert']:.3f}")
            print("-" * 60)

# =========================================================================
# Main Functions
# =========================================================================

def main_generative_finetune_p():
    """Main function for generative fine-tuning with PS dataset"""
    config = {
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_seq_length": 2048,
        "max_train_samples": None,
        "max_val_samples": None,
        "lora_r": 4,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "up_proj"
        ],
        "num_epochs": 10,
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.0,
        "output_dir": "./generative_medical_lora_p163",
        "logging_steps": 10,
        "eval_steps": 500,
        "save_steps": 1000,
        "seed": 42,
        "run_name": "gen-med-finetune-p-early-stop",
        "train_file": "./medical_dataset_p/train_dataset.json",
        "val_file": "./medical_dataset_p/val_dataset.json",
    }
    
    print("GENERATIVE MEDICAL DIAGNOSIS FINE-TUNING (PS Dataset)")
    print("="*60)
    print(json.dumps(config, indent=2))
    print("="*60)
    
    try:
        ft = GenerativeMedicalFineTuner(config)
        success = ft.run_full_pipeline(config["train_file"], config["val_file"])
        
        if success:
            print("\nGenerative fine-tuning (PS) completed successfully!")
            print(f"Model saved to: {config['output_dir']}/final_model")
        else:
            print("\nGenerative fine-tuning (PS) failed!")
            
        return success
        
    except Exception as e:
        print(f"Fine-tuning error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_generative_inference_p():
    """Main function for comprehensive generative inference and evaluation with PS dataset"""
    model_path = "./generative_medical_lora_p163/final_model"
    test_file = "./medical_dataset_p/test_dataset.json"
    
    print("COMPREHENSIVE MEDICAL INFERENCE & EVALUATION (PS Dataset)")
    print("="*80)
    
    missing_packages = []
    try:
        import nltk
        from rouge_score import rouge_scorer
        print("NLTK and ROUGE available for text generation metrics")
    except ImportError:
        missing_packages.append("nltk rouge-score")
        print("NLTK/ROUGE not available - will skip text generation metrics")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("SentenceTransformers available for BERT embeddings")
    except ImportError:
        try:
            from transformers import AutoModel
            print("Basic BERT available for embeddings")
        except ImportError:
            missing_packages.append("sentence-transformers")
            print("BERT embeddings not available")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("Plotting libraries available")
    except ImportError:
        missing_packages.append("matplotlib seaborn")
        print("Plotting not available - will skip confusion matrix visualization")
    
    if missing_packages:
        print(f"\nTo get full functionality, install: pip install {' '.join(missing_packages)}")
    
    try:
        inference = GenerativeMedicalInferencePS(model_path)
        
        if not inference.load_finetuned_model():
            print("Failed to load generative model (PS)!")
            return False
        
        print("\nTesting single prediction...")
        sample_text = "45 year old male with chest pain and shortness of breath, elevated troponins"
        pred, meta = inference.predict(sample_text)
        print(f"Sample input: {sample_text}")
        print(f"Predicted response: {pred}")
        
        pred_dict = ComprehensiveMedicalEvaluatorPS(inference).parse_generated_response(pred)
        print(f"Parsed Primary: {pred_dict['primary']}")
        print(f"Parsed Secondary: {pred_dict['secondary']}")
        
        print("\nInitializing comprehensive evaluator...")
        evaluator = ComprehensiveMedicalEvaluatorPS(inference)
        
        print("Starting comprehensive evaluation...")
        results = evaluator.comprehensive_evaluation(test_file, max_samples=None)
        
        results_file = "./comprehensive_evaluation_results_p.json"
        
        json_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                json_results[k] = float(v)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                json_results[k] = [
                    {inner_k: float(inner_v) if isinstance(inner_v, (np.floating, np.integer)) else inner_v 
                     for inner_k, inner_v in item.items()} 
                    for item in v
                ]
            else:
                json_results[k] = v
        
        with open(results_file, 'w') as f:
            json.dump({
                'evaluation_type': 'comprehensive_p_medical_diagnosis',
                'model_path': model_path,
                'test_file': test_file,
                'timestamp': pd.Timestamp.now().isoformat(),
                'results': json_results,
                'missing_packages': missing_packages,
                'evaluation_settings': {
                    'max_samples': None,
                    'semantic_thresholds': evaluator.semantic_thresholds,
                    'top_diagnoses_count': 20
                }
            }, f, indent=2)
        
        print(f"\nComprehensive results saved to: {results_file}")
        
        print(f"\nEVALUATION SUMMARY")
        print("="*60)
        print(f"Total samples evaluated: {results['n_samples']}")
        print(f"Failed predictions: {results['failed_predictions']}")
        print(f"Success rate: {(1 - results['failed_prediction_rate']):.1%}")
        
        high_performers = results.get('overall_high_rate', 0)
        medium_performers = results.get('overall_medium_rate', 0) 
        
        print(f"\nPERFORMANCE TIERS")
        print(f"High performance (≥0.7 overall): {high_performers:.1%}")
        print(f"Medium performance (≥0.5 overall): {medium_performers:.1%}")
        
        if results.get('mean_primary_bert', 0) > 0:
            print(f"\nKEY INSIGHTS")
            semantic_vs_bert = results['mean_primary_semantic'] - results['mean_primary_bert']
            if abs(semantic_vs_bert) > 0.05:
                if semantic_vs_bert > 0:
                    print(f"Jaccard similarity outperforms BERT by {semantic_vs_bert:.3f} points")
                else:
                    print(f"BERT embeddings outperform Jaccard similarity by {-semantic_vs_bert:.3f} points")
        
        if results.get('mean_bleu_1', 0) > 0:
            print(f"BLEU scores suggest {'good' if results['mean_bleu_4'] > 0.1 else 'moderate'} text generation quality")
        
        if 'top_diagnoses' in results:
            top_diagnoses_file = "./top_diagnoses_analysis.txt"
            with open(top_diagnoses_file, 'w') as f:
                f.write("TOP 20 DIAGNOSES IN PS TEST SET\n")
                f.write("="*40 + "\n")
                for i, diagnosis in enumerate(results['top_diagnoses'], 1):
                    f.write(f"{i:2d}. {diagnosis}\n")
            print(f"Top diagnoses saved to: {top_diagnoses_file}")
        
        print("\nComprehensive PS evaluation completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Comprehensive evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu_memory():
    """Check GPU memory and provide recommendations"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name} - {memory_gb:.1f} GB")
            
            if memory_gb < 8:
                print(f"GPU {i} has limited memory. Consider using smaller batch sizes.")
            elif memory_gb >= 24:
                print(f"GPU {i} has sufficient memory for full precision training.")
            else:
                print(f"GPU {i} should work well with current configuration.")
    else:
        print("No CUDA-capable GPU detected. CPU training will be very slow.")

def print_model_recommendations():
    """Print model recommendations based on available resources"""
    print("\n" + "="*60)
    print("MODEL RECOMMENDATIONS (PS Dataset)")
    print("="*60)
    print("Available Llama 3 models:")
    print("• meta-llama/Meta-Llama-3-8B-Instruct (recommended for most users)")
    print("• meta-llama/Meta-Llama-3-70B-Instruct (requires multiple GPUs or very high memory)")
    print("• meta-llama/Llama-3.1-8B-Instruct (latest version with improvements)")
    print("• meta-llama/Llama-3.1-70B-Instruct (latest large version)")
    print()
    print("PS Dataset Configuration:")
    print("• Handles comma-separated primary and secondary diagnoses")
    print("• Uses structured output format for training")
    print("• Enhanced evaluation for multi-diagnosis predictions")
    print("• Early stopping enabled (patience=3) to prevent overfitting")
    print("• Training runs for up to 10 epochs (stops early if no improvement)")
    print()
    print("Configuration tips:")
    print("• Reduce batch_size if you encounter OOM errors")
    print("• Increase gradient_accumulation_steps to maintain effective batch size")
    print("• Use dataset size limits to manage memory usage")
    print("• Adjust early_stopping_patience for more/less aggressive stopping")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    print("SYSTEM INFORMATION (PS Dataset Version with Early Stopping)")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("="*60)
    
    check_gpu_memory()
    print_model_recommendations()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "inference":
            success = main_generative_inference_p()
            
        elif command == "train":
            success = main_generative_finetune_p()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  python script.py train     - Run fine-tuning (PS dataset)")
            print("  python script.py inference - Run inference and evaluation (PS dataset)")
            success = False
    else:
        success = main_generative_finetune_p()
    
    print(f"\n{'='*60}")
    print(f"FINAL STATUS: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*60}")
    
    raise SystemExit(0 if success else 1)
