#!/usr/bin/env python3
"""
Generative Medical Diagnosis: Direct LLaMA Text Generation with Hugging Face
- Uses Hugging Face Transformers (NO TRL dependency)
- Direct text generation for medical diagnoses
- Enhanced evaluation for generative outputs
- Standard PyTorch training approach
- Direct testing after training without model reload
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
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM

# GPU Memory Monitoring
import subprocess

def log_gpu_memory(context=""):
    """Log current GPU memory usage"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            # Get nvidia-smi info
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
        
        # Remove special formatting
        text = text.replace("**", "").replace("|", " ")
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Remove very short or very long texts
        if len(text.strip()) < 20:
            return text
        
        # Basic medical text cleaning
        text = re.sub(r'\b\d+\s*y/?o\b', 'years old', text, flags=re.IGNORECASE)
        text = re.sub(r'\bc/o\b', 'complains of', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpt\b', 'patient', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_key_features(self, text: str) -> str:
        """Extract key medical features from text"""
        # For now, return cleaned text
        # Future: Could extract symptoms, medications, etc.
        return self.clean_text(text)

# =========================================================================
# Enhanced Prompt Engineering for Generation
# =========================================================================

def get_llama_chat_template(clinical_text: str, target_diagnosis: str = None) -> str:
    """Build chat template for Llama 3"""
    
    sys_prompt = """You are an expert physician specializing in medical diagnosis. 
Given a clinical presentation, provide a specific primary diagnosis using standard medical terminology.
Focus on the most likely primary condition based on the clinical findings presented.
Provide only the diagnosis name, be specific and use proper medical terminology."""

    user_prompt = f"""Based on the following clinical presentation, what is the primary medical diagnosis?

Clinical Presentation:
{clinical_text}

Primary Diagnosis:"""

    # For training, we provide the target diagnosis
    if target_diagnosis:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_diagnosis}<|eot_id|>"
    else:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

class GenerativePromptBuilder:
    """Builds prompts for generative medical diagnosis"""
    
    @staticmethod
    def build_training_prompt(clinical_text: str, target_diagnosis: str) -> str:
        """Build training prompt for generative approach"""
        return get_llama_chat_template(clinical_text, target_diagnosis)
    
    @staticmethod
    def build_inference_prompt(clinical_text: str) -> str:
        """Build inference prompt for generation"""
        return get_llama_chat_template(clinical_text)

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
        
        # Get the formatted prompt
        prompt = item["text"]
        
        # Tokenize
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
# Direct Testing Class (NEW)
# =========================================================================

class DirectModelTester:
    """Tests the model directly after training without saving/loading"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.text_processor = MedicalTextProcessor()
        self.device = next(model.parameters()).device
    
    def predict(self, clinical_text: str, max_new_tokens: int = 50) -> Tuple[str, Dict]:
        """Generate diagnosis prediction using the trained model"""
        
        # Clean and process text
        cleaned_text = self.text_processor.clean_text(clinical_text)
        
        # Build inference prompt
        prompt = GenerativePromptBuilder.build_inference_prompt(cleaned_text)
        
        try:
            # Tokenize
            inp = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=3800  # Leave room for generation
            )
            inp = {k: v.to(self.device) for k, v in inp.items()}
            
            # Get Llama-3 end-of-turn token
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_token_id is None:
                eot_token_id = self.tokenizer.eos_token_id
            
            # Set model to eval mode
            self.model.eval()
            
            # Generate
            with torch.no_grad():
                out = self.model.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,  # Low temperature for more deterministic output
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eot_token_id,  # Use Llama-3 end-of-turn token
                    repetition_penalty=1.1,
                )
            
            # Decode response
            generated_ids = out[0][inp["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Clean up response - stop at end-of-turn token
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
            
            # Clean up response
            # Remove any trailing incomplete sentences or artifacts
            if response.endswith(('.', '!', '?')):
                pass  # Good ending
            else:
                # Find last complete sentence
                last_sentence_end = max(
                    response.rfind('.'),
                    response.rfind('!'),
                    response.rfind('?')
                )
                if last_sentence_end > len(response) // 2:  # If we found a reasonable endpoint
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
    
    def evaluate_semantic_similarity(self, pred: str, target: str) -> float:
        """Calculate semantic similarity between predicted and target diagnosis"""
        if not pred or not target:
            return 0.0
            
        # Simple exact match
        if pred.lower().strip() == target.lower().strip():
            return 1.0
        
        # Partial matching strategies
        pred_words = set(pred.lower().split())
        target_words = set(target.lower().split())
        
        # Jaccard similarity
        if len(pred_words | target_words) == 0:
            return 0.0
        
        jaccard = len(pred_words & target_words) / len(pred_words | target_words)
        
        # Check if key medical terms match
        key_medical_overlap = 0
        medical_terms = {'disease', 'syndrome', 'disorder', 'infection', 'cancer', 
                        'failure', 'deficiency', 'injury', 'pain', 'inflammation',
                        'hypertension', 'diabetes', 'pneumonia', 'fracture', 'anemia'}
        
        pred_medical = pred_words & medical_terms
        target_medical = target_words & medical_terms
        
        if pred_medical and target_medical:
            key_medical_overlap = len(pred_medical & target_medical) / max(len(pred_medical), len(target_medical))
        
        # Weighted score
        return 0.7 * jaccard + 0.3 * key_medical_overlap
    
    def run_evaluation(self, test_file: str, max_samples: int = 500) -> Dict:
        """Run comprehensive evaluation of the trained model"""
        print(f"Starting direct model evaluation on up to {max_samples} samples...")
        
        # Load test data
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        test_data = test_data[:max_samples]
        print(f"Evaluating on {len(test_data)} samples")
        
        # Track results
        exact_matches = 0
        semantic_scores = []
        predictions = []
        targets = []
        failed_predictions = 0
        
        # Process samples
        for i, item in enumerate(test_data, 1):
            clinical_text = item['clinical_text']
            target = item['target']
            
            # Get prediction
            pred, metadata = self.predict(clinical_text)
            
            predictions.append(pred)
            targets.append(target)
            
            if not pred:
                failed_predictions += 1
                semantic_scores.append(0.0)
                continue
            
            # Calculate scores
            if pred.lower().strip() == target.lower().strip():
                exact_matches += 1
            
            semantic_score = self.evaluate_semantic_similarity(pred, target)
            semantic_scores.append(semantic_score)
            
            # Progress update
            if i % 100 == 0 or i == len(test_data):
                current_exact = exact_matches / i
                current_semantic = np.mean(semantic_scores)
                print(f"Progress {i}/{len(test_data)}: Exact={current_exact:.1%}, Semantic={current_semantic:.3f}")
        
        # Final metrics
        exact_accuracy = exact_matches / len(test_data)
        mean_semantic_score = np.mean(semantic_scores)
        failed_rate = failed_predictions / len(test_data)
        
        # Additional analysis
        high_semantic_matches = sum(1 for s in semantic_scores if s > 0.7)
        high_semantic_rate = high_semantic_matches / len(test_data)
        
        results = {
            'exact_accuracy': exact_accuracy,
            'mean_semantic_score': mean_semantic_score,
            'high_semantic_rate': high_semantic_rate,  # >0.7 semantic similarity
            'failed_prediction_rate': failed_rate,
            'n_samples': len(test_data),
            'exact_matches': exact_matches,
            'failed_predictions': failed_predictions,
            'predictions': predictions,
            'targets': targets,
            'semantic_scores': semantic_scores
        }
        
        print("\n" + "="*60)
        print("DIRECT EVALUATION RESULTS")
        print("="*60)
        print(f"Exact Match Accuracy: {exact_accuracy:.1%}")
        print(f"Mean Semantic Score: {mean_semantic_score:.3f}")
        print(f"High Semantic Match Rate (>0.7): {high_semantic_rate:.1%}")
        print(f"Failed Prediction Rate: {failed_rate:.1%}")
        
        # Show some examples
        print(f"\nExample Predictions:")
        for i in range(min(5, len(predictions))):
            print(f"{i+1}. Input: {test_data[i]['clinical_text'][:100]}...")
            print(f"   Predicted: {predictions[i]}")
            print(f"   Target: {targets[i]}")
            print(f"   Semantic Score: {semantic_scores[i]:.3f}")
            print()
        
        return results

# =========================================================================
# Generative Fine-tuner (MODIFIED)
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
            # Get token from environment variable
            token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if not token:
                print("❌ HUGGING_FACE_HUB_TOKEN not found in environment variables")
                return False
            
            print("✅ Found Hugging Face token")
            
            # Load tokenizer with explicit token
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
                use_fast=True,
                token=token
            )
            
            log_gpu_memory("After Tokenizer Loading")
            
            # Setup tokenizer
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "right"
            self.tokenizer.model_max_length = max_len
            
            # Load model with same token
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
                use_cache=False,  # Disable cache during training
                token=token
            )
            
            log_gpu_memory("After Model Loading")
            
            # LoRA configuration
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
            
            # Apply LoRA
            self.model = get_peft_model(self.model, peft_config)
            
            log_gpu_memory("After LoRA Application")
            
            # Print trainable parameters
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
        """Load and prepare datasets for generative training"""
        print("Loading datasets for generative training...")
        
        log_gpu_memory("Before Dataset Loading")
        
        try:
            with open(train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            with open(val_file, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            
            print(f"Datasets loaded: Train={len(train_data)}, Val={len(val_data)}")
            
            # Limit dataset size for memory management
            max_train_samples = self.config.get("max_train_samples", None)
            max_val_samples = self.config.get("max_val_samples", None)
            
            if max_train_samples and len(train_data) > max_train_samples:
                print(f"Limiting training data from {len(train_data)} to {max_train_samples} samples")
                train_data = train_data[:max_train_samples]
            
            if max_val_samples and len(val_data) > max_val_samples:
                print(f"Limiting validation data from {len(val_data)} to {max_val_samples} samples")
                val_data = val_data[:max_val_samples]
            
            # Prepare data with formatted prompts
            train_prompts = []
            val_prompts = []
            
            for item in train_data:
                clinical_text = self.text_processor.clean_text(item["clinical_text"])
                target = item["target"]
                prompt = GenerativePromptBuilder.build_training_prompt(clinical_text, target)
                train_prompts.append({"text": prompt})
            
            for item in val_data:
                clinical_text = self.text_processor.clean_text(item["clinical_text"])
                target = item["target"]
                prompt = GenerativePromptBuilder.build_training_prompt(clinical_text, target)
                val_prompts.append({"text": prompt})
            
            # Create PyTorch datasets
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
        """Fine-tune the model for generative diagnosis"""
        print("Starting generative fine-tuning...")
        
        log_gpu_memory("Before Training Setup")
        
        out_dir = self.config.get('output_dir', './generative_medical_lora')
        
        # Training arguments
        from inspect import signature
        run_name = self.config.get('run_name', 'gen-med-finetune')

        ta_kwargs = dict(
            output_dir=out_dir,
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 2),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 2),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 8),
            learning_rate=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            logging_steps=self.config.get('logging_steps', 10),
            eval_steps=self.config.get('eval_steps', 200),
            save_steps=self.config.get('save_steps', 400),
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
        
        # Remove None values
        ta_kwargs = {k: v for k, v in ta_kwargs.items() if v is not None}
        args = TrainingArguments(**ta_kwargs)

        # Use standard Trainer (NO TRL)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
                
        log_gpu_memory("After Trainer Setup")

        # Train
        try:
            print("Starting training...")
            log_gpu_memory("Training Start")
            
            # Smart checkpoint resuming logic
            resume_from = None
            if os.path.exists(out_dir):
                # Look for existing checkpoints
                checkpoints = [d for d in os.listdir(out_dir) if d.startswith('checkpoint-') and os.path.isdir(os.path.join(out_dir, d))]
                if checkpoints:
                    # Sort by checkpoint number and get the latest
                    checkpoint_nums = []
                    for cp in checkpoints:
                        try:
                            num = int(cp.split('-')[1])
                            checkpoint_nums.append((num, cp))
                        except (IndexError, ValueError):
                            continue
                    
                    if checkpoint_nums:
                        checkpoint_nums.sort(key=lambda x: x[0])
                        latest_checkpoint = checkpoint_nums[-1][1]  # Get the name of latest checkpoint
                        resume_path = os.path.join(out_dir, latest_checkpoint)
                        
                        # Verify checkpoint is valid
                        checkpoint_files = [
                            'adapter_model.safetensors', 'trainer_state.json', 'training_args.bin'
                        ]
                        is_valid = all(os.path.exists(os.path.join(resume_path, f)) for f in checkpoint_files)

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
            
            # Train with smart resume logic
            if resume_from:
                print(f"Resuming training from checkpoint: {resume_from}")
                train_result = trainer.train(resume_from_checkpoint=resume_from)
            else:
                print("Starting fresh training (no valid checkpoints to resume from)")
                train_result = trainer.train()
            
            log_gpu_memory("Training Complete")
            
            print(f"Training completed! Final loss: {float(train_result.training_loss):.4f}")
            
            # Save final model
            final_output_dir = os.path.join(out_dir, 'final_model')
            os.makedirs(final_output_dir, exist_ok=True)
            trainer.save_model(final_output_dir)
            self.tokenizer.save_pretrained(final_output_dir)
            
            # Save config
            with open(os.path.join(final_output_dir, 'training_config.json'), 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"Model saved to: {final_output_dir}")
            
            # Evaluation
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

    def run_full_pipeline(self, train_file: str, val_file: str, test_file: str = None) -> bool:
        """Run the complete training pipeline with direct testing"""
        print("GENERATIVE MEDICAL DIAGNOSIS FINE-TUNING WITH DIRECT TESTING")
        print("=" * 60)
        
        if not self.load_base_model():
            return False
            
        train_ds, val_ds = self.load_datasets(train_file, val_file)
        if train_ds is None:
            return False
            
        trainer = self.fine_tune(train_ds, val_ds)
        if trainer is None:
            return False
        
        # DIRECT TESTING AFTER TRAINING (NEW SECTION)
        if test_file:
            print("\n" + "="*60)
            print("STARTING DIRECT MODEL TESTING")
            print("="*60)
            
            try:
                # Create direct tester with the trained model
                direct_tester = DirectModelTester(self.model, self.tokenizer)
                
                # Run evaluation on test set
                max_test_samples = self.config.get("max_test_samples", 500)
                test_results = direct_tester.run_evaluation(test_file, max_test_samples)
                
                # Save test results
                out_dir = self.config.get('output_dir', './generative_medical_lora')
                final_output_dir = os.path.join(out_dir, 'final_model')
                
                test_results_file = os.path.join(final_output_dir, 'direct_test_results.json')
                with open(test_results_file, 'w') as f:
                    # Convert numpy arrays to JSON-compatible format
                    json_results = {}
                    for k, v in test_results.items():
                        if isinstance(v, np.ndarray):
                            json_results[k] = v.tolist()
                        elif isinstance(v, (np.floating, np.integer)):
                            json_results[k] = float(v)
                        else:
                            json_results[k] = v
                    json.dump(json_results, f, indent=2)
                
                print(f"\n✅ Direct test results saved to: {test_results_file}")
                
                # Print summary
                print(f"\n📊 DIRECT TESTING SUMMARY")
                print("="*40)
                print(f"Test samples: {test_results['n_samples']}")
                print(f"Exact accuracy: {test_results['exact_accuracy']:.1%}")
                print(f"Mean semantic score: {test_results['mean_semantic_score']:.3f}")
                print(f"High semantic rate: {test_results['high_semantic_rate']:.1%}")
                print(f"Failed predictions: {test_results['failed_predictions']}")
                
                return True
                
            except Exception as e:
                print(f"❌ Direct testing failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print("No test file provided - skipping direct testing")
            return True

# =========================================================================
# Enhanced Evaluation for Generative Output (KEPT FOR REFERENCE)
# =========================================================================

class GenerativeEvaluator:
    """Evaluates generative medical diagnosis outputs"""
    
    def __init__(self, inference_model):
        self.inference = inference_model
        
    def evaluate_semantic_similarity(self, pred: str, target: str) -> float:
        """Calculate semantic similarity between predicted and target diagnosis"""
        if not pred or not target:
            return 0.0
            
        # Simple exact match
        if pred.lower().strip() == target.lower().strip():
            return 1.0
        
        # Partial matching strategies
        pred_words = set(pred.lower().split())
        target_words = set(target.lower().split())
        
        # Jaccard similarity
        if len(pred_words | target_words) == 0:
            return 0.0
        
        jaccard = len(pred_words & target_words) / len(pred_words | target_words)
        
        # Check if key medical terms match
        key_medical_overlap = 0
        medical_terms = {'disease', 'syndrome', 'disorder', 'infection', 'cancer', 
                        'failure', 'deficiency', 'injury', 'pain', 'inflammation'}
        
        pred_medical = pred_words & medical_terms
        target_medical = target_words & medical_terms
        
        if pred_medical and target_medical:
            key_medical_overlap = len(pred_medical & target_medical) / max(len(pred_medical), len(target_medical))
        
        # Weighted score
        return 0.7 * jaccard + 0.3 * key_medical_overlap
    
    def comprehensive_evaluation(self, test_file: str, max_samples: int = 500) -> Dict:
        """Run comprehensive evaluation of generative model"""
        print(f"Starting generative evaluation on up to {max_samples} samples...")
        
        # Load test data
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        test_data = test_data[:max_samples]
        print(f"Evaluating on {len(test_data)} samples")
        
        # Track results
        exact_matches = 0
        semantic_scores = []
        predictions = []
        targets = []
        failed_predictions = 0
        
        # Process samples
        for i, item in enumerate(test_data, 1):
            clinical_text = item['clinical_text']
            target = item['target']
            
            # Get prediction
            pred, metadata = self.inference.predict(clinical_text)
            
            predictions.append(pred)
            targets.append(target)
            
            if not pred:
                failed_predictions += 1
                semantic_scores.append(0.0)
                continue
            
            # Calculate scores
            if pred.lower().strip() == target.lower().strip():
                exact_matches += 1
            
            semantic_score = self.evaluate_semantic_similarity(pred, target)
            semantic_scores.append(semantic_score)
            
            # Progress update
            if i % 100 == 0 or i == len(test_data):
                current_exact = exact_matches / i
                current_semantic = np.mean(semantic_scores)
                print(f"Progress {i}/{len(test_data)}: Exact={current_exact:.1%}, Semantic={current_semantic:.3f}")
        
        # Final metrics
        exact_accuracy = exact_matches / len(test_data)
        mean_semantic_score = np.mean(semantic_scores)
        failed_rate = failed_predictions / len(test_data)
        
        # Additional analysis
        high_semantic_matches = sum(1 for s in semantic_scores if s > 0.7)
        high_semantic_rate = high_semantic_matches / len(test_data)
        
        results = {
            'exact_accuracy': exact_accuracy,
            'mean_semantic_score': mean_semantic_score,
            'high_semantic_rate': high_semantic_rate,  # >0.7 semantic similarity
            'failed_prediction_rate': failed_rate,
            'n_samples': len(test_data),
            'exact_matches': exact_matches,
            'failed_predictions': failed_predictions,
            'predictions': predictions,
            'targets': targets,
            'semantic_scores': semantic_scores
        }
        
        print("\n" + "="*60)
        print("GENERATIVE EVALUATION RESULTS")
        print("="*60)
        print(f"Exact Match Accuracy: {exact_accuracy:.1%}")
        print(f"Mean Semantic Score: {mean_semantic_score:.3f}")
        print(f"High Semantic Match Rate (>0.7): {high_semantic_rate:.1%}")
        print(f"Failed Prediction Rate: {failed_rate:.1%}")
        
        # Show some examples
        print(f"\nExample Predictions:")
        for i in range(min(5, len(predictions))):
            print(f"{i+1}. Input: {test_data[i]['clinical_text'][:100]}...")
            print(f"   Predicted: {predictions[i]}")
            print(f"   Target: {targets[i]}")
            print(f"   Semantic Score: {semantic_scores[i]:.3f}")
            print()
        
        return results

# =========================================================================
# Main Functions
# =========================================================================

def main_generative_finetune():
    """Main function for generative fine-tuning with direct testing"""
    config = {
        # Model configuration
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",  # Hugging Face model
        "max_seq_length": 2048,
        
        # Dataset size limits for memory management
        "max_train_samples": 200000,  # Increased from 10k to 50k samples
        "max_val_samples": 10000,     # Increased from 2k to 25k samples
        "max_test_samples": 5000,     # NEW: Limit test samples for direct testing
        
        # LoRA configuration
        "lora_r": 4,  # Reduced from 16 to 4
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "up_proj"
        ],  # Removed "gate_proj" and "down_proj"
        
        # Training configuration
        "num_epochs": 10,
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        
        # Logging and saving
        "output_dir": "./generative_medical_lora",
        "logging_steps": 10,
        "eval_steps": 500,
        "save_steps": 1000,
        
        # Other settings
        "seed": 42,
        
        # Data files
        "train_file": "./medical_datasets_llama3_improved/train_dataset.json",
        "val_file": "./medical_datasets_llama3_improved/val_dataset.json",
        "test_file": "./medical_datasets_llama3_improved/test_dataset.json",  # NEW: Added test file
    }
    
    print("GENERATIVE MEDICAL DIAGNOSIS FINE-TUNING WITH DIRECT TESTING")
    print("="*80)
    print(json.dumps(config, indent=2))
    print("="*80)
    
    try:
        ft = GenerativeMedicalFineTuner(config)
        success = ft.run_full_pipeline(
            config["train_file"], 
            config["val_file"], 
            config["test_file"]  # Pass test file for direct testing
        )
        
        if success:
            print("\n✅ Generative fine-tuning with direct testing completed successfully!")
            print(f"Model saved to: {config['output_dir']}/final_model")
            print(f"Test results saved to: {config['output_dir']}/final_model/direct_test_results.json")
        else:
            print("\n❌ Generative fine-tuning with direct testing failed!")
            
        return success
        
    except Exception as e:
        print(f"Fine-tuning error: {e}")
        import traceback
        traceback.print_exc()
        return False

# =========================================================================
# Enhanced Comprehensive Evaluation for CSS Dataset (Single Diagnosis)
# =========================================================================

# Additional imports to add at the top of your file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
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

def setup_environment():
    """Setup environment and install required packages"""
    required_packages = [
        "torch",
        "transformers",
        "peft", 
        "accelerate",
        "datasets",
        "wandb",
        "numpy",
        "pandas",
        "scikit-learn"
    ]
    
    import subprocess
    import sys
    
    print("Setting up environment...")
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"⏳ Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")

def check_gpu_memory():
    """Check GPU memory and provide recommendations"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"GPU {i}: {props.name} - {memory_gb:.1f} GB")
            
            if memory_gb < 8:
                print(f"⚠️  GPU {i} has limited memory. Consider using smaller batch sizes.")
            elif memory_gb >= 24:
                print(f"✅ GPU {i} has sufficient memory for full precision training.")
            else:
                print(f"ℹ️  GPU {i} should work well with current configuration.")
    else:
        print("❌ No CUDA-capable GPU detected. CPU training will be very slow.")

def print_model_recommendations():
    """Print model recommendations based on available resources"""
    print("\n" + "="*60)
    print("MODEL RECOMMENDATIONS")
    print("="*60)
    print("Available Llama 3 models:")
    print("• meta-llama/Meta-Llama-3-8B-Instruct (recommended for most users)")
    print("• meta-llama/Meta-Llama-3-70B-Instruct (requires multiple GPUs or very high memory)")
    print("• meta-llama/Llama-3.1-8B-Instruct (latest version with improvements)")
    print("• meta-llama/Llama-3.1-70B-Instruct (latest large version)")
    print()
    print("Configuration tips:")
    print("• Reduce batch_size if you encounter OOM errors")
    print("• Increase gradient_accumulation_steps to maintain effective batch size")
    print("• Use dataset size limits to manage memory usage")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Print system information
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("="*60)
    
    # Check GPU memory
    check_gpu_memory()
    
    # Print model recommendations
    print_model_recommendations()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "setup":
            setup_environment()
            print("\n✅ Environment setup completed!")
            
        elif command == "train":
            success = main_generative_finetune()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  python script.py setup     - Setup environment")
            print("  python script.py train     - Run fine-tuning with direct testing")
            success = False
    else:
        # Default to training with direct testing
        success = main_generative_finetune()
    
    print(f"\n{'='*60}")
    print(f"FINAL STATUS: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*60}")
    
    raise SystemExit(0 if success else 1)
