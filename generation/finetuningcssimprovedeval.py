
#!/usr/bin/env python3
"""
Generative Medical Diagnosis: Direct LLaMA Text Generation with Hugging Face
- Uses Hugging Face Transformers (NO TRL dependency)
- Direct text generation for medical diagnoses
- Enhanced evaluation for generative outputs
- Standard PyTorch training approach
- Simplified Medical Evaluator with Multiple Similarity Metrics
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
from difflib import SequenceMatcher

# Simple imports with fallbacks for similarity calculations
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("Warning: fuzzywuzzy not available. Install with: pip install fuzzywuzzy[speedup]")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

# Text generation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    TEXT_METRICS_AVAILABLE = True
except ImportError:
    TEXT_METRICS_AVAILABLE = False
    print("Warning: NLTK/ROUGE not available. Install with: pip install nltk rouge-score")

# Additional imports for enhanced evaluation
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: plotting libraries not available. Install with: pip install matplotlib seaborn")

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
# Generative Fine-tuner
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

    def run_full_pipeline(self, train_file: str, val_file: str) -> bool:
        """Run the complete training pipeline"""
        print("GENERATIVE MEDICAL DIAGNOSIS FINE-TUNING")
        print("=" * 60)
        
        if not self.load_base_model():
            return False
            
        train_ds, val_ds = self.load_datasets(train_file, val_file)
        if train_ds is None:
            return False
            
        trainer = self.fine_tune(train_ds, val_ds)
        return trainer is not None

# =========================================================================
# Generative Inference
# =========================================================================

class GenerativeMedicalInference:
    """Inference engine for generative medical diagnosis"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.text_processor = MedicalTextProcessor()
        
        # Load config if available
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
            # Load model with PEFT adapters using AutoPeft
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True,
                use_auth_token=True
            )
            
            # Load tokenizer
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

    def predict(self, clinical_text: str, max_new_tokens: int = 50) -> Tuple[str, Dict]:
        """Generate diagnosis prediction"""
        if not self.model:
            return "", {"error": "Model not loaded"}
            
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
            inp = {k: v.to(self.model.device) for k, v in inp.items()}
            
            # Get Llama-3 end-of-turn token
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_token_id is None:
                eot_token_id = self.tokenizer.eos_token_id
            
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

# =========================================================================
# Simplified Medical Evaluator with Multiple Similarity Metrics
# =========================================================================

class SimpleMedicalEvaluator:
    """Clean and simple medical evaluator with multiple similarity metrics"""
    
    def __init__(self, inference_model):
        self.inference = inference_model
        
        # Load BERT models
        self.clinical_bert = None
        self.general_bert = None
        self._load_bert_models()
        
        # Initialize text generation scorers
        self.rouge_scorer = None
        self.smoothing_function = None
        if TEXT_METRICS_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                self.smoothing_function = SmoothingFunction().method1
                print("Text generation metrics (BLEU, ROUGE) available")
            except Exception as e:
                print(f"Warning: Text generation metrics failed to initialize: {e}")
        
        # Different thresholds for different similarity types
        self.jaccard_thresholds = {
            'high': 0.7,    # Jaccard: high overlap
            'medium': 0.4,  # Jaccard: moderate overlap
            'low': 0.2      # Jaccard: some overlap
        }
        
        self.bert_thresholds = {
            'high': 0.85,   # BERT: very similar semantically
            'medium': 0.7,  # BERT: moderately similar
            'low': 0.5      # BERT: somewhat similar
        }
        
        self.general_thresholds = {
            'high': 0.8,    # For character, cosine
            'medium': 0.6,
            'low': 0.4
        }
        
        self.text_gen_thresholds = {
            'high': 0.7,    # For BLEU, ROUGE scores
            'medium': 0.5,
            'low': 0.3
        }
    
    def _load_bert_models(self):
        """Load both BERT models with correct approach"""
        
        # Load Clinical BERT using transformers (not sentence-transformers)
        if TRANSFORMERS_AVAILABLE:
            try:
                from transformers import AutoTokenizer, AutoModel
                self.clinical_bert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                self.clinical_bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                print("✅ Loaded Clinical BERT (Bio_ClinicalBERT)")
            except Exception as e:
                print(f"⚠️ Failed to load Clinical BERT: {e}")
                self.clinical_bert_tokenizer = None
                self.clinical_bert_model = None
        else:
            print("⚠️ Transformers not available - Clinical BERT disabled")
            self.clinical_bert_tokenizer = None
            self.clinical_bert_model = None
        
        # Load General BERT using sentence-transformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.general_bert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("✅ Loaded General BERT (all-MiniLM-L6-v2)") 
            except Exception as e:
                print(f"⚠️ Failed to load General BERT: {e}")
                self.general_bert = None
        else:
            print("⚠️ SentenceTransformers not available - General BERT disabled")
            self.general_bert = None
    
    def word_similarity(self, pred: str, target: str) -> float:
        """Word-to-word similarity using Jaccard index"""
        if not pred or not target:
            return 0.0
        
        # Simple normalization - just lowercase
        pred_words = set(pred.lower().split())
        target_words = set(target.lower().split())
        
        if not pred_words or not target_words:
            return 0.0
        
        intersection = len(pred_words & target_words)
        union = len(pred_words | target_words)
        
        return intersection / union if union > 0 else 0.0
    
    def character_similarity(self, pred: str, target: str) -> float:
        """Character-to-character similarity"""
        if not pred or not target:
            return 0.0
        
        # Simple normalization
        pred_clean = pred.lower().strip()
        target_clean = target.lower().strip()
        
        if pred_clean == target_clean:
            return 1.0
        
        # Use SequenceMatcher for character-level similarity
        return SequenceMatcher(None, pred_clean, target_clean).ratio()
    
    def jaccard_index(self, pred: str, target: str) -> float:
        """Traditional Jaccard index calculation"""
        if not pred or not target:
            return 0.0
        
        # Character-level Jaccard
        pred_chars = set(pred.lower().replace(' ', ''))
        target_chars = set(target.lower().replace(' ', ''))
        
        if not pred_chars or not target_chars:
            return 0.0
        
        intersection = len(pred_chars & target_chars)
        union = len(pred_chars | target_chars)
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_text_generation_metrics(self, pred: str, target: str) -> Dict[str, float]:
        """Calculate BLEU, ROUGE, and other text generation metrics"""
        metrics = {
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
            'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0,
            'length_ratio': 0.0, 'exact_match': 0.0
        }
        
        if not pred or not target:
            return metrics
        
        # Exact match
        metrics['exact_match'] = 1.0 if pred.lower().strip() == target.lower().strip() else 0.0
        
        # Length ratio
        pred_len = len(pred.split())
        target_len = len(target.split())
        if target_len > 0:
            metrics['length_ratio'] = pred_len / target_len
        
        if not TEXT_METRICS_AVAILABLE or not self.rouge_scorer:
            return metrics
            
        try:
            # BLEU scores
            pred_tokens = pred.lower().split()
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
            
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(target, pred)
            metrics['rouge_1_f'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge_2_f'] = rouge_scores['rouge2'].fmeasure
            metrics['rouge_l_f'] = rouge_scores['rougeL'].fmeasure
                
        except Exception as e:
            print(f"Error calculating text generation metrics: {e}")
            
        return metrics
    
    def cosine_similarity_simple(self, pred: str, target: str) -> float:
        """Simple cosine similarity using character frequency"""
        if not pred or not target:
            return 0.0
        
        # Create character frequency vectors
        all_chars = set(pred.lower() + target.lower())
        
        pred_vec = [pred.lower().count(char) for char in all_chars]
        target_vec = [target.lower().count(char) for char in all_chars]
        
        if SKLEARN_AVAILABLE:
            similarity = cosine_similarity([pred_vec], [target_vec])[0][0]
            return max(0.0, float(similarity))
        else:
            # Manual cosine similarity calculation
            dot_product = sum(a * b for a, b in zip(pred_vec, target_vec))
            norm_pred = sum(a * a for a in pred_vec) ** 0.5
            norm_target = sum(b * b for b in target_vec) ** 0.5
            
            if norm_pred == 0 or norm_target == 0:
                return 0.0
            
            return dot_product / (norm_pred * norm_target)
    
    def clinical_bert_similarity(self, pred: str, target: str) -> float:
        """Clinical BERT similarity using transformers directly"""
        if not self.clinical_bert_model or not self.clinical_bert_tokenizer or not pred or not target:
            return 0.0
        
        try:
            # Tokenize both texts
            pred_inputs = self.clinical_bert_tokenizer(pred, return_tensors='pt', truncation=True, max_length=512, padding=True)
            target_inputs = self.clinical_bert_tokenizer(target, return_tensors='pt', truncation=True, max_length=512, padding=True)
            
            # Get embeddings
            with torch.no_grad():
                pred_outputs = self.clinical_bert_model(**pred_inputs)
                target_outputs = self.clinical_bert_model(**target_inputs)
                
                # Use CLS token embeddings (mean pooling alternative)
                pred_embedding = pred_outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token
                target_embedding = target_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Calculate cosine similarity
                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity(pred_embedding, target_embedding)[0][0]
                    return max(0.0, float(similarity))
                else:
                    # Manual cosine similarity
                    pred_flat = pred_embedding.flatten()
                    target_flat = target_embedding.flatten()
                    
                    dot_product = np.dot(pred_flat, target_flat)
                    norm_pred = np.linalg.norm(pred_flat)
                    norm_target = np.linalg.norm(target_flat)
                    
                    if norm_pred == 0 or norm_target == 0:
                        return 0.0
                    
                    similarity = dot_product / (norm_pred * norm_target)
                    return max(0.0, float(similarity))
                    
        except Exception as e:
            print(f"Clinical BERT error: {e}")
        
        return 0.0
    
    def general_bert_similarity(self, pred: str, target: str) -> float:
        """General BERT similarity"""
        if not self.general_bert or not pred or not target:
            return 0.0
        
        try:
            embeddings = self.general_bert.encode([pred, target])
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return max(0.0, float(similarity))
        except Exception as e:
            print(f"General BERT error: {e}")
        
        return 0.0
    
    def calculate_all_similarities(self, pred: str, target: str) -> Dict[str, float]:
        """Calculate all similarity metrics"""
        similarities = {
            'word_similarity': self.word_similarity(pred, target),
            'character_similarity': self.character_similarity(pred, target),
            'jaccard_index': self.jaccard_index(pred, target),
            'cosine_similarity': self.cosine_similarity_simple(pred, target),
            'clinical_bert': self.clinical_bert_similarity(pred, target),
            'general_bert': self.general_bert_similarity(pred, target)
        }
        
        # Add text generation metrics
        text_gen_metrics = self.calculate_text_generation_metrics(pred, target)
        similarities.update(text_gen_metrics)
        
        return similarities
    
    def calculate_threshold_stats(self, scores: List[float], metric_type: str = 'general') -> Dict[str, any]:
        """Calculate threshold statistics with appropriate thresholds for different metrics"""
        n_total = len(scores)
        if n_total == 0:
            return {}
        
        # Choose appropriate thresholds based on metric type
        if metric_type == 'jaccard':
            thresholds = self.jaccard_thresholds
            threshold_desc = "Jaccard (≥0.7, ≥0.4, ≥0.2)"
        elif metric_type == 'bert':
            thresholds = self.bert_thresholds  
            threshold_desc = "BERT (≥0.85, ≥0.7, ≥0.5)"
        elif metric_type == 'text_gen':
            thresholds = self.text_gen_thresholds
            threshold_desc = "Text Gen (≥0.7, ≥0.5, ≥0.3)"
        else:
            thresholds = self.general_thresholds
            threshold_desc = "General (≥0.8, ≥0.6, ≥0.4)"
        
        # Count samples in each category - 4 categories
        high_count = sum(1 for s in scores if s >= thresholds['high'])
        medium_count = sum(1 for s in scores if thresholds['medium'] <= s < thresholds['high'])
        low_count = sum(1 for s in scores if thresholds['low'] <= s < thresholds['medium'])
        very_low_count = sum(1 for s in scores if s < thresholds['low'])
        
        return {
            'threshold_description': threshold_desc,
            'high_count': high_count,
            'high_percentage': (high_count / n_total) * 100,
            'medium_count': medium_count, 
            'medium_percentage': (medium_count / n_total) * 100,
            'low_count': low_count,
            'low_percentage': (low_count / n_total) * 100,
            'very_low_count': very_low_count,
            'very_low_percentage': (very_low_count / n_total) * 100,
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores)
        }
    
    def extract_top_diagnoses(self, targets: List[str], top_k: int = 20) -> List[str]:
        """Extract top K most common diagnoses from targets for confusion matrix"""
        diagnoses = []
        
        for target in targets:
            if target:
                # Clean and normalize diagnosis
                diagnosis = target.lower().strip()
                # Remove common prefixes/suffixes for better grouping
                diagnosis = re.sub(r'^(acute|chronic|severe|mild)\s+', '', diagnosis)
                diagnoses.append(diagnosis)
        
        # Get top K most common
        diagnosis_counts = Counter(diagnoses)
        top_diagnoses = [diag for diag, _ in diagnosis_counts.most_common(top_k)]
        
        return top_diagnoses

    def create_confusion_matrix_data(self, predictions: List[str], targets: List[str], 
                                   top_diagnoses: List[str]) -> Tuple[List[str], List[str]]:
        """Create confusion matrix data for top diagnoses"""
        pred_labels = []
        true_labels = []
        
        for pred, target in zip(predictions, targets):
            # Normalize for comparison
            pred_clean = pred.lower().strip() if pred else ""
            target_clean = target.lower().strip() if target else ""
            
            # Remove common prefixes for better matching
            pred_clean = re.sub(r'^(acute|chronic|severe|mild)\s+', '', pred_clean)
            target_clean = re.sub(r'^(acute|chronic|severe|mild)\s+', '', target_clean)
            
            # Map to top diagnoses or "Other"
            pred_label = pred_clean if pred_clean in top_diagnoses else "Other"
            true_label = target_clean if target_clean in top_diagnoses else "Other"
            
            pred_labels.append(pred_label)
            true_labels.append(true_label)
        
        return pred_labels, true_labels

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            title: str = "Confusion Matrix - Top 20 Diagnoses"):
        """Create and save confusion matrix plot"""
        if not PLOTTING_AVAILABLE:
            print("Plotting not available - skipping confusion matrix")
            return None, None
            
        try:
            # Get unique labels
            all_labels = sorted(list(set(y_true + y_pred)))
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=all_labels)
            
            # Create plot
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=all_labels, yticklabels=all_labels)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Diagnosis', fontsize=12)
            plt.ylabel('True Diagnosis', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save plot
            plt.savefig('confusion_matrix_simple_top20.png', dpi=300, bbox_inches='tight')
            plt.savefig('confusion_matrix_simple_top20.pdf', bbox_inches='tight')
            print("Confusion matrix saved as 'confusion_matrix_simple_top20.png' and '.pdf'")
            
            return cm, all_labels
            
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            return None, None
    
    def evaluate(self, test_file: str, max_samples: int = None) -> Dict:
        """Main evaluation function"""
        print(f"Starting evaluation with simplified medical similarity...")
        
        # Load test data
        with open(test_file, "r") as f:
            test_data = json.load(f)
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        print(f"Evaluating {len(test_data)} samples")
        
        # Store all results
        all_predictions = []
        all_targets = []
        all_similarities = {
            'word_similarity': [],
            'character_similarity': [],
            'jaccard_index': [],
            'cosine_similarity': [],
            'clinical_bert': [],
            'general_bert': [],
            'bleu_1': [],
            'bleu_2': [],
            'bleu_3': [],
            'bleu_4': [],
            'rouge_1_f': [],
            'rouge_2_f': [],
            'rouge_l_f': [],
            'length_ratio': [],
            'exact_match': []
        }
        
        exact_matches = 0
        failed_predictions = 0
        
        # Process each sample
        for i, item in enumerate(test_data, 1):
            clinical_text = item['clinical_text']
            target = item['target']
            
            # Get prediction
            pred_response, _ = self.inference.predict(clinical_text, max_new_tokens=50)
            
            all_predictions.append(pred_response)
            all_targets.append(target)
            
            if not pred_response:
                failed_predictions += 1
                # Add zeros for failed predictions
                for metric in all_similarities:
                    all_similarities[metric].append(0.0)
                continue
            
            # Check exact match
            if pred_response.lower().strip() == target.lower().strip():
                exact_matches += 1
            
            # Calculate all similarities
            similarities = self.calculate_all_similarities(pred_response, target)
            
            for metric, score in similarities.items():
                all_similarities[metric].append(score)
            
            # Progress update
            if i % 100 == 0 or i == len(test_data):
                print(f"Progress: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
        
        # Calculate statistics for each metric
        results = {
            'basic_stats': {
                'total_samples': len(test_data),
                'exact_matches': exact_matches,
                'exact_accuracy': exact_matches / len(test_data),
                'failed_predictions': failed_predictions,
                'failed_rate': failed_predictions / len(test_data)
            }
        }
        
        # Add threshold statistics for each similarity metric with appropriate thresholds
        metric_threshold_map = {
            'word_similarity': 'jaccard',  # Word overlap is similar to Jaccard
            'character_similarity': 'general',
            'jaccard_index': 'jaccard',    # Classic Jaccard index
            'cosine_similarity': 'general',
            'clinical_bert': 'bert',
            'general_bert': 'bert',
            'bleu_1': 'text_gen',
            'bleu_2': 'text_gen', 
            'bleu_3': 'text_gen',
            'bleu_4': 'text_gen',
            'rouge_1_f': 'text_gen',
            'rouge_2_f': 'text_gen',
            'rouge_l_f': 'text_gen',
            'length_ratio': 'general',
            'exact_match': 'general'
        }
        
        for metric_name, scores in all_similarities.items():
            threshold_type = metric_threshold_map.get(metric_name, 'general')
            results[f'{metric_name}_stats'] = self.calculate_threshold_stats(scores, threshold_type)
        
        # Store raw data
        results['raw_data'] = {
            'predictions': all_predictions,
            'targets': all_targets,
            'similarities': all_similarities
        }
        
        # Generate confusion matrix
        print("Generating confusion matrix for top 20 diagnoses...")
        top_diagnoses = self.extract_top_diagnoses(all_targets, top_k=20)
        pred_labels, true_labels = self.create_confusion_matrix_data(all_predictions, all_targets, top_diagnoses)
        cm, cm_labels = self.plot_confusion_matrix(true_labels, pred_labels)
        
        results['confusion_matrix'] = cm.tolist() if cm is not None else None
        results['confusion_matrix_labels'] = cm_labels
        results['top_diagnoses'] = top_diagnoses
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print evaluation results in a clear format"""
        print("\n" + "="*80)
        print("SIMPLE MEDICAL EVALUATION RESULTS")
        print("="*80)
        
        basic = results['basic_stats']
        print(f"\nBASIC METRICS:")
        print(f"Total Samples: {basic['total_samples']}")
        print(f"Exact Matches: {basic['exact_matches']} ({basic['exact_accuracy']:.1%})")
        print(f"Failed Predictions: {basic['failed_predictions']} ({basic['failed_rate']:.1%})")
        
        # Print each similarity metric with appropriate threshold descriptions
        metric_info = {
            'word_similarity': ('WORD-TO-WORD SIMILARITY (Jaccard)', 'jaccard'),
            'character_similarity': ('CHARACTER-TO-CHARACTER SIMILARITY', 'general'),
            'jaccard_index': ('JACCARD INDEX (Character Level)', 'jaccard'),
            'cosine_similarity': ('COSINE SIMILARITY', 'general'),
            'clinical_bert': ('CLINICAL BERT SIMILARITY', 'bert'),
            'general_bert': ('GENERAL BERT SIMILARITY', 'bert'),
            'bleu_1': ('BLEU-1 SCORE', 'text_gen'),
            'bleu_2': ('BLEU-2 SCORE', 'text_gen'),
            'bleu_4': ('BLEU-4 SCORE', 'text_gen'),
            'rouge_1_f': ('ROUGE-1 F1 SCORE', 'text_gen'),
            'rouge_l_f': ('ROUGE-L F1 SCORE', 'text_gen'),
            'length_ratio': ('LENGTH RATIO', 'general'),
            'exact_match': ('EXACT MATCH RATE', 'general')
        }
        
        for metric, (display_name, threshold_type) in metric_info.items():
            stats = results.get(f'{metric}_stats', {})
            if not stats:
                continue
                
            print(f"\n{display_name}:")
            print(f"  Thresholds: {stats['threshold_description']}")
            print(f"  Mean Score: {stats['mean_score']:.3f} (Range: {stats['min_score']:.3f} - {stats['max_score']:.3f})")
            print(f"  High: {stats['high_count']} samples ({stats['high_percentage']:.1f}%)")
            print(f"  Medium: {stats['medium_count']} samples ({stats['medium_percentage']:.1f}%)")
            print(f"  Low: {stats['low_count']} samples ({stats['low_percentage']:.1f}%)")
            print(f"  Very Low: {stats['very_low_count']} samples ({stats['very_low_percentage']:.1f}%)")
        
        # Compare BERT models
        if results.get('clinical_bert_stats') and results.get('general_bert_stats'):
            clinical_mean = results['clinical_bert_stats']['mean_score']
            general_mean = results['general_bert_stats']['mean_score']
            
            print(f"\nBERT MODEL COMPARISON:")
            print(f"Clinical BERT Mean: {clinical_mean:.3f}")
            print(f"General BERT Mean: {general_mean:.3f}")
            
            if clinical_mean > general_mean:
                diff = clinical_mean - general_mean
                improvement = (diff / general_mean) * 100 if general_mean > 0 else 0
                print(f"Clinical BERT is better by {diff:.3f} points ({improvement:.1f}% improvement)")
            elif general_mean > clinical_mean:
                diff = general_mean - clinical_mean
                improvement = (diff / clinical_mean) * 100 if clinical_mean > 0 else 0
                print(f"General BERT is better by {diff:.3f} points ({improvement:.1f}% improvement)")
            else:
                print("Both BERT models perform similarly")
        
        # Show some examples
        print(f"\nSAMPLE PREDICTIONS (first 3):")
        predictions = results['raw_data']['predictions']
        targets = results['raw_data']['targets']
        similarities = results['raw_data']['similarities']
        
        for i in range(min(3, len(predictions))):
            print(f"\nExample {i+1}:")
            print(f"  Predicted: {predictions[i]}")
            print(f"  Target: {targets[i]}")
            print(f"  Word Sim: {similarities['word_similarity'][i]:.3f}")
            print(f"  Char Sim: {similarities['character_similarity'][i]:.3f}")
            print(f"  Jaccard: {similarities['jaccard_index'][i]:.3f}")
            print(f"  Cosine: {similarities['cosine_similarity'][i]:.3f}")
            if similarities['clinical_bert'][i] > 0:
                print(f"  Clinical BERT: {similarities['clinical_bert'][i]:.3f}")
            if similarities['general_bert'][i] > 0:
                print(f"  General BERT: {similarities['general_bert'][i]:.3f}")
            if TEXT_METRICS_AVAILABLE:
                print(f"  BLEU-1: {similarities['bleu_1'][i]:.3f}")
                print(f"  BLEU-4: {similarities['bleu_4'][i]:.3f}")
                print(f"  ROUGE-1: {similarities['rouge_1_f'][i]:.3f}")
                print(f"  ROUGE-L: {similarities['rouge_l_f'][i]:.3f}")

# =========================================================================
# Main Functions
# =========================================================================

def main_generative_finetune():
    """Main function for generative fine-tuning"""
    config = {
        # Model configuration
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",  # Hugging Face model
        "max_seq_length": 2048,
        
        # Dataset size limits for memory management
        "max_train_samples": 100000,  # Increased from 10k to 50k samples
        "max_val_samples": 10000,     # Increased from 2k to 25k samples
        
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
    }
    
    print("GENERATIVE MEDICAL DIAGNOSIS FINE-TUNING")
    print("="*60)
    print(json.dumps(config, indent=2))
    print("="*60)
    
    try:
        ft = GenerativeMedicalFineTuner(config)
        success = ft.run_full_pipeline(config["train_file"], config["val_file"])
        
        if success:
            print("\n✅ Generative fine-tuning completed successfully!")
            print(f"Model saved to: {config['output_dir']}/final_model")
        else:
            print("\n❌ Generative fine-tuning failed!")
            
        return success
        
    except Exception as e:
        print(f"Fine-tuning error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_simple_evaluation():
    """Main function using the simple evaluator"""
    model_path = "./generative_medical_lora/final_model"
    test_file = "./medical_datasets_llama3_improved/test_dataset.json"
    
    print("SIMPLE MEDICAL EVALUATION")
    print("="*50)
    
    # Check required packages
    missing_packages = []
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing_packages.append("sentence-transformers")
    if not SKLEARN_AVAILABLE:
        missing_packages.append("scikit-learn")
    if not PLOTTING_AVAILABLE:
        missing_packages.append("matplotlib seaborn")
    if not TEXT_METRICS_AVAILABLE:
        missing_packages.append("nltk rouge-score")
    # Remove the FuzzyWuzzy check since it's not used in the evaluator
    
    if missing_packages:
        print(f"\nWarning: Missing packages for full functionality:")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        print("Continuing with available features...\n")
    
    try:
        # Initialize inference
        inference = GenerativeMedicalInference(model_path)
        
        if not inference.load_finetuned_model():
            print("Failed to load model!")
            return False
        
        # Test single prediction
        print("Testing single prediction...")
        sample_text = "45 year old male with chest pain and shortness of breath, elevated troponins"
        pred, _ = inference.predict(sample_text)
        print(f"Sample input: {sample_text}")
        print(f"Predicted diagnosis: {pred}")
        
        # Initialize simple evaluator
        evaluator = SimpleMedicalEvaluator(inference)
        
        # Run evaluation
        results = evaluator.evaluate(test_file, max_samples=None)  # Test with all samples
        
        # Save results
        with open("simple_evaluation_results.json", "w") as f:
            # Convert numpy types to regular types for JSON
            clean_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    clean_results[k] = {}
                    for inner_k, inner_v in v.items():
                        if isinstance(inner_v, (np.floating, np.integer)):
                            clean_results[k][inner_k] = float(inner_v)
                        elif isinstance(inner_v, np.ndarray):
                            clean_results[k][inner_k] = inner_v.tolist()
                        else:
                            clean_results[k][inner_k] = inner_v
                else:
                    clean_results[k] = v
            
            json.dump(clean_results, f, indent=2)
        
        print(f"\nResults saved to: simple_evaluation_results.json")
        return True
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
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
        "scikit-learn",
        "fuzzywuzzy[speedup]",
        "sentence-transformers",
        "matplotlib",
        "seaborn"
    ]
    
    import subprocess
    import sys
    
    print("Setting up environment...")
    for package in required_packages:
        try:
            package_name = package.split('[')[0]  # Handle packages with extras like fuzzywuzzy[speedup]
            __import__(package_name.replace("-", "_"))
            print(f"✅ {package} is already installed")
        except ImportError:
            print(f"⏳ Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")

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
            
        elif command == "evaluate" or command == "inference":
            success = main_simple_evaluation()
            
        elif command == "train":
            success = main_generative_finetune()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  python script.py setup     - Setup environment")
            print("  python script.py train     - Run fine-tuning")
            print("  python script.py evaluate  - Run simple evaluation")
            success = False
    else:
        # Default to training
        success = main_generative_finetune()
    
    print(f"\n{'='*60}")
    print(f"FINAL STATUS: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*60}")
    
    raise SystemExit(0 if success else 1)
