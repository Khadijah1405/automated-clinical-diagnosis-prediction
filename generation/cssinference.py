#!/usr/bin/env python3
"""
Generative Medical Diagnosis: Direct LLaMA Text Generation with Hugging Face
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("GPU_ID", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import re
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, AutoModel
from transformers import EarlyStoppingCallback  # Added for early stopping
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
import subprocess

SKLEARN_AVAILABLE = True
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SKLEARN_AVAILABLE = False

SENTENCE_TRANSFORMERS_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

TRANSFORMERS_AVAILABLE = True

TEXT_METRICS_AVAILABLE = True
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    TEXT_METRICS_AVAILABLE = False

PLOTTING_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    PLOTTING_AVAILABLE = False

def log_gpu_memory(context=""):
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n=== GPU [{context}] ===")
            print(f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
            print("=" * 40)
    except Exception as e:
        print(f"GPU memory error: {e}")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    spacy = None
    nlp = None

class MedicalTextProcessor:
    def __init__(self):
        self.medical_stopwords = {'patient', 'pt', 'year', 'old', 'y/o', 'yo', 'male', 'female'}
    
    def clean_text(self, text: str) -> str:
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
        return self.clean_text(text)

def get_llama_chat_template(clinical_text: str, target_diagnosis: str = None) -> str:
    sys_prompt = """You are an expert physician specializing in medical diagnosis. 
Given a clinical presentation, provide a specific primary diagnosis using standard medical terminology.
Focus on the most likely primary condition based on the clinical findings presented.
Provide only the diagnosis name, be specific and use proper medical terminology."""
    
    user_prompt = f"""Based on the following clinical presentation, what is the primary medical diagnosis?

Clinical Presentation:
{clinical_text}

Primary Diagnosis:"""
    
    if target_diagnosis:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{target_diagnosis}<|eot_id|>"
    else:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

class GenerativePromptBuilder:
    @staticmethod
    def build_training_prompt(clinical_text: str, target_diagnosis: str) -> str:
        return get_llama_chat_template(clinical_text, target_diagnosis)
    
    @staticmethod
    def build_inference_prompt(clinical_text: str) -> str:
        return get_llama_chat_template(clinical_text)

class GenerativeMedicalDataset(Dataset):
    def __init__(self, data_items: List[Dict], tokenizer, max_length: int = 2048):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_processor = MedicalTextProcessor()
        print(f"Created dataset with {len(data_items)} items")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        prompt = item["text"]
        encoding = self.tokenizer(prompt, truncation=True, max_length=self.max_length, 
                                  padding="max_length", return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }

class GenerativeMedicalFineTuner:
    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.text_processor = MedicalTextProcessor()
        print(f"Initialized | device: {self.device}")

    def load_base_model(self) -> bool:
        base_model = self.config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct")
        max_len = self.config.get("max_seq_length", 2048)
        
        print(f"Loading model: {base_model}")
        log_gpu_memory("Before Model Loading")
        
        try:
            token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if not token:
                print("ERROR: HUGGING_FACE_HUB_TOKEN not found")
                return False
            
            print("Found HF token")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, 
                                                          use_fast=True, token=token)
            log_gpu_memory("After Tokenizer")
            
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "right"
            self.tokenizer.model_max_length = max_len
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True, use_cache=False, token=token
            )
            log_gpu_memory("After Model")
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False,
                r=self.config.get("lora_r", 16),  # Changed from 4 to 8
                lora_alpha=self.config.get("lora_alpha", 32),
                lora_dropout=self.config.get("lora_dropout", 0.1),
                target_modules=self.config.get("lora_target_modules", 
                    ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj"]),
                bias="none"
            )
            
            self.model = get_peft_model(self.model, peft_config)
            log_gpu_memory("After LoRA")
            
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Model ready | trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
            return True
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_datasets(self, train_file: str, val_file: str):
        print("Loading datasets...")
        log_gpu_memory("Before Datasets")
        
        try:
            with open(train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            with open(val_file, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            
            print(f"Loaded FULL datasets: Train={len(train_data)}, Val={len(val_data)}")
            
            
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
            
            max_len = self.config.get("max_seq_length", 2048)
            train_ds = GenerativeMedicalDataset(train_prompts, self.tokenizer, max_len)
            val_ds = GenerativeMedicalDataset(val_prompts, self.tokenizer, max_len)
            
            log_gpu_memory("After Datasets")
            return train_ds, val_ds
            
        except Exception as e:
            print(f"Dataset loading failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def fine_tune(self, train_dataset, val_dataset):
        print("Starting fine-tuning with EARLY STOPPING...")
        log_gpu_memory("Before Training Setup")
        
        out_dir = self.config.get('output_dir', './generative_medical_loracssfull16')
        
        from inspect import signature
        run_name = self.config.get('run_name', 'gen-med-finetune')
        training_args_params = signature(TrainingArguments).parameters
        
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
            load_best_model_at_end=True,  # Critical for early stopping
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
        )
        
        if "eval_strategy" in training_args_params:
            ta_kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in training_args_params:
            ta_kwargs["evaluation_strategy"] = "steps"
        
        args = TrainingArguments(**ta_kwargs)
        
        # ADDED: Early stopping callback
        early_stopping_patience = self.config.get('early_stopping_patience', 3)
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=0.0  # Any improvement counts
        )
        
        print(f"Early stopping enabled: patience={early_stopping_patience} evaluations")
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[early_stopping_callback]  # Added early stopping
        )
        
        log_gpu_memory("After Trainer Setup")

        try:
            print("Starting training...")
            log_gpu_memory("Training Start")
            
            resume_from = None
            if os.path.exists(out_dir):
                checkpoints = [d for d in os.listdir(out_dir) 
                             if d.startswith('checkpoint-') and os.path.isdir(os.path.join(out_dir, d))]
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
                        
                        checkpoint_files = ['adapter_model.safetensors', 'trainer_state.json', 
                                          'training_args.bin']
                        is_valid = all(os.path.exists(os.path.join(resume_path, f)) 
                                     for f in checkpoint_files)

                        if is_valid:
                            resume_from = resume_path
                            print(f"Resuming from: {resume_from}")
                        else:
                            print(f"Checkpoint {latest_checkpoint} incomplete - starting fresh")
                    else:
                        print("No valid checkpoints - starting fresh")
                else:
                    print("No checkpoints found - starting fresh")
            else:
                print("Output directory doesn't exist - starting fresh")
            
            if resume_from:
                train_result = trainer.train(resume_from_checkpoint=resume_from)
            else:
                train_result = trainer.train()
            
            log_gpu_memory("Training Complete")
            
            # Check if early stopping was triggered
            if hasattr(trainer.state, 'best_metric'):
                print(f"Best validation loss: {trainer.state.best_metric:.4f}")
            
            print(f"Training completed! Final loss: {float(train_result.training_loss):.4f}")
            
            final_output_dir = os.path.join(out_dir, 'final_model')
            os.makedirs(final_output_dir, exist_ok=True)
            trainer.save_model(final_output_dir)
            self.tokenizer.save_pretrained(final_output_dir)
            
            with open(os.path.join(final_output_dir, 'training_config.json'), 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"Model saved to: {final_output_dir}")
            
            try:
                print("Running validation evaluation...")
                log_gpu_memory("Evaluation Start")
                metrics = trainer.evaluate()
                print(f"Validation metrics: {metrics}")
                
                with open(os.path.join(final_output_dir, 'eval_metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                log_gpu_memory("Evaluation Complete")
            except Exception as e:
                print(f"Validation evaluation failed: {e}")
                log_gpu_memory("Evaluation Failed")
            
            return trainer
            
        except Exception as e:
            print(f"Training failed: {e}")
            log_gpu_memory("Training Failed")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_on_test_set(self, test_file: str):
        print("\n" + "="*80)
        print("EVALUATING ON TEST SET")
        print("="*80)
        
        if not self.model or not self.tokenizer:
            print("Model not trained yet!")
            return None
        
        class InferenceWrapper:
            def __init__(self, model, tokenizer, text_processor):
                self.model = model
                self.tokenizer = tokenizer
                self.text_processor = text_processor
                self.device = (model.device if hasattr(model, 'device') 
                             else next(model.parameters()).device)
                
            def predict(self, clinical_text: str, max_new_tokens: int = 50) -> Tuple[str, Dict]:
                if not self.model:
                    return "", {"error": "Model not loaded"}
                
                cleaned_text = self.text_processor.clean_text(clinical_text)
                prompt = GenerativePromptBuilder.build_inference_prompt(cleaned_text)
                
                try:
                    inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3800)
                    inp = {k: v.to(self.device) for k, v in inp.items()}
                    
                    eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    if eot_token_id is None:
                        eot_token_id = self.tokenizer.eos_token_id
                    
                    with torch.no_grad():
                        out = self.model.generate(
                            **inp, max_new_tokens=max_new_tokens, temperature=0.3,
                            do_sample=True, top_p=0.9, top_k=50,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=eot_token_id, repetition_penalty=1.1
                        )
                    
                    generated_ids = out[0][inp["input_ids"].shape[1]:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    
                    if "<|eot_id|>" in response:
                        response = response.split("<|eot_id|>")[0].strip()
                    
                    if not response.endswith(('.', '!', '?')):
                        last_sentence_end = max(response.rfind('.'), response.rfind('!'), 
                                               response.rfind('?'))
                        if last_sentence_end > len(response) // 2:
                            response = response[:last_sentence_end + 1]
                    
                    return response, {'cleaned_input': cleaned_text, 'raw_response': response}
                except Exception as e:
                    print(f"Prediction error: {e}")
                    return "", {'error': str(e)}
        
        inference_wrapper = InferenceWrapper(self.model, self.tokenizer, self.text_processor)
        print("Initializing SimpleMedicalEvaluator...")
        evaluator = SimpleMedicalEvaluator(inference_wrapper)
        
        print("Starting evaluation on FULL test dataset...")
        results = evaluator.evaluate(test_file, max_samples=None)
        
        results_file = os.path.join(self.config.get('output_dir', './generative_medical_loracssfull16'), 
                                   'test_evaluation_results.json')
        
        json_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                json_results[k] = float(v)
            elif isinstance(v, dict):
                json_results[k] = {
                    inner_k: (float(inner_v) if isinstance(inner_v, (np.floating, np.integer)) 
                            else inner_v)
                    for inner_k, inner_v in v.items()
                }
            else:
                json_results[k] = v
        
        with open(results_file, 'w') as f:
            json.dump({
                'evaluation_type': 'test_set_evaluation_after_training',
                'test_file': test_file,
                'timestamp': pd.Timestamp.now().isoformat(),
                'results': json_results
            }, f, indent=2)
        
        print(f"\nTest evaluation results saved to: {results_file}")
        return results

    def run_full_pipeline(self, train_file: str, val_file: str, test_file: str = None) -> bool:
        print("GENERATIVE MEDICAL DIAGNOSIS FINE-TUNING")
        print("=" * 60)
        
        if not self.load_base_model():
            return False
            
        train_ds, val_ds = self.load_datasets(train_file, val_file)
        if train_ds is None:
            return False
            
        trainer = self.fine_tune(train_ds, val_ds)
        if trainer is None:
            return False
        
        if test_file:
            print("\n" + "="*80)
            print("RUNNING TEST SET EVALUATION")
            print("="*80)
            self.evaluate_on_test_set(test_file)
        
        return True

class GenerativeMedicalInference:
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
        print(f"Loading fine-tuned model from: {self.model_path}")
        try:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True, use_auth_token=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, use_auth_token=True
            )
            
            self.tokenizer.model_max_length = 2048
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            self.model.eval()
            torch.set_grad_enabled(False)
            
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, clinical_text: str, max_new_tokens: int = 50) -> Tuple[str, Dict]:
        if not self.model:
            return "", {"error": "Model not loaded"}
            
        cleaned_text = self.text_processor.clean_text(clinical_text)
        prompt = GenerativePromptBuilder.build_inference_prompt(cleaned_text)
        
        try:
            inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3800)
            inp = {k: v.to(self.model.device) for k, v in inp.items()}
            
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_token_id is None:
                eot_token_id = self.tokenizer.eos_token_id
            
            with torch.no_grad():
                out = self.model.generate(
                    **inp, max_new_tokens=max_new_tokens, temperature=0.3,
                    do_sample=True, top_p=0.9, top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eot_token_id, repetition_penalty=1.1
                )
            
            generated_ids = out[0][inp["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
            
            if not response.endswith(('.', '!', '?')):
                last_sentence_end = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
                if last_sentence_end > len(response) // 2:
                    response = response[:last_sentence_end + 1]
            
            return response, {
                'cleaned_input': cleaned_text,
                'raw_response': response,
                'prompt_length': len(prompt)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "", {'error': str(e)}

class SimpleMedicalEvaluator:
    def __init__(self, inference_model):
        self.inference = inference_model
        self.clinical_bert = None
        self.general_bert = None
        self._load_bert_models()
        
        self.rouge_scorer = None
        self.smoothing_function = None
        if TEXT_METRICS_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                                            use_stemmer=True)
                self.smoothing_function = SmoothingFunction().method1
                print("Text generation metrics available")
            except Exception as e:
                print(f"Warning: Text generation metrics failed: {e}")
        
        self.jaccard_thresholds = {'high': 0.7, 'medium': 0.4, 'low': 0.2}
        self.bert_thresholds = {'high': 0.85, 'medium': 0.7, 'low': 0.5}
        self.general_thresholds = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
        self.text_gen_thresholds = {'high': 0.7, 'medium': 0.5, 'low': 0.3}
    
    def _load_bert_models(self):
        if TRANSFORMERS_AVAILABLE:
            try:
                self.clinical_bert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                self.clinical_bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                print("Loaded Clinical BERT")
            except Exception as e:
                print(f"Failed to load Clinical BERT: {e}")
                self.clinical_bert_tokenizer = None
                self.clinical_bert_model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.general_bert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("Loaded General BERT")
            except Exception as e:
                print(f"Failed to load General BERT: {e}")
                self.general_bert = None
    
    def word_similarity(self, pred: str, target: str) -> float:
        if not pred or not target:
            return 0.0
        pred_words = set(pred.lower().split())
        target_words = set(target.lower().split())
        if not pred_words or not target_words:
            return 0.0
        intersection = len(pred_words & target_words)
        union = len(pred_words | target_words)
        return intersection / union if union > 0 else 0.0
    
    def character_similarity(self, pred: str, target: str) -> float:
        if not pred or not target:
            return 0.0
        pred_clean = pred.lower().strip()
        target_clean = target.lower().strip()
        if pred_clean == target_clean:
            return 1.0
        return SequenceMatcher(None, pred_clean, target_clean).ratio()
    
    def jaccard_index(self, pred: str, target: str) -> float:
        if not pred or not target:
            return 0.0
        pred_chars = set(pred.lower().replace(' ', ''))
        target_chars = set(target.lower().replace(' ', ''))
        if not pred_chars or not target_chars:
            return 0.0
        intersection = len(pred_chars & target_chars)
        union = len(pred_chars | target_chars)
        return intersection / union if union > 0 else 0.0
    
    def calculate_text_generation_metrics(self, pred: str, target: str) -> Dict[str, float]:
        metrics = {
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0,
            'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0,
            'length_ratio': 0.0, 'exact_match': 0.0
        }
        
        if not pred or not target:
            return metrics
        
        metrics['exact_match'] = 1.0 if pred.lower().strip() == target.lower().strip() else 0.0
        
        pred_len = len(pred.split())
        target_len = len(target.split())
        if target_len > 0:
            metrics['length_ratio'] = pred_len / target_len
        
        if not TEXT_METRICS_AVAILABLE or not self.rouge_scorer:
            return metrics
            
        try:
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
            
            rouge_scores = self.rouge_scorer.score(target, pred)
            metrics['rouge_1_f'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge_2_f'] = rouge_scores['rouge2'].fmeasure
            metrics['rouge_l_f'] = rouge_scores['rougeL'].fmeasure
        except Exception as e:
            print(f"Error calculating text metrics: {e}")
            
        return metrics
    
    def cosine_similarity_simple(self, pred: str, target: str) -> float:
        if not pred or not target:
            return 0.0
        all_chars = set(pred.lower() + target.lower())
        pred_vec = [pred.lower().count(char) for char in all_chars]
        target_vec = [target.lower().count(char) for char in all_chars]
        
        if SKLEARN_AVAILABLE:
            similarity = cosine_similarity([pred_vec], [target_vec])[0][0]
            return max(0.0, float(similarity))
        else:
            dot_product = sum(a * b for a, b in zip(pred_vec, target_vec))
            norm_pred = sum(a * a for a in pred_vec) ** 0.5
            norm_target = sum(b * b for b in target_vec) ** 0.5
            if norm_pred == 0 or norm_target == 0:
                return 0.0
            return dot_product / (norm_pred * norm_target)
    
    def clinical_bert_similarity(self, pred: str, target: str) -> float:
        if not self.clinical_bert_model or not self.clinical_bert_tokenizer or not pred or not target:
            return 0.0
        
        try:
            pred_inputs = self.clinical_bert_tokenizer(pred, return_tensors='pt', truncation=True, 
                                                       max_length=512, padding=True)
            target_inputs = self.clinical_bert_tokenizer(target, return_tensors='pt', truncation=True, 
                                                         max_length=512, padding=True)
            
            with torch.no_grad():
                pred_outputs = self.clinical_bert_model(**pred_inputs)
                target_outputs = self.clinical_bert_model(**target_inputs)
                pred_embedding = pred_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                target_embedding = target_outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                if SKLEARN_AVAILABLE:
                    similarity = cosine_similarity(pred_embedding, target_embedding)[0][0]
                    return max(0.0, float(similarity))
                else:
                    pred_flat = pred_embedding.flatten()
                    target_flat = target_embedding.flatten()
                    dot_product = np.dot(pred_flat, target_flat)
                    norm_pred = np.linalg.norm(pred_flat)
                    norm_target = np.linalg.norm(target_flat)
                    if norm_pred == 0 or norm_target == 0:
                        return 0.0
                    return max(0.0, float(dot_product / (norm_pred * norm_target)))
        except Exception as e:
            print(f"Clinical BERT error: {e}")
        return 0.0
    
    def general_bert_similarity(self, pred: str, target: str) -> float:
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
        similarities = {
            'word_similarity': self.word_similarity(pred, target),
            'character_similarity': self.character_similarity(pred, target),
            'jaccard_index': self.jaccard_index(pred, target),
            'cosine_similarity': self.cosine_similarity_simple(pred, target),
            'clinical_bert': self.clinical_bert_similarity(pred, target),
            'general_bert': self.general_bert_similarity(pred, target)
        }
        text_gen_metrics = self.calculate_text_generation_metrics(pred, target)
        similarities.update(text_gen_metrics)
        return similarities
    
    def calculate_threshold_stats(self, scores: List[float], metric_type: str = 'general') -> Dict:
        n_total = len(scores)
        if n_total == 0:
            return {}
        
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
        diagnoses = []
        for target in targets:
            if target:
                diagnosis = target.lower().strip()
                diagnosis = re.sub(r'^(acute|chronic|severe|mild)\s+', '', diagnosis)
                diagnoses.append(diagnosis)
        diagnosis_counts = Counter(diagnoses)
        return [diag for diag, _ in diagnosis_counts.most_common(top_k)]

    def create_confusion_matrix_data(self, predictions: List[str], targets: List[str],
                                   top_diagnoses: List[str]) -> Tuple[List[str], List[str]]:
        pred_labels = []
        true_labels = []
        for pred, target in zip(predictions, targets):
            pred_clean = pred.lower().strip() if pred else ""
            target_clean = target.lower().strip() if target else ""
            pred_clean = re.sub(r'^(acute|chronic|severe|mild)\s+', '', pred_clean)
            target_clean = re.sub(r'^(acute|chronic|severe|mild)\s+', '', target_clean)
            pred_label = pred_clean if pred_clean in top_diagnoses else "Other"
            true_label = target_clean if target_clean in top_diagnoses else "Other"
            pred_labels.append(pred_label)
            true_labels.append(true_label)
        return pred_labels, true_labels

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str],
                            title: str = "Confusion Matrix - Top 20 Diagnoses"):
        if not PLOTTING_AVAILABLE:
            print("Plotting not available - skipping confusion matrix")
            return None, None
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
            plt.savefig('confusion_matrix_simple_top20.png', dpi=300, bbox_inches='tight')
            plt.savefig('confusion_matrix_simple_top20.pdf', bbox_inches='tight')
            print("Confusion matrix saved")
            return cm, all_labels
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            return None, None
    
    def evaluate(self, test_file: str, max_samples: int = None) -> Dict:
        print("Starting evaluation...")
        with open(test_file, "r") as f:
            test_data = json.load(f)
        
        if max_samples:
            test_data = test_data[:max_samples]
            print(f"Evaluating {len(test_data)} samples (limited)")
        else:
            print(f"Evaluating full dataset: {len(test_data)} samples")
        
        all_predictions = []
        all_targets = []
        all_similarities = {
            'word_similarity': [], 'character_similarity': [], 'jaccard_index': [],
            'cosine_similarity': [], 'clinical_bert': [], 'general_bert': [],
            'bleu_1': [], 'bleu_2': [], 'bleu_3': [], 'bleu_4': [],
            'rouge_1_f': [], 'rouge_2_f': [], 'rouge_l_f': [],
            'length_ratio': [], 'exact_match': []
        }
        exact_matches = 0
        failed_predictions = 0
        
        for i, item in enumerate(test_data, 1):
            pred_response, _ = self.inference.predict(item['clinical_text'], max_new_tokens=50)
            all_predictions.append(pred_response)
            all_targets.append(item['target'])
            
            if not pred_response:
                failed_predictions += 1
                for metric in all_similarities:
                    all_similarities[metric].append(0.0)
                continue
            
            if pred_response.lower().strip() == item['target'].lower().strip():
                exact_matches += 1
            
            similarities = self.calculate_all_similarities(pred_response, item['target'])
            for metric, score in similarities.items():
                all_similarities[metric].append(score)
            
            if i % 100 == 0 or i == len(test_data):
                print(f"Progress: {i}/{len(test_data)} ({i/len(test_data)*100:.1f}%)")
        
        results = {
            'basic_stats': {
                'total_samples': len(test_data),
                'exact_matches': exact_matches,
                'exact_accuracy': exact_matches / len(test_data),
                'failed_predictions': failed_predictions,
                'failed_rate': failed_predictions / len(test_data)
            }
        }
        
        metric_threshold_map = {
            'word_similarity': 'jaccard', 'character_similarity': 'general',
            'jaccard_index': 'jaccard', 'cosine_similarity': 'general',
            'clinical_bert': 'bert', 'general_bert': 'bert',
            'bleu_1': 'text_gen', 'bleu_2': 'text_gen', 'bleu_3': 'text_gen', 'bleu_4': 'text_gen',
            'rouge_1_f': 'text_gen', 'rouge_2_f': 'text_gen', 'rouge_l_f': 'text_gen',
            'length_ratio': 'general', 'exact_match': 'general'
        }
        
        for metric_name, scores in all_similarities.items():
            threshold_type = metric_threshold_map.get(metric_name, 'general')
            results[f'{metric_name}_stats'] = self.calculate_threshold_stats(scores, threshold_type)
        
        results['raw_data'] = {
            'predictions': all_predictions,
            'targets': all_targets,
            'similarities': all_similarities
        }
        
        print("Generating confusion matrix...")
        top_diagnoses = self.extract_top_diagnoses(all_targets, top_k=20)
        pred_labels, true_labels = self.create_confusion_matrix_data(all_predictions, all_targets, 
                                                                      top_diagnoses)
        cm, cm_labels = self.plot_confusion_matrix(true_labels, pred_labels)
        
        results['confusion_matrix'] = cm.tolist() if cm is not None else None
        results['confusion_matrix_labels'] = cm_labels
        results['top_diagnoses'] = top_diagnoses
        
        self._print_results(results)
        return results
    
    def _print_results(self, results: Dict):
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        basic = results['basic_stats']
        print(f"\nBASIC METRICS:")
        print(f"Total: {basic['total_samples']}, Exact: {basic['exact_matches']} ({basic['exact_accuracy']:.1%})")
        print(f"Failed: {basic['failed_predictions']} ({basic['failed_rate']:.1%})")

def main_generative_finetune():
    config = {
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_seq_length": 2048,
        "lora_r": 16, 
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj"],
        "num_epochs": 10,
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 3e-5,  
        "weight_decay": 0.01,
        "output_dir": "./generative_medical_loracssfull16",
        "logging_steps": 10,
        "eval_steps": 500,
        "save_steps": 1000,
        "seed": 42,
        "early_stopping_patience": 3,  # ADDED: Stop if no improvement for 3 evaluations
        "train_file": "./medical_datasets_llama3_improved/train_dataset.json",
        "val_file": "./medical_datasets_llama3_improved/val_dataset.json",
        "test_file": "./medical_datasets_llama3_improved/test_dataset.json",
    }
    
    print("GENERATIVE MEDICAL DIAGNOSIS TRAINING WITH EARLY STOPPING")
    print("="*60)
    print(json.dumps(config, indent=2))
    print("="*60)
    
    try:
        ft = GenerativeMedicalFineTuner(config)
        success = ft.run_full_pipeline(config["train_file"], config["val_file"], config["test_file"])
        
        if success:
            print("\nTraining and evaluation completed!")
            print(f"Model: {config['output_dir']}/final_model")
            print(f"Results: {config['output_dir']}/test_evaluation_results.json")
        else:
            print("\nTraining failed!")
        return success
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main_generative_inference():
    model_path = "./generative_medical_loracssfull16/final_model"
    test_file = "./medical_datasets_llama3_improved/test_dataset.json"
    
    print("STANDALONE INFERENCE & EVALUATION")
    print("="*80)
    
    try:
        inference = GenerativeMedicalInference(model_path)
        if not inference.load_finetuned_model():
            print("Failed to load model!")
            return False
        
        print("\nTesting prediction...")
        sample = "45 year old male with chest pain and shortness of breath, elevated troponins"
        pred, meta = inference.predict(sample)
        print(f"Input: {sample}")
        print(f"Prediction: {pred}")
        
        print("\nInitializing evaluator...")
        evaluator = SimpleMedicalEvaluator(inference)
        print("Starting evaluation on full dataset...")
        results = evaluator.evaluate(test_file, max_samples=None)
        
        results_file = "./standalone_evaluation_results.json"
        json_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                json_results[k] = float(v)
            elif isinstance(v, dict):
                json_results[k] = {
                    ik: (float(iv) if isinstance(iv, (np.floating, np.integer)) else iv)
                    for ik, iv in v.items()
                }
            else:
                json_results[k] = v
        
        with open(results_file, 'w') as f:
            json.dump({
                'evaluation_type': 'standalone_saved_model_evaluation',
                'model_path': model_path,
                'test_file': test_file,
                'timestamp': pd.Timestamp.now().isoformat(),
                'results': json_results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print("\nEvaluation completed!")
        return True
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    print("SYSTEM INFO")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPUs: {torch.cuda.device_count()}")
    print("="*60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "train":
            success = main_generative_finetune()
        elif command == "inference":
            success = main_generative_inference()
        else:
            print(f"Unknown command: {command}")
            print("Available: python script.py train | inference")
            success = False
    else:
        success = main_generative_finetune()
    
    print(f"\n{'='*60}")
    print(f"STATUS: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*60}")
    
    raise SystemExit(0 if success else 1)
