#!/usr/bin/env python3
"""
Generative Medical Primary Diagnosis: Direct LLaMA Text Generation with Hugging Face
- Uses Hugging Face Transformers (NO TRL dependency)
- Direct text generation for medical primary diagnoses
- Direct evaluation after training (no separate inference model)
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
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model

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

# Additional imports for comprehensive evaluation
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

# =========================================================================
# Medical Text Preprocessor for Primary Diagnosis
# =========================================================================

class MedicalTextProcessor:
    """Processes and cleans medical text for better primary diagnosis generation"""
    
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
        return self.clean_text(text)

# =========================================================================
# Enhanced Prompt Engineering for Primary Diagnosis Generation
# =========================================================================

def get_llama_chat_template_primary(clinical_text: str, target_diagnosis: str = None) -> str:
    """Build chat template for Llama 3 - Primary Diagnosis focused"""
    
    sys_prompt = """You are an expert physician specializing in primary medical diagnosis. 
Given a clinical presentation, provide the most likely primary diagnosis using standard medical terminology.
Focus on identifying the main underlying condition that best explains the patient's symptoms and findings.
Provide only the primary diagnosis name, be specific and use proper medical terminology."""

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
    """Builds prompts for generative primary diagnosis prediction"""
    
    @staticmethod
    def build_training_prompt(clinical_text: str, target_diagnosis: str) -> str:
        """Build training prompt for generative approach"""
        return get_llama_chat_template_primary(clinical_text, target_diagnosis)
    
    @staticmethod
    def build_inference_prompt(clinical_text: str) -> str:
        """Build inference prompt for generation"""
        return get_llama_chat_template_primary(clinical_text)

# =========================================================================
# Generative Training Dataset for Primary Diagnosis
# =========================================================================

class GenerativePrimaryDiagnosisDataset(Dataset):
    """Dataset for generative primary diagnosis training"""
    
    def __init__(self, data_items: List[Dict], tokenizer, max_length: int = 2048):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_processor = MedicalTextProcessor()
        
        print(f"Created generative primary diagnosis dataset with {len(data_items)} items")

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
# Direct Evaluation Functions
# =========================================================================

class DirectPrimaryDiagnosisEvaluator:
    """Direct evaluator that works with the trained model without saving/loading"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.text_processor = MedicalTextProcessor()
        
        # Initialize BERT model for embeddings
        self.bert_model = None
        self.bert_tokenizer = None
        self._init_bert_model()
        
        # Initialize text generation scorers
        self.rouge_scorer = None
        self.smoothing_function = None
        if NLTK_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.smoothing_function = SmoothingFunction().method1
        
        # EVALUATION THRESHOLDS
        self.jaccard_thresholds = {
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        
        self.bert_thresholds = {
            'high': 0.6,
            'medium': 0.4,
            'low': 0.2
        }
        
    def _init_bert_model(self):
        """Initialize BERT model for semantic similarity"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("✅ Loaded SentenceTransformer model for BERT embeddings")
            elif BERT_TRANSFORMERS_AVAILABLE:
                model_name = 'bert-base-uncased'
                self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.bert_model = AutoModel.from_pretrained(model_name)
                print("✅ Loaded BERT model for embeddings")
            else:
                print("⚠️  BERT embeddings not available")
        except Exception as e:
            print(f"⚠️  Failed to load BERT model: {e}")
            self.bert_model = None

    def predict(self, clinical_text: str, max_new_tokens: int = 50) -> Tuple[str, Dict]:
        """Generate primary diagnosis prediction directly from the trained model"""
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
            
            # Set model to eval mode and disable gradients
            self.model.eval()
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
            
            # Decode response
            generated_ids = out[0][inp["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # Clean up response
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
            
            # Remove incomplete sentences
            if response.endswith(('.', '!', '?')):
                pass  # Good ending
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
            print(f"Primary diagnosis prediction error: {e}")
            return "", {'error': str(e)}

    def calculate_bert_similarity(self, pred: str, target: str) -> float:
        """Calculate BERT-based semantic similarity"""
        if not self.bert_model or not pred or not target:
            return 0.0
            
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                embeddings = self.bert_model.encode([pred, target], convert_to_tensor=False)
                embeddings = np.array(embeddings)
            elif BERT_TRANSFORMERS_AVAILABLE:
                embeddings = []
                for text in [pred, target]:
                    inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
                    with torch.no_grad():
                        outputs = self.bert_model(**inputs)
                        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                        embeddings.append(embedding[0])
                embeddings = np.array(embeddings)
            else:
                return 0.0
            
            if len(embeddings) != 2:
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
        
        # Exact match
        metrics['exact_match'] = 1.0 if predicted.lower().strip() == target.lower().strip() else 0.0
        
        # Length ratio
        pred_len = len(predicted.split())
        target_len = len(target.split())
        if target_len > 0:
            metrics['length_ratio'] = pred_len / target_len
        
        if not NLTK_AVAILABLE:
            return metrics
            
        try:
            # BLEU scores
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
            
            # ROUGE scores
            if self.rouge_scorer:
                rouge_scores = self.rouge_scorer.score(target, predicted)
                metrics['rouge_1_f'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge_2_f'] = rouge_scores['rouge2'].fmeasure
                metrics['rouge_l_f'] = rouge_scores['rougeL'].fmeasure
                
        except Exception as e:
            print(f"Error calculating text generation metrics: {e}")
            
        return metrics

    def evaluate_semantic_similarity(self, pred: str, target: str) -> float:
        """Calculate Jaccard semantic similarity"""
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

    def extract_top_diagnoses(self, targets: List[str], top_k: int = 20) -> List[str]:
        """Extract top K most common primary diagnoses"""
        diagnoses = []
        
        for target in targets:
            if target:
                diagnosis = target.lower().strip()
                diagnosis = re.sub(r'^(acute|chronic|severe|mild)\s+', '', diagnosis)
                diagnoses.append(diagnosis)
        
        diagnosis_counts = Counter(diagnoses)
        top_diagnoses = [diag for diag, _ in diagnosis_counts.most_common(top_k)]
        return top_diagnoses

    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            title: str = "Confusion Matrix - Top 20 Primary Diagnoses"):
        """Create and save confusion matrix plot"""
        try:
            all_labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=all_labels)
            
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=all_labels, yticklabels=all_labels)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel('Predicted Primary Diagnosis', fontsize=12)
            plt.ylabel('True Primary Diagnosis', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plt.savefig('confusion_matrix_primary_diagnosis_direct.png', dpi=300, bbox_inches='tight')
            plt.savefig('confusion_matrix_primary_diagnosis_direct.pdf', bbox_inches='tight')
            print("📊 Confusion matrix saved as 'confusion_matrix_primary_diagnosis_direct.png/pdf'")
            
            return cm, all_labels
            
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            return None, None

    def comprehensive_evaluation(self, test_file: str, max_samples: int = None) -> Dict:
        """Run comprehensive evaluation directly on the trained model"""
        if max_samples is None:
            print(f"Starting comprehensive Primary Diagnosis evaluation on FULL dataset...")
        else:
            print(f"Starting comprehensive Primary Diagnosis evaluation on up to {max_samples} samples...")
        
        # Load test data
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        
        if max_samples is not None:
            test_data = test_data[:max_samples]
            
        print(f"Evaluating on {len(test_data)} samples")
        
        # Initialize tracking variables
        all_predictions = []
        all_targets = []
        exact_matches = 0
        semantic_scores = []
        bert_scores = []
        all_text_gen_metrics = []
        failed_predictions = 0
        
        # Separate counters for Jaccard and BERT similarities
        jaccard_counters = {'high': 0, 'medium': 0, 'low': 0}
        bert_counters = {'high': 0, 'medium': 0, 'low': 0}
        
        # Process each sample
        print("Processing samples...")
        for i, item in enumerate(test_data, 1):
            clinical_text = item['clinical_text']
            target = item['target']
            
            # Get prediction
            pred_response, metadata = self.predict(clinical_text, max_new_tokens=50)
            all_predictions.append(pred_response)
            all_targets.append(target)
            
            if not pred_response:
                failed_predictions += 1
                semantic_scores.append(0.0)
                bert_scores.append(0.0)
                all_text_gen_metrics.append({})
                continue
            
            # Calculate exact match
            if pred_response.lower().strip() == target.lower().strip():
                exact_matches += 1
            
            # Calculate Jaccard semantic similarity
            semantic_score = self.evaluate_semantic_similarity(pred_response, target)
            semantic_scores.append(semantic_score)
            
            # Calculate BERT similarity
            bert_score = self.calculate_bert_similarity(pred_response, target)
            bert_scores.append(bert_score)
            
            # Calculate text generation metrics
            text_gen_metrics = self.calculate_text_generation_metrics(pred_response, target)
            all_text_gen_metrics.append(text_gen_metrics)
            
            # Update counters
            if semantic_score >= self.jaccard_thresholds['high']:
                jaccard_counters['high'] += 1
            elif semantic_score >= self.jaccard_thresholds['medium']:
                jaccard_counters['medium'] += 1
            elif semantic_score >= self.jaccard_thresholds['low']:
                jaccard_counters['low'] += 1
            
            if bert_score >= self.bert_thresholds['high']:
                bert_counters['high'] += 1
            elif bert_score >= self.bert_thresholds['medium']:
                bert_counters['medium'] += 1
            elif bert_score >= self.bert_thresholds['low']:
                bert_counters['low'] += 1
            
            # Progress update
            if i % 100 == 0 or i == len(test_data):
                progress = i / len(test_data) * 100
                print(f"Progress: {i}/{len(test_data)} ({progress:.1f}%)")
        
        # Calculate final metrics
        n_samples = len(test_data)
        results = {
            # Basic accuracy metrics
            'exact_accuracy': exact_matches / n_samples,
            'failed_prediction_rate': failed_predictions / n_samples,
            
            # Jaccard semantic similarity metrics
            'mean_semantic_score': np.mean(semantic_scores),
            'median_semantic_score': np.median(semantic_scores),
            'std_semantic_score': np.std(semantic_scores),
            
            # BERT similarity metrics
            'mean_bert_score': np.mean(bert_scores) if bert_scores else 0.0,
            'median_bert_score': np.median(bert_scores) if bert_scores else 0.0,
            'std_bert_score': np.std(bert_scores) if bert_scores else 0.0,
            
            # Text generation metrics
            'mean_bleu_1': np.mean([m.get('bleu_1', 0) for m in all_text_gen_metrics]) if all_text_gen_metrics else 0.0,
            'mean_bleu_2': np.mean([m.get('bleu_2', 0) for m in all_text_gen_metrics]) if all_text_gen_metrics else 0.0,
            'mean_bleu_4': np.mean([m.get('bleu_4', 0) for m in all_text_gen_metrics]) if all_text_gen_metrics else 0.0,
            'mean_rouge_1': np.mean([m.get('rouge_1_f', 0) for m in all_text_gen_metrics]) if all_text_gen_metrics else 0.0,
            'mean_rouge_l': np.mean([m.get('rouge_l_f', 0) for m in all_text_gen_metrics]) if all_text_gen_metrics else 0.0,
            'exact_match_rate': np.mean([m.get('exact_match', 0) for m in all_text_gen_metrics]) if all_text_gen_metrics else 0.0,
            'mean_length_ratio': np.mean([m.get('length_ratio', 0) for m in all_text_gen_metrics]) if all_text_gen_metrics else 0.0,
            
            # Threshold-based rates
            'jaccard_high_rate': jaccard_counters['high'] / n_samples,
            'jaccard_medium_rate': jaccard_counters['medium'] / n_samples,
            'jaccard_low_rate': jaccard_counters['low'] / n_samples,
            
            'bert_high_rate': bert_counters['high'] / n_samples,
            'bert_medium_rate': bert_counters['medium'] / n_samples,
            'bert_low_rate': bert_counters['low'] / n_samples,
            
            # Count metrics
            'n_samples': n_samples,
            'exact_matches': exact_matches,
            'failed_predictions': failed_predictions,
            
            # Raw data
            'predictions': all_predictions,
            'targets': all_targets,
            'semantic_scores': semantic_scores,
            'bert_scores': bert_scores,
            'all_text_gen_metrics': all_text_gen_metrics
        }
        
        # Generate confusion matrix
        print("Generating confusion matrix for top 20 primary diagnoses...")
        top_diagnoses = self.extract_top_diagnoses(all_targets, top_k=20)
        
        # Create confusion matrix data
        pred_labels = []
        true_labels = []
        
        for pred, target in zip(all_predictions, all_targets):
            pred_clean = pred.lower().strip() if pred else ""
            target_clean = target.lower().strip() if target else ""
            
            pred_clean = re.sub(r'^(acute|chronic|severe|mild)\s+', '', pred_clean)
            target_clean = re.sub(r'^(acute|chronic|severe|mild)\s+', '', target_clean)
            
            pred_label = pred_clean if pred_clean in top_diagnoses else "Other"
            true_label = target_clean if target_clean in top_diagnoses else "Other"
            
            pred_labels.append(pred_label)
            true_labels.append(true_label)
        
        cm, cm_labels = self.plot_confusion_matrix(true_labels, pred_labels)
        results['confusion_matrix'] = cm.tolist() if cm is not None else None
        results['confusion_matrix_labels'] = cm_labels
        results['top_diagnoses'] = top_diagnoses
        
        # Print results
        self._print_results(results)
        
        # Show sample examples
        self._show_examples(test_data, all_predictions, all_targets, semantic_scores, bert_scores)
        
        return results

    def _print_results(self, results: Dict):
        """Print comprehensive evaluation results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PRIMARY DIAGNOSIS EVALUATION RESULTS (DIRECT)")
        print("="*80)
        
        print("\n📊 BASIC ACCURACY METRICS")
        print(f"Exact Match Accuracy:             {results['exact_accuracy']:.1%}")
        print(f"Failed Prediction Rate:           {results['failed_prediction_rate']:.1%}")
        
        print("\n🔍 JACCARD SEMANTIC SIMILARITY METRICS")
        print(f"Mean Jaccard Score:               {results['mean_semantic_score']:.3f}")
        print(f"Median Jaccard Score:             {results['median_semantic_score']:.3f}")
        print(f"Std Dev Jaccard Score:            {results['std_semantic_score']:.3f}")
        
        print(f"Mean BERT Similarity:             {results['mean_bert_score']:.3f}")
        print(f"Median BERT Similarity:           {results['median_bert_score']:.3f}")
        print(f"Std Dev BERT Similarity:          {results['std_bert_score']:.3f}")
        
        if results['mean_bleu_1'] > 0:
            print("\n📝 TEXT GENERATION METRICS")
            print(f"BLEU-1 Score:                    {results['mean_bleu_1']:.3f}")
            print(f"BLEU-2 Score:                    {results['mean_bleu_2']:.3f}")
            print(f"BLEU-4 Score:                    {results['mean_bleu_4']:.3f}")
            print(f"ROUGE-1 F1 Score:                {results['mean_rouge_1']:.3f}")
            print(f"ROUGE-L F1 Score:                {results['mean_rouge_l']:.3f}")
            print(f"Exact Match Rate:                {results['exact_match_rate']:.1%}")
            print(f"Mean Length Ratio:               {results['mean_length_ratio']:.2f}")
            print("\n🎯 JACCARD SIMILARITY THRESHOLD ANALYSIS")
            print(f"Jaccard High (≥0.7):             {results['jaccard_high_rate']:.1%}")
            print(f"Jaccard Medium (≥0.5):           {results['jaccard_medium_rate']:.1%}")
            print(f"Jaccard Low (≥0.3):              {results['jaccard_low_rate']:.1%}")
            
        if results['mean_bert_score'] > 0:
            print("\n🤖 BERT SIMILARITY THRESHOLD ANALYSIS")
            print(f"BERT High (≥0.6):               {results['bert_high_rate']:.1%}")
            print(f"BERT Medium (≥0.4):             {results['bert_medium_rate']:.1%}")
            print(f"BERT Low (≥0.2):                {results['bert_low_rate']:.1%}")
    
    def _show_examples(self, test_data: List[Dict], predictions: List[str], 
                      targets: List[str], semantic_scores: List[float], bert_scores: List[float]):
        """Show sample examples with all metrics"""
        print(f"\n📋 SAMPLE PRIMARY DIAGNOSIS PREDICTIONS (Top 5):")
        print("="*80)
        
        for i in range(min(5, len(predictions))):
            print(f"\n🔸 Example {i+1}:")
            print(f"Clinical Text: {test_data[i]['clinical_text'][:150]}...")
            print(f"Predicted: {predictions[i]}")
            print(f"Target: {targets[i]}")
            
            print(f"\n📈 Scores:")
            print(f"  Semantic Score: {semantic_scores[i]:.3f}")
            if i < len(bert_scores) and bert_scores[i] > 0:
                print(f"  BERT Score: {bert_scores[i]:.3f}")
            exact_match = "Yes" if predictions[i].lower().strip() == targets[i].lower().strip() else "No"
            print(f"  Exact Match: {exact_match}")
            print("-" * 60)

# =========================================================================
# Generative Primary Diagnosis Fine-tuner with Direct Evaluation
# =========================================================================

class GenerativePrimaryDiagnosisFineTuner:
    """Fine-tuner for generative primary diagnosis with direct evaluation after training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.text_processor = MedicalTextProcessor()
        print("Generative Primary Diagnosis Fine-tuner with Direct Evaluation initialized | device:", self.device)

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

    def load_datasets(self, train_file: str, val_file: str, test_file: str):
        """Load and prepare datasets for training and evaluation"""
        print("Loading datasets for generative primary diagnosis training...")
        
        log_gpu_memory("Before Dataset Loading")
        
        try:
            with open(train_file, "r", encoding="utf-8") as f:
                train_data = json.load(f)
            with open(val_file, "r", encoding="utf-8") as f:
                val_data = json.load(f)
            with open(test_file, "r", encoding="utf-8") as f:
                test_data = json.load(f)
            
            print(f"Datasets loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            
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
            train_ds = GenerativePrimaryDiagnosisDataset(train_prompts, self.tokenizer, max_len)
            val_ds = GenerativePrimaryDiagnosisDataset(val_prompts, self.tokenizer, max_len)
            
            log_gpu_memory("After Dataset Creation")
            
            return train_ds, val_ds, test_data
            
        except Exception as e:
            print("Dataset loading failed:", e)
            log_gpu_memory("Dataset Loading Failed")
            return None, None, None

    def fine_tune(self, train_dataset, val_dataset):
        """Fine-tune the model for generative primary diagnosis"""
        print("Starting generative primary diagnosis fine-tuning...")
        
        log_gpu_memory("Before Training Setup")
        
        out_dir = self.config.get('output_dir', './generative_medical_lora_p')
        
        # Training arguments
        from inspect import signature
        run_name = self.config.get('run_name', 'gen-primary-diag-finetune')

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

        # Use standard Trainer
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
            print("Starting primary diagnosis training...")
            log_gpu_memory("Training Start")
            
            # Smart checkpoint resuming logic
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
                print(f"Resuming primary diagnosis training from checkpoint: {resume_from}")
                train_result = trainer.train(resume_from_checkpoint=resume_from)
            else:
                print("Starting fresh primary diagnosis training")
                train_result = trainer.train()
            
            log_gpu_memory("Training Complete")
            
            print(f"Primary diagnosis training completed! Final loss: {float(train_result.training_loss):.4f}")
            
            # Save final model (optional - we can test directly without saving)
            if self.config.get('save_model', True):
                final_output_dir = os.path.join(out_dir, 'final_model')
                os.makedirs(final_output_dir, exist_ok=True)
                trainer.save_model(final_output_dir)
                self.tokenizer.save_pretrained(final_output_dir)
                
                # Save config
                with open(os.path.join(final_output_dir, 'training_config.json'), 'w') as f:
                    json.dump(self.config, f, indent=2)
                
                print(f"Primary diagnosis model saved to: {final_output_dir}")
            
            return trainer
            
        except Exception as e:
            print(f"Primary diagnosis training failed: {e}")
            log_gpu_memory("Training Failed")
            import traceback
            traceback.print_exc()
            return None

    def run_full_pipeline_with_direct_evaluation(self, train_file: str, val_file: str, test_file: str) -> bool:
        """Run the complete training pipeline with direct evaluation"""
        print("GENERATIVE PRIMARY DIAGNOSIS FINE-TUNING WITH DIRECT EVALUATION")
        print("=" * 80)
        
        if not self.load_base_model():
            return False
            
        train_ds, val_ds, test_data = self.load_datasets(train_file, val_file, test_file)
        if train_ds is None:
            return False
            
        trainer = self.fine_tune(train_ds, val_ds)
        if trainer is None:
            return False
        
        # Direct evaluation after training
        print("\n" + "="*80)
        print("STARTING DIRECT EVALUATION ON TRAINED MODEL")
        print("="*80)
        
        try:
            # Create direct evaluator with the trained model
            evaluator = DirectPrimaryDiagnosisEvaluator(trainer.model, self.tokenizer)
            
            # Test single prediction first
            print("\nTesting single primary diagnosis prediction...")
            sample_text = "45 year old male with chest pain and shortness of breath, elevated troponins"
            pred, meta = evaluator.predict(sample_text)
            print(f"Sample input: {sample_text}")
            print(f"Predicted primary diagnosis: {pred}")
            
            # Run comprehensive evaluation
            print("\nStarting comprehensive evaluation...")
            max_eval_samples = self.config.get('max_eval_samples', None)
            
            # Save test data temporarily for evaluation
            temp_test_file = "./temp_test_data.json"
            with open(temp_test_file, 'w') as f:
                json.dump(test_data, f)
            
            results = evaluator.comprehensive_evaluation(temp_test_file, max_samples=max_eval_samples)
            
            # Clean up temporary file
            if os.path.exists(temp_test_file):
                os.remove(temp_test_file)
            
            # Save results
            results_file = "./direct_evaluation_results_primary_diagnosis.json"
            
            # Convert numpy arrays to JSON-compatible format
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
            
            # Save results with metadata
            with open(results_file, 'w') as f:
                json.dump({
                    'evaluation_type': 'direct_primary_diagnosis_after_training',
                    'training_config': self.config,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'results': json_results,
                    'evaluation_settings': {
                        'max_eval_samples': max_eval_samples,
                        'jaccard_thresholds': evaluator.jaccard_thresholds,
                        'bert_thresholds': evaluator.bert_thresholds,
                        'top_diagnoses_count': 20
                    }
                }, f, indent=2)
            
            print(f"\n📊 Direct evaluation results saved to: {results_file}")
            
            # Additional analysis
            print(f"\n📋 EVALUATION SUMMARY")
            print("="*60)
            print(f"Total samples evaluated: {results['n_samples']}")
            print(f"Failed predictions: {results['failed_predictions']}")
            print(f"Success rate: {(1 - results['failed_prediction_rate']):.1%}")
            
            # Performance insights
            if results.get('mean_bert_score', 0) > 0:
                semantic_vs_bert = results['mean_semantic_score'] - results['mean_bert_score']
                print(f"\n🎯 KEY INSIGHTS")
                if abs(semantic_vs_bert) > 0.05:
                    if semantic_vs_bert > 0:
                        print(f"Jaccard similarity outperforms BERT by {semantic_vs_bert:.3f} points")
                    else:
                        print(f"BERT embeddings outperform Jaccard similarity by {-semantic_vs_bert:.3f} points")
            
            # Save top diagnoses analysis
            if 'top_diagnoses' in results:
                top_diagnoses_file = "./top_primary_diagnoses_direct_analysis.txt"
                with open(top_diagnoses_file, 'w') as f:
                    f.write("TOP 20 PRIMARY DIAGNOSES IN TEST SET (DIRECT EVALUATION)\n")
                    f.write("="*60 + "\n")
                    for i, diagnosis in enumerate(results['top_diagnoses'], 1):
                        f.write(f"{i:2d}. {diagnosis}\n")
                print(f"📋 Top primary diagnoses saved to: {top_diagnoses_file}")
            
            print("\n✅ Direct evaluation after training completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"Direct evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return False

# =========================================================================
# Main Functions
# =========================================================================

def main_generative_primary_diagnosis_with_direct_evaluation():
    """Main function for training and direct evaluation"""
    config = {
        # Model configuration
        "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "max_seq_length": 2048,
        
        # Dataset size limits for memory management
        "max_train_samples": 100000,
        "max_val_samples": 10000,
        "max_eval_samples": 10000, 
        
        # LoRA configuration
        "lora_r": 4,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "up_proj"
        ],
        
        # Training configuration
        "num_epochs": 3,  # Reduced for faster testing
        "batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        
        # Logging and saving
        "output_dir": "./generative_medical_lora_p",
        "logging_steps": 10,
        "eval_steps": 500,
        "save_steps": 1000,
        "save_model": True,  # Set to False to skip saving if you only want direct evaluation
        
        # Other settings
        "seed": 42,
        
        # Data files
        "train_file": "./medical_dataset_p/train_dataset.json",
        "val_file": "./medical_dataset_p/val_dataset.json",
        "test_file": "./medical_dataset_p/test_dataset.json",
    }
    
    print("GENERATIVE PRIMARY DIAGNOSIS FINE-TUNING WITH DIRECT EVALUATION")
    print("="*80)
    print(json.dumps(config, indent=2))
    print("="*80)
    
    try:
        ft = GenerativePrimaryDiagnosisFineTuner(config)
        success = ft.run_full_pipeline_with_direct_evaluation(
            config["train_file"], 
            config["val_file"], 
            config["test_file"]
        )
        
        if success:
            print("\n✅ Training and direct evaluation completed successfully!")
        else:
            print("\n❌ Training or evaluation failed!")
            
        return success
        
    except Exception as e:
        print(f"Pipeline error: {e}")
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
    """Print model recommendations"""
    print("\n" + "="*60)
    print("PRIMARY DIAGNOSIS MODEL RECOMMENDATIONS (DIRECT EVALUATION)")
    print("="*60)
    print("Available Llama 3 models:")
    print("• meta-llama/Meta-Llama-3-8B-Instruct (recommended)")
    print("• meta-llama/Llama-3.1-8B-Instruct (latest version)")
    print()
    print("Direct Evaluation Benefits:")
    print("• No model saving/loading overhead")
    print("• Immediate results after training")
    print("• Memory efficient evaluation")
    print("• Faster iteration cycles")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Print system information
    print("PRIMARY DIAGNOSIS SYSTEM INFORMATION (DIRECT EVALUATION)")
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
            
        elif command in ["train", "run"]:
            success = main_generative_primary_diagnosis_with_direct_evaluation()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands:")
            print("  python primary_diagnosis_direct.py setup - Setup environment")
            print("  python primary_diagnosis_direct.py train - Run training with direct evaluation")
            print("  python primary_diagnosis_direct.py run   - Same as train")
            success = False
    else:
        # Default to training with direct evaluation
        success = main_generative_primary_diagnosis_with_direct_evaluation()
    
    print(f"\n{'='*60}")
    print(f"FINAL STATUS: {'SUCCESS' if success else 'FAILED'}")
    print(f"{'='*60}")
    
    raise SystemExit(0 if success else 1)
