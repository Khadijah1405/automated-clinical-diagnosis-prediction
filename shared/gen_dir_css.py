#!/usr/bin/env python3
"""
Standalone Evaluation Script for Already-Trained Medical Diagnosis Model
Usage: python evaluate_model.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("GPU_ID", "0")

import json
import re
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from peft import AutoPeftModelForCausalLM

# Check for optional libraries
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    import nltk
    nltk.download('punkt', quiet=True)
    TEXT_METRICS_AVAILABLE = True
except ImportError:
    TEXT_METRICS_AVAILABLE = False
    print("Warning: NLTK/ROUGE not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available")

TRANSFORMERS_AVAILABLE = True  # Already imported above

# =========================================================================
# Medical Text Preprocessor
# =========================================================================

class MedicalTextProcessor:
    """Processes and cleans medical text"""
    
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
        
        # Expand common medical abbreviations
        text = re.sub(r'\b\d+\s*y/?o\b', 'years old', text, flags=re.IGNORECASE)
        text = re.sub(r'\bc/o\b', 'complains of', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpt\b', 'patient', text, flags=re.IGNORECASE)
        text = re.sub(r'\bs/p\b', 'status post', text, flags=re.IGNORECASE)
        text = re.sub(r'\bh/o\b', 'history of', text, flags=re.IGNORECASE)
        
        return text.strip()

# =========================================================================
# Prompt Builder
# =========================================================================

def get_llama_chat_template(clinical_text: str) -> str:
    """Build inference prompt for Llama 3"""
    
    sys_prompt = """You are an expert physician with extensive training in differential diagnosis. 

Your task: Analyze clinical presentations and provide accurate diagnoses using standard medical terminology.

CRITICAL INSTRUCTIONS:
1. Provide the PRIMARY DIAGNOSIS first - this is the main condition causing the patient's presentation
3. Use precise medical terminology (e.g., "Acute myocardial infarction" not "heart attack")
4. Be specific about anatomy and pathophysiology when relevant

REQUIRED FORMAT:
Primary Diagnosis: [Specific medical condition with relevant qualifiers]

DO NOT include explanations, probabilities, or differential diagnoses - only the final diagnosis."""

    user_prompt = f"""Clinical Presentation:
{clinical_text}

Provide the diagnosis using the exact format specified:"""

    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

class ImprovedPromptBuilder:
    """Builds prompts for inference"""
    
    @staticmethod
    def build_inference_prompt(clinical_text: str) -> str:
        """Build inference prompt"""
        return get_llama_chat_template(clinical_text)

# =========================================================================
# Comprehensive Medical Evaluator
# =========================================================================

class ComprehensiveMedicalEvaluator:
    """Comprehensive evaluator with multiple similarity metrics"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.text_processor = MedicalTextProcessor()
        self.device = next(model.parameters()).device
        
        # Load BERT models
        self.clinical_bert_tokenizer = None
        self.clinical_bert_model = None
        self.general_bert = None
        self._load_bert_models()
        
        # Initialize text generation scorers
        self.rouge_scorer = None
        self.smoothing_function = None
        if TEXT_METRICS_AVAILABLE:
            try:
                self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                self.smoothing_function = SmoothingFunction().method1
                print("✅ Text generation metrics available")
            except Exception as e:
                print(f"⚠️ Text generation metrics failed: {e}")
        
        # Thresholds for different similarity types
        self.jaccard_thresholds = {'high': 0.7, 'medium': 0.4, 'low': 0.2}
        self.bert_thresholds = {'high': 0.85, 'medium': 0.7, 'low': 0.5}
        self.general_thresholds = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
        self.text_gen_thresholds = {'high': 0.7, 'medium': 0.5, 'low': 0.3}
    
    def _load_bert_models(self):
        """Load BERT models"""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.clinical_bert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                self.clinical_bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
                print("✅ Loaded Clinical BERT")
            except Exception as e:
                print(f"⚠️ Failed to load Clinical BERT: {e}")
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.general_bert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                print("✅ Loaded General BERT")
            except Exception as e:
                print(f"⚠️ Failed to load General BERT: {e}")
    
    def predict(self, clinical_text: str, max_new_tokens: int = 50) -> Tuple[str, Dict]:
        """Generate diagnosis prediction"""
        cleaned_text = self.text_processor.clean_text(clinical_text)
        prompt = ImprovedPromptBuilder.build_inference_prompt(cleaned_text)
        
        try:
            inp = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3700)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_token_id is None:
                eot_token_id = self.tokenizer.eos_token_id
            
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
            
            generated_ids = out[0][inp["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if "<|eot_id|>" in response:
                response = response.split("<|eot_id|>")[0].strip()
            
            metadata = {'cleaned_input': cleaned_text, 'raw_response': response}
            return response, metadata
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "", {'error': str(e)}
    
    def word_similarity(self, pred: str, target: str) -> float:
        """Word-to-word Jaccard similarity"""
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
        """Character-level similarity using SequenceMatcher"""
        if not pred or not target:
            return 0.0
        pred_clean = pred.lower().strip()
        target_clean = target.lower().strip()
        if pred_clean == target_clean:
            return 1.0
        return SequenceMatcher(None, pred_clean, target_clean).ratio()
    
    def jaccard_index(self, pred: str, target: str) -> float:
        """Character-level Jaccard index"""
        if not pred or not target:
            return 0.0
        pred_chars = set(pred.lower().replace(' ', ''))
        target_chars = set(target.lower().replace(' ', ''))
        if not pred_chars or not target_chars:
            return 0.0
        intersection = len(pred_chars & target_chars)
        union = len(pred_chars | target_chars)
        return intersection / union if union > 0 else 0.0
    
    def cosine_similarity_simple(self, pred: str, target: str) -> float:
        """Simple cosine similarity using character frequency"""
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
        """Clinical BERT similarity"""
        if not self.clinical_bert_model or not self.clinical_bert_tokenizer or not pred or not target:
            return 0.0
        try:
            pred_inputs = self.clinical_bert_tokenizer(pred, return_tensors='pt', truncation=True, max_length=512, padding=True)
            target_inputs = self.clinical_bert_tokenizer(target, return_tensors='pt', truncation=True, max_length=512, padding=True)
            
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
                    return dot_product / (norm_pred * norm_target)
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
    
    def calculate_text_generation_metrics(self, pred: str, target: str) -> Dict[str, float]:
        """Calculate BLEU, ROUGE metrics"""
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
        text_gen_metrics = self.calculate_text_generation_metrics(pred, target)
        similarities.update(text_gen_metrics)
        return similarities
    
    def calculate_threshold_stats(self, scores: List[float], metric_type: str = 'general') -> Dict:
        """Calculate threshold statistics"""
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
    
    def evaluate(self, test_file: str, max_samples: int = None) -> Dict:
        """Main evaluation function"""
        print(f"Starting comprehensive evaluation...")
        
        with open(test_file, "r") as f:
            test_data = json.load(f)
        
        if max_samples:
            test_data = test_data[:max_samples]
        
        print(f"Evaluating {len(test_data)} samples")
        
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
            clinical_text = item['clinical_text']
            target = item['target']
            
            pred_response, _ = self.predict(clinical_text, max_new_tokens=50)
            
            all_predictions.append(pred_response)
            all_targets.append(target)
            
            if not pred_response:
                failed_predictions += 1
                for metric in all_similarities:
                    all_similarities[metric].append(0.0)
                continue
            
            if pred_response.lower().strip() == target.lower().strip():
                exact_matches += 1
            
            similarities = self.calculate_all_similarities(pred_response, target)
            for metric, score in similarities.items():
                all_similarities[metric].append(score)
            
            if i % 500 == 0 or i == len(test_data):
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
        
        self._print_results(results)
        return results
    
    def _print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("SIMPLE MEDICAL EVALUATION RESULTS")
        print("="*80)
        
        basic = results['basic_stats']
        print(f"\nBASIC METRICS:")
        print(f"Total Samples: {basic['total_samples']}")
        print(f"Exact Matches: {basic['exact_matches']} ({basic['exact_accuracy']:.1%})")
        print(f"Failed Predictions: {basic['failed_predictions']} ({basic['failed_rate']:.1%})")
        
        metric_info = {
            'word_similarity': 'WORD-TO-WORD SIMILARITY (Jaccard)',
            'character_similarity': 'CHARACTER-TO-CHARACTER SIMILARITY',
            'jaccard_index': 'JACCARD INDEX (Character Level)',
            'cosine_similarity': 'COSINE SIMILARITY',
            'clinical_bert': 'CLINICAL BERT SIMILARITY',
            'general_bert': 'GENERAL BERT SIMILARITY',
            'bleu_1': 'BLEU-1 SCORE',
            'bleu_2': 'BLEU-2 SCORE',
            'bleu_4': 'BLEU-4 SCORE',
            'rouge_1_f': 'ROUGE-1 F1 SCORE',
            'rouge_l_f': 'ROUGE-L F1 SCORE',
            'length_ratio': 'LENGTH RATIO',
            'exact_match': 'EXACT MATCH RATE'
        }
        
        for metric, display_name in metric_info.items():
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
        
        predictions = results['raw_data']['predictions']
        targets = results['raw_data']['targets']
        similarities = results['raw_data']['similarities']
        
        print(f"\nSAMPLE PREDICTIONS (first 3):")
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
# Main Evaluation Function
# =========================================================================

def main():
    """Main evaluation function"""
    
    # ========== CONFIGURATION - CHANGE THESE PATHS ==========
    MODEL_PATH = "./generative_medical_lora/final_model"  # Path to your trained model
    TEST_FILE = "./medical_datasets_llama3_improved/test_dataset.json"  # Path to test data
    OUTPUT_FILE = "evaluation_results.json"  # Where to save results
    MAX_SAMPLES = None  # Set to a number (e.g., 1000) to test on subset, or None for full dataset
    # =========================================================
    
    print("="*80)
    print("STANDALONE MODEL EVALUATION")
    print("="*80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Test file: {TEST_FILE}")
    print(f"Max samples: {MAX_SAMPLES if MAX_SAMPLES else 'ALL'}")
    print("="*80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Running on CPU will be very slow!")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Exiting...")
            return False
    else:
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model path not found: {MODEL_PATH}")
        return False
    
    if not os.path.exists(TEST_FILE):
        print(f"❌ ERROR: Test file not found: {TEST_FILE}")
        return False
    
    try:
        # Load the model
        print("\nLoading model...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        print("✅ Model loaded successfully!")
        
        # Test single prediction
        print("\n" + "="*80)
        print("TESTING SINGLE PREDICTION")
        print("="*80)
        
        evaluator = ComprehensiveMedicalEvaluator(model, tokenizer)
        
        test_text = "45 year old male with chest pain and shortness of breath, elevated troponins"
        pred, _ = evaluator.predict(test_text)
        print(f"Input: {test_text}")
        print(f"Prediction: {pred}")
        
        # Run full evaluation
        print("\n" + "="*80)
        print("RUNNING FULL EVALUATION")
        print("="*80)
        
        results = evaluator.evaluate(TEST_FILE, max_samples=MAX_SAMPLES)
        
        # Save results to JSON
        print("\nSaving results...")
        with open(OUTPUT_FILE, "w") as f:
            # Convert numpy types to regular Python types for JSON
            json_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    json_results[k] = {}
                    for inner_k, inner_v in v.items():
                        if isinstance(inner_v, (np.floating, np.integer)):
                            json_results[k][inner_k] = float(inner_v)
                        elif isinstance(inner_v, np.ndarray):
                            json_results[k][inner_k] = inner_v.tolist()
                        elif inner_k == 'similarities' and isinstance(inner_v, dict):
                            # Handle nested similarities dict
                            json_results[k][inner_k] = {}
                            for sim_k, sim_v in inner_v.items():
                                if isinstance(sim_v, list):
                                    json_results[k][inner_k][sim_k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in sim_v]
                                else:
                                    json_results[k][inner_k][sim_k] = sim_v
                        else:
                            json_results[k][inner_k] = inner_v
                else:
                    json_results[k] = v
            
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved to: {OUTPUT_FILE}")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        basic = results['basic_stats']
        print(f"Total Samples: {basic['total_samples']}")
        print(f"Exact Accuracy: {basic['exact_accuracy']:.1%}")
        
        if results.get('clinical_bert_stats'):
            print(f"Clinical BERT Mean: {results['clinical_bert_stats']['mean_score']:.3f}")
        
        if results.get('word_similarity_stats'):
            print(f"Word Similarity Mean: {results['word_similarity_stats']['mean_score']:.3f}")
        
        print("="*80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("MEDICAL DIAGNOSIS MODEL - STANDALONE EVALUATION")
    print("="*80)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
    print("="*80)
    
    success = main()
    
    sys.exit(0 if success else 1)
