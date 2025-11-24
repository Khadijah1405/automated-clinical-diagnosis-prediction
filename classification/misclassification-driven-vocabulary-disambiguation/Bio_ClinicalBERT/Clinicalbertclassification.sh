#!/bin/bash
#SBATCH --requeue
#SBATCH -p capella
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J clinical_bert_h100
#SBATCH -o clinical_bert_h100_%j.log
#SBATCH -e clinical_bert_h100_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G

module purge

# Source conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Error: Conda not found"
    exit 1
fi

# List available environments to debug
echo "Available conda environments:"
conda env list

# Try to activate the environment, create if it doesn't exist
if conda env list | grep -q "^kal "; then
    echo "Activating existing 'kal' environment..."
    conda activate kal
else
    echo "Environment 'kal' not found. Creating it..."
    # Create environment with required packages
    conda create -y -n kal python=3.10
    conda activate kal
    
    # Install required packages
    echo "Installing required packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers peft accelerate datasets wandb numpy pandas scikit-learn
    pip install spacy nltk matplotlib seaborn rouge-score 
    python -m spacy download en_core_web_sm
    
    # Download NLTK data
    echo "Downloading NLTK data..."
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
fi

# Ensure NLTK data is downloaded (even if environment exists)
echo "Verifying NLTK data..."
python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

# Verify Python is available
echo "Python version:"
python --version
echo "Python path:"
which python

# Set environment variables
export WANDB_ENTITY="..."
export WANDB_PROJECT="quickstart_playground"
export WANDB_RUN_GROUP="clinical-bert-classification"
export WANDB_NAME="h100-finetune-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES="0"
export HUGGING_FACE_HUB_TOKEN="..."

# H100 optimization flags
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Check if HF token is available
echo "Checking Hugging Face authentication..."
if python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "✅ Hugging Face authentication successful"
else
    echo "❌ Hugging Face authentication failed"
    echo "Please run 'huggingface-cli login' before submitting the job"
    exit 1
fi

echo "GPU Information:"
nvidia-smi

echo "============================================"
echo "CONFIGURATION SUMMARY"
echo "============================================"
echo "Time limit: 48 hours"
echo "Balanced samples per class: 1200"
echo "Batch size: 96 (H100 optimized)"
echo "Max sequence length: 192"
echo "Analysis samples: 100 (safe for small classes)"
echo "============================================"

python -u automated-clinical-diagnosis-prediction/classification/misclassification-driven-vocabulary-disambiguation/Bio_ClinicalBERT/clinicalbert_classification_pipeline.py \
  --csv_file /hcup_processed_medical_dataset_with_labels.csv \
  --synonyms true \
  --balance true \
  --batch_size 96 \
  --max_length 192 \
  --max_samples_per_class 1200 \
  --samples_for_analysis 100

echo "Job finished: $(date)"
