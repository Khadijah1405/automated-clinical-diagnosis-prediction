#!/bin/bash
#SBATCH --requeue
#SBATCH -p capella
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J medical
#SBATCH -o classificationfull1.log
#SBATCH -e classificationfull1.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G

module purge

# Check if conda is available and source it properly
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Error: Conda not found in expected locations"
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
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers peft accelerate datasets wandb numpy pandas scikit-learn
    pip install spacy nltk
    python -m spacy download en_core_web_sm

    # Download NLTK data (stopwords and other common datasets)
    python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
        python -m spacy download en_core_web_sm
fi

# Verify Python is available
echo "Python version:"
python --version
echo "Python path:"
which python

# Set environment variables
export WANDB_ENTITY=""
export WANDB_PROJECT="quickstart_playground"
export WANDB_RUN_GROUP="llama3-gen-med"
export WANDB_NAME="finetune-${SLURM_JOB_ID}"
export CUDA_VISIBLE_DEVICES="0"
export HUGGING_FACE_HUB_TOKEN=""

# Check if HF token is available
echo "Checking Hugging Face authentication..."
if python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "✅ Hugging Face authentication successful"
else
    echo "❌ Hugging Face authentication failed"
    echo "Please run 'huggingface-cli login' before submitting the job"
    exit 1
fi

# ============================================================================
# Run Main Pipeline
# ============================================================================

echo "=================================="
echo "Starting Main Pipeline"
echo "=================================="
echo ""

# Run the main script with unbuffered output
python -u automated-clinical-diagnosis-prediction/classification/misclassification-driven-vocabulary-disambiguation/DeBERTa/main_memory_optimized.py \ 
  --csv_file /hcup_processed_medical_dataset_with_labels.csv \
  --synonyms true \
  --balance true \
  --batch_size 16 \
  --max_length 128

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=================================="
echo "Job Complete"
echo "=================================="
echo "Exit code: $EXIT_CODE"
echo "Finished at: $(date)"
echo "=================================="

# Exit with the same code as the Python script
exit $EXIT_CODE
