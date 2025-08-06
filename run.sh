#!/bin/bash
#SBATCH --job-name=financial_rag_hf
#SBATCH --partition=A5000
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --ntasks=1         
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=sbatch_log/rag_hf_%j.out
#SBATCH --error=sbatch_log/rag_hf_%j.err

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU Info:"
nvidia-smi

# Load modules
module load Miniforge3
module load GCCcore/13.3.0

# Activate environment
conda activate rag

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export HF_HOME="$HOME/.cache/huggingface"

# Clear any potential GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Download model if needed (this caches it for future use)
echo "Pre-downloading model..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'microsoft/Phi-4-mini-instruct'
print(f'Downloading {model_name}...')

try:
    # Just download, don't load to GPU yet
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print('✅ Tokenizer downloaded')
    
    # Download model weights
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print('✅ Model downloaded')
    del model  # Free memory
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f'Warning: Could not pre-download model: {e}')
"

echo "Starting RAG processing with HuggingFace models..."
python runner.py \
    --pdf-dir ./data/all \
    --batch-size 50 \
    --max-files 3 \
    --collection report_3 \
    --extract-entities

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ RAG processing completed successfully"
else
    echo "❌ RAG processing failed"
fi

echo "Job completed at: $(date)"