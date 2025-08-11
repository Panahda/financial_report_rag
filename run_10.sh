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

echo "========================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "========================================"

# Display GPU info
echo "GPU Information:"
nvidia-smi
echo "========================================"

# Load modules
module load Miniforge3
module load GCCcore/13.3.0

# Activate environment
conda activate rag

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export HF_HOME="$HOME/.cache/huggingface"
export PYTHONUNBUFFERED=1  # Ensure real-time output

# Create necessary directories
mkdir -p sbatch_log
mkdir -p results
mkdir -p chroma_db
mkdir -p processed_chunks

# Clear any potential GPU memory
echo "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')" 2>/dev/null || true

# Check Python and package versions
echo "========================================"
echo "Environment Check:"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo "========================================"

# Set collection name
COLLECTION_NAME="report_3"
PDF_DIR="./data/all"

echo "Starting COMPLETE FRESH RAG pipeline..."
echo "Collection: $COLLECTION_NAME"
echo "PDF Directory: $PDF_DIR"
echo "========================================"

# python runner.py \
#     --pdf-dir "$PDF_DIR" \
#     --collection "$COLLECTION_NAME" \
#     --mode both \
#     --batch-size 200 \
#     --extraction-batch-size 100 \
#     --validate \
#     --quality-check \
#     --max-files 3

    # --reset-collection \# Uncomment if you want to reset the collection

python extractor.py \
    --collection "$COLLECTION_NAME" \
    --batch-size 200

    
# Capture exit code
EXIT_CODE=$?