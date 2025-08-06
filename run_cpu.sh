#!/bin/bash
#SBATCH --job-name=financial_rag_cpu
#SBATCH --partition=A5000
#SBATCH --ntasks=1         
#SBATCH --mem=30G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=sbatch_log/rag_cpu_%j.out
#SBATCH --error=sbatch_log/rag_cpu_%j.err
# NOTE: Removed --gres=gpu:1 to avoid GPU allocation

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Available CPUs: $(nproc)"
echo "Available Memory: $(free -h)"

# Force CPU-only mode from the start
export CUDA_VISIBLE_DEVICES=""
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1

# CRITICAL: Force Ollama to use CPU only
export OLLAMA_CUDA_DISABLE=1
export OLLAMA_NO_GPU=1
export OLLAMA_CPU_ONLY=1

# Load modules
module load Miniforge3
module load GCCcore/13.3.0

echo "Setting up CPU-only environment..."

# Activate environment
conda activate rag

# Install CPU-only ollama if needed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama CPU version..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start ollama in CPU mode with explicit CPU-only flags
echo "Starting Ollama in CPU-only mode..."
ollama serve --gpu-layers 0 &
OLLAMA_PID=$!
echo "Started Ollama with PID: $OLLAMA_PID"

# Wait for ollama to start
echo "Waiting for Ollama to initialize..."
sleep 20

# Test ollama connection
echo "Testing Ollama connection..."
for i in {1..10}; do
    if ollama list >/dev/null 2>&1; then
        echo "✅ Ollama is ready"
        break
    else
        echo "⏳ Waiting for Ollama... (attempt $i/10)"
        sleep 10
    fi
done

# Pull a CPU-optimized model - using the EXACT name your Python expects
echo "Pulling CPU-optimized model..."
OLLAMA_MODEL="llama3.2"  # Small version for CPU

# First try the 1b version
if ollama pull $OLLAMA_MODEL; then
    echo "✅ Successfully pulled $OLLAMA_MODEL"
    # Create an alias for the model name your Python expects
    ollama cp $OLLAMA_MODEL llama3.2
    echo "✅ Created alias 'llama3.2' for $OLLAMA_MODEL"
else
    echo "Trying alternative model..."
    OLLAMA_MODEL="gemma2:2b"
    if ollama pull $OLLAMA_MODEL; then
        echo "✅ Successfully pulled $OLLAMA_MODEL"
        # Create an alias for compatibility
        ollama cp $OLLAMA_MODEL llama3.2
        echo "✅ Created alias 'llama3.2' for $OLLAMA_MODEL"
    else
        echo "❌ Failed to pull models, exiting..."
        kill $OLLAMA_PID
        exit 1
    fi
fi

# List available models
echo "Available models:"
ollama list

# ============================================
# RUN YOUR PYTHON SCRIPT HERE
# ============================================
echo "Starting RAG processing..."

# Example: Process 10 files from the data directory
python runner.py \
    --pdf-dir ./data/all \
    --max-files 10 \
    --batch-size 5 \
    --collection financial_reports_cpu_test \
    --extract-entities

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ RAG processing completed successfully"
else
    echo "❌ RAG processing failed with exit code $?"
fi

# ============================================
# CLEANUP
# ============================================
echo "Cleaning up..."

# Kill ollama server
if [ ! -z "$OLLAMA_PID" ]; then
    echo "Stopping Ollama (PID: $OLLAMA_PID)..."
    kill $OLLAMA_PID 2>/dev/null
    sleep 5
    # Force kill if still running
    kill -9 $OLLAMA_PID 2>/dev/null
fi

echo "Job completed at: $(date)"