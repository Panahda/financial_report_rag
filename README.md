## Method

### EEE cluster specific
```bash
# new gpu instance (12hrs 16GB)
srun --pty -p A5000 --time=24:00:00 --gres gpu:1 --mem 30G --cpus-per-task 8 bash

module load Miniforge3
```

### Create env
```bash
conda create -n "rag" python=3.10
conda activate rag
```

### install requirements
```bash
pip install gpustat
pip install tqdm
pip install --upgrade transformers==4.51.3 accelerate==1.3.0 bitsandbytes
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install auto-gptq optimum

pip install flash_attn==2.7.4.post1 --no-build-isolation
pip install --upgrade numpy scipy scikit-learn
pip install --upgrade unstructured pdf2image pdfplumber pytesseract
pip install --upgrade pi-heif
pip install --upgrade unstructured-inference
pip install --upgrade pillow pypdf
pip install --upgrade langchain langchain-community chromadb
pip install --upgrade sentence-transformers
pip install --upgrade safetensors huggingface-hub
pip install --upgrade sentencepiece protobuf tiktoken
pip install PyPDF2
conda install -c conda-forge tesseract
conda install -c conda-forge poppler
conda install -c conda-forge pytesseract
pip install unstructured-pytesseract
```

### env settings
```bash
export CHROMADB_SQLITE_SYNCHRONOUS=0
export TOKENIZERS_PARALLELISM=true
```
