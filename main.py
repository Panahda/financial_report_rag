import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime
import subprocess
import torch
import random
import csv

# Core libraries
import chromadb
from chromadb.config import Settings

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# HuggingFace imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# PDF processing imports
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import dict_to_elements

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration - using Phi-4-mini as the default
DEFAULT_MODEL = "microsoft/Phi-4-mini-instruct"

class FinancialReportRAG:
    def __init__(self, 
                 model_name: str = "microsoft/Phi-4-mini-instruct",  
                 collection_name: str = "financial_reports",
                 persist_directory: str = "./chroma_db",
                 use_gpu: bool = None,
                 use_quantization: bool = True,  # Add quantization option
                 save_chunks: bool = True):  # Add option to save chunks
    
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.save_chunks = save_chunks
        
        # Create directory for processed chunks
        if self.save_chunks:
            self.chunks_dir = Path(f"processed_chunks/{collection_name}")
            self.chunks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Chunks will be saved to: {self.chunks_dir}")
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                logger.info("❌ No GPU detected, using CPU")
        
        self.device = 'cuda' if use_gpu else 'cpu'
        
        # Check OCR availability
        self.check_ocr_setup()
        
        # Initialize embeddings
        logger.info(f"🔧 Using device: {self.device} for embeddings")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 128 if use_gpu else 32
            }
        )
        
        # Initialize LLM with HuggingFace model
        self.llm = self._initialize_hf_model(use_quantization)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize stores
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Entity patterns for extraction
        self.entity_patterns = {
            'CAR': [
                r'CAR[-\s]?\d+',
                r'NCR[-\s]?\d+', 
                r'NC[-\s]?\d+',
                r'Corrective\s+Action\s+Request',
                r'Non[-\s]?Conformity\s+Report',
                r'Non[-\s]?Conformance'
            ],
            'CL': [
                r'CL[-\s]?\d+',
                r'NIR[-\s]?\d+',
                r'Clarification\s+Request',
                r'New\s+Information\s+Request',
                r'Information\s+Request'
            ],
            'FAR': [
                r'FAR[-\s]?\d+',
                r'OFI[-\s]?\d+',
                r'Forward\s+Action\s+Request',
                r'Opportunity\s+for\s+Improvement',
                r'Observation'
            ]
        }
    
    def _initialize_hf_model(self, use_quantization: bool):
        """Initialize HuggingFace model without Ollama"""
        logger.info(f"🤖 Loading model: {self.model_name}")
        
        try:
            # Configure quantization for memory efficiency
            if use_quantization and self.device == 'cuda':
                logger.info("Using 4-bit quantization for memory efficiency")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                # Load model normally (CPU or non-quantized GPU)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if self.device == 'cuda' else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float32 if self.device == 'cpu' else torch.float16,
                    low_cpu_mem_usage=True
                )
                
                if self.device == 'cpu':
                    model = model.to('cpu')
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                device_map="auto" if self.device == 'cuda' else None
            )
            
            # Wrap in LangChain
            llm = HuggingFacePipeline(pipeline=pipe)
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            logger.info("Falling back to a smaller model...")
            
            # Fallback to a smaller model
            try:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                
                model_name = "gpt2"  # Very small model for testing
                model = GPT2LMHeadModel.from_pretrained(model_name)
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=256,
                    temperature=0.1
                )
                
                llm = HuggingFacePipeline(pipeline=pipe)
                logger.info(f"✅ Fallback model (GPT-2) loaded")
                return llm
                
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def check_ocr_setup(self):
        """Check if OCR tools are properly configured"""
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Tesseract OCR available")
                self.ocr_available = True
            else:
                logger.warning("⚠️ Tesseract OCR not found")
                self.ocr_available = False
        except:
            logger.warning("⚠️ Tesseract OCR not available")
            self.ocr_available = False
    
    def setup_vector_store(self):
        """Initialize or load existing vector store"""
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
        except Exception as e:
            logger.info(f"Creating new collection: {self.collection_name}")
            # Create new collection
            self.chroma_client.create_collection(name=self.collection_name)
            
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file and extract content"""
        logger.info(f"Processing PDF: {pdf_path}")
        documents = []
        
        try:
            # Use unstructured to partition PDF
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res" if self.ocr_available else "fast",
                infer_table_structure=True,
                include_page_breaks=True,
                extract_images_in_pdf=False  # Disable image extraction for now
            )
            
            # Chunk elements by title
            chunks = chunk_by_title(
                elements,
                max_characters=1500,
                combine_text_under_n_chars=500,
                new_after_n_chars=1200
            )
            
            # Convert chunks to LangChain documents
            for i, chunk in enumerate(chunks):
                # Extract text content
                text = str(chunk)
                
                # Skip empty chunks
                if not text.strip():
                    continue
                
                # Extract page number if available - FIX: Handle ElementMetadata object properly
                page_num = None
                if hasattr(chunk, 'metadata'):
                    metadata_obj = chunk.metadata
                    if metadata_obj is not None:
                        # ElementMetadata is an object with attributes, not a dictionary
                        if hasattr(metadata_obj, 'page_number'):
                            page_num = metadata_obj.page_number
                        elif hasattr(metadata_obj, '__dict__'):
                            # Try to access via __dict__ if it exists
                            page_num = metadata_obj.__dict__.get('page_number', None)
                
                # Search for entities in the text
                entities_found = self.extract_entities_from_text(text)
                
                # Create document
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': pdf_path,
                        'file_name': os.path.basename(pdf_path),
                        'chunk_index': i,
                        'page': page_num,
                        'entities': entities_found,
                        'processed_date': datetime.now().isoformat()
                    }
                )
                documents.append(doc)
            
            logger.info(f"✅ Extracted {len(documents)} chunks from {os.path.basename(pdf_path)}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path} with unstructured: {e}")
            
            # Fallback: Try simple text extraction
            try:
                logger.info("Attempting fallback text extraction...")
                with open(pdf_path, 'rb') as file:
                    # Try using PyPDF2 as fallback
                    try:
                        import PyPDF2
                        pdf_reader = PyPDF2.PdfReader(file)
                        
                        for page_num, page in enumerate(pdf_reader.pages):
                            text = page.extract_text()
                            if text.strip():
                                # Split into chunks
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=1500,
                                    chunk_overlap=200
                                )
                                chunks = text_splitter.split_text(text)
                                
                                for i, chunk_text in enumerate(chunks):
                                    entities_found = self.extract_entities_from_text(chunk_text)
                                    doc = Document(
                                        page_content=chunk_text,
                                        metadata={
                                            'source': pdf_path,
                                            'file_name': os.path.basename(pdf_path),
                                            'page': page_num + 1,
                                            'chunk_index': i,
                                            'entities': entities_found,
                                            'processed_date': datetime.now().isoformat()
                                        }
                                    )
                                    documents.append(doc)
                        
                        logger.info(f"✅ Fallback extraction successful: {len(documents)} chunks")
                    except Exception as pypdf_error:
                        logger.error(f"Failed to process PDF with fallback method: {pypdf_error}")
                        
            except Exception as e2:
                logger.error(f"Fallback extraction failed: {e2}")
        
        return documents
    
    def extract_entities_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using regex patterns"""
        found_entities = {'CAR': [], 'CL': [], 'FAR': []}
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found_entities[entity_type].extend(matches)
        
        # Remove duplicates
        for entity_type in found_entities:
            found_entities[entity_type] = list(set(found_entities[entity_type]))
        
        return found_entities
    
    def _save_chunks_to_file(self, pdf_name: str, documents: List[Document], all_chunks_data: List[Dict]):
        """Save chunks from a PDF to individual files and collect for consolidated file"""
        try:
            # Create subdirectory for this PDF
            pdf_dir = self.chunks_dir / pdf_name.replace('.pdf', '')
            pdf_dir.mkdir(parents=True, exist_ok=True)
            
            # Save individual chunks
            for i, doc in enumerate(documents):
                chunk_data = {
                    'file_name': pdf_name,
                    'chunk_index': i,
                    'page': doc.metadata.get('page', 'unknown'),
                    'content': doc.page_content,
                    'entities': doc.metadata.get('entities', {}),
                    'content_length': len(doc.page_content),
                    'processed_date': doc.metadata.get('processed_date', datetime.now().isoformat())
                }
                
                # Save individual chunk as JSON
                chunk_file = pdf_dir / f"chunk_{i:04d}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, indent=2, ensure_ascii=False)
                
                # Add to consolidated data
                all_chunks_data.append(chunk_data)
            
            # Save summary for this PDF
            summary = {
                'pdf_name': pdf_name,
                'total_chunks': len(documents),
                'total_characters': sum(len(doc.page_content) for doc in documents),
                'pages_covered': list(set(doc.metadata.get('page', 'unknown') for doc in documents)),
                'entities_found': {
                    'CAR': sum(len(doc.metadata.get('entities', {}).get('CAR', [])) for doc in documents),
                    'CL': sum(len(doc.metadata.get('entities', {}).get('CL', [])) for doc in documents),
                    'FAR': sum(len(doc.metadata.get('entities', {}).get('FAR', [])) for doc in documents)
                },
                'processed_date': datetime.now().isoformat()
            }
            
            summary_file = pdf_dir / 'summary.json'
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"  💾 Saved {len(documents)} chunks to {pdf_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save chunks for {pdf_name}: {e}")
    
    def _save_consolidated_chunks(self, all_chunks_data: List[Dict]):
        """Save all chunks to a consolidated file"""
        try:
            # Create consolidated JSON file with all chunks
            consolidated_file = self.chunks_dir / f"all_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            consolidated_data = {
                'collection_name': self.collection_name,
                'total_chunks': len(all_chunks_data),
                'total_files': len(set(chunk['file_name'] for chunk in all_chunks_data)),
                'extraction_date': datetime.now().isoformat(),
                'chunks': all_chunks_data
            }
            
            with open(consolidated_file, 'w', encoding='utf-8') as f:
                json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
            
            # Also create a CSV for easy viewing
            csv_file = self.chunks_dir / f"chunks_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'file_name', 'chunk_index', 'page', 'content_length',
                    'car_count', 'cl_count', 'far_count', 'content_preview'
                ])
                writer.writeheader()
                
                for chunk in all_chunks_data:
                    entities = chunk.get('entities', {})
                    writer.writerow({
                        'file_name': chunk['file_name'],
                        'chunk_index': chunk['chunk_index'],
                        'page': chunk.get('page', 'unknown'),
                        'content_length': chunk['content_length'],
                        'car_count': len(entities.get('CAR', [])),
                        'cl_count': len(entities.get('CL', [])),
                        'far_count': len(entities.get('FAR', [])),
                        'content_preview': chunk['content'][:100] + '...' if len(chunk['content']) > 100 else chunk['content']
                    })
            
            logger.info(f"📊 Saved consolidated chunks to:")
            logger.info(f"   JSON: {consolidated_file}")
            logger.info(f"   CSV: {csv_file}")
            
        except Exception as e:
            logger.error(f"Failed to save consolidated chunks: {e}")
    
    def batch_process_pdfs(self, 
                          pdf_directory: str,
                          batch_size: int = 10,
                          max_files: Optional[int] = None,
                          file_pattern: Optional[str] = None,
                          shuffle: bool = False) -> None:
        """Process multiple PDFs in batches"""
        pdf_dir = Path(pdf_directory)
        
        # Get all PDF files
        if file_pattern:
            pdf_files = [f for f in pdf_dir.glob("*.pdf") if re.match(file_pattern, f.name)]
        else:
            pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return
        
        # Shuffle if requested
        if shuffle:
            random.shuffle(pdf_files)
        
        # Limit files if specified
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files to process")
        
        # Ensure vector store is initialized
        if not self.vector_store:
            logger.error("Vector store not initialized! Please run setup_vector_store() first.")
            return
        
        # Initialize chunks storage
        all_chunks_data = []
        
        # Process in batches
        total_processed = 0
        total_chunks = 0
        
        for batch_start in range(0, len(pdf_files), batch_size):
            batch_end = min(batch_start + batch_size, len(pdf_files))
            batch_files = pdf_files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                       f"Files {batch_start + 1}-{batch_end} of {len(pdf_files)}")
            
            batch_documents = []
            
            for pdf_file in batch_files:
                try:
                    # Process individual PDF
                    documents = self.process_pdf(str(pdf_file))
                    
                    if documents:
                        batch_documents.extend(documents)
                        total_processed += 1
                        total_chunks += len(documents)
                        logger.info(f"  ✅ {pdf_file.name}: {len(documents)} chunks")
                        
                        # Save chunks if enabled
                        if self.save_chunks:
                            self._save_chunks_to_file(pdf_file.name, documents, all_chunks_data)
                    else:
                        logger.warning(f"  ⚠️ {pdf_file.name}: No content extracted")
                        
                except Exception as e:
                    logger.error(f"  ❌ {pdf_file.name}: {e}")
            
            # Add batch to vector store
            if batch_documents:
                try:
                    logger.info(f"Adding {len(batch_documents)} chunks to vector store...")
                    
                    # Add documents in smaller sub-batches to avoid memory issues
                    sub_batch_size = 20
                    for i in range(0, len(batch_documents), sub_batch_size):
                        sub_batch = batch_documents[i:i+sub_batch_size]
                        self.vector_store.add_documents(sub_batch)
                        logger.info(f"  Added sub-batch {i//sub_batch_size + 1}: {len(sub_batch)} documents")
                    
                    logger.info(f"✅ Batch added to vector store")
                    
                    # Persist the collection to ensure data is saved
                    self.vector_store.persist()
                    
                except Exception as e:
                    logger.error(f"Failed to add batch to vector store: {e}")
                    logger.error(f"Error details: {str(e)}")
            
            # Clear memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        # Save consolidated chunks file
        if self.save_chunks and all_chunks_data:
            self._save_consolidated_chunks(all_chunks_data)
        
        # Final persist to ensure all data is saved
        if self.vector_store:
            try:
                self.vector_store.persist()
                logger.info("✅ Vector store persisted successfully")
            except Exception as e:
                logger.error(f"Failed to persist vector store: {e}")
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"   Files processed: {total_processed}/{len(pdf_files)}")
        logger.info(f"   Total chunks created: {total_chunks}")
        if self.save_chunks:
            logger.info(f"   Chunks saved to: {self.chunks_dir}")
    
    def setup_qa_chain(self):
        """Set up the QA chain for querying"""
        if not self.retriever:
            raise ValueError("Vector store not initialized. Run setup_vector_store() first.")
        
        # Define the prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        logger.info("✅ QA chain initialized")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.qa_chain:
            self.setup_qa_chain()
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            # Extract source information
            sources = []
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    sources.append({
                        'file': doc.metadata.get('file_name', 'Unknown'),
                        'page': doc.metadata.get('page', 'Unknown'),
                        'content': doc.page_content[:200] + "..."
                    })
            
            return {
                'answer': result.get('result', 'No answer found'),
                'sources': sources
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'answer': f"Error processing query: {e}",
                'sources': []
            }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of the collection"""
        try:
            # Get collection
            collection = self.chroma_client.get_collection(name=self.collection_name)
            count = collection.count()
            
            logger.info(f"Collection '{self.collection_name}' has {count} documents")
            
            # Get sample of documents to analyze entities (limit to avoid memory issues)
            sample_size = min(count, 1000)
            
            if count > 0:
                all_docs = collection.get(
                    include=["metadatas"],
                    limit=sample_size
                )
                
                # Count entities
                entity_counts = {'CAR': 0, 'CL': 0, 'FAR': 0}
                file_set = set()
                
                for metadata in all_docs['metadatas']:
                    file_set.add(metadata.get('file_name', 'Unknown'))
                    entities = metadata.get('entities', {})
                    for entity_type in entity_counts:
                        entity_counts[entity_type] += len(entities.get(entity_type, []))
                
                result = {
                    'total_documents': count,
                    'unique_files': len(file_set),
                    'entities_found': entity_counts,
                    'files': list(file_set),
                    'sample_size': sample_size
                }
                
                # Add chunks directory info if saving chunks
                if self.save_chunks and self.chunks_dir.exists():
                    chunk_files = list(self.chunks_dir.glob('**/*.json'))
                    result['chunks_saved'] = len(chunk_files)
                    result['chunks_directory'] = str(self.chunks_dir)
                
                return result
            else:
                return {
                    'total_documents': 0,
                    'unique_files': 0,
                    'entities_found': {'CAR': 0, 'CL': 0, 'FAR': 0},
                    'files': [],
                    'sample_size': 0
                }
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            logger.error(f"Error details: {str(e)}")
            return {
                'total_documents': 0,
                'unique_files': 0,
                'entities_found': {'CAR': 0, 'CL': 0, 'FAR': 0},
                'files': [],
                'error': str(e)
            }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG System without Ollama')
    parser.add_argument('--no-quantization', action='store_true',
                       help='Disable 4-bit quantization')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage')
    parser.add_argument('--no-save-chunks', action='store_true',
                       help='Do not save processed chunks to files')
    
    args = parser.parse_args()
    
    use_quantization = not args.no_quantization
    use_gpu = None if not args.force_cpu else False
    save_chunks = not args.no_save_chunks
    
    print(f"Initializing RAG with Phi-4-mini...")
    print(f"Model path: {DEFAULT_MODEL}")
    print(f"Quantization: {use_quantization}")
    print(f"Save chunks: {save_chunks}")
    
    # Initialize RAG
    rag = FinancialReportRAG(
        model_name=DEFAULT_MODEL,
        collection_name="financial_reports",
        use_gpu=use_gpu,
        use_quantization=use_quantization,
        save_chunks=save_chunks
    )
    
    # Test the model
    test_prompt = "What is a CAR in financial auditing?"
    print(f"\nTest prompt: {test_prompt}")
    response = rag.llm.invoke(test_prompt)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()