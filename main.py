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
                 use_quantization: bool = True):  # Add quantization option
    
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
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
                logger.warning("⚠️ Tfesseract OCR not found")
                self.ocr_available = False
        except:
            logger.warning("⚠️ Tesseract OCR not available")
            self.ocr_available = False
    
    def setup_vector_store(self):
        """Initialize or load existing vector store"""
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
        except Exception as e:
            logger.info(f"Creating new collection: {self.collection_name}")
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
                
                # Extract page number if available
                page_num = None
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    page_num = chunk.metadata.get('page_number', None)
                
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
            logger.error(f"❌ Error processing {pdf_path}: {e}")
            
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
                    except:
                        logger.error(f"Failed to process PDF with fallback method")
                        
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
                    else:
                        logger.warning(f"  ⚠️ {pdf_file.name}: No content extracted")
                        
                except Exception as e:
                    logger.error(f"  ❌ {pdf_file.name}: {e}")
            
            # Add batch to vector store
            if batch_documents and self.vector_store:
                try:
                    logger.info(f"Adding {len(batch_documents)} chunks to vector store...")
                    self.vector_store.add_documents(batch_documents)
                    logger.info(f"✅ Batch added to vector store")
                except Exception as e:
                    logger.error(f"Failed to add batch to vector store: {e}")
            
            # Clear memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        logger.info(f"🎉 Processing complete!")
        logger.info(f"   Files processed: {total_processed}/{len(pdf_files)}")
        logger.info(f"   Total chunks created: {total_chunks}")
    
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
            # Get collection stats
            collection = self.chroma_client.get_collection(name=self.collection_name)
            count = collection.count()
            
            # Get all documents to analyze entities
            all_docs = collection.get(
                include=["metadatas"],
                limit=count
            )
            
            # Count entities
            entity_counts = {'CAR': 0, 'CL': 0, 'FAR': 0}
            file_set = set()
            
            for metadata in all_docs['metadatas']:
                file_set.add(metadata.get('file_name', 'Unknown'))
                entities = metadata.get('entities', {})
                for entity_type in entity_counts:
                    entity_counts[entity_type] += len(entities.get(entity_type, []))
            
            return {
                'total_documents': count,
                'unique_files': len(file_set),
                'entities_found': entity_counts,
                'files': list(file_set)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {
                'total_documents': 0,
                'unique_files': 0,
                'entities_found': {'CAR': 0, 'CL': 0, 'FAR': 0},
                'files': []
            }


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG System without Ollama')
    parser.add_argument('--no-quantization', action='store_true',
                       help='Disable 4-bit quantization')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    use_quantization = not args.no_quantization
    use_gpu = None if not args.force_cpu else False
    
    print(f"Initializing RAG with Phi-4-mini...")
    print(f"Model path: {DEFAULT_MODEL}")
    print(f"Quantization: {use_quantization}")
    
    # Initialize RAG
    rag = FinancialReportRAG(
        model_name=DEFAULT_MODEL,
        collection_name="financial_reports",
        use_gpu=use_gpu,
        use_quantization=use_quantization
    )
    
    # Test the model
    test_prompt = "What is a CAR in financial auditing?"
    print(f"\nTest prompt: {test_prompt}")
    response = rag.llm.invoke(test_prompt)
    print(f"Response: {response}")

if __name__ == "__main__":
    main()