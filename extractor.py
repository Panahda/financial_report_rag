#!/usr/bin/env python3
"""
Entity Extractor for Financial Report RAG System using HuggingFace models
Extracts and analyzes CARs, CLs, and FARs from the vector database
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import csv
import torch

import chromadb
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db", 
                 model_name: str = "microsoft/Phi-4-mini-instruct",
                 use_quantization: bool = True,
                 use_gpu: bool = None):
        """
        Initialize the EntityExtractor with HuggingFace models
        
        Args:
            collection_name: Name of the ChromaDB collection to extract from
            persist_directory: Directory where ChromaDB is persisted
            model_name: HuggingFace model to use for extraction
            use_quantization: Whether to use 4-bit quantization
            use_gpu: Whether to use GPU (None for auto-detect)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("❌ No GPU detected, using CPU")
        
        self.device = 'cuda' if use_gpu else 'cpu'
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"✅ Connected to collection: {collection_name}")
            
            # Check if collection has documents
            doc_count = self.collection.count()
            if doc_count == 0:
                logger.warning(f"⚠️ Collection '{collection_name}' is empty (0 documents)")
            else:
                logger.info(f"📊 Collection has {doc_count} documents")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to collection: {e}")
            raise
        
        # Initialize LLM with HuggingFace model
        self.llm = self._initialize_hf_model(use_quantization)
        logger.info(f"🤖 Using model: {model_name}")
        
        # Create results directory
        self.results_dir = Path(f"results/{collection_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Entity extraction prompt
        self.extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""You are an expert auditor analyzing financial audit reports. Extract ONLY actual audit findings from the following text.

Text to analyze:
{text}

Identify and extract:
1. CARs (Corrective Action Requests) - Also known as NCRs (Non-Conformity Reports) or NCs
2. CLs (Clarification Requests) - Also known as NIRs (New Information Requests)  
3. FARs (Forward Action Requests) - Also known as OFIs (Opportunities for Improvement)

For each finding, extract:
- Reference number (e.g., CAR-001, NCR-2023-05, CL-42)
- Brief description (1-2 sentences max)
- Category (CAR, CL, or FAR)

Return ONLY a JSON object with this structure:
{{
  "entities": [
    {{
      "reference": "XXX-###",
      "description": "Brief description",
      "category": "CAR|CL|FAR"
    }}
  ]
}}

If no entities are found, return: {{"entities": []}}

IMPORTANT: Only extract items that are clearly identified as audit findings. Do not make up or infer entities."""
        )
    
    def _initialize_hf_model(self, use_quantization: bool):
        """Initialize HuggingFace model (same as in main_hf.py)"""
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
            raise
    
    def extract_entities_from_chunk(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from a single text chunk using LLM"""
        try:
            prompt = self.extraction_prompt.format(text=text[:3000])  # Limit text length
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            try:
                result = json.loads(response)
                return result.get("entities", [])
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        return result.get("entities", [])
                    except:
                        pass
                logger.warning(f"Failed to parse LLM response as JSON")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_entities_from_collection(self) -> Tuple[List[Dict], Dict[str, List]]:
        """Extract all entities from the collection"""
        logger.info("Starting entity extraction from collection...")
        
        # Check if collection is empty
        doc_count = self.collection.count()
        if doc_count == 0:
            logger.warning("⚠️ Collection is empty. No documents to process.")
            # Return empty results
            self._save_results([], {"CAR": [], "CL": [], "FAR": []})
            return [], {"CAR": [], "CL": [], "FAR": []}
        
        # Get all documents from collection
        try:
            # Get documents in batches
            all_docs = []
            batch_size = 100
            
            # Use the actual count from the collection
            total_batches = (doc_count + batch_size - 1) // batch_size
            
            for batch_num in range(total_batches):
                offset = batch_num * batch_size
                limit = min(batch_size, doc_count - offset)
                
                logger.info(f"Retrieving batch {batch_num + 1}/{total_batches} (limit: {limit}, offset: {offset})")
                
                # ChromaDB's get method - proper parameter order
                batch = self.collection.get(
                    limit=limit,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                
                if batch and batch.get("documents"):
                    for doc, meta in zip(batch["documents"], batch["metadatas"]):
                        all_docs.append((doc, meta))
                    logger.info(f"  Retrieved {len(batch['documents'])} documents in this batch")
                else:
                    logger.warning(f"  No documents in batch {batch_num + 1}")
            
            logger.info(f"Total documents retrieved: {len(all_docs)}")
            
            if not all_docs:
                logger.warning("No documents retrieved from collection")
                self._save_results([], {"CAR": [], "CL": [], "FAR": []})
                return [], {"CAR": [], "CL": [], "FAR": []}
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            self._save_results([], {"CAR": [], "CL": [], "FAR": []})
            return [], {"CAR": [], "CL": [], "FAR": []}
        
        # Group documents by file
        docs_by_file = {}
        for doc_text, metadata in all_docs:
            file_name = metadata.get("file_name", "unknown") if metadata else "unknown"
            if file_name not in docs_by_file:
                docs_by_file[file_name] = []
            docs_by_file[file_name].append((doc_text, metadata))
        
        logger.info(f"Processing {len(docs_by_file)} unique files")
        
        # Extract entities from each file
        all_entities = {"CAR": [], "CL": [], "FAR": []}
        file_summaries = []
        
        for file_idx, (file_name, file_docs) in enumerate(docs_by_file.items(), 1):
            logger.info(f"Processing file {file_idx}/{len(docs_by_file)}: {file_name}")
            
            file_entities = {"CAR": [], "CL": [], "FAR": []}
            
            # Process each chunk from the file
            for chunk_idx, (text, metadata) in enumerate(file_docs):
                if chunk_idx % 10 == 0:
                    logger.info(f"  Processing chunk {chunk_idx}/{len(file_docs)}")
                
                # Extract entities using LLM
                entities = self.extract_entities_from_chunk(text)
                
                # Categorize entities
                for entity in entities:
                    category = entity.get("category", "").upper()
                    if category in file_entities:
                        entity["source_file"] = file_name
                        entity["page"] = metadata.get("page", "unknown") if metadata else "unknown"
                        file_entities[category].append(entity)
                        all_entities[category].append(entity)
            
            # Create file summary
            file_summary = {
                "file_name": file_name,
                "car_count": len(file_entities["CAR"]),
                "cl_count": len(file_entities["CL"]),
                "far_count": len(file_entities["FAR"]),
                "total_entities": sum(len(v) for v in file_entities.values()),
                "entities": file_entities
            }
            file_summaries.append(file_summary)
            
            # Log progress
            logger.info(f"  Found: {file_summary['car_count']} CARs, "
                       f"{file_summary['cl_count']} CLs, "
                       f"{file_summary['far_count']} FARs")
        
        # Save results
        self._save_results(file_summaries, all_entities)
        
        return file_summaries, all_entities
    
    def _save_results(self, file_summaries: List[Dict], all_entities: Dict[str, List]):
        """Save extraction results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        detailed_report = {
            "extraction_date": timestamp,
            "collection": self.collection_name,
            "model": self.model_name,
            "total_files": len(file_summaries),
            "total_entities": {
                "CAR": len(all_entities["CAR"]),
                "CL": len(all_entities["CL"]),
                "FAR": len(all_entities["FAR"])
            },
            "file_summaries": file_summaries,
            "all_entities": all_entities
        }
        
        json_path = self.results_dir / f"detailed_report_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(detailed_report, f, indent=2)
        logger.info(f"📁 Saved detailed report: {json_path}")
        
        # Save CSV files for each entity type (only if entities exist)
        for entity_type in ["CAR", "CL", "FAR"]:
            if all_entities[entity_type]:
                csv_path = self.results_dir / f"{entity_type.lower()}_entities_{timestamp}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["reference", "description", 
                                                           "source_file", "page"])
                    writer.writeheader()
                    writer.writerows(all_entities[entity_type])
                logger.info(f"📁 Saved {entity_type} CSV: {csv_path}")
        
        # Save summary report
        summary_path = self.results_dir / f"summary_report_{timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write(f"Entity Extraction Summary\n")
            f.write(f"=" * 50 + "\n\n")
            f.write(f"Extraction Date: {timestamp}\n")
            f.write(f"Collection: {self.collection_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Total Files Processed: {len(file_summaries)}\n\n")
            
            f.write(f"Overall Statistics:\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"Total CARs: {len(all_entities['CAR'])}\n")
            f.write(f"Total CLs: {len(all_entities['CL'])}\n")
            f.write(f"Total FARs: {len(all_entities['FAR'])}\n")
            f.write(f"Total Entities: {sum(len(v) for v in all_entities.values())}\n\n")
            
            if file_summaries:
                f.write(f"Per-File Breakdown:\n")
                f.write(f"-" * 30 + "\n")
                for fs in sorted(file_summaries, key=lambda x: x["total_entities"], reverse=True):
                    if fs["total_entities"] > 0:
                        f.write(f"\n{fs['file_name']}:\n")
                        f.write(f"  CARs: {fs['car_count']}\n")
                        f.write(f"  CLs: {fs['cl_count']}\n")
                        f.write(f"  FARs: {fs['far_count']}\n")
            else:
                f.write("No files were processed (collection is empty)\n")
        
        logger.info(f"📁 Saved summary report: {summary_path}")

def main():
    """Test the entity extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract entities from ChromaDB collection')
    parser.add_argument('--collection', type=str, required=True,
                       help='ChromaDB collection name')
    parser.add_argument('--model', type=str, default='microsoft/Phi-4-mini-instruct',
                       help='HuggingFace model to use')
    parser.add_argument('--no-quantization', action='store_true',
                       help='Disable 4-bit quantization')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage')
    
    args = parser.parse_args()
    
    use_quantization = not args.no_quantization
    use_gpu = None if not args.force_cpu else False
    
    extractor = EntityExtractor(
        args.collection, 
        model_name=args.model,
        use_quantization=use_quantization,
        use_gpu=use_gpu
    )
    file_summaries, all_entities = extractor.extract_entities_from_collection()
    
    print(f"\n✅ Extraction complete!")
    print(f"📊 Results saved to: results/{args.collection}/")
    
    # Print summary
    if file_summaries:
        total_cars = sum(fs['car_count'] for fs in file_summaries)
        total_cls = sum(fs['cl_count'] for fs in file_summaries)
        total_fars = sum(fs['far_count'] for fs in file_summaries)
        
        print(f"\nSummary:")
        print(f"  Files processed: {len(file_summaries)}")
        print(f"  Total CARs: {total_cars}")
        print(f"  Total CLs: {total_cls}")
        print(f"  Total FARs: {total_fars}")
    else:
        print("\nNo documents found in collection to process.")

if __name__ == "__main__":
    main()