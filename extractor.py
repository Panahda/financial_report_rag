#!/usr/bin/env python3
"""
LLM-Focused Entity Extractor for Financial Report RAG System
Primarily uses LLM for accurate extraction of audit findings
"""
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import csv
import torch
from collections import defaultdict
from tqdm import tqdm

import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set seed for reproducibility
torch.manual_seed(0)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMEntityExtractor:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db",
                 model_name: str = "microsoft/Phi-4-mini-instruct",
                 use_quantization: bool = True,
                 use_gpu: bool = None):
        """
        Initialize the LLM-focused EntityExtractor
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

            doc_count = self.collection.count()
            if doc_count == 0:
                logger.warning(f"⚠️ Collection '{collection_name}' is empty")
            else:
                logger.info(f"📊 Collection has {doc_count} documents")

        except Exception as e:
            logger.error(f"❌ Failed to connect to collection: {e}")
            raise

        # Initialize model and tokenizer
        self._initialize_hf_model(use_quantization)
        logger.info(f"🤖 Using model: {model_name}")

        # Create results directory
        self.results_dir = Path(f"results/{collection_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_hf_model(self, use_quantization: bool):
        """Initialize HuggingFace model with optimized settings"""
        logger.info(f"🤖 Loading model: {self.model_name}")

        try:
            if use_quantization and self.device == 'cuda':
                logger.info("Using 4-bit quantization for memory efficiency")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cuda" if self.device == 'cuda' else "cpu",
                    trust_remote_code=True,
                    torch_dtype="auto" if self.device == 'cuda' else torch.float32,
                    low_cpu_mem_usage=True
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"✅ Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise

    def extract_entities_from_text(self, query: str, context_text: str, source_file: str, page: int) -> List[Dict[str, Any]]:
        """
        Runs the LLM on a given text context to extract entities.

        - query: The specific instruction for the LLM (e.g., "Extract all CAR findings").
        - context_text: The full text of the document to be analyzed.
        - source_file: The name of the source file for metadata.
        - page: The starting page number for metadata.
        """
        # 1. Build the prompt for the LLM
        prompt = f"""<|user|>
You are an expert auditor reviewing a financial report. Your task is to extract all specified findings from the provided context.
Strictly adhere to the following instructions:
1.  **Instruction**: {query}
2.  **Context**: Analyze the text below to find all relevant entities.
3.  **Output Format**: Return the findings strictly as a JSON list. Do not include any explanations or introductory text outside of the JSON structure. If no findings are found, return an empty list `[]`.

**Context to Analyze:**
---
{context_text}
---

**Required JSON Schema:**
```json
[
  {{
    "reference": "string (e.g., CAR (1), NCR-05)",
    "category": "string (CAR/NCR/NC, CL/NIR, FAR/OFI)",
    "description": "string (Detailed description of the finding)"
  }}
]
```<|end|><|assistant|>
"""

        # 2. LLM inference
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=2048,  # Increased token limit for potentially large extractions
            temperature=0.1,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode only the newly generated tokens to avoid including the prompt
        output_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        # 3. Extract JSON from model output using a more robust regex
        json_match = re.search(r"```json\s*(\[.*?\])\s*```|(\[.*?\])", output_text, re.DOTALL)
        if not json_match:
            logger.warning(f"No valid JSON found in LLM output for file: '{source_file}'")
            return []

        # The regex has two groups, one for the markdown block and one for raw json.
        json_str = json_match.group(1) or json_match.group(2)
        try:
            extracted_entities = json.loads(json_str)
            # Add source metadata to each extracted entity
            for entity in extracted_entities:
                entity['source_file'] = source_file
                entity['page'] = page
            return extracted_entities
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format from LLM for file '{source_file}': {e}\nRaw output: {json_str}")
            return []

    def extract_entities_from_collection(self, batch_size: int = 50, sample_mode: bool = False) -> Tuple[List[Dict], Dict[str, List]]:
        """
        Processes all documents in the collection to extract entities by grouping chunks by file.
        """
        logger.info("🚀 Starting entity extraction from the entire collection...")

        # 1. Fetch all document chunks from the collection
        total_docs = self.collection.count()
        if total_docs == 0:
            logger.warning("Collection is empty. Nothing to process.")
            return [], defaultdict(list)

        limit = 100 if sample_mode else total_docs
        logger.info(f"Fetching {limit if sample_mode else 'all'} document chunks from collection '{self.collection_name}'...")

        docs = self.collection.get(
            limit=limit,
            include=["metadatas", "documents"]
        )

        # 2. Group chunks by their source file
        docs_by_file = defaultdict(list)
        for i in range(len(docs['ids'])):
            meta = docs['metadatas'][i]
            source_file = meta.get('source', 'unknown_file')
            docs_by_file[source_file].append({
                'content': docs['documents'][i],
                'page': meta.get('page', 0)
            })

        logger.info(f"Grouped {len(docs['ids'])} chunks into {len(docs_by_file)} unique files.")

        all_entities = defaultdict(list)
        file_summaries = []

        # 3. Process each file's complete text
        file_iterator = tqdm(docs_by_file.items(), desc="Processing Files")
        for file_name, file_docs in file_iterator:
            file_iterator.set_postfix_str(file_name)

            # Sort pages and combine all chunks into a single text for context
            sorted_docs = sorted(file_docs, key=lambda x: x['page'])
            full_text = "\n\n".join(doc['content'] for doc in sorted_docs)
            first_page = sorted_docs[0]['page'] if sorted_docs else 0
            
            file_entity_count = {"CAR": 0, "CL": 0, "FAR": 0}

            # Define queries for different entity types
            queries = {
                "CAR": "Extract all Corrective Action Requests (CAR), Non-Conformance Reports (NCR), and Non-Compliances (NC).",
                "CL": "Extract all Clarification Lists (CL) and Needs for Information Requests (NIR).",
                "FAR": "Extract all Forward Action Requests (FAR) and Opportunities for Improvement (OFI)."
            }

            for entity_type, query in queries.items():
                try:
                    # Call the LLM for each type of finding
                    extracted = self.extract_entities_from_text(query, full_text, file_name, first_page)
                    all_entities[entity_type].extend(extracted)
                    file_entity_count[entity_type] += len(extracted)
                except Exception as e:
                    logger.error(f"Failed extraction for {entity_type} in {file_name}: {e}")

            file_summaries.append({
                "file_name": file_name,
                "car_count": file_entity_count["CAR"],
                "cl_count": file_entity_count["CL"],
                "far_count": file_entity_count["FAR"],
                "total_entities": sum(file_entity_count.values())
            })

        # 4. Deduplicate and save the final results
        for entity_type in all_entities:
            all_entities[entity_type] = self._deduplicate_entities(all_entities[entity_type])

        self._save_results(file_summaries, all_entities)

        return file_summaries, all_entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on reference, keeping the best description."""
        seen = {}
        for entity in entities:
            ref = entity.get("reference", "").strip().upper()
            if not ref:
                continue  # Skip entities without a reference

            # If we haven't seen this reference, or the new one has a longer description, keep it.
            if ref not in seen or len(entity.get("description", "")) > len(seen[ref].get("description", "")):
                seen[ref] = entity

        return list(seen.values())

    def _save_results(self, file_summaries: List[Dict], all_entities: Dict[str, List]):
        """Save extraction results to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        detailed_report = {
            "extraction_date": timestamp,
            "collection": self.collection_name,
            "model": self.model_name,
            "extraction_method": "LLM-focused",
            "total_files": len(file_summaries),
            "total_entities": {
                "CAR": len(all_entities["CAR"]),
                "CL": len(all_entities["CL"]),
                "FAR": len(all_entities["FAR"])
            },
            "file_summaries": file_summaries,
            "all_entities": all_entities
        }

        json_path = self.results_dir / f"llm_report_{timestamp}.json"
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        logger.info(f"📁 Saved detailed report: {json_path}")

        # Save each entity type to its own CSV file
        for entity_type in ["CAR", "CL", "FAR"]:
            if all_entities[entity_type]:
                csv_path = self.results_dir / f"{entity_type.lower()}_llm_{timestamp}.csv"
                with open(csv_path, "w", newline="", encoding='utf-8') as f:
                    fieldnames = ["reference", "category", "description", "source_file", "page"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for entity in all_entities[entity_type]:
                        row = {
                            "reference": entity.get("reference", ""),
                            "category": entity.get("category", ""),
                            "description": entity.get("description", "")[:1000], # Limit description length for CSV
                            "source_file": entity.get("source_file", ""),
                            "page": entity.get("page", "")
                        }
                        writer.writerow(row)

                logger.info(f"📁 Saved {entity_type} CSV: {csv_path}")

        # Generate a human-readable summary text file
        summary_path = self.results_dir / f"llm_summary_{timestamp}.txt"
        with open(summary_path, "w", encoding='utf-8') as f:
            f.write("LLM-Focused Entity Extraction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Extraction Date: {timestamp}\n")
            f.write(f"Collection: {self.collection_name}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Method: LLM-focused\n")
            f.write(f"Total Files Processed: {len(file_summaries)}\n\n")

            f.write("Overall Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total CARs/NCRs/NCs: {len(all_entities['CAR'])}\n")
            f.write(f"Total CLs/NIRs: {len(all_entities['CL'])}\n")
            f.write(f"Total FARs/OFIs: {len(all_entities['FAR'])}\n")
            f.write(f"Total Valid Entities: {sum(len(v) for v in all_entities.values())}\n\n")

            files_with_entities = [fs for fs in file_summaries if fs["total_entities"] > 0]
            f.write(f"Files with entities: {len(files_with_entities)}/{len(file_summaries)}\n\n")

            if files_with_entities:
                f.write("Top Files by Entity Count:\n")
                f.write("-" * 30 + "\n")
                sorted_files = sorted(files_with_entities, key=lambda x: x["total_entities"], reverse=True)[:10]
                for fs in sorted_files:
                    f.write(f"\n{fs['file_name']}:\n")
                    f.write(f"   CARs: {fs['car_count']}\n")
                    f.write(f"   CLs: {fs['cl_count']}\n")
                    f.write(f"   FARs: {fs['far_count']}\n")
                    f.write(f"   Total: {fs['total_entities']}\n")

            f.write("\n\nSample Extracted Entities:\n")
            f.write("-" * 30 + "\n")
            for entity_type in ["CAR", "CL", "FAR"]:
                if all_entities[entity_type]:
                    f.write(f"\n{entity_type} Examples (up to 5):\n")
                    for entity in all_entities[entity_type][:5]:
                        ref = entity.get('reference', 'N/A')
                        desc = entity.get('description', 'N/A')[:100]
                        f.write(f"   - {ref}: {desc}...\n")

        logger.info(f"📁 Saved summary report: {summary_path}")

def main():
    """Run the LLM-focused entity extractor from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description='LLM-Focused Entity Extraction from ChromaDB')
    parser.add_argument('--collection', type=str, required=True,
                        help='Name of the ChromaDB collection to process.')
    parser.add_argument('--model', type=str, default='microsoft/Phi-4-mini-instruct',
                        help='HuggingFace model to use for extraction.')
    parser.add_argument('--no-quantization', action='store_true',
                        help='Disable 4-bit quantization (uses more memory).')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage even if a GPU is available.')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for processing documents (currently unused but kept for future).')
    parser.add_argument('--sample', action='store_true',
                        help='Sample mode - process only the first 100 document chunks found.')

    args = parser.parse_args()

    use_quantization = not args.no_quantization
    use_gpu = None if not args.force_cpu else False

    print("\n" + "="*70)
    print("🚀 LLM-Focused Entity Extraction System")
    print("="*70)
    print(f"📊 Collection: {args.collection}")
    print(f"🤖 Model: {args.model}")
    print(f"⚡ Quantization: {'Enabled' if use_quantization else 'Disabled'}")
    if args.sample:
        print("🔍 SAMPLE MODE: Processing first 100 document chunks only")
    print("="*70 + "\n")

    try:
        extractor = LLMEntityExtractor(
            args.collection,
            model_name=args.model,
            use_quantization=use_quantization,
            use_gpu=use_gpu
        )
    except Exception as e:
        logger.error(f"Failed to initialize the extractor. Please check your collection name and model. Error: {e}")
        return

    file_summaries, all_entities = extractor.extract_entities_from_collection(
        batch_size=args.batch_size,
        sample_mode=args.sample
    )

    print(f"\n" + "="*70)
    print("✅ Extraction Complete!")
    print("="*70)
    print(f"📊 Results saved to: results/{args.collection}/")

    if file_summaries:
        total_cars = len(all_entities["CAR"])
        total_cls = len(all_entities["CL"])
        total_fars = len(all_entities["FAR"])

        print(f"\n📈 Extraction Summary:")
        print(f"   Files processed: {len(file_summaries)}")
        print(f"   Files with entities: {sum(1 for fs in file_summaries if fs['total_entities'] > 0)}")
        print(f"   Total CARs/NCRs/NCs: {total_cars}")
        print(f"   Total CLs/NIRs: {total_cls}")
        print(f"   Total FARs/OFIs: {total_fars}")
        print(f"   Total valid entities: {total_cars + total_cls + total_fars}")
    else:
        print("\n⚠️ No documents found in collection to process.")

    print("="*70)

if __name__ == "__main__":
    main()
