#!/usr/bin/env python3
"""
Runner for Financial Report RAG using HuggingFace models (no Ollama)
"""
import argparse
import os
from pathlib import Path

# Import the modified RAG class
from main import FinancialReportRAG, DEFAULT_MODEL
from extractor import EntityExtractor

def main():
    parser = argparse.ArgumentParser(description='Process PDFs into RAG system using HuggingFace models')
    parser.add_argument('--pdf-dir', type=str, default='./data/all',
                       help='Directory containing PDF files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (default: all)')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Regex pattern to filter files (e.g., "VCS_.*\.pdf")')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of files to process per batch')
    parser.add_argument('--shuffle', action='store_true',
                       help='Randomly shuffle files before processing')
    parser.add_argument('--collection', type=str, default='financial_reports',
                       help='ChromaDB collection name')
    parser.add_argument('--extract-entities', action='store_true',
                       help='Extract entities after processing')
    parser.add_argument('--skip-processing', action='store_true',
                       help='Skip PDF processing and only extract entities')
    
    # Quantization option
    parser.add_argument('--no-quantization', action='store_true',
                       help='Disable 4-bit quantization (uses more memory)')
    
    # CPU/GPU control
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU-only mode even if GPU is available')
    
    # Reset collection option
    parser.add_argument('--reset-collection', action='store_true',
                       help='Delete and reset the collection before processing')
    
    args = parser.parse_args()
    
    # Fixed model configuration
    model_name = DEFAULT_MODEL  # Always use Phi-4-mini
    use_quantization = not args.no_quantization
    use_gpu = None if not args.force_cpu else False
    
    if not args.skip_processing:
        print("="*60)
        print("🚀 Initializing Financial Report RAG System")
        print("🤗 Using HuggingFace Models (No Ollama Required)")
        print("="*60)
        print(f"📁 PDF Directory: {args.pdf_dir}")
        print(f"📊 Collection: {args.collection}")
        print(f"🤖 Model: {model_name}")
        print(f"⚡ Quantization: {'Enabled (4-bit)' if use_quantization else 'Disabled'}")
        
        if args.reset_collection:
            print(f"🔄 Reset collection: Enabled")
        
        if args.force_cpu:
            print("🖥️ Forcing CPU-only mode")
        
        if args.max_files:
            print(f"🔢 Max files: {args.max_files}")
        if args.pattern:
            print(f"🔍 File pattern: {args.pattern}")
        if args.shuffle:
            print(f"🔀 Random shuffle: enabled")
        
        # Initialize the system with HuggingFace model
        print("\n" + "="*60)
        print("Loading model... (this may take a few minutes)")
        print("="*60)
        
        rag = FinancialReportRAG(
            model_name=model_name,
            collection_name=args.collection,
            persist_directory="./chroma_db",
            use_gpu=use_gpu,
            use_quantization=use_quantization,
            reset_collection=args.reset_collection  # Pass the reset flag
        )
        
        # Set up vector store
        print("\n📊 Setting up vector store...")
        rag.setup_vector_store()
        
        # Verify vector store is ready
        if rag.vector_store is None:
            print("❌ ERROR: Vector store failed to initialize!")
            print("Please check the logs and try again.")
            return
        else:
            print("✅ Vector store ready")
        
        # Process PDFs
        print("\n📄 Processing PDFs...")
        rag.batch_process_pdfs(
            pdf_directory=args.pdf_dir,
            batch_size=args.batch_size,
            max_files=args.max_files,
            file_pattern=args.pattern,
            shuffle=args.shuffle
        )
        
        print("\n✅ Processing complete!")
        
        # Show summary
        summary = rag.generate_summary_report()
        print("\n📊 Summary:")
        print(f"Total chunks in database: {summary['total_documents']}")
        print(f"Entities found (pattern matches):")
        print(f"  - CARs: {summary['entities_found']['CAR']}")
        print(f"  - CLs: {summary['entities_found']['CL']}")
        print(f"  - FARs: {summary['entities_found']['FAR']}")
    
    # Extract entities if requested
    if args.extract_entities or args.skip_processing:
        print("\n" + "="*60)
        print("🔍 Starting Entity Extraction with LLM Analysis")
        print("="*60)
        
        extractor = EntityExtractor(
            collection_name=args.collection,
            model_name=model_name,
            use_quantization=use_quantization,
            use_gpu=use_gpu
        )
        
        file_summaries, all_entities = extractor.extract_entities_from_collection()
        
        print("\n✅ Entity extraction complete!")
        print(f"📊 Results saved to: results/{args.collection}/")
        
        # Print summary statistics
        total_cars = sum(fs['car_count'] for fs in file_summaries)
        total_cls = sum(fs['cl_count'] for fs in file_summaries)
        total_fars = sum(fs['far_count'] for fs in file_summaries)
        
        print(f"\n📈 Entity Extraction Summary:")
        print(f"  Files analyzed: {len(file_summaries)}")
        print(f"  Actual CARs identified: {total_cars}")
        print(f"  Actual CLs identified: {total_cls}")
        print(f"  Actual FARs identified: {total_fars}")
        print(f"  Total entities: {total_cars + total_cls + total_fars}")

if __name__ == "__main__":
    main()