#!/usr/bin/env python3
"""
Enhanced Runner for Financial Report RAG with Improved Entity Extraction
"""
import argparse
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict

# Import the RAG class and improved extractor
from main import FinancialReportRAG, DEFAULT_MODEL
from extractor import LLMEntityExtractor

def run_extraction_validation(collection_name: str, model_name: str, use_quantization: bool, use_gpu: bool):
    """Run extraction and validate results"""
    print("\n" + "="*60)
    print("🔍 Running Entity Extraction Validation")
    print("="*60)
    
    extractor = LLMEntityExtractor(
        collection_name=collection_name,
        model_name=model_name,
        use_quantization=use_quantization,
        use_gpu=use_gpu
    )
    
    # Run extraction
    file_summaries, all_entities = extractor.extract_entities_from_collection()
    
    # Validation statistics
    validation_stats = {
        "total_files": len(file_summaries),
        "files_with_entities": sum(1 for fs in file_summaries if fs["total_entities"] > 0),
        "total_cars": len(all_entities["CAR"]),
        "total_cls": len(all_entities["CL"]),
        "total_fars": len(all_entities["FAR"]),
        "total_entities": sum(len(v) for v in all_entities.values()),
        "avg_entities_per_file": sum(fs["total_entities"] for fs in file_summaries) / len(file_summaries) if file_summaries else 0
    }
    
    # Check for common issues
    issues = []
    
    # Check if extraction rate is too low
    if validation_stats["total_entities"] < validation_stats["total_files"] * 0.5:
        issues.append("Low extraction rate - fewer than 0.5 entities per file on average")
    
    # Check for imbalanced categories
    total = validation_stats["total_entities"]
    if total > 0:
        car_pct = validation_stats["total_cars"] / total * 100
        cl_pct = validation_stats["total_cls"] / total * 100
        far_pct = validation_stats["total_fars"] / total * 100
        
        if car_pct > 80 or cl_pct > 80 or far_pct > 80:
            issues.append(f"Imbalanced categories - CAR: {car_pct:.1f}%, CL: {cl_pct:.1f}%, FAR: {far_pct:.1f}%")
    
    # Print validation report
    print("\n📊 Validation Report:")
    print("-" * 40)
    print(f"Files processed: {validation_stats['total_files']}")
    print(f"Files with entities: {validation_stats['files_with_entities']}")
    print(f"Total entities extracted: {validation_stats['total_entities']}")
    print(f"Average entities per file: {validation_stats['avg_entities_per_file']:.2f}")
    print(f"\nBreakdown by category:")
    print(f"  CARs/NCRs/NCs: {validation_stats['total_cars']}")
    print(f"  CLs/NIRs: {validation_stats['total_cls']}")
    print(f"  FARs/OFIs: {validation_stats['total_fars']}")
    
    if issues:
        print("\n⚠️ Potential Issues Detected:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No major issues detected")
    
    return validation_stats, all_entities

def run_quality_check(all_entities: Dict, sample_size: int = 10):
    """Perform quality check on extracted entities"""
    print("\n" + "="*60)
    print("🔬 Quality Check on Sample Entities")
    print("="*60)
    
    for entity_type in ["CAR", "CL", "FAR"]:
        if all_entities[entity_type]:
            print(f"\n{entity_type} Samples (up to {sample_size}):")
            print("-" * 40)
            
            samples = all_entities[entity_type][:sample_size]
            for i, entity in enumerate(samples, 1):
                ref = entity.get("reference", "N/A")
                desc = entity.get("description", "N/A")[:150]
                file = entity.get("source_file", "N/A")
                
                print(f"{i}. Reference: {ref}")
                print(f"   File: {file}")
                print(f"   Description: {desc}...")
                print()

def main():
    parser = argparse.ArgumentParser(description='RAG System with Improved Entity Extraction')
    
    # Data processing arguments
    parser.add_argument('--pdf-dir', type=str, default='./data/all',
                       help='Directory containing PDF files')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of files to process per batch')
    parser.add_argument('--collection', type=str, default='financial_reports',
                       help='ChromaDB collection name')
    
    # Processing modes
    parser.add_argument('--mode', type=str, choices=['process', 'extract', 'both', 'validate'],
                       default='both',
                       help='Mode: process PDFs, extract entities, both, or validate existing')
    parser.add_argument('--reset-collection', action='store_true',
                       help='Reset the collection before processing')
    
    # Model configuration
    parser.add_argument('--no-quantization', action='store_true',
                       help='Disable 4-bit quantization')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU-only mode')
    parser.add_argument('--no-progress', action='store_true',
                       help='Disable progress monitoring')
    
    # Extraction options
    parser.add_argument('--extraction-batch-size', type=int, default=50,
                       help='Batch size for entity extraction')
    parser.add_argument('--quality-check', action='store_true',
                       help='Perform quality check on extracted entities')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after extraction')
    
    args = parser.parse_args()
    
    # Configuration
    model_name = DEFAULT_MODEL
    use_quantization = not args.no_quantization
    use_gpu = None if not args.force_cpu else False
    show_progress = not args.no_progress
    
    print("="*70)
    print("🚀 Enhanced Financial Report RAG System")
    print("="*70)
    print(f"📁 PDF Directory: {args.pdf_dir}")
    print(f"📊 Collection: {args.collection}")
    print(f"🤖 Model: {model_name}")
    print(f"⚡ Quantization: {'Enabled (4-bit)' if use_quantization else 'Disabled'}")
    print(f"🔄 Mode: {args.mode}")
    print("="*70)
    
    # Process PDFs if needed
    if args.mode in ['process', 'both']:
        print("\n📄 Processing PDFs into Vector Store...")
        print("-" * 60)
        
        rag = FinancialReportRAG(
            model_name=model_name,
            collection_name=args.collection,
            persist_directory="./chroma_db",
            use_gpu=use_gpu,
            use_quantization=use_quantization,
            reset_collection=args.reset_collection,
            show_progress=show_progress
        )
        
        # Set up vector store
        rag.setup_vector_store()
        
        if rag.vector_store is None:
            print("❌ ERROR: Vector store failed to initialize!")
            return
        
        # Process PDFs
        rag.batch_process_pdfs(
            pdf_directory=args.pdf_dir,
            batch_size=args.batch_size,
            max_files=args.max_files
        )
        
        # Show processing summary
        summary = rag.generate_summary_report()
        print("\n📊 Processing Summary:")
        print(f"  Total chunks: {summary['total_documents']}")
        print(f"  Unique files: {summary['unique_files']}")
        print(f"  Pattern matches found:")
        print(f"    - CARs: {summary['entities_found']['CAR']}")
        print(f"    - CLs: {summary['entities_found']['CL']}")
        print(f"    - FARs: {summary['entities_found']['FAR']}")
    
    # Extract entities if needed
    if args.mode in ['extract', 'both', 'validate']:
        print("\n🔍 Starting Enhanced Entity Extraction...")
        print("-" * 60)
        
        extractor = LLMEntityExtractor(
            collection_name=args.collection,
            model_name=model_name,
            use_quantization=use_quantization,
            use_gpu=use_gpu
        )
        
        # Run extraction
        file_summaries, all_entities = extractor.extract_entities_from_collection(
            batch_size=args.extraction_batch_size
        )
        
        print("\n✅ Entity extraction complete!")
        print(f"📊 Results saved to: results/{args.collection}/")
        
        # Print extraction summary
        total_cars = sum(fs['car_count'] for fs in file_summaries)
        total_cls = sum(fs['cl_count'] for fs in file_summaries)
        total_fars = sum(fs['far_count'] for fs in file_summaries)
        
        print(f"\n📈 Extraction Summary:")
        print(f"  Files analyzed: {len(file_summaries)}")
        print(f"  CARs/NCRs/NCs identified: {total_cars}")
        print(f"  CLs/NIRs identified: {total_cls}")
        print(f"  FARs/OFIs identified: {total_fars}")
        print(f"  Total entities: {total_cars + total_cls + total_fars}")
        
        # Show distribution
        if file_summaries:
            files_with_cars = sum(1 for fs in file_summaries if fs['car_count'] > 0)
            files_with_cls = sum(1 for fs in file_summaries if fs['cl_count'] > 0)
            files_with_fars = sum(1 for fs in file_summaries if fs['far_count'] > 0)
            
            print(f"\n📊 File Distribution:")
            print(f"  Files with CARs: {files_with_cars}/{len(file_summaries)}")
            print(f"  Files with CLs: {files_with_cls}/{len(file_summaries)}")
            print(f"  Files with FARs: {files_with_fars}/{len(file_summaries)}")
        
        # Run validation if requested
        if args.validate or args.mode == 'validate':
            validation_stats, _ = run_extraction_validation(
                args.collection, model_name, use_quantization, use_gpu
            )
        
        # Run quality check if requested
        if args.quality_check:
            run_quality_check(all_entities)
    
    print("\n" + "="*70)
    print("🎉 All operations completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()