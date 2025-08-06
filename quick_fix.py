#!/usr/bin/env python3
"""
Quick fix script to reprocess PDFs and ensure they're added to ChromaDB
"""
import os
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import FinancialReportRAG, DEFAULT_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_fix_process(
    pdf_directory: str = "./data/all",
    collection_name: str = "report_3",
    max_files: int = 3,
    reset_collection: bool = True
):
    """Quick fix to reprocess PDFs with proper vector store handling"""
    
    print("="*60)
    print("🔧 QUICK FIX: Reprocessing PDFs with fixed vector store")
    print("="*60)
    
    # Initialize ChromaDB client
    import chromadb
    persist_dir = "./chroma_db"
    client = chromadb.PersistentClient(path=persist_dir)
    
    if reset_collection:
        # Delete existing collection if it exists
        try:
            client.delete_collection(name=collection_name)
            print(f"✅ Deleted existing collection: {collection_name}")
        except:
            print(f"Collection {collection_name} didn't exist")
    
    # Initialize RAG system
    print("\n🚀 Initializing RAG system...")
    rag = FinancialReportRAG(
        model_name=DEFAULT_MODEL,
        collection_name=collection_name,
        persist_directory=persist_dir,
        use_quantization=True
    )
    
    # Setup vector store
    print("\n📊 Setting up vector store...")
    rag.setup_vector_store()
    
    # Get PDF files
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("*.pdf"))[:max_files]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_directory}")
        return
    
    print(f"\n📄 Processing {len(pdf_files)} PDF files...")
    
    # Process each PDF individually and add to vector store immediately
    total_chunks = 0
    successful_files = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        
        try:
            # Process PDF
            documents = rag.process_pdf(str(pdf_file))
            
            if documents:
                print(f"  ✅ Extracted {len(documents)} chunks")
                
                # Add to vector store immediately
                print(f"  📝 Adding to vector store...")
                rag.vector_store.add_documents(documents)
                
                # Persist after each file
                rag.vector_store.persist()
                
                total_chunks += len(documents)
                successful_files += 1
                
                # Verify documents were added
                collection = client.get_collection(name=collection_name)
                current_count = collection.count()
                print(f"  📊 Collection now has {current_count} documents")
                
            else:
                print(f"  ⚠️ No content extracted")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    # Final check
    try:
        collection = client.get_collection(name=collection_name)
        final_count = collection.count()
        
        print(f"✅ Processing complete!")
        print(f"  Files processed: {successful_files}/{len(pdf_files)}")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Documents in collection: {final_count}")
        
        if final_count == 0:
            print("\n⚠️ WARNING: Collection is still empty!")
            print("Possible issues:")
            print("  1. PDFs might be corrupted or unreadable")
            print("  2. OCR might not be working properly")
            print("  3. ChromaDB persistence might be failing")
            print("\nTry running: python debug_chromadb.py")
        else:
            print(f"\n✅ SUCCESS: Collection has {final_count} documents!")
            
    except Exception as e:
        print(f"❌ Error checking final collection: {e}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick fix for empty collection issue')
    parser.add_argument('--pdf-dir', type=str, default='./data/all',
                       help='Directory containing PDF files')
    parser.add_argument('--collection', type=str, default='report_3',
                       help='Collection name')
    parser.add_argument('--max-files', type=int, default=3,
                       help='Maximum number of files to process')
    parser.add_argument('--no-reset', action='store_true',
                       help='Do not reset the collection')
    
    args = parser.parse_args()
    
    quick_fix_process(
        pdf_directory=args.pdf_dir,
        collection_name=args.collection,
        max_files=args.max_files,
        reset_collection=not args.no_reset
    )

if __name__ == "__main__":
    main()