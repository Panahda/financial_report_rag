#!/usr/bin/env python3
"""
Debug script for ChromaDB collections
Helps diagnose issues with document storage and retrieval
"""
import chromadb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_chromadb(persist_directory: str = "./chroma_db"):
    """Debug ChromaDB collections and document counts"""
    
    logger.info(f"🔍 Debugging ChromaDB at: {persist_directory}")
    
    # Check if directory exists
    chroma_dir = Path(persist_directory)
    if not chroma_dir.exists():
        logger.error(f"❌ ChromaDB directory does not exist: {persist_directory}")
        return
    
    logger.info(f"✅ ChromaDB directory exists")
    
    # Initialize client
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        logger.info("✅ ChromaDB client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ChromaDB client: {e}")
        return
    
    # List all collections
    try:
        collections = client.list_collections()
        logger.info(f"📊 Found {len(collections)} collections:")
        
        if not collections:
            logger.warning("  No collections found in database")
            return
        
        for collection in collections:
            logger.info(f"\n  Collection: {collection.name}")
            
            # Get collection object
            col = client.get_collection(collection.name)
            
            # Get document count
            count = col.count()
            logger.info(f"    Document count: {count}")
            
            # Get sample documents if any exist
            if count > 0:
                # Get first 3 documents
                sample = col.get(
                    limit=min(3, count),
                    include=["documents", "metadatas"]
                )
                
                logger.info(f"    Sample documents:")
                for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
                    logger.info(f"      Doc {i+1}:")
                    logger.info(f"        File: {meta.get('file_name', 'Unknown')}")
                    logger.info(f"        Page: {meta.get('page', 'Unknown')}")
                    logger.info(f"        Content preview: {doc[:100]}...")
                    
                # Check for unique files
                all_meta = col.get(
                    limit=count,
                    include=["metadatas"]
                )
                
                unique_files = set()
                for meta in all_meta["metadatas"]:
                    unique_files.add(meta.get("file_name", "Unknown"))
                
                logger.info(f"    Unique files: {len(unique_files)}")
                for file in list(unique_files)[:5]:  # Show first 5 files
                    logger.info(f"      - {file}")
                    
    except Exception as e:
        logger.error(f"❌ Error listing collections: {e}")
        return

def reset_collection(collection_name: str, persist_directory: str = "./chroma_db"):
    """Reset a specific collection (delete and recreate)"""
    
    logger.info(f"🔄 Resetting collection: {collection_name}")
    
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Try to delete existing collection
        try:
            client.delete_collection(name=collection_name)
            logger.info(f"✅ Deleted existing collection: {collection_name}")
        except:
            logger.info(f"Collection {collection_name} didn't exist")
        
        # Create new collection
        client.create_collection(name=collection_name)
        logger.info(f"✅ Created new collection: {collection_name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to reset collection: {e}")

def test_add_document(collection_name: str, persist_directory: str = "./chroma_db"):
    """Test adding a document to a collection"""
    
    logger.info(f"🧪 Testing document addition to: {collection_name}")
    
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
        
        # Count before
        count_before = collection.count()
        logger.info(f"Documents before: {count_before}")
        
        # Add test document
        collection.add(
            documents=["This is a test document for debugging ChromaDB"],
            metadatas=[{
                "file_name": "test_file.pdf",
                "page": 1,
                "source": "debug_test"
            }],
            ids=[f"test_doc_{count_before + 1}"]
        )
        
        # Count after
        count_after = collection.count()
        logger.info(f"Documents after: {count_after}")
        
        if count_after > count_before:
            logger.info(f"✅ Successfully added document! New count: {count_after}")
        else:
            logger.error(f"❌ Document was not added. Count unchanged: {count_after}")
            
    except Exception as e:
        logger.error(f"❌ Failed to test document addition: {e}")

def main():
    """Main debug function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug ChromaDB collections')
    parser.add_argument('--persist-dir', type=str, default='./chroma_db',
                       help='ChromaDB persist directory')
    parser.add_argument('--reset', type=str,
                       help='Reset a specific collection')
    parser.add_argument('--test-add', type=str,
                       help='Test adding a document to a collection')
    
    args = parser.parse_args()
    
    # Run debug
    debug_chromadb(args.persist_dir)
    
    # Reset collection if requested
    if args.reset:
        print(f"\n{'='*60}")
        reset_collection(args.reset, args.persist_dir)
        print(f"{'='*60}\n")
        # Show collections again
        debug_chromadb(args.persist_dir)
    
    # Test adding document if requested
    if args.test_add:
        print(f"\n{'='*60}")
        test_add_document(args.test_add, args.persist_dir)
        print(f"{'='*60}")

if __name__ == "__main__":
    main()