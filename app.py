"""
German Document RAG System - Main Entry Point

This system processes German language documents (.docx, .pdf) and implements
a Retrieval-Augmented Generation (RAG) pipeline for interactive Q&A.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import argparse

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from chat_interface import ChatInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_argparse() -> argparse.Namespace:
    """Configure command line argument parsing.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="German Document RAG System")
    parser.add_argument(
        "--docs_dir", 
        type=str, 
        default="documents",
        help="Directory containing documents to process"
    )
    parser.add_argument(
        "--vector_db_path", 
        type=str, 
        default="vector_db",
        help="Path to vector database storage"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=1024,
        help="Size of text chunks for processing"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=512,
        help="Overlap between consecutive chunks"
    )
    parser.add_argument(
        "--embedding_model", 
        type=str, 
        # default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Hugging Face model ID for embeddings"
    )
    return parser.parse_args()

def main():
    """Main entry point for the German Document RAG System."""
    try:
        # Parse command line arguments
        args = setup_argparse()
        
        # Initialize document processor
        doc_processor = DocumentProcessor(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Initialize vector store
        vector_store = VectorStore(
            db_path=args.vector_db_path,
            embedding_model=args.embedding_model
        )
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            document_processor=doc_processor,
            vector_store=vector_store
        )
        
        # Process documents and build index
        if os.path.exists(args.docs_dir):
            rag_pipeline.build_knowledge_base(args.docs_dir)
        else:
            logger.error(f"Document directory not found: {args.docs_dir}")
            return
        
        # Initialize and start chat interface
        chat_interface = ChatInterface(rag_pipeline=rag_pipeline)
        chat_interface.start()
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()