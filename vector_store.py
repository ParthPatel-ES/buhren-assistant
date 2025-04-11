"""
Vector Store for German document embeddings.

This module handles the creation, storage, and retrieval of document
embeddings using a vector database optimized for German language.
"""

import os
import logging
import json
import shutil
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

# Vector DB and embedding libraries
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from document_processor import DocumentChunk

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages vector embeddings for German document chunks.
    
    Handles embedding generation, storage, retrieval, and backup
    using ChromaDB as the underlying vector database.
    """
    
    def __init__(
        self,
        db_path: str = "vector_db",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "german_documents"
    ):
        """
        Initialize the vector store.
        
        Args:
            db_path: Path to vector database storage
            embedding_model: HuggingFace model ID for embeddings
            collection_name: Name of the vector collection
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # Create database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize Chroma vector database
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Create or get collection
        self.collection = self._get_or_create_collection()
        
        # Load embedding model optimized for German language
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Could not initialize embedding model: {str(e)}")
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one.
        
        Returns:
            chromadb.Collection: ChromaDB collection
        """
        try:
            # Check if collection exists
            collections = self.client.list_collections()
            for collection in collections:
                if collection.name == self.collection_name:
                    logger.info(f"Using existing collection: {self.collection_name}")
                    return self.client.get_collection(self.collection_name)
            
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "German document chunks", "created": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Error accessing collection: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Process document chunks and add them to the vector store.
        
        Args:
            chunks: List of document chunks to embed and store
        """
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return
        
        try:
            # Prepare data for batch insertion
            ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                # Create a unique ID for each chunk
                chunk_id = f"{os.path.basename(chunk.metadata.source)}_{chunk.chunk_id}"
                ids.append(chunk_id)
                
                # Add the document text
                documents.append(chunk.text)
                
                # Convert metadata to dictionary for storage
                metadata_dict = {
                    "source": chunk.metadata.source,
                    "title": chunk.metadata.title or "",
                    "author": chunk.metadata.author or "",
                    "created_date": chunk.metadata.created_date or "",
                    "doc_type": chunk.metadata.doc_type or "",
                    "chunk_id": chunk.chunk_id,
                    "language": "de"
                }
                metadatas.append(metadata_dict)
            
            # Generate embeddings
            embeddings = [self.get_embedding(text) for text in documents]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Added {len(chunks)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using the sentence transformer model.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        # Handle empty or invalid text
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text")
            # Return zero vector of appropriate size
            return [0.0] * self.embedding_model.get_sentence_embedding_dimension()
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_model.get_sentence_embedding_dimension()
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents similar to query.
        
        Args:
            query: Query text
            k: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            List of document chunks with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filters
            )
            
            # Format results
            formatted_results = []
            
            if results and results["documents"]:
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (1.0 is perfect match)
                    similarity = 1.0 - min(1.0, float(distance))
                    
                    formatted_results.append({
                        "id": doc_id,
                        "text": document,
                        "metadata": metadata,
                        "similarity": similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def create_backup(self, backup_dir: Optional[str] = None) -> bool:
        """
        Create a backup of the vector database.
        
        Args:
            backup_dir: Directory to store backup (defaults to db_path + "_backup")
            
        Returns:
            True if backup successful, False otherwise
        """
        if not backup_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{self.db_path}_backup_{timestamp}"
        
        try:
            # Create backup directory
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy database files
            shutil.copytree(self.db_path, backup_dir, dirs_exist_ok=True)
            
            # Save metadata about the backup
            backup_meta = {
                "original_path": self.db_path,
                "backup_time": datetime.now().isoformat(),
                "embedding_model": self.embedding_model_name,
                "collection": self.collection_name
            }
            
            with open(os.path.join(backup_dir, "backup_metadata.json"), "w") as f:
                json.dump(backup_meta, f, indent=2)
            
            logger.info(f"Created database backup at {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            return False
    
    def restore_from_backup(self, backup_dir: str) -> bool:
        """
        Restore vector database from backup.
        
        Args:
            backup_dir: Path to backup directory
            
        Returns:
            True if restore successful, False otherwise
        """
        try:
            # Verify backup metadata
            metadata_path = os.path.join(backup_dir, "backup_metadata.json")
            
            if not os.path.exists(metadata_path):
                logger.error(f"Invalid backup: missing metadata file in {backup_dir}")
                return False
            
            # Close current DB connection
            del self.client
            del self.collection
            
            # Copy backup files to database path
            shutil.rmtree(self.db_path, ignore_errors=True)
            shutil.copytree(backup_dir, self.db_path, dirs_exist_ok=True)
            
            # Reinitialize database connection
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self._get_or_create_collection()
            
            logger.info(f"Restored database from backup at {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {str(e)}")
            return False