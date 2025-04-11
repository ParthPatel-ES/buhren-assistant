# german_document_rag/rag_pipeline.py
"""
RAG Pipeline for German document question answering.

This module integrates document processing, vector retrieval, and
language model generation for German language Q&A.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import glob

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate

from document_processor import DocumentProcessor
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Implements Retrieval-Augmented Generation for German documents.
    
    Integrates document processing, vector retrieval, and language
    model generation for accurate question answering on German documents.
    """
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_store: VectorStore,
        model_name: str = "model-name-here",
        max_new_tokens: int = 1024,
        temperature: float = 0.85
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            document_processor: Component for processing documents
            vector_store: Component for storing and retrieving vectors
            model_name: HuggingFace model ID for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
        """
        self.document_processor = document_processor
        self.vector_store = vector_store
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Initialize language model for generation
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                clean_up_tokenization_spaces=False,
                trust_remote_code=True
            )
            logger.info(f"Tokenizer loaded successfully for {model_name}")

            # Configure quantization with CPU offloading
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offloading
            )

            # Custom device map to offload some layers to CPU
            device_map = {
                "transformer.word_embeddings": 0,
                "transformer.final_layernorm": 0,
                "lm_head": 0,
                "transformer.h": "cpu"  # Offload transformer layers to CPU
            }
            print('-----\n\n\n')
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # device_map=device_map,  # Use custom device mapping
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create text generation pipeline
            self.generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                trust_remote_code=True,
                device_map=device_map,
                do_sample=True,
                top_p=0.95,
            )
            
            logger.info(f"Loaded language model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load language model: {str(e)}")
            logger.warning("Running in retrieve-only mode (no generation)")
            self.generation_pipeline = None
        
        # Define prompts for German language
        self.qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Du bist ein hilfreicher KI-Assistent, der Fragen auf Deutsch beantwortet.
            
Basierend auf den folgenden Dokumentauszügen, beantworte bitte die Frage. 
Wenn du die Antwort nicht in den Dokumenten findest, sage ehrlich "Ich kann diese Frage basierend auf den vorliegenden Dokumenten nicht beantworten."
Antworte immer auf Deutsch.

Kontext:
{context}

Frage: {question}

Antwort:"""
        )
    
    def build_knowledge_base(self, documents_path: str) -> None:
        """
        Process documents and build vector index.
        
        Args:
            documents_path: Path to directory or file
        """
        logger.info(f"Building knowledge base from: {documents_path}")
        
        try:
            if os.path.isdir(documents_path):
                # Process directory of documents
                chunks = self.document_processor.process_directory(documents_path)
            elif os.path.isfile(documents_path):
                # Process single file
                if documents_path.lower().endswith('.pdf'):
                    chunks = self.document_processor.process_pdf(documents_path)
                elif documents_path.lower().endswith('.docx'):
                    chunks = self.document_processor.process_docx(documents_path)
                else:
                    logger.error(f"Unsupported file type: {documents_path}")
                    return
            else:
                logger.error(f"Path not found: {documents_path}")
                return
            
            # Add chunks to vector store
            if chunks:
                self.vector_store.add_documents(chunks)
                logger.info(f"Added {len(chunks)} chunks to knowledge base")
            else:
                logger.warning(f"No document chunks extracted from {documents_path}")
                
        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User query in German
            top_k: Number of results to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of document chunks with similarity scores
        """
        try:
            # Get similar documents from vector store
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                filters=filters
            )
            
            return results
        except Exception as e:
            logger.error(f"Error in retrieval: {str(e)}")
            return []
    
    def generate(self, query: str, context: str) -> str:
        """
        Generate answer based on retrieved context.
        
        Args:
            query: User query in German
            context: Retrieved document context
            
        Returns:
            Generated answer
        """
        if not self.generation_pipeline:
            return "Language model is not available for generation."
        
        try:
            # Format prompt with context and query
            prompt = self.qa_prompt_template.format(
                context=context,
                question=query
            )
            
            # Generate response
            response = self.generation_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95
            )
            
            # Extract generated text from response
            generated_text = response[0]["generated_text"]
            
            # Extract only the answer part (after the prompt)
            answer = generated_text[len(prompt):].strip()
            
            return answer
        except Exception as e:
            logger.error(f"Error in generation: {str(e)}")
            return "Es ist ein Fehler bei der Generierung der Antwort aufgetreten."
    
    def answer_question(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        End-to-end question answering using the RAG pipeline.
        
        Args:
            query: User query in German
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            Dictionary with answer and supporting documents
        """
        try:
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(query, top_k, filters)
            
            if not retrieved_docs:
                return {
                    "answer": "Ich konnte keine relevanten Dokumente zu dieser Frage finden.",
                    "sources": [],
                    "context": ""
                }
            
            # Build context from retrieved documents
            context = self._format_context(retrieved_docs)
            
            # Generate answer
            answer = self.generate(query, context)
            
            # Format source information
            sources = []
            for doc in retrieved_docs:
                source = {
                    "text": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"],
                    "source": doc["metadata"]["source"],
                    "title": doc["metadata"]["title"],
                    "similarity": doc["similarity"]
                }
                sources.append(source)
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}")
            return {
                "answer": "Es ist ein Fehler bei der Beantwortung aufgetreten.",
                "sources": [],
                "context": ""
            }
    
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for generation.
        
        Args:
            retrieved_docs: List of retrieved document chunks
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            # Format each document chunk with source information
            source = doc["metadata"]["source"]
            title = doc["metadata"]["title"] if doc["metadata"]["title"] else os.path.basename(source)
            
            context_part = f"Dokument {i+1}: {title}\n"
            context_part += f"Quelle: {source}\n"
            context_part += f"Inhalt: {doc['text']}\n\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def evaluate(self, test_questions: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline on test questions.
        
        Args:
            test_questions: List of dictionaries with 'question' and 'expected_answer'
            
        Returns:
            Evaluation metrics and results
        """
        results = []
        total_questions = len(test_questions)
        
        for i, test_item in enumerate(test_questions):
            question = test_item["question"]
            expected = test_item.get("expected_answer", "")
            
            logger.info(f"Evaluating question {i+1}/{total_questions}: {question[:50]}...")
            
            # Get answer
            response = self.answer_question(question)
            answer = response["answer"]
            
            # Store result
            result = {
                "question": question,
                "expected": expected,
                "actual": answer,
                "sources": response["sources"]
            }
            results.append(result)
        
        # Return overall evaluation
        return {
            "total_questions": total_questions,
            "results": results
        }