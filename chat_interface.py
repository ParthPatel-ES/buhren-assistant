# german_document_rag/chat_interface.py
"""
Chat interface for interacting with the German RAG system.

This module provides a command-line interface for asking questions
about German documents using the RAG pipeline.
"""

import logging
import cmd
import os
import readline
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class ChatInterface(cmd.Cmd):
    """
    Interactive command-line interface for the German RAG system.
    
    Provides commands for asking questions, viewing document sources,
    and managing the conversation history.
    """
    
    intro = """
German Document RAG System
--------------------------
Stellen Sie Fragen zu Ihren Dokumenten auf Deutsch.
Geben Sie 'help' ein für verfügbare Befehle oder 'quit' zum Beenden.
    """
    prompt = "Frage > "
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Initialize the chat interface.
        
        Args:
            rag_pipeline: RAG pipeline for question answering
        """
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.history = []
        self.last_sources = []
        self.log_dir = "chat_logs"
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def default(self, line: str) -> bool:
        """
        Handle input that isn't a recognized command.
        
        Args:
            line: User input
            
        Returns:
            False to continue, True to exit
        """
        if line.lower() in ('quit', 'exit', 'ende', 'tschüss'):
            return self.do_quit(line)
        
        # Treat input as a question
        self._process_question(line)
        return False
    
    def do_quit(self, arg: str) -> bool:
        """
        Exit the application.
        
        Args:
            arg: Command arguments
            
        Returns:
            True to exit
        """
        print("Auf Wiedersehen!")
        self._save_chat_history()
        return True
    
    def do_sources(self, arg: str) -> None:
        """
        Show sources for the last answer.
        
        Args:
            arg: Command arguments
        """
        if not self.last_sources:
            print("Keine Quellen verfügbar. Stellen Sie zuerst eine Frage.")
            return
        
        print("\nQuellen für die letzte Antwort:")
        print("---------------------------------")
        
        for i, source in enumerate(self.last_sources):
            print(f"\nQuelle {i+1}:")
            print(f"Titel: {source['title']}")
            print(f"Datei: {source['source']}")
            print(f"Relevanz: {source['similarity']:.2f}")
            print(f"Auszug: {source['text']}")
    
    def do_history(self, arg: str) -> None:
        """
        Show conversation history.
        
        Args:
            arg: Command arguments
        """
        if not self.history:
            print("Noch keine Konversationshistorie.")
            return
        
        print("\nKonversationshistorie:")
        print("----------------------")
        
        for i, item in enumerate(self.history):
            timestamp = item["timestamp"].strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Frage: {item['question']}")
            print(f"Antwort: {item['answer'][:100]}...")
    
    def do_help(self, arg: str) -> None:
        """
        Show help information.
        
        Args:
            arg: Command arguments
        """
        commands = {
            "Frage stellen": "Geben Sie einfach Ihre Frage ein",
            "sources": "Zeigt die Quellen für die letzte Antwort",
            "history": "Zeigt die Konversationshistorie",
            "quit": "Beendet das Programm"
        }
        
        print("\nVerfügbare Befehle:")
        print("------------------")
        
        for cmd, desc in commands.items():
            print(f"{cmd:<15} - {desc}")
    
    def _process_question(self, question: str) -> None:
        """
        Process a user question and display the answer.
        
        Args:
            question: User query in German
        """
        if not question.strip():
            return
        
        try:
            print("\nVerarbeite Ihre Frage...\n")
            
            # Get answer from RAG pipeline
            response = self.rag_pipeline.answer_question(question)
            
            # Display answer
            print(f"Antwort: {response['answer']}")
            
            # Store sources for later reference
            self.last_sources = response["sources"]
            
            # Add to history
            history_item = {
                "timestamp": datetime.now(),
                "question": question,
                "answer": response["answer"],
                "sources": response["sources"]
            }
            self.history.append(history_item)
            
            # Provide hint about sources
            if self.last_sources:
                print(f"\nHinweis: Geben Sie 'sources' ein, um die {len(self.last_sources)} verwendeten Quellen anzuzeigen.")
                
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            print("Es ist ein Fehler bei der Verarbeitung Ihrer Frage aufgetreten.")
    
    def _save_chat_history(self) -> None:
        """Save the current chat history to a log file."""
        if not self.history:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, f"chat_log_{timestamp}.json")
            
            # Convert history to serializable format
            serializable_history = []
            for item in self.history:
                serializable_item = {
                    "timestamp": item["timestamp"].isoformat(),
                    "question": item["question"],
                    "answer": item["answer"],
                    "sources": item["sources"]
                }
                serializable_history.append(serializable_item)
            
            # Save to file
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(serializable_history, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Chat history saved to {log_file}")
            print(f"\nKonversationshistorie gespeichert in {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save chat history: {str(e)}")
    
    def start(self) -> None:
        """Start the interactive chat interface."""
        try:
            self.cmdloop()
        except KeyboardInterrupt:
            print("\nProgramm unterbrochen.")
            self._save_chat_history()
        except Exception as e:
            logger.error(f"Unexpected error in chat interface: {str(e)}")
            print("\nEin unerwarteter Fehler ist aufgetreten.")
            self._save_chat_history()