# buhren-assistant
A Chatbot designed to help students clarify doubts from lectures.

A comprehensive Retrieval-Augmented Generation (RAG) system for processing and analyzing German-language documents. This system integrates document parsing, vector database storage, and language model generation to provide accurate answers to questions about your German documents.

## Features

- **Robust Document Processing**: Parse and extract text from PDF and DOCX files with special handling for German language characteristics including umlauts and document structure.
- **Optimized for German Language**: Built with German language processing in mind, using models and tools specifically tuned for German.
- **Modular Architecture**: Clean separation between document processing, vector storage, retrieval, and generation components.
- **Interactive CLI**: User-friendly command-line interface for asking questions and exploring document sources.
- **Vector Database Integration**: Efficient storage and retrieval of document embeddings using ChromaDB.
- **Customizable Parameters**: Easily adjust chunk sizes, retrieval parameters, and generation settings.

## System Architecture

The system is built with a modular design:

1. **Document Processor**: Handles parsing, text extraction, and chunking of German documents.
2. **Vector Store**: Manages document embeddings and similarity search using ChromaDB.
3. **RAG Pipeline**: Integrates retrieval and generation for question answering.
4. **Chat Interface**: Provides a user-friendly interface for interacting with the system.

## Installation

### Prerequisites

- Python 3.11 or higher
- Required libraries (see `requirements.txt`)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/ParthPatel-ES/buhren-assistant.git
   cd buhren-assistant
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the German language model for spaCy:
   ```
   python -m spacy download de_core_news_sm
   ```

## Usage

### Basic Usage

1. Place your German PDF and DOCX documents in a directory (e.g., `documents/`).

2. Run the system:
   ```
   python -m app.py --docs_dir documents/ --vector_db_path vector_db/
   ```

3. Ask questions in the interactive interface.

### Command Line Arguments

- `--docs_dir`: Directory containing documents to process (default: "documents")
- `--vector_db_path`: Path to vector database storage (default: "vector_db")
- `--chunk_size`: Size of text chunks for processing (default: 512)
- `--chunk_overlap`: Overlap between consecutive chunks (default: 50)
- `--embedding_model`: Hugging Face model ID for embeddings (default: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
- `--language_model`: Hugging Face model ID for language model (default: "openGPT-X/Teuken-7B-instruct-commercial-v0.4")

### Chat Interface Commands

- Ask a question: Simply type your question in German
- `sources`: Show sources for the last answer
- `history`: Show conversation history
- `help`: Show available commands
- `quit`: Exit the application

## Customization

### Changing Language Models

You can customize the embedding and generation models:

1. For embeddings, modify the `embedding_model` parameter in `VectorStore`:
   ```python
   vector_store = VectorStore(
       embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v1"
   )
   ```

2. For generation, modify the `model_name` parameter in `RAGPipeline`:
   ```python
   rag_pipeline = RAGPipeline(
       model_name="openGPT-X/Teuken-7B-instruct-commercial-v0.4"
   )
   ```

### Optimizing for Different Document Types

You can adjust chunking parameters based on your document characteristics:

```python
doc_processor = DocumentProcessor(
    chunk_size=256,  # Smaller chunks for more precise retrieval
    chunk_overlap=100  # Higher overlap for better context preservation
)
```

## Project Structure

```
german_document_rag/
├── __init__.py
├── main.py                 # Main entry point
├── document_processor.py   # Document parsing & chunking
├── vector_store.py         # Vector database integration
├── rag_pipeline.py         # RAG implementation
└── chat_interface.py       # Interactive CLI
```

## Troubleshooting

### Common Issues

- **Memory Issues**: If you encounter memory errors, try reducing the chunk size or using a smaller language model.
- **German Character Encoding**: If German umlauts appear incorrectly, check that your files use UTF-8 encoding.
- **Performance**: For faster processing, consider using GPU acceleration where available.

### Optional:

German language support
> Run: python -m spacy download de_core_news_sm


## License

This project is licensed under the MIT License - see the LICENSE file for details.

