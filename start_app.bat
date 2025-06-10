@echo off
REM Default values
set PORT=8000
set DOCS_DIR=documents/
set VECTOR_DB_PATH=vector_db/
set LLM_MODEL=openGPT-X/Teuken-7B-instruct-commercial-v0.4
REM Other options for language models:
REM "openGPT-X/Teuken-7B-instruct-commercial-v0.4"
REM "LeoLM/leo-mistral-hessianai-7b"
REM "meta-llama/Llama-4-Scout-17B-16E-Instruct"

set EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
REM Other options for embedding models:
REM "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REM "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
REM "sentence-transformers/all-MiniLM-L6-v2"

REM Container name
set CONTAINER_NAME=buhren-assistant

REM Run the container
docker run -d ^
    --name %CONTAINER_NAME% ^
    -v %cd%\documents:/app/documents ^
    -v %cd%\vector_db:/app/vector_db ^
    -p %PORT%:8000 ^
    buhren-assistant:1.0 ^
    python app.py ^
    --docs_dir %DOCS_DIR% ^
    --vector_db_path %VECTOR_DB_PATH% ^
    --language_model %LLM_MODEL% ^
    --embedding_model %EMBEDDING_MODEL%