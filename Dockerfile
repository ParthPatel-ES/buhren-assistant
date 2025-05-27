FROM python:3.12-slim

# Add labels for better image identification
LABEL maintainer="Parth Patel"
LABEL version="1.0"
LABEL description="Buhren Assistant Chatbot"

# Install git and update all system packages to fix vulnerabilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download de_core_news_sm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/ParthPatel-ES/buhren-assistant.git /app

WORKDIR /app
COPY documents /app/documents
COPY requirements.txt /app/requirements.txt

# Expose port
EXPOSE 8000

# Create volume mount points
VOLUME ["/app/documents", "/app/vector_db"]

# Install dependencies
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt 
RUN python -m spacy download de_core_news_sm
