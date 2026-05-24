FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required by SemanticChunker
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy application source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
