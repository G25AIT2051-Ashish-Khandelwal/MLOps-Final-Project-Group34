FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY id2label.json .

# Create volume mount point for data
VOLUME /app/data

# Expose port for API
EXPOSE 8000

# Default command: run inference
CMD ["python", "src/inference.py"]

# Alternative: To run training, use:
# docker run -e WANDB_API_KEY=<key> imdb-sentiment python src/train.py

# Alternative: To run API server (requires FastAPI main.py)
# docker run -p 8000:8000 imdb-sentiment python main.py
