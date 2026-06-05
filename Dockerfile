FROM python:3.11-slim

WORKDIR /app

ARG HF_MODEL_NAME=your-username/your-model-repo
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY src/inference.py ./src/inference.py

CMD ["python", "src/inference.py"]
