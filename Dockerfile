FROM python:3.11-slim

ARG HF_MODEL_NAME=ashishk-G25AIT2051/MLOps_Final_Group_34

ENV HF_MODEL_NAME=${HF_MODEL_NAME}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/inference.py .

CMD ["python", "inference.py"]