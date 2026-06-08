FROM python:3.11-slim

ARG HF_MODEL_NAME=ashishk-G25AIT2051/MLOps_Final_Group_34
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir numpy==1.26.4

RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    huggingface_hub==0.22.0 \
    scikit-learn \
    pandas

COPY src/inference.py .

CMD ["python", "inference.py"]