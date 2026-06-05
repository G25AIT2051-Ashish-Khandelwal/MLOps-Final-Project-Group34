# training script
"""
Model loading test — run locally to verify setup.
Actual training is done on Kaggle (see Task 4).
"""

import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load label mapping
with open("id2label.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}
label2id = {v: k for k, v in id2label.items()}

model_name = "distilbert-base-uncased"
num_labels  = len(id2label)

print(f"Loading tokenizer for: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model with {num_labels} output labels...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

print(f"\nModel loaded successfully!")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Labels: {id2label}")