"""
Inference script — loads model from Hugging Face and classifies input text.
Used by both Docker and GitHub Actions.
"""

import os
import json
from transformers import pipeline

HF_MODEL = os.environ.get("HF_MODEL_NAME", "ashishk-G25AIT2051/MLOps_Final_Group_34")
INPUT     = os.environ.get("INPUT_TEXT",    "This movie was absolutely amazing!")

print(f"Model  : {HF_MODEL}")
print(f"Input  : {INPUT}")
print("Loading model (first run downloads weights)...")

classifier = pipeline(
    "text-classification",
    model=HF_MODEL,
    token=os.environ.get("HF_TOKEN", None),
)

result = classifier(INPUT)
print(f"\nResult : {result[0]['label']} (confidence: {result[0]['score']:.4f})")