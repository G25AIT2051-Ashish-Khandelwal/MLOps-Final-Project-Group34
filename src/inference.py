import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_MODEL = os.environ.get("HF_MODEL_NAME", "ashishk-G25AIT2051/MLOps_Final_Group_34")
INPUT = os.environ.get("INPUT_TEXT", "This movie was absolutely amazing!")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

print(f"Model  : {HF_MODEL}")
print(f"Input  : {INPUT}")
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, token=HF_TOKEN)
model.eval()

inputs = tokenizer(INPUT, return_tensors="pt", truncation=True, max_length=512)
inputs.pop("token_type_ids", None)

with torch.no_grad():
    outputs = model(**inputs)

predicted_id = outputs.logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_id]
confidence = torch.softmax(outputs.logits, dim=-1)[0][predicted_id].item()

print(f"\nResult : {predicted_label} (confidence: {confidence:.4f})")
