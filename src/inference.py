import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "somnathchakraborty/distilbert-imdb-sentiment")
INPUT_TEXT = os.environ.get("INPUT_TEXT", "This movie was great!")


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_id = torch.argmax(probs).item()

    id2label = model.config.id2label
    label = id2label[pred_id]
    confidence = probs[pred_id].item()

    return {
        "text": text,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {id2label[i]: round(probs[i].item(), 4) for i in range(len(probs))}
    }


if __name__ == "__main__":
    print(f"Loading model: {HF_MODEL_NAME}")
    tokenizer, model = load_model(HF_MODEL_NAME)

    print(f"Input text: {INPUT_TEXT}")
    result = predict(INPUT_TEXT, tokenizer, model)

    print(f"\nPrediction: {result['label']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Probabilities: {result['probabilities']}")
