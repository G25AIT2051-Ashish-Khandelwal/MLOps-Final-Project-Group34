import json
import logging
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentClassifier:
    def __init__(self, model_path: str = "models/best_model"):
        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        with open("id2label.json") as f:
            self.id2label = json.load(f)

    def predict(self, texts: List[str]) -> List[Dict]:
        results = []

        for text in texts:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                prediction_id = torch.argmax(logits, dim=1).item()

            results.append({
                "text": text,
                "prediction": self.id2label[str(prediction_id)],
                "confidence": float(probabilities[prediction_id]),
                "probabilities": {
                    self.id2label[str(i)]: float(probabilities[i])
                    for i in range(len(probabilities))
                },
            })

        return results


def batch_inference(texts: List[str]) -> List[Dict]:
    classifier = SentimentClassifier()
    return classifier.predict(texts)


if __name__ == "__main__":
    test_texts = [
        "This movie was absolutely amazing! I loved every second of it.",
        "Terrible film. Complete waste of time and money.",
    ]

    classifier = SentimentClassifier()
    results = classifier.predict(test_texts)

    for result in results:
        print(f"\nText: {result['text'][:50]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
