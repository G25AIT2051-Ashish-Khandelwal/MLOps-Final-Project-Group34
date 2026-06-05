import os
import json
import logging
import numpy as np
import pandas as pd

import torch
import wandb
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models"
DATA_DIR = "data"


def load_data(data_dir: str = DATA_DIR):
    train_df = pd.read_csv(f"{data_dir}/train_clean.csv")
    test_df = pd.read_csv(f"{data_dir}/test_clean.csv")
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


def tokenize_data(df: pd.DataFrame, tokenizer, max_length: int = 512):
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(df["label"].values),
    })
    return dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_model(train_df, test_df, epochs=3, batch_size=16, learning_rate=2e-5, wandb_project="mlops-group34"):
    wandb.init(
        project=wandb_project,
        name="imdb-sentiment-distilbert",
        config={
            "model": MODEL_NAME,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        },
    )

    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    logger.info("Tokenizing data...")
    train_dataset = tokenize_data(train_df, tokenizer)
    test_dataset = tokenize_data(test_df, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["wandb"],
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    best_model_path = f"{OUTPUT_DIR}/best_model"
    model.save_pretrained(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    logger.info(f"Model saved to {best_model_path}")

    final_metrics = trainer.evaluate()
    logger.info(f"Final metrics: {final_metrics}")
    wandb.log({"final_metrics": final_metrics})

    wandb.finish()


if __name__ == "__main__":
    train_df, test_df = load_data()
    train_model(
        train_df=train_df,
        test_df=test_df,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        wandb_project="mlops-group34",
    )
