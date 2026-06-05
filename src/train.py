import os
import numpy as np
import pandas as pd
import wandb
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from huggingface_hub import login

MODEL_NAME = "distilbert-base-uncased"
HF_REPO = "somnathchakraborty/distilbert-imdb-sentiment"


def load_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
        login(token=secrets.get_secret("HF_TOKEN"))
    except ImportError:
        login(token=os.environ.get("HF_TOKEN", ""))
    wandb.login()


def load_data():
    train_df = pd.read_csv("data/train_clean.csv")
    test_df = pd.read_csv("data/test_clean.csv")
    return train_df, test_df


def tokenize_data(df, tokenizer, max_length=512):
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


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train_version(train_dataset, test_dataset, model, tokenizer, version_config):
    wandb.init(
        project="mlops-group34",
        name=f"run-{version_config['version']}",
        config={
            "model": MODEL_NAME,
            "epochs": version_config["epochs"],
            "batch_size": version_config["batch_size"],
            "learning_rate": version_config["learning_rate"],
            "version": version_config["version"],
            "platform": "Kaggle",
        },
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=version_config["epochs"],
        per_device_train_batch_size=version_config["batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"run-{version_config['version']}",
        learning_rate=version_config["learning_rate"],
        weight_decay=version_config.get("weight_decay", 0.01),
        warmup_steps=version_config.get("warmup_steps", 100),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer


if __name__ == "__main__":
    load_secrets()

    train_df, test_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = tokenize_data(train_df, tokenizer)
    test_dataset = tokenize_data(test_df, tokenizer)

    v1_config = {
        "version": "v1",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 100,
    }

    model_v1 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    trainer_v1 = train_version(train_dataset, test_dataset, model_v1, tokenizer, v1_config)
    wandb.finish()

    v2_config = {
        "version": "v2",
        "epochs": 4,
        "batch_size": 32,
        "learning_rate": 5e-5,
        "weight_decay": 0.02,
        "warmup_steps": 50,
    }

    model_v2 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    trainer_v2 = train_version(train_dataset, test_dataset, model_v2, tokenizer, v2_config)

    best_trainer = trainer_v2
    best_trainer.model.push_to_hub(HF_REPO)
    tokenizer.push_to_hub(HF_REPO)

    hf_url = f"https://huggingface.co/{HF_REPO}"
    wandb.run.summary["huggingface_model"] = hf_url
    print(f"Model pushed to: {hf_url}")

    wandb.finish()
