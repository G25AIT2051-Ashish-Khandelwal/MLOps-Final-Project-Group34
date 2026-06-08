"""
Data Preparation Script
Downloads, cleans, and saves the IMDb sentiment dataset.
Run this once on your local machine.
"""

import os
import re
import json
import re

from datasets import load_dataset


def clean_text(text):
    """Remove HTML tags and extra whitespace."""
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


print("Loading IMDb dataset from Hugging Face...")
dataset = load_dataset("stanfordnlp/imdb")

print("\n=== RAW DATA INSPECTION ===")
print(f"Train samples : {len(dataset['train'])}")
print(f"Test  samples : {len(dataset['test'])}")
print(f"Features      : {dataset['train'].features}")
print(f"\nFirst sample  :\n{dataset['train'][0]}")

train_df = dataset['train'].to_pandas()
print(f"\nClass distribution (train):\n{train_df['label'].value_counts()}")

print("\n=== CLEANING ===")

train_df['text'] = train_df['text'].apply(clean_text)
train_df = train_df.sample(n=5000, random_state=42).reset_index(drop=True)

print(f"Missing values: {train_df.isnull().sum().to_dict()}")
print(f"Duplicates removed: {train_df.duplicated(subset='text').sum()}")
train_df = train_df.drop_duplicates(subset='text').reset_index(drop=True)

test_df = dataset['test'].to_pandas()
test_df['text'] = test_df['text'].apply(clean_text)
test_df = test_df.sample(n=1000, random_state=42).reset_index(drop=True)

id2label = {0: "negative", 1: "positive"}

with open("id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)
print("Saved id2label.json")

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train_clean.csv", index=False)
test_df.to_csv("data/test_clean.csv", index=False)
print(f"Saved data/train_clean.csv ({len(train_df)} rows)")
print(f"Saved data/test_clean.csv  ({len(test_df)} rows)")
print("\nData preparation complete!")
