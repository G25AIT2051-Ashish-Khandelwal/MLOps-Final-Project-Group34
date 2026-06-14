# MLOps Final Project - Group 34

IMDb Sentiment Classification using DistilBERT | IIT Jodhpur

## Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  IMDb Data   │────>│  Data Prep   │────>│  Kaggle GPU   │────>│  HuggingFace │
│  (HF Hub)    │     │  data_prep.py│     │  Training     │     │  Model Hub   │
└─────────────┘     └──────────────┘     └───────┬───────┘     └──────┬───────┘
                                                 │                    │
                                          ┌──────▼───────┐           │
                                          │  Weights &   │           │
                                          │  Biases      │           │
                                          │  (Tracking)  │           │
                                          └──────────────┘           │
                                                                     │
┌─────────────┐     ┌──────────────┐     ┌───────────────┐          │
│  GitHub      │────>│  CI Pipeline │     │  Inference    │<─────────┘
│  Actions     │────>│  (Linting)   │     │  (from HF)   │
│              │────>│  Inference   │────>│  inference.py │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                 │
                                          ┌──────▼───────┐
                                          │   Docker     │
                                          │   Container  │
                                          └──────────────┘
```

## Project Structure

```
├── src/
│   ├── data_prep.py          # Downloads and cleans IMDb dataset
│   ├── train.py              # DistilBERT fine-tuning with W&B
│   └── inference.py          # Loads model from HF, runs predictions
├── notebooks/
│   └── train_v1.ipynb        # Kaggle notebook for training
├── .github/workflows/
│   ├── ci.yml                # Linting on push to develop
│   └── inference.yml         # Manual inference trigger
├── Dockerfile                # Containerized inference
├── requirements.txt          # Full dependencies
├── requirements-inference.txt# Slim deps for Docker
└── id2label.json             # Label mapping
```

## Links

| Resource | URL |
|----------|-----|
| GitHub Repo | https://github.com/G25AIT2051-Ashish-Khandelwal/MLOps-Final-Project-Group34 |
| Kaggle Notebook | https://www.kaggle.com/code/somnathg25ait2107/mlops-project-gr34 |
| HuggingFace Model | https://huggingface.co/somnathchakraborty/distilbert-imdb-sentiment |
| Docker Image | `docker pull somnchak/mlops-group34-inference:latest` |
| W&B Dashboard | https://wandb.ai/ashish-iit-jodhpur-25ait2051/mlops-assignment3 |

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/G25AIT2051-Ashish-Khandelwal/MLOps-Final-Project-Group34.git
cd MLOps-Final-Project-Group34
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python src/data_prep.py
```

Outputs `data/train_clean.csv` (5000 samples) and `data/test_clean.csv` (1000 samples).

### 3. Train on Kaggle

- Import `notebooks/train_v1.ipynb` into Kaggle
- Enable GPU T4 under Settings > Accelerator
- Add secrets: `HF_TOKEN`, `WANDB_API_KEY`
- Run all cells

### 4. Run Inference Locally

```bash
export HF_TOKEN=your_token
export INPUT_TEXT="This movie was great"
python src/inference.py
```

### 5. Run via Docker

```bash
docker pull somnchak/mlops-group34-inference:latest
docker run --rm -e HF_TOKEN=your_token -e INPUT_TEXT="Great film" somnchak/mlops-group34-inference:latest
```

### 6. Trigger GitHub Actions Inference

Go to Actions > Inference > Run workflow > enter text to classify.


## Model

- **Base:** distilbert-base-uncased (~67MB, 66.9M parameters)
- **Task:** Binary sentiment classification (positive / negative)
- **Dataset:** Stanford IMDb (25K reviews, sampled to 5K train + 1K test)

## CI/CD

- **CI (ci.yml):** Runs `flake8` linting on every push to `develop` / `develop2`
- **Inference (inference.yml):** Manual trigger via `workflow_dispatch`, accepts input text, loads model from HuggingFace, returns prediction

## Docker

```bash
# Build locally
docker build --build-arg HF_MODEL_NAME=somnathchakraborty/distilbert-imdb-sentiment \
  -t somnchak/mlops-group34-inference:latest .

# Test
docker run --rm -e HF_TOKEN=your_token -e INPUT_TEXT="Sample text" \
  somnchak/mlops-group34-inference:latest
```

## Team

Group 34 - IIT Jodhpur PGD AI Program
