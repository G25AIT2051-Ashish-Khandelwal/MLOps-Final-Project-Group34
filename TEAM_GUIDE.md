# Team Guide - How to Use This Pipeline

This document explains what has been built, how each piece connects, and how teammates can trigger and test the pipeline.

## Pipeline Flow

```
                              ┌─────────────────────────────┐
                              │     KAGGLE NOTEBOOK         │
                              │                             │
  ┌──────────┐   ┌─────────┐ │  ┌─────────┐  ┌──────────┐ │  ┌──────────────┐
  │ IMDb     │──>│ data_   │ │  │ Load    │─>│ Train    │ │─>│ HuggingFace  │
  │ Dataset  │   │ prep.py │ │  │ Secrets │  │ DistilBERT│ │  │ Model Hub    │
  └──────────┘   └────┬────┘ │  └─────────┘  └────┬─────┘ │  │ (public)     │
                      │      │                     │       │  └──────┬───────┘
                      ▼      │                     ▼       │         │
                 ┌─────────┐ │              ┌──────────┐   │         │
                 │ CSV     │ │              │ W&B      │   │         │
                 │ Files   │ │              │ Tracking │   │         │
                 └─────────┘ │              └──────────┘   │         │
                              └─────────────────────────────┘         │
                                                                      │
         ┌────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────┐    ┌─────────────────────────────────────────┐
  │ inference.py │    │          GITHUB ACTIONS                 │
  │ (loads from  │    │                                         │
  │  HF Hub)     │    │  Push to develop ──> CI (flake8 lint)   │
  └──────┬───────┘    │                                         │
         │            │  Manual trigger ──> Inference workflow   │
         ▼            │   (enter text)      (runs inference.py) │
  ┌──────────────┐    └─────────────────────────────────────────┘
  │   DOCKER     │
  │  Container   │
  │  (portable)  │
  └──────────────┘
```

## What Has Been Done

### Code (in `develop2` branch)

| File | What It Does |
|------|-------------|
| `src/data_prep.py` | Downloads IMDb dataset from HuggingFace, cleans HTML tags, removes duplicates, creates train (5000) and test (1000) CSVs |
| `src/train.py` | Fine-tunes DistilBERT with W&B experiment tracking, supports two versions with different hyperparameters, pushes best model to HuggingFace Hub |
| `src/inference.py` | Loads trained model from HuggingFace Hub, accepts input text via environment variable, outputs prediction with confidence |
| `notebooks/train_v1.ipynb` | Kaggle notebook that runs the full training pipeline with GPU |
| `Dockerfile` | Packages inference.py into a container that pulls model from HuggingFace |
| `.github/workflows/ci.yml` | Runs flake8 linting on every push to develop/develop2 |
| `.github/workflows/inference.yml` | Manual workflow - enter any text, it loads model from HF and classifies it |

### Accounts and Services

| Service | What's Set Up | Who Has Access |
|---------|--------------|----------------|
| GitHub Repo | Public, secrets added (HF_TOKEN, WANDB_API_KEY) | All collaborators |
| Kaggle | Notebook with GPU runs, secrets configured | Somnath |
| HuggingFace | Model pushed, set to public | Somnath |
| W&B | Runs tracked under ashish-iit-jodhpur-25ait2051/mlops-assignment3 | All team members |
| Docker Hub | Image pushed as somnchak/mlops-group34-inference | Public |

### Training Results

Two experiment versions were run on Kaggle with GPU T4:

| | V1 (baseline) | V2 (tuned) |
|--|---------------|------------|
| Epochs | 3 | 4 |
| Batch Size | 16 | 32 |
| Learning Rate | 2e-5 | 5e-5 |
| Weight Decay | 0.01 | 0.02 |
| Warmup Steps | 100 | 50 |
| **Accuracy** | 0.8950 | **0.9060** |
| **F1** | 0.8948 | **0.9060** |
| **Loss** | 0.5752 | **0.5129** |

V2 performed better due to the higher learning rate and additional epoch.

---

## How Teammates Can Use This

### Prerequisites

1. Git access to the repo (you should already be a collaborator)
2. Python 3.10+ installed
3. Docker Desktop installed (for container testing)
4. Kaggle account (for training)
5. W&B account (for experiment tracking)

### Step 1: Clone the Repo

```bash
git clone https://github.com/G25AIT2051-Ashish-Khandelwal/MLOps-Final-Project-Group34.git
cd MLOps-Final-Project-Group34
git checkout develop2
pip install -r requirements.txt
```

### Step 2: Run Data Preparation (optional - just to verify)

```bash
python src/data_prep.py
```

This downloads IMDb data, cleans it, and saves CSVs locally.

### Step 3: Run Training on Kaggle

1. Go to https://www.kaggle.com
2. Create a new notebook or fork the existing one
3. Copy the cells from `notebooks/train_v1.ipynb`
4. Under Settings:
   - Set Accelerator to GPU T4 x2
   - Enable Internet
   - Add secrets: `HF_TOKEN` and `WANDB_API_KEY`
5. For V1: keep the config cell as is
6. For V2: change the config cell:
   ```python
   VERSION = "v2"
   EPOCHS = 4
   BATCH_SIZE = 32
   LEARNING_RATE = 5e-5
   WEIGHT_DECAY = 0.02
   WARMUP_STEPS = 50
   ```
7. Run All
8. Save the notebook version after each run

### Step 4: Run Inference Locally

```bash
export HF_TOKEN=your_huggingface_token
export INPUT_TEXT="This movie was fantastic"
python src/inference.py
```

Expected output:
```
Loading model: somnathchakraborty/distilbert-imdb-sentiment
Input text: This movie was fantastic
Prediction: positive
Confidence: 0.98
```

### Step 5: Run Inference via Docker

```bash
docker pull somnchak/mlops-group34-inference:latest

docker run --rm \
  -e HF_TOKEN=your_token \
  -e INPUT_TEXT="Worst movie ever" \
  somnchak/mlops-group34-inference:latest
```

### Step 6: Trigger GitHub Actions Inference

1. Go to https://github.com/G25AIT2051-Ashish-Khandelwal/MLOps-Final-Project-Group34/actions
2. Click "Inference" workflow on the left
3. Click "Run workflow" button
4. Select branch: `develop2`
5. Enter any text in the input field
6. Click "Run workflow"
7. Wait ~2 minutes, then check the run logs for the prediction

### Step 7: View Experiment Tracking

Go to https://wandb.ai/ashish-iit-jodhpur-25ait2051/mlops-assignment3 to see all training runs, metrics, and comparisons.

---

## How to Make Your Own Commits

```bash
git checkout develop2
git pull origin develop2

# make your changes

git add .
git commit -m "your message"
git push origin develop2
```

This will automatically trigger the CI pipeline (linting).

---

## Secrets Reference

These are already configured but if you need to set up your own:

| Secret | Where to Get It | Where to Add It |
|--------|----------------|-----------------|
| `HF_TOKEN` | https://huggingface.co/settings/tokens | Kaggle Secrets + GitHub Repo Secrets |
| `WANDB_API_KEY` | https://wandb.ai/authorize | Kaggle Secrets + GitHub Repo Secrets |

---

## Troubleshooting

**Docker "gated repo" error:** Make sure the HuggingFace model is set to Public and you're passing a valid HF_TOKEN.

**CI failing:** Run `flake8 src/ --max-line-length=120` locally to check for lint errors before pushing.

**Kaggle out of GPU hours:** Reduce `n=5000` to `n=2000` in data_prep section of the notebook, or reduce epochs.

**token_type_ids error:** Already fixed in inference.py. DistilBERT doesn't use token_type_ids, so they're stripped before inference.
