# Football Player Valuation & Breakout Prediction

What will a football player be worth next season? Clubs, agents, and fans all want to know.

This project uses machine learning to **predict a player's next-period Transfermarkt market value** and **flag breakout talents and declining veterans** before the market catches up. An interactive Streamlit dashboard lets you explore any player's momentum, valuation history, and model predictions based on latest data available.

**Data source:** [Transfermarkt dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores/data) (`davidcariboo/player-scores`)

---

## Real-World Applications

- **Scouting:** rank hidden gems by predicted burst probability before competitors bid
- **Transfer negotiation:** ground asking prices in model-derived fair values
- **Squad planning:** identify aging players at decline risk ahead of the transfer window
- **Fantasy football:** use PMI and valuation momentum to inform team selection

---


## What it does

Three models are trained on half-season player performance snapshots:

| Model | Type | Task |
|---|---|---|
| LightGBM | Regression | Predict next half-season market value (EUR) |
| XGBoost Burst | Classification | Flag players likely to surge in value (top 5% growth, currently below median value) |
| XGBoost Decline | Classification | Flag veterans (30+) likely to drop sharply in value (bottom 5%) |

Features include: minutes/90, goals/90, assists/90, win rate, team strength, league tier, age, career stage, and etc,.

The Streamlit app (`app.py`) provides:
- **Player Momentum Index (PMI):** 10-game rolling performance score
- **Actual vs Predicted Valuations:** model output overlaid on real Transfermarkt values
- **Player Profile:** career stats, market value timeline, transfer history

---

## Model Performance

### Valuation Model — LightGBM (next half-season market value)

Five models were evaluated (Baseline, Random Forest, XGBoost, LightGBM, MLP). LightGBM was the clear winner:

| Model | R² (log) | R² (EUR) | RMSE (EUR) | MAPE |
|---|---|---|---|---|
| Baseline (no change) | 0.9043 | 0.9084 | €3,138,803 | 25.52% |
| Random Forest | 0.9206 | 0.8766 | €3,642,548 | 27.56% |
| XGBoost | 0.9317 | 0.9243 | €2,853,249 | 24.37% |
| **LightGBM** | **0.9325** | **0.9293** | **€2,755,924** | **23.93%** |
| MLP | 0.9027 | 0.7372 | €5,315,592 | 30.94% |

**LightGBM explains 93% of variance in player market value** and predicts the typical player's valuation within ~24% — beating even a strong "no-change" baseline by a meaningful margin.

---

### Breakout & Decline Models — XGBoost classifiers

Both targets are heavily imbalanced (burst: 5% positive rate; decline: 1% positive rate), so **Precision@K** is the primary metric: how many of the model's top-K flagged players are genuine cases.

#### Burst (hidden gem) detection

| Metric | Value |
|---|---|
| ROC-AUC | **0.9576** |
| PR-AUC | **0.5257** (random baseline ≈ 0.05) |
| Recall (at threshold 0.2) | 82.9% — catches 34 of 41 true breakout players |
| Precision@10 | **60%** — 6 of every 10 top picks are genuine breakouts |
| Precision@10 vs. random | **60× better** |

The model's top-10 picks average a **~20× value increase over 2 years** (log-change ≈ 3.0). A young-player-first heuristic baseline only achieves 4× growth on the same picks.

#### Rapid decline detection

| Metric | Value |
|---|---|
| ROC-AUC | **0.9764** |
| PR-AUC | **0.2860** (random baseline ≈ 0.01) |
| Recall (at threshold 0.2) | 87.5% — catches 7 of 8 true decline cases |
| Precision@10 | **40%** — 4 of every 10 flagged veterans are genuine decline cases |
| Precision@10 vs. random | **40× better** |

The model's top picks show strongly negative valuation trajectories, confirming it genuinely identifies decline rather than just filtering by age.



---

## Quick Start

### Prerequisites
- Python 3.9+
- A Kaggle account with an API token (`~/.kaggle/kaggle.json`) — needed for the data download step

### Option A — Docker (recommended)

The easiest way to run the project with no environment setup.

**Prerequisite:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

**Step 1 — Train the models**

This downloads the dataset, processes features, and saves trained models to your local `models/` folder:

```bash
docker compose run --rm pipeline
```

Your Kaggle credentials are mounted read-only from `~/.kaggle/kaggle.json`. Make sure that file exists before running.

**Step 2 — Launch the app**

```bash
docker compose up app
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> Data and models are stored on your machine (not inside the container) via volume mounts, so they persist across runs.

---

### Option B — Local Python

### 1. Environment setup & data download

```bash
bash setup.sh
```

This creates a `.venv`, installs all dependencies, and downloads the dataset from Kaggle automatically.

Activate the environment afterwards:

```bash
source .venv/bin/activate
```

### 2. Train the models

```bash
python src/pipeline.py
```

This runs the full pipeline: download → process → train → save models to `models/`.

**Flags:**

```bash
# Data already downloaded
python src/pipeline.py --skip-download

# Data already processed (dataset.csv exists)
python src/pipeline.py --skip-download --skip-processing
```

### 3. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── src/
│   ├── pipeline.py         # End-to-end training pipeline
│   ├── player_profile.py   # Player profile helpers
│   └── modules/
│       ├── config.py       # Paths, constants, hyperparameters
│       ├── data_download.py
│       ├── data_processing.py
│       ├── model_valuation.py   # LightGBM regressor
│       ├── model_classification.py  # XGBoost burst/decline
│       └── predictor.py    # Inference for the app
├── notebooks/              # EDA and modelling notebooks
├── models/                 # Saved model artifacts (gitignored)
├── data/                   # Raw Kaggle CSVs (gitignored)
├── processed/              # Engineered features (gitignored)
├── setup.sh
└── requirements.txt
```

---

## Dependencies

Key packages: `lightgbm`, `xgboost`, `scikit-learn`, `streamlit`, `pandas`, `plotly`, `kagglehub`

Full list in `requirements.txt`.
