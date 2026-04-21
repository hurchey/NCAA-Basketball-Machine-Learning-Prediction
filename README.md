# NCAA Basketball Machine Learning Prediction

Predicting the 2024 NCAA Division I Men's Basketball Tournament (March Madness) champion using historical tournament data, regular-season statistics, and a deep-learning model.

## Overview

Every March, 68 college basketball teams face off in a single-elimination tournament where a single bad game ends a championship run. The odds of filling out a perfect bracket are roughly 1 in 9.2 quintillion — but with enough historical data and the right model, you can dramatically improve your picks over guessing.

This project trains models on recent seasons of NCAA men's basketball data (from the Kaggle *March Machine Learning Mania 2023* competition, plus FiveThirtyEight's tournament forecasts) and uses an LSTM neural network to predict which team is most likely to win the 2024 tournament. The entire workflow — data loading, feature engineering, model training, and final prediction — lives in a single Google Colab notebook.

## Project Structure

```
NCAA-Basketball-Machine-Learning-Prediction/
├── March_Madness_2023.ipynb    # End-to-end analysis and modeling notebook
└── README.md
```

## Dataset

Two data sources are used:

- **[March Machine Learning Mania 2023](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/data)** — Kaggle's official competition dataset, including:
  - `MNCAATourneyDetailedResults.csv` — detailed box scores for every tournament game
  - `MNCAATourneySeeds.csv` — seeding for each team in each tournament
  - `MRegularSeasonDetailedResults.csv` — detailed box scores for every regular-season game
- **FiveThirtyEight NCAA forecasts** (`fivethirtyeight_ncaa_forecasts.csv`) — used for team-ID-to-name mapping and as an external benchmark

The notebook expects these files in `/content/drive/MyDrive/March_Madness_2023/`, since it's set up to run in Google Colab with Google Drive mounted. If you're running locally, update the `pd.read_csv` paths in Step 1 to point to wherever you've placed the CSVs.

## Approach

The notebook is organized into four clear steps:

### Step 1 — Setting up the data
Loads the tournament results, seeds, regular-season results, and FiveThirtyEight forecasts into pandas DataFrames.

### Step 2 — Feature engineering and data representation
- Builds a `team_id → team_name` lookup from the FiveThirtyEight file
- Computes field-goal percentage (FG%) for winning and losing teams in every game
- Aggregates per-team season statistics (average score, average FG%)
- Filters to the most recent **5 seasons** to keep the dataset focused and recent
- A **Random Forest Regressor** (`n_estimators=100`) is trained on these aggregate stats to rank team performance and surface top-performing teams

### Step 3 — LSTM model training
- Normalizes the recent-season stats with `MinMaxScaler`
- Builds time-series sequences (5-season lookback window) using Keras' `TimeseriesGenerator`
- Trains a stacked **LSTM** network:
  - `LSTM(50, return_sequences=True)`
  - `LSTM(50)`
  - `Dense(1)` output
  - Optimizer: `adam`, loss: `mean_squared_error`

### Step 4 — 2024 prediction
Applies the trained LSTM to the 2023 season data to rank teams, takes the `argmax` of the model's outputs, and maps the resulting team ID back to a team name to declare the predicted 2024 champion.

Evaluation metrics referenced in the notebook include **Brier score loss** (the official Kaggle competition metric) and **mean absolute error** as a proxy for win-margin prediction.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook, JupyterLab, or Google Colab (the notebook is set up for Colab by default)

### Installation

Clone the repository:

```bash
git clone https://github.com/hurchey/NCAA-Basketball-Machine-Learning-Prediction.git
cd NCAA-Basketball-Machine-Learning-Prediction
```

Install the required packages:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn xgboost tensorflow jupyter
```

Or, if you prefer a `requirements.txt`:

```
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
xgboost
tensorflow
jupyter
```

### Usage

**Option A — Google Colab (recommended, matches the notebook as-written):**
1. Open the notebook in Colab via the badge at the top of the file.
2. Download the [March Machine Learning Mania 2023](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/data) dataset plus the FiveThirtyEight forecasts CSV.
3. Upload the CSVs to `My Drive/March_Madness_2023/` on Google Drive.
4. Run the cells in order — the notebook will mount Drive and read the files from there.

**Option B — Local Jupyter:**
1. Download the datasets as above.
2. Place them in a local `data/` folder inside the project directory.
3. Remove (or comment out) the `drive.mount(...)` cell.
4. Update the `pd.read_csv(...)` paths to point to your local `data/` folder.
5. Launch Jupyter and run the notebook:
   ```bash
   jupyter notebook March_Madness_2023.ipynb
   ```

## Results

Running the full notebook produces a single predicted 2024 champion, printed at the end of Step 4 in the form:

```
Predicted top team for 2024: <team name>
```

The ranking is derived from the LSTM's output scores across the 2023 teams, with the top `argmax` treated as the predicted champion. Secondary metrics (Brier score loss and MAE) are set up in Step 1 for evaluation against held-out tournaments.

> Note: The LSTM training loop currently uses placeholder arrays for `X_train` / `y_train` as a scaffold — a natural next step is to wire the `TimeseriesGenerator` output directly into `model.fit` and report validation metrics on held-out seasons.

## Future Work

- Replace the scaffolded LSTM inputs with the real `TimeseriesGenerator` outputs and report validation Brier score
- Expand the feature set beyond average score and FG% (rebounds, turnovers, assists, strength of schedule)
- Incorporate advanced external metrics (KenPom, BPI, NET rankings)
- Compare the LSTM against simpler baselines (logistic regression, XGBoost) using log-loss
- Extend the pipeline to predict *every* tournament matchup, not just the champion, and generate a full bracket
- Apply the same pipeline to the women's tournament (`WNCAATourney*` files from the same Kaggle dataset)

## Acknowledgments

- [Kaggle's March Machine Learning Mania 2023](https://www.kaggle.com/competitions/march-machine-learning-mania-2023) for the tournament and regular-season data
- [FiveThirtyEight](https://projects.fivethirtyeight.com/) for their NCAA tournament forecasts
- The broader March Machine Learning Mania community for years of open-sourced approaches and ideas

## Author

**Eric Hurchey** — [@hurchey](https://github.com/hurchey)

## License

Released under the [MIT License](https://opensource.org/licenses/MIT).
