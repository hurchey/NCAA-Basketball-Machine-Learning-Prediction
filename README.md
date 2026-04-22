# NCAA Basketball Machine Learning Prediction

A machine learning project for predicting NCAA Tournament outcomes using detailed regular-season box-score data. Three modeling approaches are implemented end-to-end — an XGBoost margin regressor with spline-calibrated win probabilities, a Random Forest next-season scoring projection, and an LSTM time-series experiment — each evaluated against an appropriate baseline.

This project prioritizes honest evaluation over model complexity. One model works modestly, one works on a different task, one failed to outperform a naive baseline and is documented as a failed experiment.

---

## Results at a Glance

| Model | Task | Metric | Result | Baseline |
|---|---|---|---|---|
| XGBoost + Spline | Predict winner of 2022 tournament games | Brier score | 0.2467 | 0.2500 (50/50) |
| XGBoost + Spline | Predict winner of 2022 tournament games | Accuracy | 55.2% | 50.0% |
| Random Forest | Predict team's next-season PPG | MAE / R² | 3.59 / 0.074 | 3.90 / -0.151 |
| LSTM | Predict team's next-season PPG (time series) | Outcome | Collapsed to constant predictions | — |

**Bracket simulation:** The XGBoost model was run on the full 2022 bracket and predicted **Gonzaga** as the champion — matching the consensus pre-tournament favorite (Gonzaga was the #1 overall seed). Kansas, the actual champion, was knocked out in the Sweet 16 of the simulation.

---

## Data

Source: [Kaggle — March Machine Learning Mania 2023](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/data)

Files used:
- `MRegularSeasonDetailedResults.csv` — 120k+ regular-season games, 2003-2023, with full box scores
- `MNCAATourneyDetailedResults.csv` — 1,248 tournament games, 2003-2022
- `MNCAATourneySeeds.csv` — tournament seeding and regions
- `MTeams.csv` — team IDs and names
- `fivethirtyeight_ncaa_forecasts.csv` — 538's 2023 tournament forecasts (reference)

All Kaggle competition files go in a single folder; the notebook expects them at `/content/drive/MyDrive/March_Madness_2023/`.

---

## Approach

### 1. Feature engineering

For every (season, team) pair, the notebook computes games-weighted averages from the detailed box scores — points per game, field goal / 3P / FT percentages, rebounds, assists, turnovers, and win percentage — aggregated separately from games the team won and lost, then combined with proper weighting.

Each game becomes a training row with **difference features** (`team1_stat - team2_stat`) and a **signed margin** target (`team1_score - team2_score`). Team1 is always the lower TeamID for orientation consistency.

### 2. XGBoost + Spline (primary model)

- **Stage 1**: XGBoost regressor (400 trees, depth 4, lr 0.05) predicts signed win margin from the difference features.
- **Stage 2**: A smoothing spline (`scipy.interpolate.UnivariateSpline`) maps predicted margin → empirical win probability, fit on training predictions.

The spline is clipped to `[0.20, 0.80]` to reflect the empirical tournament upset rate. Without clipping, the model was overconfident in its highest-probability predictions: the 80-100% predicted bucket only won 57% of the time, which is the classic March Madness upset effect.

Evaluated with **Brier score**, the same calibration metric used to grade the Kaggle competition and real-money sports prediction markets.

### 3. Random Forest (secondary model)

A separate task: given a team's prior-season stats, predict their next-season points per game. Uses lag features (`prev_ppg`, `prev_fg_pct`, `prev_win_pct`) with a strictly chronological train/test split to avoid leakage.

The RF modestly outperforms a naive "predict last season unchanged" baseline. The baseline's negative R² is itself informative — it means teams regress toward the mean between seasons, and a model that blindly assumes persistence does worse than predicting the league average. The RF correctly learns to apply some mean-reversion.

### 4. LSTM (failed experiment)

Two-layer LSTM on 3-season sequences per team, predicting next-season PPG. The model collapsed to near-constant predictions (~18 pts across all teams) while actual scoring ranged 65-88 pts. Retained in the notebook to document the failure honestly — the Random Forest captures the main signal (mean-reversion) with far less data and complexity.

Known causes: only two input features, short per-team sequence length after windowing, and limited overall training volume. Not worth further iteration on without substantially richer features.

---

## Limitations and Future Work

The XGBoost model captures real signal but the edge over a 50/50 baseline on tournament games is modest (1.3% Brier improvement on the 2022 tournament, ~55% accuracy). This is consistent with what basic box-score-only features can achieve; competitive public models (Kaggle winners, KenPom, FiveThirtyEight) typically land in the 0.17-0.19 Brier range with richer feature sets.

The biggest limitation surfaced clearly in the 2022 bracket simulation: the model placed **Murray State** and **South Dakota State** in the Elite Eight — both deep-tournament picks that didn't happen in reality. Without strength-of-schedule adjustments, mid-majors with dominant in-conference records get overvalued relative to power-conference teams that played tougher opponents.

The primary next steps would be:
- **Pace-adjusted efficiency metrics** (points per 100 possessions, KenPom-style)
- **Strength-of-schedule adjustments** based on opponent quality
- **Team-rating features** from external sources (Elo, KenPom, or the fivethirtyeight file already in this repo)
- **Rolling-form features** (last-10-games weighted higher than full-season averages)

Seed differential alone (from `MNCAATourneySeeds.csv`) is the single highest-leverage feature that isn't currently used — a known strong predictor of tournament outcomes that would likely drop the Brier score meaningfully with ~30 minutes of work.

---

## How to Run

1. Clone the repo
2. Download the Kaggle competition files listed above and place them in a folder named `March_Madness_2023/` in your Google Drive
3. Open `March_Madness_2023.ipynb` in Google Colab
4. `Runtime → Restart and run all`

The notebook will train all three models, produce the calibration plot, simulate the 2022 bracket round by round, and print a final summary.

---

## Tools

Python · XGBoost · scikit-learn · TensorFlow/Keras · SciPy · pandas · NumPy · matplotlib
