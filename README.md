# NCAA Basketball Machine Learning Prediction

Predicting the outcomes of the 2023 NCAA Division I Men's Basketball Tournament (March Madness) using machine learning.

## Overview

Every March, 68 college basketball teams face off in a single-elimination tournament where a single bad game ends a championship run. The odds of filling out a perfect bracket are roughly 1 in 9.2 quintillion — but with enough historical data and the right model, you can dramatically improve your picks over guessing.

This project uses regular-season and historical tournament data to train a classification model that predicts, for any given matchup, which team is more likely to win. The entire workflow — data loading, feature engineering, model training, evaluation, and bracket generation — lives in a single Jupyter notebook.

## Project Structure

```
NCAA-Basketball-Machine-Learning-Prediction/
├── March_Madness_2023.ipynb    # End-to-end analysis and modeling notebook
└── README.md
```

## Dataset

This project uses the [March Machine Learning Mania 2023](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/data) dataset from Kaggle, which includes:

- **Regular season results** — game-by-game scores and box-score statistics
- **Historical tournament results** — every NCAA tournament game since 1985
- **Team metadata** — team IDs, names, conferences, and seeding
- **Seeds** — tournament seeding for each team in each season

To run the notebook, download the competition data from Kaggle and place the CSV files in a local `data/` directory (or update the file paths in the notebook to match your setup).

## Approach

The notebook walks through a standard machine-learning pipeline:

1. **Data loading** — reading in regular-season, tournament, and seeding CSVs
2. **Exploratory data analysis** — examining seed-vs-win distributions, team performance trends, and feature correlations
3. **Feature engineering** — building per-team season aggregates (points per game, field-goal %, rebounds, turnovers, etc.) and computing the *difference* between Team A and Team B as the input vector for each matchup
4. **Model training** — training classification models on historical tournament matchups to predict win probability
5. **Evaluation** — measuring accuracy and log-loss on held-out seasons
6. **2023 predictions** — applying the trained model to the 2023 bracket and generating round-by-round picks

> **TODO:** Update this section with the specific model(s) you used (e.g., Logistic Regression, Random Forest, XGBoost) and your final validation accuracy / log-loss.

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

Clone the repository:

```bash
git clone https://github.com/hurchey/NCAA-Basketball-Machine-Learning-Prediction.git
cd NCAA-Basketball-Machine-Learning-Prediction
```

Install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

> **TODO:** If you used additional libraries (e.g., `xgboost`, `lightgbm`), add them to the install command above or include a `requirements.txt`.

### Usage

1. Download the [March Machine Learning Mania 2023](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/data) dataset from Kaggle.
2. Place the CSVs in a `data/` folder inside the project directory (or update the paths in the notebook).
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook March_Madness_2023.ipynb
   ```
4. Run the cells in order to reproduce the analysis and predictions.

## Results

> **TODO:** Summarize your model's performance here. For example:
> - Validation accuracy: XX%
> - Log-loss: X.XX
> - Bracket score vs. chalk / ESPN average
> - Notable correct upsets predicted
> - Predicted Final Four / Champion

## Future Work

- Incorporate advanced metrics (KenPom ratings, BPI, NET rankings)
- Experiment with ensemble methods and hyperparameter tuning
- Extend the pipeline to predict the women's tournament
- Build a simple web UI to visualize the predicted bracket
- Generalize the notebook to run on any tournament year with a single config change

## Acknowledgments

- Kaggle and the NCAA for the tournament dataset
- The broader March Machine Learning Mania community for years of open-sourced approaches and ideas

## Author

**Eric Hurchey** — [@hurchey](https://github.com/hurchey)

## License

> **TODO:** Add a license (MIT is a common choice for personal projects) and include a `LICENSE` file.
