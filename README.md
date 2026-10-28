# Ecuadorian Banking Delinquency Forecasting

### Reproducible Machine Learning for Credit Portfolio Risk Analysis in Ecuador

An end-to-end machine learning project for forecasting monthly loan delinquency ratios across Ecuadorian private banks, with an emphasis on temporal validation, experiment tracking, reproducibility, and transparent model evaluation.

**Tech stack:** Python 3.12 · pandas · NumPy · scikit-learn · MLflow · pytest · Ruff · Jupyter · micromamba

## 1. Project Overview

Loan delinquency is an important indicator of credit portfolio quality and financial sector performance.

This project investigates whether machine learning models can forecast the monthly delinquency ratio of Ecuadorian private banks using historical financial indicators and temporal features.

The main goals are to:

- Construct and validate a historical bank-level financial dataset.
- Engineer predictive features without introducing future information.
- Compare statistical benchmarks and machine learning models.
- Evaluate forecast horizons of one, two, and three months.
- Track experiments using MLflow.
- Provide reusable Python modules and reproducible inference.
- Document model performance, limitations, and potential applications.

The project forecasts institutional delinquency ratios, not individual loan defaults.

## 2. Data

The dataset was constructed from historical monthly financial reports published by the [Superintendencia de Bancos del Ecuador](https://www.superbancos.gob.ec/).

| Attribute | Value |
|---|---|
| Country | Ecuador |
| Sector | Private banking |
| Financial institutions | 23 |
| Historical coverage | January 2003 – September 2026 |
| Bank-month observations | 5,972 |
| Frequency | Monthly |
| Forecast horizons | 1, 2, and 3 months |
| Prediction task | Regression |

The target variable is the delinquency ratio:

\[
M_{b,t} =
\frac{\text{Non-performing portfolio}_{b,t}}
{\text{Gross loan portfolio}_{b,t}}
\times 100
\]

where \(b\) represents the bank and \(t\) the observation month.

### Data integrity

The preparation pipeline includes:

- Bank identifier normalization.
- Financial indicator consistency checks.
- Duplicate bank-month detection.
- Missing-value analysis.
- Historical panel validation.
- SHA-256 verification of original data files.

The original Excel reports are not automatically distributed with the repository. See the reproduction instructions below.

## 3. Methodology

### Feature engineering

Predictive information includes historical delinquency values, lagged observations, monthly changes, rolling statistics, solvency, liquidity, profitability and portfolio-related indicators.

The modeling pipeline also incorporates cyclical calendar features and institution identifiers.

The selected one-month Random Forest uses **22 predictors**.

### Models evaluated

| Approach | Description |
|---|---|
| Persistence | Future delinquency equals current delinquency |
| Seasonal persistence | Current delinquency plus a learned calendar-month adjustment |
| Ridge Regression | Regularized linear regression |
| Random Forest | Ensemble of decision trees |
| HistGradientBoosting | Histogram-based gradient boosting |

Machine learning models were studied using both direct forecasting and change-based formulations.

For the delta formulation:

\[
\Delta M_{b,t+h}=M_{b,t+h}-M_{b,t}
\]

The forecast is reconstructed as:

\[
\widehat M_{b,t+h}
=M_{b,t}+\widehat{\Delta M}_{b,t+h}
\]

### Temporal validation

The experimental design uses five expanding-window chronological validation folds covering 2021–2025.

All institutions share chronological validation boundaries. Training target dates are restricted to precede the relevant forecast origins.

The final held-out period covers target months through September 2026, with observation counts varying by horizon.

Test data was excluded from model selection and hyperparameter tuning.

## 4. Final Forecasting System

The selected system is hybrid:

| Horizon | Strategy |
|---|---|
| 1 month | Random Forest — flexible delta configuration |
| 2 months | Seasonally adjusted persistence |
| 3 months | Seasonally adjusted persistence |

The one-month Random Forest configuration uses 150 trees, maximum depth 14, minimum leaf size 3, and `max_features=0.7`.

For the seasonal strategy, the adjustment for each target calendar month is the historical median change learned from the training period. Predictions are clipped to the interval from 0 to 100 percentage points.

## 5. Experimental Results

### Final held-out test

All prediction errors are expressed in percentage points (pp).

| Horizon | Observations | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| 1 month | 276 | 0.2839 | 0.4765 | 0.9590 |
| 2 months | 253 | 0.3902 | 0.6937 | 0.9111 |
| 3 months | 230 | 0.4636 | 0.8031 | 0.8798 |

At the one-month horizon, the selected Random Forest achieved an **8.19% lower MAE** than seasonal persistence.

The improvements were not uniform across banks and periods.

For two- and three-month forecasts, seasonal persistence is the selected model itself, so its results equal the baseline by design.

### Early-warning analysis

The ability to forecast delinquency levels does not necessarily imply the ability to anticipate unusually large increases.

| Monthly increase | Real events | Detected by RF | Recall |
|---|---:|---:|---:|
| At least 0.5 pp | 23 | 5 | 21.7% |
| At least 1.0 pp | 9 | 1 | 11.1% |

The model is more promising for short-term delinquency forecasting than for autonomous early-warning detection of substantial portfolio deterioration.

## 6. Installation

### Requirements

- Linux is the primary tested development environment.
- Git.
- micromamba or Conda.
- Python and project dependencies managed through `environment.yml`.
- A notebook-capable editor such as VS Code, or a compatible Jupyter installation.

### Clone the repository

```bash
git clone https://github.com/tavomx45/audit-ecuador-banking-data.git
cd audit-ecuador-banking-data
```

### Create and activate the environment

Using micromamba:

```bash
micromamba create -f environment.yml
micromamba activate morosidad-bancos-ecuador
```

If the environment already exists:

```bash
micromamba update -n morosidad-bancos-ecuador -f environment.yml
micromamba activate morosidad-bancos-ecuador
```

Verify the environment:

```bash
python --version
python -c "import sklearn, pandas, mlflow; print('Dependencies OK')"
```

The published `environment.yml` must include MLflow for the final verification command to work. See the dependency note below.

## 7. Reproducing the Dataset

### Obtain the original financial reports

Download the corresponding historical reports from the Ecuadorian banking regulator.

The reconstruction pipeline expects the source Excel files in:

```text
data/raw/series_entidad/
```

The currently implemented pipeline selects the institution-level Excel files matching:

```text
*2026_09.xlsx
```

The repository includes a metadata manifest at:

```text
data/metadata/raw_manifest.json
```

This manifest records the expected source files, sizes, and SHA-256 hashes.

**Important:** placing a different collection of files in `data/raw/` is not equivalent to reproducing the original dataset.

### Verify original files

From the project root:

```bash
python -m src.data_manifest verificar
```

A successful verification means that the supplied raw files match the recorded manifest.

If the original files are unavailable, this step cannot be reproduced exactly.

Do not replace the original manifest with one generated from different files and describe the resulting data as an exact reproduction.

### Rebuild the panel and features

```bash
python -m src.pipeline --write
```

This command validates the original data, reconstructs the banking panel, performs feature engineering, and writes the processed outputs.

The modeling dataset is saved to:

```text
data/processed/rebuild/panel_features_multihorizonte.parquet
```

The generated files remain local and are excluded from Git.

An additional optional check is available:

```bash
python -m src.pipeline --verify-reference
```

This comparison requires the original academic reference Parquet files, which are not distributed automatically with the repository. It is not necessary for the standard rebuild.

## 8. Reproducing the Experiments

Once the feature dataset has been reconstructed, use the modeling notebooks in the `notebooks/` directory.

The primary modeling notebooks are:

```text
notebooks/03_modelado.ipynb
notebooks/04_experimentos_delta.ipynb
```

The workflow includes:

1. Preparing datasets separately for each forecast horizon.
2. Creating chronological expanding-window folds.
3. Evaluating persistence and seasonal persistence benchmarks.
4. Training Ridge, Random Forest, and HistGradientBoosting models.
5. Comparing direct and delta formulations.
6. Performing feature ablation and Random Forest hyperparameter experiments.
7. Selecting a hybrid strategy using development results.
8. Conducting the final held-out test.
9. Analyzing bank-level, monthly, and early-warning errors.

Run the notebook cells in their original order using the environment created above.

The original experimental results are reported separately from the held-out test results. Re-running the experiments may require additional computation, depending on the number of model-fold combinations.

## 9. MLflow Experiment Tracking

MLflow is used to record:

- Model configurations and hyperparameters.
- Validation metrics.
- Final test metrics.
- Experiment metadata and tags.
- The registered Random Forest pipeline.
- Supporting model artifacts.

Some historical experiment results were imported retrospectively from previously executed notebook experiments. Those records should not be interpreted as original MLflow-tracked training runs.

### Start the local tracking server

Open a terminal and run:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

In another terminal, activate the environment and configure:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

Open the tracking interface at:

http://127.0.0.1:5000

The main experiment is:

```text
morosidad-bancos-ecuador
```

The registered one-month forecasting model is:

```text
MorosidadRFDeltaH1
```

Model registry version numbers are specific to each MLflow server. The original local experiment used version 2; a fresh installation may assign a different version.

The MLflow database and stored model binaries are not automatically included in the Git repository.

## 10. Rebuilding the Final Models

After reconstructing the feature dataset, the selected model can be retrained using the training-safe temporal cutoff.

The following example shows the preparation and fitting procedure for the one-month Random Forest.

```python
import numpy as np
import pandas as pd

from src.modeling.data import preparar_datasets
from src.modeling.models import construir_rf_delta_flexible

features = pd.read_parquet(
    "data/processed/rebuild/panel_features_multihorizonte.parquet"
)

datasets, _ = preparar_datasets(features)

config = datasets[1]
desarrollo = config["desarrollo"]
test = config["test"]

primer_origen_test = (
    pd.to_datetime(test["fecha"])
    .dt.to_period("M")
    .min()
)

fechas_target = (
    pd.to_datetime(desarrollo[config["fecha_target"]])
    .dt.to_period("M")
)

train = desarrollo.loc[
    fechas_target < primer_origen_test
].copy()

X_train = train[config["predictoras"]].replace(
    [np.inf, -np.inf],
    np.nan,
)

y_delta = (
    train[config["target"]] - train["morosidad_pp"]
)

modelo = construir_rf_delta_flexible(
    config["numericas"]
)

modelo.fit(X_train, y_delta)
```

For the original experiment, this procedure uses 5,643 training observations.

The fitted pipeline contains the preprocessing and estimator used for one-month delta forecasting.

To persist it in MLflow, use `mlflow.sklearn.log_model()` and register it under `MorosidadRFDeltaH1`, following the final-model registration notebook section.

The seasonal strategies for horizons 2 and 3 use the function `aprender_ajustes_estacionales()` and the same training-safe cutoff. Their serialized parameters are stored in:

```text
configs/modeling/seasonal_h2.json
configs/modeling/seasonal_h3.json
```

These JSON files must be available to the inference module. They do not require a running MLflow server.

## 11. Running Inference

The project includes the function:

```python
from src.modeling.inference import pronosticar_morosidad
```

It accepts a DataFrame containing the institution, forecast origin, current delinquency, and engineered predictive features.

Example using the reconstructed dataset:

```python
import pandas as pd

from src.modeling.data import preparar_datasets
from src.modeling.inference import pronosticar_morosidad

features = pd.read_parquet(
    "data/processed/rebuild/panel_features_multihorizonte.parquet"
)

datasets, _ = preparar_datasets(features)

entrada = datasets[1]["test"].copy()

entrada = entrada.drop(
    columns=[
        col for col in entrada.columns
        if col.startswith("target_")
    ]
)

pronosticos = pronosticar_morosidad(
    datos=entrada,
    horizonte=1,
    model_uri="models:/MorosidadRFDeltaH1/2",
)

print(pronosticos.head())
```

Replace version `2` with the version available in your own MLflow Model Registry.

The output contains the bank, forecast origin, target date, current delinquency, predicted change, and predicted delinquency.

For horizons 2 and 3, call the same function with `horizonte=2` or `horizonte=3`. The interface then uses the serialized seasonal parameters.

**Note:** the inference function requires already engineered features. It is not a raw-Excel-to-prediction interface.

## 12. Automated Tests and Code Quality

The project uses pytest and Ruff.

Run the full automated test suite:

```bash
python -m pytest -q
```

Check code quality:

```bash
ruff check src tests
ruff format --check src tests
```

The suite covers data preparation, financial validations, feature engineering, temporal modeling, and inference behavior.

Inference tests include validations for missing features, duplicate bank-month observations, future target columns, seasonal adjustments, and model loading behavior.

## 13. Project Structure

The main project components include:

```text
audit-ecuador-banking-data/
├── configs/
│   └── modeling/
│       ├── seasonal_h2.json
│       └── seasonal_h3.json
├── data/
│   ├── metadata/
│   │   └── raw_manifest.json
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 03_modelado.ipynb
│   └── 04_experimentos_delta.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── boletines.py
│   ├── data_manifest.py
│   ├── features.py
│   ├── pipeline.py
│   └── modeling/
│       ├── baselines.py
│       ├── data.py
│       ├── inference.py
│       ├── models.py
│       └── ...
├── tests/
├── environment.yml
└── README.md
```

Only the principal files are shown. Some outputs and intermediate files are created locally during execution.

## 14. Limitations

The project is a research and portfolio implementation, not a deployed banking risk platform.

Important limitations include:

- Banking data publication delays have not been fully incorporated into operational backtesting.
- Performance varies across institutions and historical periods.
- The one-month model has limited sensitivity to large delinquency increases.
- The held-out evaluation is retrospective rather than a verified real-time deployment simulation.
- Exact reconstruction requires access to the original source files recorded in the manifest.
- The MLflow model registry must be reconstructed or made accessible separately.
- Strong overall R² values do not guarantee accurate detection of sudden deterioration.

The system should not be treated as an autonomous regulatory early-warning system.

## 15. Future Improvements

Potential extensions include publication-date-aware backtesting, automated monthly data updates, uncertainty intervals, model monitoring, improved deterioration alerts, and an interactive dashboard for exploring institutional forecasts.

## Data Source and Disclaimer

The underlying data originates from Ecuador's banking regulator. The analysis, feature engineering, models, and conclusions are independent research outputs and do not represent official regulatory forecasts.

This repository is intended for educational, research, and professional portfolio purposes.
