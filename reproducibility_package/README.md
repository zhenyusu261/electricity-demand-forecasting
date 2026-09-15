# Full Reproducibility Package

This package supports inspection and reproduction of the forecasting and
statistical analyses reported in the manuscript, including the 2024 and 2025
Changzhou evaluations and the Guangzhou cross-city validation.

## Contents

- `data/changzhou.csv`: Changzhou monthly electricity demand used in the
  primary evaluation. Source: Changzhou Municipal Bureau of Statistics.
- `data/guangzhou.csv`: Guangzhou monthly electricity demand used in the
  cross-city evaluation. Source: Guangzhou Municipal Statistics Bureau.
- `code_2024/`: source code, result files, and selected model configurations
  for the 2024 evaluation.
- `code_2025/`: source code, result files, and selected model configurations
  for the 2025 evaluation.
- All X13-based models use automatic log selection (`log=None`). If X13 selects
  a log transformation, the holiday effect is applied multiplicatively to the
  reconstructed original-scale forecast.
- `statistical_analysis/`: DM, paired-bootstrap, and Holm-adjustment scripts
  and results.
- `cross_city_results/`: monthly forecasts and annual metrics for the
  Guangzhou evaluation.
- `supplementary_tables/Table_S1_final_hyperparameters.csv`: final selected
  Hyperband configurations for all models and components.
- `supplementary_tables/Table_S2_statistical_results.csv`: pooled DM,
  paired-bootstrap, block-bootstrap, and Holm-adjusted results.

## Environment

The analyses were implemented in Python with NumPy, pandas, SciPy,
statsmodels, scikit-learn, TensorFlow/Keras, and Keras Tuner. X13 processing
requires the X-13ARIMA-SEATS executable.

## Reproduction note

The scripts retain the original absolute input paths used during the study.
Before rerunning, set `DATA_PATH` and `X13_PATH` to the locations of
`data/changzhou.csv` and the local `x13as` executable.
