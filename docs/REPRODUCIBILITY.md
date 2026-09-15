# Reproducibility Notes

## Full evaluation package

The `reproducibility_package/` directory contains the exact 2024 and 2025
evaluation code, monthly result files, Guangzhou cross-city outputs,
statistical analysis scripts, and supplementary tables. It preserves the
original path variables used during the study. Before rerunning a model, set
`DATA_PATH` and `X13_PATH` for the local data and X-13ARIMA-SEATS executable.

## Evaluation Protocol

The repository implements the 2025 Changzhou evaluation:

- observations through December 2024 are used for model construction;
- the final 20 percent of chronologically ordered training samples form the
  validation set;
- the selected model is held fixed for the 12 monthly one-step forecasts;
- X13 adjustment and HP decomposition are recomputed before each target month
  using only observations strictly before that month;
- no target-month or future observation enters the scaler, X13 estimation, HP
  decomposition, or neural-network input.

## X13 Log Transformation

All X13-based models call `x13_arima_analysis(..., log=None)`. X13 therefore
selects the logarithmic transformation automatically for each estimation
window.

The returned adjusted series is on the original scale. When X13 selects a log
transformation, the estimated holiday effect is therefore applied as:

```text
final_forecast = adjusted_forecast * exp(holiday_effect)
```

When no log transformation is selected, the correction is additive:

```text
final_forecast = adjusted_forecast + holiday_effect
```

## Shared Settings

- sequence length: 36 months;
- HP smoothing parameter: 14,400;
- random seed: 42;
- Hyperband iterations: 3 for reported results;
- training shuffle: disabled;
- trend Hyperband: 20 maximum epochs, 30 search epochs;
- cycle Hyperband: 30 maximum epochs, 50 search epochs;
- early stopping: validation loss, patience 8, restore best weights.

## Recommended Execution Order

1. `src/CNN-ATT.py`
2. `src/X13-CNN-ATT.py`
3. `src/HP-CNN-ATT.py`
4. `src/X13-HP-CNN-ATT.py`
5. `src/X13-HP-CNN-B-ATT.py`
6. `src/X13-HP-CNN-C-ATT.py`
7. `src/X13-HP-CNN-D-ATT.py`
8. `src/X13-HP-CNN-E-ATT.py`
9. `src/X13-HP-BiLSTM-ATT.py`
10. `src/X13-HP-LSTM-ATT.py`
11. `src/X13-HP-RNN-ATT.py`
12. `src/X13-HP-CNN-BiLSTM.py`

Alternatively, run `python scripts/run_all.py` from the repository root.

## Expected Outputs

Each model creates:

- `<MODEL>_2025_results.csv`;
- `<MODEL>_trend_hyperparameters.json`;
- `<MODEL>_cycle_hyperparameters.json`;
- saved Keras model files under `outputs/models/`;
- Keras Tuner trial information under `outputs/tuners/`.

Run-to-run values can differ slightly across TensorFlow builds, GPU providers,
and X13 binaries. The manuscript used the versions and environment described
in the article. Archive the exact environment with a release when making a
permanent reproducibility deposit.
