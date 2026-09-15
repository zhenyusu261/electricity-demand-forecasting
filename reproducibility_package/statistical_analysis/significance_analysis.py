import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(r"D:\computers")
MODELS = [
    ("X13-HP-CNN-ATT", "guangzhou24/X13-HP-CNN-ATT_2024_log_auto_retrain_results.csv",
     "guangzhou25/X13-HP-CNN-ATT_2025_retrain_auto_log_results.csv"),
    ("CNN-ATT", "guangzhou24/CNN-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/CNN-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-CNN-ATT", "guangzhou24/X13-CNN-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-CNN-ATT_2025_retrain_auto_log_results.csv"),
    ("HP-CNN-ATT", "guangzhou24/HP-CNN-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/HP-CNN-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-HP-CNN-B-ATT", "guangzhou24/X13-HP-CNN-B-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-HP-CNN-B-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-HP-CNN-C-ATT", "guangzhou24/X13-HP-CNN-C-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-HP-CNN-C-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-HP-CNN-D-ATT", "guangzhou24/X13-HP-CNN-D-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-HP-CNN-D-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-HP-CNN-E-ATT", "guangzhou24/X13-HP-CNN-E-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-HP-CNN-E-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-HP-BiLSTM-ATT", "guangzhou24/X13-HP-BiLSTM-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-HP-BiLSTM-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-HP-LSTM-ATT", "guangzhou24/X13-HP-LSTM-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-HP-LSTM-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-HP-RNN-ATT", "guangzhou24/X13-HP-RNN-ATT_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-HP-RNN-ATT_2025_retrain_auto_log_results.csv"),
    ("X13-HP-CNN-BiLSTM", "guangzhou24/X13-HP-CNN-BiLSTM_2024_retrain_auto_log_results.csv",
     "guangzhou25/X13-HP-CNN-BiLSTM_2025_retrain_auto_log_results.csv"),
]

PROPOSED = "X13-HP-CNN-ATT"
WINNER = "X13-HP-CNN-BiLSTM"
RNG_SEED = 42
BOOTSTRAP_REPS = 10000
BLOCK_LENGTH = 5


def read_actuals():
    raw = pd.read_csv(ROOT / "guangzhou24" / "guangzhou.csv", parse_dates=["date"])
    return raw.set_index("date")["elec"]


def read_forecast(relative_path):
    frame = pd.read_csv(ROOT / relative_path)
    frame = frame[frame["month"].astype(str).str.match(r"^\d{4}-\d{2}$")].copy()
    frame["date"] = pd.to_datetime(frame["month"] + "-01")
    return frame.set_index("date")["forecast"].astype(float)


def metric_difference(actual, left, right, metric):
    err_left = left - actual
    err_right = right - actual
    if metric == "RMSE":
        return math.sqrt(np.mean(err_right**2)) - math.sqrt(np.mean(err_left**2))
    return np.mean(np.abs(err_right)) - np.mean(np.abs(err_left))


def hac_variance(values, max_lag):
    centered = values - values.mean()
    n = len(centered)
    gamma0 = np.dot(centered, centered) / n
    total = gamma0
    for lag in range(1, max_lag + 1):
        gamma = np.dot(centered[lag:], centered[:-lag]) / n
        weight = 1 - lag / (max_lag + 1)
        total += 2 * weight * gamma
    return max(total, np.finfo(float).eps)


def dm_test(loss_left, loss_right):
    diff = loss_right - loss_left
    n = len(diff)
    max_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    variance = hac_variance(diff, max_lag)
    statistic = diff.mean() / math.sqrt(variance / n)
    p_value = 2 * t.sf(abs(statistic), df=n - 1)
    return diff.mean(), statistic, p_value


def paired_bootstrap(actual, left, right, metric, block=False):
    rng = np.random.default_rng(RNG_SEED)
    n = len(actual)
    differences = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        if block:
            block_count = math.ceil(n / BLOCK_LENGTH)
            starts = rng.integers(0, n, size=block_count)
            indices = np.concatenate(
                [np.arange(start, start + BLOCK_LENGTH) % n for start in starts]
            )[:n]
        else:
            indices = rng.integers(0, n, size=n)
        differences[i] = metric_difference(
            actual.iloc[indices].to_numpy(),
            left.iloc[indices].to_numpy(),
            right.iloc[indices].to_numpy(),
            metric,
        )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    observed = metric_difference(actual, left, right, metric)
    null_distribution = differences - observed
    p_value = float(np.mean(np.abs(null_distribution) >= abs(observed)))
    return float(lower), float(upper), p_value


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


actual = read_actuals()
actual_2024 = actual.loc["2024-01-01":"2024-12-01"]
actual_2025 = actual.loc["2025-01-01":"2025-12-01"]
actual_pooled = pd.concat([actual_2024, actual_2025])

forecasts = {}
for name, file_2024, file_2025 in MODELS:
    forecasts[name] = {
        "2024": read_forecast(file_2024),
        "2025": read_forecast(file_2025),
    }
    forecasts[name]["pooled"] = pd.concat(
        [forecasts[name]["2024"], forecasts[name]["2025"]]
    )

actuals = {"2024": actual_2024, "2025": actual_2025, "pooled": actual_pooled}
metric_rows = []
for period in ["2024", "2025", "pooled"]:
    for name, _, _ in MODELS:
        prediction = forecasts[name][period]
        error = prediction - actuals[period]
        metric_rows.append(
            {
                "period": period,
                "model": name,
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "MAE": float(np.mean(np.abs(error))),
                "MAPE": float(np.mean(np.abs(error / actuals[period])) * 100),
                "R2": float(
                    1
                    - np.sum(error**2)
                    / np.sum((actuals[period] - actuals[period].mean()) ** 2)
                ),
            }
        )
metrics = pd.DataFrame(metric_rows)

key_rows = []
for period in ["2024", "2025", "pooled"]:
    for metric in ["RMSE", "MAE"]:
        left_name = PROPOSED
        right_name = WINNER
        left = forecasts[left_name][period]
        right = forecasts[right_name][period]
        left_error = left - actuals[period]
        right_error = right - actuals[period]
        if metric == "RMSE":
            loss_left = left_error**2
            loss_right = right_error**2
            metric_diff = (
                np.sqrt(np.mean(loss_right)) - np.sqrt(np.mean(loss_left))
            )
        else:
            loss_left = np.abs(left_error)
            loss_right = np.abs(right_error)
            metric_diff = np.mean(loss_right) - np.mean(loss_left)
        dm_mean, dm_stat, dm_p = dm_test(loss_left, loss_right)
        ci_simple_low, ci_simple_high, bootstrap_p = paired_bootstrap(
            actuals[period], left, right, metric, block=False
        )
        ci_block_low, ci_block_high, _ = paired_bootstrap(
            actuals[period], left, right, metric, block=True
        )
        key_rows.append(
            {
                "period": period,
                "metric": metric,
                "left_model": left_name,
                "right_model": right_name,
                "metric_difference_right_minus_left": float(metric_diff),
                "dm_mean_loss_difference": float(dm_mean),
                "dm_statistic": float(dm_stat),
                "dm_p_value": float(dm_p),
                "bootstrap_p_value": bootstrap_p,
                "bootstrap_ci_low": ci_simple_low,
                "bootstrap_ci_high": ci_simple_high,
                "block_bootstrap_ci_low": ci_block_low,
                "block_bootstrap_ci_high": ci_block_high,
            }
        )
key_results = pd.DataFrame(key_rows)

comparison_rows = []
for metric in ["RMSE", "MAE"]:
    p_values = []
    for name, _, _ in MODELS:
        if name == PROPOSED:
            continue
        left = forecasts[PROPOSED]["pooled"]
        right = forecasts[name]["pooled"]
        left_error = left - actual_pooled
        right_error = right - actual_pooled
        if metric == "RMSE":
            loss_left = left_error**2
            loss_right = right_error**2
            metric_diff = (
                np.sqrt(np.mean(loss_right)) - np.sqrt(np.mean(loss_left))
            )
        else:
            loss_left = np.abs(left_error)
            loss_right = np.abs(right_error)
            metric_diff = np.mean(loss_right) - np.mean(loss_left)
        dm_mean, dm_stat, dm_p = dm_test(loss_left, loss_right)
        ci_simple_low, ci_simple_high, bootstrap_p = paired_bootstrap(
            actual_pooled, left, right, metric, block=False
        )
        ci_block_low, ci_block_high, _ = paired_bootstrap(
            actual_pooled, left, right, metric, block=True
        )
        p_values.append(dm_p)
        comparison_rows.append(
            {
                "metric": metric,
                "model": name,
                "metric_difference_right_minus_proposed": float(metric_diff),
                "dm_mean_loss_difference": float(dm_mean),
                "dm_statistic": float(dm_stat),
                "dm_p_value": float(dm_p),
                "bootstrap_p_value": bootstrap_p,
                "bootstrap_ci_low": ci_simple_low,
                "bootstrap_ci_high": ci_simple_high,
                "block_bootstrap_ci_low": ci_block_low,
                "block_bootstrap_ci_high": ci_block_high,
            }
        )
    adjusted = holm_adjust(np.asarray(p_values))
    start = len(comparison_rows) - len(p_values)
    for idx, adjusted_p in enumerate(adjusted):
        comparison_rows[start + idx]["holm_adjusted_p_value"] = float(adjusted_p)
        comparison_rows[start + idx]["significance_holm_5pct"] = bool(adjusted_p < 0.05)
comparison_results = pd.DataFrame(comparison_rows)

matrix_results = {}
for metric in ["RMSE", "MAE"]:
    n_models = len(MODELS)
    raw_p = np.ones((n_models, n_models), dtype=float)
    stat = np.zeros((n_models, n_models), dtype=float)
    holm = np.ones((n_models, n_models), dtype=float)
    pairs = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            left_name = MODELS[i][0]
            right_name = MODELS[j][0]
            left = forecasts[left_name]["pooled"]
            right = forecasts[right_name]["pooled"]
            left_error = left - actual_pooled
            right_error = right - actual_pooled
            if metric == "RMSE":
                loss_left = left_error**2
                loss_right = right_error**2
            else:
                loss_left = np.abs(left_error)
                loss_right = np.abs(right_error)
            _, dm_stat, dm_p = dm_test(loss_left, loss_right)
            pairs.append((i, j, dm_stat, dm_p))
    adjusted = holm_adjust(np.asarray([pair[3] for pair in pairs]))
    for pair, adjusted_p in zip(pairs, adjusted):
        i, j, dm_stat, dm_p = pair
        raw_p[i, j] = raw_p[j, i] = dm_p
        stat[i, j] = dm_stat
        stat[j, i] = -dm_stat
    matrix_results[metric] = {
        "raw_p": raw_p.tolist(),
        "holm_p": holm.tolist(),
        "statistic": stat.tolist(),
    }
    for pair, adjusted_p in zip(pairs, adjusted):
        i, j, _, _ = pair
        holm[i, j] = holm[j, i] = adjusted_p
    matrix_results[metric]["holm_p"] = holm.tolist()

payload = {
    "models": [name for name, _, _ in MODELS],
    "metrics": metrics.to_dict(orient="records"),
    "key_comparison": key_results.to_dict(orient="records"),
    "proposed_comparisons": comparison_results.to_dict(orient="records"),
    "pairwise_matrix": matrix_results,
}

out_dir = ROOT / "_spreadsheet_build_gz"
with open(out_dir / "significance_results.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)

print("KEY COMPARISONS")
print(key_results.to_string(index=False))
print("\nPROPOSED VS OTHER MODELS")
print(comparison_results.to_string(index=False))
print("\nMETRICS")
print(metrics.sort_values(["period", "RMSE"]).to_string(index=False))
