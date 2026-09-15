import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import keras_tuner as kt
from statsmodels.tsa.filters import hp_filter
from statsmodels.tsa.x13 import x13_arima_analysis
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed


CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = CODE_DIR / "models"
TUNER_DIR = CODE_DIR / "tuners"
DATA_PATH = "D:/x13-changzhou/changzhou.csv"
X13_PATH = "D:/suzhou/x13as.exe"

SEQ_LENGTH = 36
HP_LAMBDA = 14400
RANDOM_SEED = 42
FORECAST_YEAR = "2025"
FAST_MODE = False
HYPERBAND_ITERATIONS = 1 if FAST_MODE else 3

TREND_SEARCH = {"max_epochs": 20, "search_epochs": 30}
CYCLE_SEARCH = {"max_epochs": 30, "search_epochs": 50}

BRANCH_KERNELS = [1, 3, 6, 9, 12]
ATT_MAX = 4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def run_x13(endog, exog):
    result = x13_arima_analysis(
        endog=endog,
        x12path=X13_PATH,
        prefer_x13=True,
        outlier=True,
        trading=False,
        exog=exog,
        print_stdout=False,
        log=None,
        forecast_periods=0,
    )
    adjusted = result.d1
    cycle, trend = hp_filter.hpfilter(adjusted, lamb=HP_LAMBDA)
    return result, adjusted, cycle, trend


def parse_x13_result(result):
    content = result.results
    logt = "prefers log transformation" in content
    table_text = content.split("User-defined")[1].split("/n/n")[0]
    rows = [
        line.strip()
        for line in table_text.split("\n")
        if line.strip() and not line.startswith("---")
    ]
    coef_data = []
    for row in rows:
        parts = re.split(r"\s{2,}", row.strip())
        if len(parts) >= 4:
            coef_data.append(
                {
                    "variable": parts[0],
                    "estimate": parts[1],
                    "stderr": parts[2],
                    "tstat": parts[3],
                }
            )
    reg_table = pd.DataFrame(coef_data)
    for col in ["estimate", "stderr", "tstat"]:
        if col in reg_table.columns:
            reg_table[col] = reg_table[col].replace(
                r"[^\d\.\-eE]", "", regex=True
            )
            reg_table[col] = pd.to_numeric(reg_table[col], errors="coerce")
    return reg_table, logt


def get_coefficient(reg_table, variable_name):
    match = reg_table[reg_table["variable"] == variable_name]
    if not match.empty:
        return match["estimate"].values[0]
    for pattern in [f"^{variable_name}$", f".*{variable_name}.*"]:
        fuzzy = reg_table[reg_table["variable"].str.contains(pattern, regex=True)]
        if not fuzzy.empty:
            return fuzzy["estimate"].values[0]
    raise ValueError(f"未找到变量 '{variable_name}' 的系数")


def create_sequences(values, seq_length=SEQ_LENGTH):
    values = np.asarray(values, dtype="float64")
    X, y = [], []
    for i in range(len(values) - seq_length):
        X.append(values[i : i + seq_length])
        y.append(values[i + seq_length])
    X = np.asarray(X).reshape(-1, seq_length, 1)
    y = np.asarray(y).reshape(-1, 1)
    return X, y


def prepare_component(values):
    X, y = create_sequences(values)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(
        X_train.reshape(-1, 1)
    ).reshape(X_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))
    return X_train, y_train, X_val, y_val, scaler_X, scaler_y


def make_rolling_input(values, scaler_X):
    values = np.asarray(values, dtype="float64")
    test = values[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    scaled = scaler_X.transform(test.reshape(-1, 1)).reshape(test.shape)
    return scaled


class ChannelAttention(layers.Layer):
    def __init__(self, bottleneck_units=4, **kwargs):
        super().__init__(**kwargs)
        self.bottleneck_units = bottleneck_units

    def build(self, input_shape):
        channels = input_shape[-1]
        if self.bottleneck_units >= channels:
            raise ValueError(
                "bottleneck_units must be smaller than channels "
                f"({channels} channels, got {self.bottleneck_units})."
            )
        self.shared_dense = layers.Dense(
            self.bottleneck_units, activation="relu"
        )
        self.channel_dense = layers.Dense(channels)
        super().build(input_shape)

    def call(self, inputs):
        ndim = K.ndim(inputs)
        if ndim == 3:
            avg_pool = K.mean(inputs, axis=1, keepdims=True)
            max_pool = K.max(inputs, axis=1, keepdims=True)
        elif ndim == 4:
            avg_pool = K.mean(inputs, axis=[1, 2], keepdims=True)
            max_pool = K.max(inputs, axis=[1, 2], keepdims=True)
        else:
            raise ValueError(f"Unsupported input dimension: {ndim}")
        avg_out = self.channel_dense(self.shared_dense(avg_pool))
        max_out = self.channel_dense(self.shared_dense(max_pool))
        channel_att = layers.Activation("sigmoid")(avg_out + max_out)
        return inputs * channel_att


def build_cnn(hp):
    input_layer = layers.Input(shape=(SEQ_LENGTH, 1), name="sequence")
    branch_outputs = []
    for kernel in sorted(BRANCH_KERNELS):
        if kernel == 1:
            out = layers.Conv1D(
                filters=1, kernel_size=1, padding="same", name=f"conv_{kernel}"
            )(input_layer)
        else:
            out = layers.Conv1D(
                filters=1,
                kernel_size=kernel,
                strides=kernel,
                padding="same",
                name=f"conv_{kernel}",
            )(input_layer)
            out = layers.UpSampling1D(size=kernel)(out)
        branch_outputs.append(out)

    concat = layers.concatenate(list(reversed(branch_outputs)))
    att_units = hp.Int("att_units", 1, ATT_MAX, step=1)
    x = ChannelAttention(bottleneck_units=att_units)(concat)

    x = layers.Flatten()(x)
    x = layers.Dense(
        hp.Int("dense_units", 32, 256, step=12), activation="relu"
    )(x)
    x = layers.Dense(
        hp.Int("dense_units1", 64, 128, step=12), activation="relu"
    )(x)
    x = layers.Dropout(hp.Float("dropout", 0.1, 0.5))(x)
    output = layers.Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, step=10, sampling="log")
        ),
        loss="mean_squared_error",
        metrics=["mae", "mape"],
    )
    return model


def tune_component(name, X_train, y_train, X_val, y_val, settings):
    tuner = kt.Hyperband(
        build_cnn,
        objective="val_loss",
        max_epochs=settings["max_epochs"],
        hyperband_iterations=HYPERBAND_ITERATIONS,
        overwrite=True,
        directory=str(TUNER_DIR / name),
        project_name=name,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        mode="min",
        restore_best_weights=True,
    )
    tuner.search(
        X_train,
        y_train,
        epochs=settings["search_epochs"],
        batch_size=8,
        shuffle=False,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters()[0].values
    best_model.save(str(MODEL_DIR / f"{name}_model.keras"))
    with open(
        MODEL_DIR / f"{name}_hyperparameters.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_hps, f, ensure_ascii=False, indent=2)
    print(f"{name} best hyperparameters: {best_hps}")
    return best_model


def run_variant(variant_name, kernels):
    global BRANCH_KERNELS, ATT_MAX
    BRANCH_KERNELS = kernels
    ATT_MAX = max(1, len(kernels) - 1)

    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TUNER_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH, header=0)
    month_end = pd.date_range(
        start="2012-01-31", periods=len(raw), freq="ME"
    )
    elec = pd.Series(
        raw["elec"].interpolate(method="linear").dropna().values,
        index=month_end,
        name="elec",
    )
    exog = pd.DataFrame(
        raw[["aft", "bef"]].values,
        index=month_end,
        columns=["aft", "bef"],
    )

    train_elec = elec.loc[: "2024-12-31"]
    train_exog = exog.loc[: "2024-12-31"]
    forecast_months = pd.date_range(
        start=f"{FORECAST_YEAR}-01-31", periods=12, freq="ME"
    )

    _, _, cycle_train, trend_train = run_x13(train_elec, train_exog)
    X_trend, y_trend, Xv_trend, yv_trend, scaler_trend_X, scaler_trend_y = (
        prepare_component(trend_train.values)
    )
    X_cycle, y_cycle, Xv_cycle, yv_cycle, scaler_cycle_X, scaler_cycle_y = (
        prepare_component(cycle_train.values)
    )

    trend_tag = f"{variant_name}_trend"
    cycle_tag = f"{variant_name}_cycle"
    print(f"开始 {variant_name} 趋势项 Hyperband 搜索")
    best_trend = tune_component(
        trend_tag, X_trend, y_trend, Xv_trend, yv_trend, TREND_SEARCH
    )
    print(f"开始 {variant_name} 周期项 Hyperband 搜索")
    best_cycle = tune_component(
        cycle_tag, X_cycle, y_cycle, Xv_cycle, yv_cycle, CYCLE_SEARCH
    )

    def predict_component(model, known_values, scaler_X, scaler_y):
        test_input = make_rolling_input(known_values, scaler_X)
        pred = model.predict(test_input, verbose=0)
        return scaler_y.inverse_transform(pred).ravel()[0]

    forecasts = []
    actuals = []
    for forecast_month in forecast_months:
        known_mask = elec.index < forecast_month
        known_elec = elec.loc[known_mask]
        known_exog = exog.loc[known_mask]

        result, _, cycle_known, trend_known = run_x13(
            known_elec, known_exog
        )
        reg_table, logt = parse_x13_result(result)
        aft_est = get_coefficient(reg_table, "aft")
        bef_est = get_coefficient(reg_table, "bef")

        pred_trend = predict_component(
            best_trend, trend_known.values, scaler_trend_X, scaler_trend_y
        )
        pred_cycle = predict_component(
            best_cycle, cycle_known.values, scaler_cycle_X, scaler_cycle_y
        )
        adjusted_pred = pred_trend + pred_cycle
        holiday_effect = (
            aft_est * exog.loc[forecast_month, "aft"]
            + bef_est * exog.loc[forecast_month, "bef"]
        )
        if logt:
            final_pred = float(adjusted_pred * np.exp(holiday_effect))
        else:
            final_pred = float(adjusted_pred + holiday_effect)

        forecasts.append(final_pred)
        actuals.append(float(elec.loc[forecast_month]))
        print(f"{forecast_month:%Y-%m} forecast={final_pred:.4f}")

    forecasts = np.asarray(forecasts)
    actuals = np.asarray(actuals)

    rmse = float(np.sqrt(np.mean((forecasts - actuals) ** 2)))
    mae = float(np.mean(np.abs(forecasts - actuals)))
    mape = float(np.mean(np.abs((forecasts - actuals) / actuals)) * 100)
    ss_res = np.sum((forecasts - actuals) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot)

    print(
        f"{variant_name}: RMSE={rmse:.4f}, MAPE={mape:.4f}, "
        f"MAE={mae:.4f}, R2={r2:.4f}"
    )
    monthly_rows = pd.DataFrame(
        {
            "month": [d.strftime("%Y-%m") for d in forecast_months],
            "forecast": np.round(forecasts, 6),
        }
    )
    metric_rows = pd.DataFrame(
        {
            "month": ["RMSE", "MAPE", "MAE", "R2"],
            "forecast": [
                round(rmse, 6),
                round(mape, 6),
                round(mae, 6),
                round(r2, 6),
            ],
        }
    )
    result_table = pd.concat([monthly_rows, metric_rows], ignore_index=True)
    result_path = CODE_DIR / f"{variant_name}_{FORECAST_YEAR}_results.csv"
    result_table.to_csv(result_path, index=False)
    print(f"结果已保存至: {result_path}")
