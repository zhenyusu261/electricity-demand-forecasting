import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import keras_tuner as kt
from x13 import x13_arima_analysis
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed


# ----------------------------- configuration -----------------------------
CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_TAG = "retrain_auto_log"
MODEL_NAME = "X13-CNN-ATT"
MODEL_DIR = CODE_DIR / "models" / f"{MODEL_NAME}_{EXPERIMENT_TAG}"
TUNER_DIR = CODE_DIR / "tuners" / f"{MODEL_NAME}_{EXPERIMENT_TAG}"
DATA_PATH = "D:/x13-changzhou/changzhou.csv"
X13_PATH = "D:/suzhou/x13as.exe"

SEQ_LENGTH = 36
RANDOM_SEED = 42
FORECAST_YEAR = "2024"

# X13-CNN-ATT：只做 X13 regARIMA 调整，不使用 HP 分解。
# 模型结构、搜索空间、滚动一步预测与结果保存方式同 X13-HP-CNN-ATT。
FAST_MODE = False
HYPERBAND_ITERATIONS = 1 if FAST_MODE else 3
CNN_SEARCH = {"max_epochs": 30, "search_epochs": 50}


# ----------------------------- shared helpers -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def run_x13(endog, exog):
    """执行 X13 regARIMA，由 X13 自动判断是否采用对数变换。"""
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
    return result, result.d1


def parse_x13_result(result):
    """解析回归系数，并识别 X13 自动建模时是否选择了对数变换。"""
    content = result.results
    use_log_transform = "prefers log transformation" in content

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
    return reg_table, use_log_transform


def get_coefficient(reg_table, variable_name):
    """按变量名获取 X13 回归系数。"""
    match = reg_table[reg_table["variable"] == variable_name]
    if not match.empty:
        return match["estimate"].values[0]
    for pattern in [f"^{variable_name}$", f".*{variable_name}.*"]:
        fuzzy = reg_table[reg_table["variable"].str.contains(pattern, regex=True)]
        if not fuzzy.empty:
            return fuzzy["estimate"].values[0]
    raise ValueError(f"未找到变量 '{variable_name}' 的系数")


def create_sequences(values, seq_length=SEQ_LENGTH):
    """把一维序列转换为 (样本数, seq_length, 1) 的监督学习数据。"""
    values = np.asarray(values, dtype="float64")
    X, y = [], []
    for i in range(len(values) - seq_length):
        X.append(values[i : i + seq_length])
        y.append(values[i + seq_length])
    X = np.asarray(X).reshape(-1, seq_length, 1)
    y = np.asarray(y).reshape(-1, 1)
    return X, y


def prepare_series(values):
    """按时间顺序切分训练/验证集，并返回对应 scaler。"""
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
    """使用最近 SEQ_LENGTH 个观测构造单步预测输入。"""
    values = np.asarray(values, dtype="float64")
    test = values[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)
    scaled = scaler_X.transform(test.reshape(-1, 1)).reshape(test.shape)
    return scaled


# ----------------------------- model definition -----------------------------
class ChannelAttention(layers.Layer):
    """通道注意力。bottleneck_units 直接表示压缩后的通道数。"""

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
    """多分支 CNN：1/3/6/9/12 周期卷积、上采样、通道注意力和全连接输出。"""
    input_layer = layers.Input(shape=(SEQ_LENGTH, 1), name="sequence")

    x1 = layers.Conv1D(
        filters=1, kernel_size=1, padding="same", name="conv_1"
    )(input_layer)
    x3 = layers.Conv1D(
        filters=1, kernel_size=3, strides=3, padding="same", name="conv_3"
    )(input_layer)
    x3 = layers.UpSampling1D(size=3)(x3)
    x6 = layers.Conv1D(
        filters=1, kernel_size=6, strides=6, padding="same", name="conv_6"
    )(input_layer)
    x6 = layers.UpSampling1D(size=6)(x6)
    x9 = layers.Conv1D(
        filters=1, kernel_size=9, strides=9, padding="same", name="conv_9"
    )(input_layer)
    x9 = layers.UpSampling1D(size=9)(x9)
    x12 = layers.Conv1D(
        filters=1, kernel_size=12, strides=12, padding="same", name="conv_12"
    )(input_layer)
    x12 = layers.UpSampling1D(size=12)(x12)

    concat = layers.concatenate([x12, x9, x6, x3, x1])
    # concat 后为 5 个通道，因此瓶颈单元取 1-4 才自洽。
    att_units = hp.Int("att_units", 1, 4, step=1)
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


def tune_cnn(X_train, y_train, X_val, y_val):
    """对 X13-CNN-ATT 执行 Hyperband 搜索，并保存最优模型。"""
    tuner = kt.Hyperband(
        build_cnn,
        objective="val_loss",
        max_epochs=CNN_SEARCH["max_epochs"],
        hyperband_iterations=HYPERBAND_ITERATIONS,
        seed=RANDOM_SEED,
        overwrite=True,
        directory=str(TUNER_DIR / "x13_cnn_att"),
        project_name="x13_cnn_att_model",
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
        epochs=CNN_SEARCH["search_epochs"],
        batch_size=8,
        shuffle=False,
        verbose=0,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters()[0].values
    best_model.save(str(MODEL_DIR / "x13_cnn_att_model.keras"))
    with open(
        MODEL_DIR / "x13_cnn_att_hyperparameters.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_hps, f, ensure_ascii=False, indent=2)
    print(f"X13-CNN-ATT best hyperparameters: {best_hps}")
    return best_model


# ----------------------------- data preparation -----------------------------
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

train_elec = elec.loc[: "2023-12-31"]
train_exog = exog.loc[: "2023-12-31"]
forecast_months = pd.date_range(
    start=f"{FORECAST_YEAR}-01-31", periods=12, freq="ME"
)

_, adjusted_train = run_x13(train_elec, train_exog)
X_train, y_train, X_val, y_val, scaler_X, scaler_y = prepare_series(
    adjusted_train.values
)

print("开始 X13-CNN-ATT Hyperband 搜索")
best_model = tune_cnn(X_train, y_train, X_val, y_val)


# ----------------------------- rolling one-step forecast -----------------------------
def predict_next(known_values):
    test_input = make_rolling_input(known_values, scaler_X)
    pred = best_model.predict(test_input, verbose=0)
    return scaler_y.inverse_transform(pred).ravel()[0]


forecasts = []
actuals = []
for forecast_month in forecast_months:
    known_mask = elec.index < forecast_month
    known_elec = elec.loc[known_mask]
    known_exog = exog.loc[known_mask]

    result, adjusted = run_x13(known_elec, known_exog)
    reg_table, use_log_transform = parse_x13_result(result)
    aft_est = get_coefficient(reg_table, "aft")
    bef_est = get_coefficient(reg_table, "bef")

    adjusted_pred = predict_next(adjusted.values)
    holiday_effect = (
        aft_est * exog.loc[forecast_month, "aft"]
        + bef_est * exog.loc[forecast_month, "bef"]
    )
    if use_log_transform:
        # d1 已由 X13 反变换回原尺度；对数变换下的回归效应应按乘法还原。
        final_pred = float(adjusted_pred * np.exp(holiday_effect))
    else:
        final_pred = float(adjusted_pred + holiday_effect)

    forecasts.append(final_pred)
    actuals.append(float(elec.loc[forecast_month]))
    print(f"{forecast_month:%Y-%m} forecast={final_pred:.4f}")

forecasts = np.asarray(forecasts)
actuals = np.asarray(actuals)

# ----------------------------- metrics and output -----------------------------
rmse = float(np.sqrt(np.mean((forecasts - actuals) ** 2)))
mae = float(np.mean(np.abs(forecasts - actuals)))
mape = float(np.mean(np.abs((forecasts - actuals) / actuals)) * 100)
ss_res = np.sum((forecasts - actuals) ** 2)
ss_tot = np.sum((actuals - actuals.mean()) ** 2)
r2 = float(1 - ss_res / ss_tot)

print(f"RMSE={rmse:.4f}, MAPE={mape:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

monthly_rows = pd.DataFrame(
    {
        "month": [d.strftime("%Y-%m") for d in forecast_months],
        "forecast": np.round(forecasts, 6),
    }
)
metric_rows = pd.DataFrame(
    {
        "month": ["RMSE", "MAPE", "MAE", "R2"],
        "forecast": [round(rmse, 6), round(mape, 6), round(mae, 6), round(r2, 6)],
    }
)
result_table = pd.concat([monthly_rows, metric_rows], ignore_index=True)
result_path = (
    CODE_DIR
    / f"{MODEL_NAME}_{FORECAST_YEAR}_{EXPERIMENT_TAG}_results.csv"
)
result_table.to_csv(result_path, index=False)
print(f"结果已保存至: {result_path}")
