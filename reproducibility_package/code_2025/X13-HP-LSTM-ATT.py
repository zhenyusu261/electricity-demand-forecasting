from tensorflow.keras import layers

from _hp_rnn_common import run_variant


if __name__ == "__main__":
    run_variant(
        "X13-HP-LSTM-ATT",
        layers.LSTM,
        ["lstm_units1", "lstm_units2"],
    )
