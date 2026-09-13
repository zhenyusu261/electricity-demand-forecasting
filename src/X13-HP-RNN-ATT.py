from tensorflow.keras import layers

from _hp_rnn_common import run_variant


if __name__ == "__main__":
    run_variant(
        "X13-HP-RNN-ATT",
        layers.SimpleRNN,
        ["rnn_units1", "rnn_units2"],
    )
