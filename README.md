# Electricity Demand Forecasting

Reproducible code and data for the 2024-2025 Changzhou experiments and the
2024-2025 Guangzhou cross-city evaluation reported in:

**An Integrated regARIMA-HP Filter-CNN Framework with Attention Mechanism for
Monthly Electricity Demand Forecasting**

Repository: <https://github.com/zhenyusu261/electricity-demand-forecasting>

Current release: `v1.0.1`

Repository description: AI and forecasting.

## Scope

This repository contains the evaluation code and result files for the primary
Changzhou dataset in 2024 and 2025, together with the Guangzhou cross-city
validation under the same causal rolling one-step protocol. It includes the
proposed model, decomposition and preprocessing ablations, branch ablations,
and recurrent baselines.

The protocol uses 2012-2024 observations for model construction and evaluates
12 causally updated one-step forecasts for 2025 in the primary workflow. The
`reproducibility_package/` directory additionally contains the exact 2024 code
and result files, Guangzhou data, cross-city results, statistical analysis,
and supplementary tables. X13 is allowed to select a logarithmic
transformation automatically. If the transformation is selected, holiday
effects are applied multiplicatively after the X13-adjusted series is returned
to its original scale.

## Models

| Script | Model |
|---|---|
| `src/CNN-ATT.py` | CNN with channel attention |
| `src/X13-CNN-ATT.py` | X13 adjustment plus CNN-ATT |
| `src/HP-CNN-ATT.py` | HP decomposition plus CNN-ATT |
| `src/X13-HP-CNN-ATT.py` | Proposed X13-HP-CNN-ATT |
| `src/X13-HP-CNN-B-ATT.py` | Full model without kernels 9 and 12 |
| `src/X13-HP-CNN-C-ATT.py` | Full model without kernel 12 |
| `src/X13-HP-CNN-D-ATT.py` | Full model without kernel 9 |
| `src/X13-HP-CNN-E-ATT.py` | Full model without kernel 1 |
| `src/X13-HP-BiLSTM-ATT.py` | X13-HP with Bi-LSTM and attention |
| `src/X13-HP-LSTM-ATT.py` | X13-HP with LSTM and attention |
| `src/X13-HP-RNN-ATT.py` | X13-HP with SimpleRNN and attention |
| `src/X13-HP-CNN-BiLSTM.py` | CNN-ATT followed by Bi-LSTM and attention |

## Repository Structure

```text
electricity-demand-forecasting/
├── data/
│   ├── README.md
│   ├── changzhou.csv
│   └── guangzhou.csv
├── docs/
│   └── REPRODUCIBILITY.md
├── reference_results/
│   └── 2025/
├── reproducibility_package/
│   ├── code_2024/
│   ├── code_2025/
│   ├── cross_city_results/
│   ├── data/
│   ├── statistical_analysis/
│   └── supplementary_tables/
├── scripts/
│   ├── check_environment.py
│   └── run_all.py
├── src/
│   ├── repo_config.py
│   ├── _hp_cnn_common.py
│   ├── _hp_rnn_common.py
│   └── model scripts
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10 or 3.11. Python 3.11 is recommended.
- X-13ARIMA-SEATS, including the `x13as` executable.
- A C compiler or compatible binary is not required when the official X13
  executable is used.

Install Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow may not support every newly released Python version. Use Python
3.10 or 3.11 if installation fails.

## X13 Configuration

The repository does not redistribute the X-13ARIMA-SEATS executable. Set its
path before running any X13-based model:

```powershell
$env:X13AS_PATH = "C:\path\to\x13as.exe"
```

The code supports `log=None`, which lets X13 choose whether a logarithmic
transformation is appropriate for each estimation window.

## Data

The default primary data path is:

```text
data/changzhou.csv
```

The file contains monthly observations with these columns:

| Column | Description |
|---|---|
| `date` | Observation month |
| `elec` | Monthly electricity demand |
| `indu` | Industrial electricity demand |
| `leap` | Leap-year or calendar indicator retained from the source data |
| `bef` | Normalized pre-Spring-Festival effect variable |
| `aft` | Normalized post-Spring-Festival effect variable |

An alternative data path can be supplied as:

```powershell
$env:CHANGZHOU_DATA = "C:\path\to\changzhou.csv"
```

The Guangzhou cross-city data are provided in:

```text
data/guangzhou.csv
```

The complete source data and result package used for the 2024 evaluation,
2025 evaluation, and Guangzhou cross-city comparison is also available under
`reproducibility_package/`.

## Environment Check

Run:

```powershell
python scripts/check_environment.py
```

The script checks the Python version, required packages, data file, and X13
executable.

## Run One Model

Example for the proposed model:

```powershell
python src/X13-HP-CNN-ATT.py
```

Each model writes its monthly forecasts, annual metrics, trained model files,
tuner state, and selected hyperparameters under `outputs/`.

## Run All Models

```powershell
python scripts/run_all.py
```

The full Hyperband protocol can be computationally expensive. Set
`FAST_MODE = True` in a model script only for a quick smoke test; do not use
fast mode for reported results.

## Outputs

```text
outputs/
├── models/
├── tuners/
└── <MODEL_NAME>_2025_results.csv
```

Each results file contains 12 monthly forecasts followed by RMSE, MAPE, MAE,
and R2 rows.

The `reference_results/2025/` directory contains the primary 2025 result
files. The `reproducibility_package/cross_city_results/` directory contains
the Guangzhou result files. Generated files in `outputs/` are excluded from
Git by default.

## Reproducibility

The implementation follows these rules:

- chronological train/validation split;
- fixed random seed 42;
- three Hyperband iterations for reported results;
- no shuffling during model training;
- causal rolling one-step forecasts;
- X13 and HP filtering recomputed only from observations before each target
  month;
- X13 automatic log selection in every X13-based model.

See `docs/REPRODUCIBILITY.md` for a detailed execution order and model mapping.

## Data Source

Changzhou electricity demand data were obtained from the Changzhou Municipal
Bureau of Statistics:

<https://tjjyw.changzhou.gov.cn/cztjj/mbWeb_CZ.action>

Guangzhou electricity demand data were obtained from the Guangzhou Municipal
Statistics Bureau:

<https://tjj.gz.gov.cn/datav/admin/home/www_report>

Users are responsible for checking the source terms before redistributing the
data.

## Citation

Citation metadata are provided in `CITATION.cff`. Add the final article DOI or
archived repository DOI after publication.

## License

The code is released under the MIT License. Data files remain subject to the
terms of their original providers.
