# Data

`changzhou.csv` contains the monthly Changzhou electricity demand series used
in the 2025 primary evaluation.

## Columns

| Column | Description |
|---|---|
| `date` | Observation month |
| `elec` | Monthly electricity demand |
| `indu` | Industrial electricity demand |
| `leap` | Calendar indicator retained from the source dataset |
| `bef` | Normalized pre-Spring-Festival holiday-effect variable |
| `aft` | Normalized post-Spring-Festival holiday-effect variable |

## Source

Changzhou Municipal Bureau of Statistics:

<https://tjjyw.changzhou.gov.cn/cztjj/mbWeb_CZ.action>

The repository distributes the processed data file for reproducibility.
Users should verify the source provider's redistribution terms before reusing
or republishing the data.
