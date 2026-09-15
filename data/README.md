# Data

This directory contains the monthly electricity demand series used by the
study:

- `changzhou.csv`: primary Changzhou evaluation data.
- `guangzhou.csv`: Guangzhou cross-city validation data.

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

Guangzhou Municipal Statistics Bureau:

<https://tjj.gz.gov.cn/datav/admin/home/www_report>

The repository distributes the processed data files for reproducibility.
Users should verify the source provider's redistribution terms before reusing
or republishing the data.
