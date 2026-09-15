# Third-Party Notices

## statsmodels X13 interface

The files below are unmodified copies of
`statsmodels/tsa/x13.py` from statsmodels 0.14.4:

- `src/x13.py`
- `reproducibility_package/code_2024/x13.py`
- `reproducibility_package/code_2025/x13.py`

Upstream project:

- Project: statsmodels
- Version: 0.14.4
- Source: <https://github.com/statsmodels/statsmodels/blob/v0.14.4/statsmodels/tsa/x13.py>
- Original SHA-256:
  `3D7B291B3E3B16DD1EE672B28A940946EE68F54ACA06BCCB19E88F5CDB707C6C`

The vendored file is licensed under the BSD 3-Clause License. The complete
license text is provided in `third_party/statsmodels/LICENSE.txt`.

The X13-based model scripts in this repository import `x13.py` as their Python
X13 interface. The external `x13as` executable is still required to perform
the X-13ARIMA-SEATS estimation.
