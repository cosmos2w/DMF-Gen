# Validation

The [elasticity demo](../README.md#demo-elasticity-reconstruction) was checked on the CPU environment documented in the README, using the four downloads and SHA-256 values in the [dataset guide](../dataset/README.md#elasticity-demo-files).

| Check | Result |
| --- | --- |
| Downloaded arrays | Both `[41, 41, 2000]`, finite; stress float64 and binary material mask int64 |
| Supplied checkpoint and statistics | Strict model loading, 432,392 parameters; matching means, scales, and field order |
| 400-case evaluation | Mean stress relative L2 about 0.13; mean void IoU about 0.95; JSON and both per-case CSV files produced |
| Reconstruction and plotting | Held-out indices 28 and 178 produced full-grid NPZ files and a two-case PNG |
| One-update training smoke check | Finite training and validation losses; `best.pt` and `last.pt` produced |

The commands sample new sensor and source draws. Their outputs demonstrate the software interfaces; they do not establish reproduction of every manuscript result.

## Package checks

Install the existing test tools and run the same checks as [CPU CI](../.github/workflows/ci.yml):

```bash
python -m pip install -e '.[test,plot,physics]'
python -m ruff check src tests docs/figures
python -m pytest -q
```
