# Validation and reproducibility

The checks below describe the scope of source-package verification. Geometry comparisons used the manuscript's selected checkpoints and per-case evaluation arrays; the standard CLI uses independent sensor and source draws, so its aggregate scores need not equal values calculated from the manuscript's saved draws. Dataset layout and normalization requirements are in the [dataset guide](../dataset/README.md).

## Package checks

| Check | Result | Scope |
| --- | --- | --- |
| `PYTHONPATH=src python -m pytest -q` | 30 passed | Data contracts, model behavior, geometry masking, metrics, training, evaluation, reconstruction, and checkpoint paths. |
| `python -m ruff check src tests docs/figures` | Passed | Static lint for package, tests, and figure scripts. |
| Package wheel build | Passed | Source can be packaged for installation. |

The package checks exercise interfaces and small examples; they are not full manuscript training runs.

## Changing geometry

The elasticity and airfoil adapters use the manuscript's seed-42 80/20 case partitions: 1,600/400 unit cells and 1,992/498 airfoils. A recorded comparison across all held-out cases found the same case order, coordinates, reference fields, and prescribed sensor candidates as the manuscript arrays; airfoil density and pressure differed only by up to `1.2e-7` after the float32 log/exp round trip. The geometry metrics applied to those arrays reproduced the manuscript's per-case stress, mask, concentration, and drag values, with Mach relative L2 agreeing within `1.4e-17`.

Both selected geometry checkpoints loaded strictly into the supplied model profiles (432,392 elasticity parameters and 1,983,865 airfoil parameters). The airfoil checkpoint required a tensor-only export because its original file also contains a pickled NumPy random state; the exported weights and statistics were unchanged. Matching statistics are necessary for checkpoint evaluation: the stored elasticity scales differ from exact double-precision training moments by up to `0.83%`.

| Recorded end-to-end check | Result | Interpretation |
| --- | --- | --- |
| Elasticity `dmf-evaluate` on 400 cases, 16 stress sensors, 8 draws, 4 Euler steps | Mean stress relative L2 `0.1297`; geometry relative L2 `0.216`; stress-concentration mean absolute relative error `0.095` | The manuscript's saved sensor draw yielded `0.1238` stress relative L2. |
| Airfoil `dmf-evaluate` on 498 cases, 32 pressure sensors, 8 draws, 32 Euler steps | Mean Mach relative L2 `0.0120`; geometry relative L2 `0.106`; drag mean absolute relative error `0.154` | The manuscript's saved draw yielded `0.0122` Mach relative L2. |
| One or two optimizer updates with each geometry profile, followed by evaluation | Finite train and validation losses; checkpoints loaded and held-out cases were scored | Exercises real-data training and evaluation paths without claiming convergence. |
| Full-grid reconstruction and plotting for one held-out case of each task | NPZ reconstruction and PNG field panels were produced | Exercises reconstruction and visualization entry points. |

The CLI samples sensors from the prescribed candidate sets but does not replay the manuscript's saved sensor or source draws. The reported CLI means therefore describe fresh draws. Method comparisons should use matched draws.

## Combustion and mixed-resolution CFD

Recorded data checks accepted one combustion sample with shape `(40300, 5)` and the field order `CH4, CO, T, U_1, p`, plus CFD low, medium, and high samples with 1,024, 4,096, and 16,384 query points. Recomputed combustion moments over all 9,000 training frames matched the stored means; the largest relative standard-deviation difference was `4.3e-7`. Small optimizer and checkpoint checks exercised `GL_rbf`, `GL_rbf_ENH`, and `GL_rbf_ENH_CQ`, including mixed-resolution updates. One full-grid Cond-T reconstruction used 40,300 queries and 256 observations; observed-value replacement had maximum physical difference `0.0`.

These checks do not establish paper-scale combustion or CFD scores. Dataset arrays, normalization files, trained checkpoints, and generated runs are outside the source package.
