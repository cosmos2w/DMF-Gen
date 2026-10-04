# Datasets

Download each dataset from the links below once they are available, then place the files at the paths shown here.

| Dataset | Download | Expected files under `dataset/` |
| --- | --- | --- |
| Five-field turbulent combustion | **Download link pending** | `combustion.h5` |
| PDEBench CFD at three resolutions | **Download link pending** | `pdebench/Processed/CFD_L_res.h5`, `CFD_M_res.h5`, `CFD_H_res.h5` |
| Elasticity unit cells with a void (Geo-FNO) | **Download link pending** | `elasticity/Interp/Random_UnitCell_sigma_10_interp.npy`, `Random_UnitCell_mask_10_interp.npy` |
| Transonic NACA airfoils (Geo-FNO), resampled to 101 × 101 | **Download link pending** | `airfoil/naca_interp_5f/NACA_Q_interp.npy`, `NACA_mask_interp.npy`, `NACA_X_interp.npy`, `NACA_Y_interp.npy` |

The resulting directory layout should be:

```text
dataset/
├── README.md
├── combustion.h5
├── pdebench/
│   └── Processed/
│       ├── CFD_L_res.h5
│       ├── CFD_M_res.h5
│       └── CFD_H_res.h5
├── elasticity/
│   └── Interp/
│       ├── Random_UnitCell_sigma_10_interp.npy
│       └── Random_UnitCell_mask_10_interp.npy
└── airfoil/
    └── naca_interp_5f/
        ├── NACA_Q_interp.npy
        ├── NACA_mask_interp.npy
        ├── NACA_X_interp.npy
        └── NACA_Y_interp.npy
```

Keep the CFD filenames as shown, and rename the combustion HDF5 file to `combustion.h5` if its downloaded name differs. The loaders validate the expected HDF5 shapes and field order (`CH4, CO, T, U_1, p` for combustion; `Vx, Vy, density, pressure` for processed CFD).

The changing-geometry arrays are stored as NumPy files:

- **Elasticity:** stress and mask arrays of shape `[41, 41, 2000]`. The mask is 1 in the material and 0 in the void, where the stored stress is zero.
- **Airfoil:** 2,490 cases on a uniform 101 × 101 grid over `x ∈ [−0.5, 1.5]`, `y ∈ [−1, 1]` (chord 1). `NACA_Q_interp.npy` is `[2490, 5, 101, 101]` with freestream-normalized channels `rho, u, v, p, Ma`, all zero inside the airfoil; `NACA_mask_interp.npy` is boolean and true in the fluid. The loader checks the channel order with `Ma = |(u, v)| / sqrt(1.4 p / rho)` and drops the stored Mach channel; Mach is derived at evaluation.

The supplied configs refer to optional precomputed statistics at `dataset/combustion.stats.pt`, `dataset/cfd_hml.stats.pt`, `dataset/elasticity.stats.pt`, or `dataset/airfoil.stats.pt`. If you have only the data files, set `data.stats_path` to `null` in a copy of the chosen YAML config so the loader computes normalization statistics from its training split before the run; this can take time on the full datasets. Reuse a precomputed statistics file only when its field order and training split match your config. A checkpoint must be evaluated with the statistics it was trained with; the evaluation command compares them and stops on a mismatch.

Dataset files and statistics under this directory are ignored by Git.
