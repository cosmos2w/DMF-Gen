# Dataset sources and prepared representations

The mixed-resolution CFD examples use the [PDEBench 2D CFD data](https://github.com/pdebench/PDEBench/tree/main/pdebench/data_download), and the elasticity and airfoil examples use the datasets linked from the [Geo-FNO repository](https://github.com/neuraloperator/Geo-FNO#datasets). Obtain the original benchmark data from those projects and arrange the representations used by DMF-Gen as described below. The turbulent-combustion data were generated for this study; because the archive is large, they are available from the corresponding author upon reasonable request. Its file contract is included here to make the data interface clear.

## Prepared directory layout

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

These are the **prepared inputs**, rather than filenames promised by the source repositories. The paths are defaults in [`configs/`](../configs/); `data.path` or `data.root` can point to the same representation elsewhere. The large arrays and locally computed statistics are excluded from Git.

## PDEBench: mixed-resolution CFD

The source is PDEBench's two-dimensional CFD archive. For this study, stack each raw case's `Vx`, `Vy`, `density`, and `pressure` variables in that order on the native 128 × 128 grid. Form the 64 × 64 and 32 × 32 grids by taking nonoverlapping 2 × 2 and 4 × 4 spatial means of each high-resolution field; use the corresponding block-averaged coordinates. The paper's reconstruction target is **`Vx` only**. Keep case and frame indices aligned across all three files so that a case can be assigned to one training resolution and evaluated on the high-resolution grid.

Each `CFD_{L,M,H}_res.h5` has `fields` with shape `[cases, frames, points, 1, 1, 4]`, `coordinates` with one 3-component coordinate per point, and a `time` vector with one entry per frame. The `fields` channel order is `Vx, Vy, density, pressure`; set its `selected_fields` attribute to that comma-separated order or retain that exact default order. If the source time coordinate gives 22 frame edges, use their adjacent midpoints for the 21 stored frames. The study uses 10,000 cases and 21 frames, with `points` equal to 1,024, 4,096, or 16,384 for L, M, or H. The loader splits by case, keeps all frames of a case together, and uses high-resolution training-case statistics at every resolution.

## Geo-FNO: changing geometry

**Elasticity.** Use the released interpolated unit-cell stress and material-mask arrays on the shared 41 × 41 Cartesian grid. Both NumPy files have shape `[41, 41, 2000]`, with cases on the last axis. The stress channel is von Mises stress; the mask is `1` in material and `0` in the hole, and stress is zero in the hole. DMF-Gen predicts stress and the material mask together. The loader samples stress sensors from material cells near each hole and derives normalization from the training cases.

**Airfoil.** Start from the Geo-FNO NACA flow cases on their body-fitted grids, interpolate each flow field with piecewise-linear `scipy.interpolate.griddata` to a shared 101 × 101 Cartesian grid spanning `x ∈ [−0.5, 1.5]` and `y ∈ [−1, 1]`, and use zero fill outside the interpolation hull. A point-in-polygon test on the airfoil wall defines the fluid mask; set source values inside the body to zero. The prepared `NACA_Q_interp.npy` has shape `[2490, 5, 101, 101]` in the freestream-normalized order `rho, u, v, p, Ma`. `NACA_mask_interp.npy` is a boolean `[2490, 101, 101]` array that is true in the fluid; `NACA_X_interp.npy` and `NACA_Y_interp.npy` each have that same case/grid shape and contain the common Cartesian coordinates. DMF-Gen predicts `rho, u, v, p` and the fluid mask; the stored Mach channel is used to check the source order, then Mach is derived from the reconstructed fields for evaluation. Density and pressure are normalized in log space after fluid values are selected.

## In-house turbulent combustion

The backward-facing-step methane–air simulation is represented by a single HDF5 file, `combustion.h5`. Its `fields` dataset has shape `[1, frames, 40300, 1, 1, 5]` and channel order `CH4, CO, T, U_1, p`, corresponding to methane and carbon-monoxide mass fractions, temperature, streamwise velocity, and pressure. The study uses 10,000 frames on a 403 × 100 nonuniform rectilinear point set. The file also contains `coordinates` with one 3-component spatial coordinate per point and may contain a `time` vector; a `selected_fields` attribute may record the channel names. The loader splits frames with a fixed seed and fits per-channel normalization from training frames. Observation configurations select which physical channels supply sparse sensors, while every target retains all five fields.

## Normalization statistics

The example configs name optional statistics files such as `combustion.stats.pt`, `cfd_hml.stats.pt`, `elasticity.stats.pt`, and `airfoil.stats.pt`. To compute statistics from the configured training split, set `data.stats_path: null` in a copy of the config; full-archive preparation can take time. Keep the resulting statistics with trained checkpoints, since evaluation must use the same channel order, transforms, split, and normalization values as training.
