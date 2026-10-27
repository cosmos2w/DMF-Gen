# DMF-Gen

**DMF-Gen** reconstructs a physical field from sparse measurements. Each measurement carries a coordinate, scalar value, and physical-channel label; the caller supplies separate coordinates where the complete field is needed. The model learns a rectified-flow velocity on normalized field values, conditioned by global latent attention and local RBF-weighted sensor evidence; evaluation and reconstruction convert outputs back to physical units.

**What you can run:** The repository provides the GL-RBF model family, data interfaces, and commands for training, evaluation, and reconstruction on turbulent combustion, mixed-resolution CFD, elasticity, and airfoil examples.

**Try the [small elasticity demo](#small-elasticity-demo):** download about 61 MB of example data and pretrained weights, then evaluate and generate reconstruction images on a CPU in about one minute.

## Small elasticity demo

This example reconstructs von Mises stress and the material mask of a simulated elastic unit cell from 16 sparse stress measurements. It includes the Geo-FNO arrays for 2,000 unit cells, matching training normalization, and a pretrained `GL_rbf_ENH` checkpoint. The commands evaluate the 400 held-out cases, reconstruct two selected cases, and plot their reference and predicted stress, sensors, and errors. No training or GPU is needed to run this demo.

### 1. Download and place the files

Download the following four individual files from the [elasticity demo folder](https://drive.google.com/drive/folders/1ebHxgUIBfr-IAJ9kQwwgNfO8F3nNXZxS) and place them at the paths below, relative to the repository root. Rename `elasticity_stats.pt` and `best.pt` as shown.

```bash
mkdir -p dataset/elasticity/Interp checkpoints
```

| Download | Required local path | Contents |
| --- | --- | --- |
| [Random_UnitCell_sigma_10_interp.npy](https://drive.google.com/file/d/19EPJ_F1-pRG5112xy4BEs7qGxC4OdvY8/view) | `dataset/elasticity/Interp/Random_UnitCell_sigma_10_interp.npy` | Geo-FNO stress array, float64 `[41, 41, 2000]` |
| [Random_UnitCell_mask_10_interp.npy](https://drive.google.com/file/d/1JuTLUNg57ERahU7JQpfZMm5gkiin-t0c/view) | `dataset/elasticity/Interp/Random_UnitCell_mask_10_interp.npy` | Geo-FNO binary material mask, int64 `[41, 41, 2000]` |
| [elasticity_stats.pt](https://drive.google.com/file/d/178rc5ub2HDMEOOjCuiqRPmla-kkykDEz/view) | `dataset/elasticity.stats.pt` | Training normalization matching the checkpoint |
| [best.pt](https://drive.google.com/file/d/1pwNcx3S_schcI17tixEL9gIjSogUmqP5/view) | `checkpoints/elasticity_gl_rbf_enh.pt` | Epoch-12,605 checkpoint |

The [dataset guide](dataset/README.md#elasticity-demo-files) provides the SHA-256 checksums and Geo-FNO attribution.

### 2. Install

**Requirements:** Python 3.10 or newer. From the repository root, create an environment and install CPU PyTorch plus the package and demo plotting dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[physics]'
```

**Tested environment:** Ubuntu 22.04.5 LTS (x86-64), Python 3.10.19, PyTorch 2.14.1+cpu, NumPy 2.2.6, h5py 3.16.0, PyYAML 6.0.3, SciPy 1.15.3, and Matplotlib 3.10.9. These are tested versions; [pyproject.toml](pyproject.toml) specifies the minimum dependency versions. Installation into an empty environment took about 30 seconds on the test workstation with a fast network, excluding Python setup and demo downloads; allow several minutes on a typical desktop connection.

**Hardware:** The elasticity evaluation and reconstruction demo runs on a standard CPU desktop; no GPU or other non-standard hardware is required. The timings below use four CPU threads on an AMD Ryzen Threadripper PRO 5975WX; peak process memory was about 0.4 GB for evaluation/reconstruction and 0.5 GB for the training smoke check. For optional full training, use a CUDA-capable GPU and install a matching PyTorch build in place of the CPU build above.

**Optional acceleration:** PyKeOps can be installed for large point clouds (`pip install -e '.[keops]'`). Choose the neighbor backend in the YAML config and use an appropriate PyTorch build for your accelerator before starting a full training run.

### 3. Evaluate, reconstruct, and plot

Run these commands from the repository root with the environment above activated:

```bash
# Use four CPU threads for the timings below
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Evaluate the supplied checkpoint on all 400 held-out unit cells
dmf-evaluate --config configs/elasticity_gl_rbf_enh.yaml \
  --checkpoint checkpoints/elasticity_gl_rbf_enh.pt --device cpu \
  --output-dir runs/elasticity_demo/evaluation

# Reconstruct two held-out cells and render the stress fields
for i in 28 178; do
  dmf-reconstruct --config configs/elasticity_gl_rbf_enh.yaml \
    --checkpoint checkpoints/elasticity_gl_rbf_enh.pt --sample-index "$i" --device cpu \
    --output runs/elasticity_demo/case_$i.npz
done
python docs/figures/render_elasticity_reconstruction.py \
  runs/elasticity_demo/case_28.npz runs/elasticity_demo/case_178.npz \
  --output runs/elasticity_demo/reconstruction.png
```

| Step | Expected output | Approximate CPU run time |
| --- | --- | --- |
| 1. Evaluate | `runs/elasticity_demo/evaluation/summary.json`, `per_sample_field.csv`, and `per_sample_task.csv`; mean stress relative L2 about 0.1305 and mean void IoU about 0.953 | 48 s |
| 2. Reconstruct and plot | `runs/elasticity_demo/case_28.npz`, `case_178.npz`, and `reconstruction.png` | 5 s |

**Configuration.** The [runnable elasticity profile](configs/elasticity_gl_rbf_enh.yaml) specifies the supplied checkpoint's architecture (`GL_rbf_ENH`, 432,392 parameters), 16–32 stress sensors per training case, and 16 at evaluation. Evaluation averages 8 draws with 4 Euler steps each, using the supplied training statistics. The checkpoint records epoch 12,605. Sensor and source draws are generated by the commands, so scores and per-case reconstructions can vary across environments and from the manuscript's saved draws.

<a href="docs/figures/elasticity_reconstruction.png"><img src="docs/figures/elasticity_reconstruction.png" alt="Reference and DMF-Gen von Mises stress for two held-out unit cells, the 16 stress sensors next to each void, and the signed reconstruction errors with their relative L2 errors" width="75%"></a>

*Elasticity reconstruction.* Held-out samples 28 and 178 (unit cells 161 and 893), reconstructed with the supplied checkpoint using step 2. Top: reference von Mises stress and the DMF-Gen mean of 8 draws. Bottom: the 16 stress sensors on the reference geometry and the signed reconstruction error, with its relative L2 error over the material. Per-case errors depend on the sensor draw; the commands draw their own sensors rather than replaying the manuscript's.

### Optional: start a new elasticity training run

To check the training interface with one optimizer update and two validation cases:

```bash
dmf-train --config configs/elasticity_gl_rbf_enh.yaml --epochs 1 --batch-size 8 \
  --query-points 256 --max-steps 1 --validation-samples 2 --device cpu \
  --output-dir runs/elasticity_smoke
```

This writes `run_config.yaml`, `train.log`, `metrics.jsonl`, `best.pt`, and `last.pt` in `runs/elasticity_smoke/` and took about 2 seconds on the tested CPU. It checks execution only. To start full training on your own GPU:

```bash
dmf-train --config configs/elasticity_gl_rbf_enh.yaml --device cuda:0 \
  --output-dir runs/elasticity_train
```

The profile configures 20,000 epochs. A previously measured 50-epoch NVIDIA L40S run was about 1.4 seconds per epoch (roughly 8 hours extrapolated for 20,000 epochs). This is a runtime estimate; the public training command does not guarantee recovery of the supplied checkpoint or manuscript scores.

## Models and tasks

[![DMF-Gen model structure: sparse channel-labeled measurements, global and local sensor conditioning, rectified-flow generation, and changing task settings](docs/_resources/Figure_ModelStructure.png)](docs/_resources/Figure_ModelStructure.pdf)

**Measurements and queries.** On the left, each measured scalar carries its own coordinate and channel label, while separate query coordinates determine where the full field is reconstructed.

**Global and local conditioning.** The GL-RBF backbone turns measurements into sensor tokens, exchanges context with global latent tokens, and uses nearby RBF-weighted sensors to predict velocity at each query. The enhanced profile also includes a global query readout.

**Generation.** The middle row follows a Gaussian reference field as the learned velocity carries it toward a reconstruction.

**Tasks.** The bottom row illustrates changes in geometry, output resolution, observed channels, and conditional draws. The examples cover [mixed-resolution CFD](configs/mixed_hml.yaml), [five-field combustion](configs/combustion_cond_t.yaml), [elasticity](configs/elasticity_gl_rbf_enh.yaml), and [airfoil flow](configs/airfoil_gl_rbf_cq.yaml).

### Available model profiles

| Profile | Model features | Example config |
| --- | --- | --- |
| `GL_rbf` | Global latent context and local RBF conditioning | [Base example](configs/gl_rbf_base.yaml) |
| `GL_rbf_ENH` | Fourier sensor features, repeated latent reads, and query latent readout | [Combustion](configs/combustion_cond_t.yaml), [mixed-resolution CFD](configs/mixed_hml.yaml), and [elasticity](configs/elasticity_gl_rbf_enh.yaml) |
| `GL_rbf_ENH_CQ` | Compact-query head used for the airfoil task | [Airfoil](configs/airfoil_gl_rbf_cq.yaml) and [optional profile](configs/gl_rbf_enh_cq.yaml) |

*Choose a profile through its YAML config; the model, observation pattern, training options, and evaluation options are kept together.*

## Data

**<ins>Before training:</ins>** See the [dataset guide](dataset/README.md) for dataset availability and placement. Put each task's data under `dataset/` in the documented layout. The loaders check field order and shape:

| Task | Fields | Native query points |
| --- | --- | --- |
| Combustion | `CH4, CO, T, U_1, p` | 40,300 |
| PDEBench CFD | processed `Vx` for paper examples | L 1,024; M 4,096; H 16,384 |
| Elasticity | von Mises stress and material mask | 1,681 |
| Airfoil | density, two velocity components, pressure, and fluid mask | 10,201 |

**Normalization statistics:** The example configs refer to optional precomputed files. If you have the data without matching statistics, set `data.stats_path` to `null` in your chosen config to compute statistics from the training split before use; this can take time on the full datasets. Evaluate a trained checkpoint with its original statistics.

*The figures below show raw held-out dataset states, not model reconstructions.*

### Five-field turbulent combustion

<a href="docs/figures/combustion_multifield.png"><img src="docs/figures/combustion_multifield.png" alt="Five raw combustion fields and an illustrative set of 256 temperature measurements from one held-out frame" width="50%"></a>

*Combustion example.* One held-out frame shows the five physical target fields. The colored dots show an illustrative set of 256 sparse temperature observations.

### Mixed-resolution CFD

<a href="docs/figures/cfd_resolution_hierarchy.png"><img src="docs/figures/cfd_resolution_hierarchy.png" alt="One held-out CFD streamwise-velocity field at low, medium, and high resolution on a shared color scale" width="50%"></a>

*CFD example.* The same held-out state appears at 32 × 32, 64 × 64, and 128 × 128 points. The lower-resolution fields are spatial averages of the high-resolution field.

### Changing geometry

<a href="docs/figures/geometry_elasticity.png"><img src="docs/figures/geometry_elasticity.png" alt="Von Mises stress in three held-out elasticity unit cells with differently shaped voids, and 16 stress sensors in the material next to each void" width="50%"></a>

*Elasticity example.* Three held-out unit cells share one 41 × 41 grid, but each void (grey) has its own shape. The model reconstructs the stress and the material mask from 16 stress sensors (white) drawn from the material next to the void (small dots).

<a href="docs/figures/geometry_airfoil.png"><img src="docs/figures/geometry_airfoil.png" alt="Static pressure around three held-out airfoils of different thickness and angle, and 32 pressure sensors on a fixed ellipse" width="50%"></a>

*Airfoil example.* Three held-out airfoils, cropped from one 101 × 101 grid. The model reconstructs density, velocity, pressure, and the fluid mask from 32 static-pressure sensors (white) drawn from one fixed ellipse (small dots); Mach number and drag are derived at evaluation.

## Use with your data

Follow the [dataset guide](dataset/README.md#use-with-your-data) for input shapes, field order, path settings, and training-split normalization. Copy a suitable [task config](configs/), set `data.path` or `data.root` and `data.stats_path`, and use the same `dmf-train`, `dmf-evaluate`, and `dmf-reconstruct` commands with your config and checkpoint. Keep normalization statistics paired with their trained model.

## Outputs and interpretation

**Training** writes a resolved config, metrics, a run log, and checkpoints under the chosen `runs/` folder. Set `training.plot_every_epochs` to a positive integer to save `loss_history.csv` and `loss_history.png`, or set `evaluation.every_epochs` to run periodic physical-field evaluation.

**Evaluation** writes per-sample and per-field physical relative-L2 results. Set `evaluation.plot_samples: 1` to also save one full-field truth/reconstruction/absolute-error PNG per channel for the first deterministic held-out sample. Scheduled evaluations place these panels beside `summary.json` in `evaluation/epoch_XXXX/`; the summary records the sample ID and filenames. Plotting requires the optional `plot` dependencies (`pip install -e '.[plot]'`). **Reconstruction** writes coordinates, observations, targets, and sampled predictions to an NPZ file that `dmf-plot` can visualize.

## License

Original DMF-Gen source code is distributed under the [MIT License](LICENSE). The license does not automatically relicense Geo-FNO or PDEBench datasets, pretrained checkpoints, third-party dependencies, or article materials; their applicable terms remain with their respective providers. Dataset sources and attribution are in the [dataset guide](dataset/README.md).
