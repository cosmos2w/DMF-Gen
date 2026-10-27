# Elasticity demo

This example reconstructs von Mises stress and the material mask of a simulated elastic unit cell from 16 sparse stress measurements. It includes the Geo-FNO arrays for 2,000 unit cells, matching training normalization, and a pretrained `GL_rbf_ENH` checkpoint. The commands evaluate the 400 held-out cases, reconstruct two selected cases, and plot their reference and predicted stress, sensors, and errors. No training or GPU is needed to run this demo.

Run all commands below from the repository root, which contains `pyproject.toml` and `configs/`.

## 1. Download and place the files

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

The [dataset guide](../../dataset/README.md#elasticity-demo-files) provides the SHA-256 checksums and Geo-FNO attribution.

## 2. Install

**Requirements:** Python 3.10 or newer. From the repository root, create an environment and install CPU PyTorch plus the package and demo plotting dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[physics]'
```

**Tested environment:** Ubuntu 22.04.5 LTS (x86-64), Python 3.10.19, PyTorch 2.14.1+cpu, NumPy 2.2.6, h5py 3.16.0, PyYAML 6.0.3, SciPy 1.15.3, and Matplotlib 3.10.9. These are tested versions; [pyproject.toml](../../pyproject.toml) specifies the minimum dependency versions. Installation into an empty environment took about 30 seconds on the test workstation with a fast network, excluding Python setup and demo downloads; allow several minutes on a typical desktop connection.

**Hardware:** The elasticity evaluation and reconstruction demo runs on a standard CPU desktop; no GPU or other non-standard hardware is required. The timings below use four CPU threads on an AMD Ryzen Threadripper PRO 5975WX; peak process memory was about 0.4 GB for evaluation/reconstruction and 0.5 GB for the training smoke check. For optional full training, use a CUDA-capable GPU and install a matching PyTorch build in place of the CPU build above.

**Optional acceleration:** PyKeOps can be installed for large point clouds (`pip install -e '.[keops]'`). Choose the neighbor backend in the YAML config and use an appropriate PyTorch build for your accelerator before starting a full training run.

## 3. Evaluate, reconstruct, and plot

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

**Configuration.** The [runnable elasticity profile](../../configs/elasticity_gl_rbf_enh.yaml) specifies the supplied checkpoint's architecture (`GL_rbf_ENH`, 432,392 parameters), 16–32 stress sensors per training case, and 16 at evaluation. Evaluation averages 8 draws with 4 Euler steps each, using the supplied training statistics. The checkpoint records epoch 12,605. Sensor and source draws are generated by the commands, so scores and per-case reconstructions can vary across environments and from the manuscript's saved draws.

<a href="../../docs/figures/elasticity_reconstruction.png"><img src="../../docs/figures/elasticity_reconstruction.png" alt="Reference and DMF-Gen von Mises stress for two held-out unit cells, the 16 stress sensors next to each void, and the signed reconstruction errors with their relative L2 errors" width="75%"></a>

*Elasticity reconstruction.* Held-out samples 28 and 178 (unit cells 161 and 893), reconstructed with the supplied checkpoint using step 2. Top: reference von Mises stress and the DMF-Gen mean of 8 draws. Bottom: the 16 stress sensors on the reference geometry and the signed reconstruction error, with its relative L2 error over the material. Per-case errors depend on the sensor draw; the commands draw their own sensors rather than replaying the manuscript's.

## Optional: start a new elasticity training run

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
