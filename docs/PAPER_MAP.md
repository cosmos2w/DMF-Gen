# Paper-to-code map

The manuscript and its Supplementary Information specify the scientific model and evaluation protocols. Their authoring sources are not part of this package.

| Paper item | Release implementation | Evidence and limit |
| --- | --- | --- |
| Marked observations `(xi, channel, value)` and independent query coordinates | `dmf_gen.data.ObservationBatch`; GL-RBF model inputs | Main Results, direct measurement-to-field generation, Eqs. 1 and 3–5 |
| Sensor → latent → sensor conditioning | `dmf_gen.models` GL-RBF family | Main Eq. 3 and architecture methods |
| Local RBF and global query readout | `dmf_gen.models` GL-RBF family | Main Eqs. 4–5 describe the enhanced path; the base example uses local RBF and a pooled global summary without query-to-latent readout or the learned importance bias |
| Smooth Gaussian source and straight rectified-flow loss | `dmf_gen.priors`, `dmf_gen.models.PointCloudFFM` | Main Eqs. 2 and 6; SI model section |
| Euler/Heun generation and observed-entry replacement | `dmf_gen.models.PointCloudFFM.sample` | Main direct-generation section; SI evaluation sections |
| L/M/H mixed-resolution CFD | `dmf_gen.data.PDEBenchCFD` adapter | SI mixed-resolution data and training; same query interface at every scale |
| Five-channel combustion with subset observations | `dmf_gen.data.CombustionH5` adapter | SI combustion data/training; target remains five channels |
| Changing geometry | `dmf_gen.data.ElasticityDataset` and `AirfoilDataset` through `SupportMaskAdapter`; `dmf_gen.metrics` | SI elasticity and airfoil sections; see the [validation record](VALIDATION.md#changing-geometry) and [dataset guide](../dataset/README.md) |

## Model identity

`GL_rbf` is the earlier global/local backbone, not the backbone used for the manuscript's combustion results. The base example reads the sensor set into latents once, returns latent context to sensor anchors, and combines local RBF evidence with a pooled global summary. It does not enable repeated sensor reads, coordinate-based query-to-latent readout, or the `topk_rbf_glres` importance-bias and coarse-residual paths. Its RBF width can be fixed or learned according to the config. The main manuscript's full query-readout and fusion equations describe the enhanced implementation; they should not be read as a list of mandatory features for every earlier profile.

`GL_rbf_ENH` enables Fourier sensor tokenization, repeated latent sensor reads, query-to-latent readout, and normalized fusion. The manuscript's combustion models and H-only/H-limited/Mixed-HML CFD models use this enhanced family. The two zero-H CFD recipes use an earlier global/local backbone. Every included YAML names its model explicitly; the builder's empty-config default remains `GL_rbf_ENH` for checkpoint compatibility.

`GL_rbf_ENH_CQ` is an optional compact-query variant. The SI describes an additive query head for the airfoil setting, and [the airfoil profile](../configs/airfoil_gl_rbf_cq.yaml) uses it. It is not a substitute for the enhanced combustion or mixed-resolution checkpoints. The manuscript's elasticity model is `GL_rbf_ENH` with a fixed RBF width ([elasticity profile](../configs/elasticity_gl_rbf_enh.yaml)); its airfoil model is this compact-query variant with a learned width. Each profile reproduces its manuscript checkpoint's architecture, so that checkpoint loads with strict state-dictionary checks.

## Provenance boundaries

- The combustion adapter requires `CH4,CO,T,U_1,p` in that order. The observed channels are `T=[2]`, `T,U_1=[2,3]`, and `CO,T,U_1,p=[1,2,3,4]`; the adapter rejects HDF5 data with a different field order.
- The manuscript reports selected combustion epochs 6005, 6060, and 6045 for its three regimes. Those selected weights and reported metrics are not bundled here. A filename such as `best.pt` is not evidence that it contains a manuscript-selected checkpoint.
- The CFD paper variable is processed channel zero, `Vx`. Verify HDF5 metadata and the processed-file field order before scoring.
- H-only, H-limited, Mixed-HML, and both zero-H recipes use different training partitions. A generic mixed-resolution template is a runnable interface example, not by itself an exact recipe reproduction.
- The combustion HDF5 metadata records `dt=0.0001`, while the manuscript uses `5e-5` seconds. This provenance discrepancy remains unresolved; no time-dependent physical claim is made from that metadata in this package.
- Elasticity and airfoil use seeded 80/20 case splits (seed 42): 1,600/400 unit cells and 1,992/498 airfoils. Their statistics are moments over every grid point of the training cases, with airfoil density and pressure in log space. The manuscript checkpoints stored moments accumulated in single precision; for elasticity these differ from exact moments by up to 0.83 % (mask standard deviation), so evaluating those checkpoints requires their stored statistics.
- Elasticity sensors are drawn from material points within three grid steps of each case's void, so their placement follows the case geometry; airfoil sensors come from one fixed elliptic ring for every case. Evaluation uses 16 stress or 32 pressure sensors, eight averaged draws, and 4 (elasticity) or 32 (airfoil) Euler steps; draws are averaged in normalized space, which is a geometric mean for log-space density and pressure.
- Airfoil Mach number is derived from density, velocity, and pressure for reference and reconstruction alike. Drag is integrated on the reference airfoil surface for both, so its error measures surface pressure, not shape; per-case relative errors of drag and stress concentration divide by `max(|reference|, 0.15 × RMS(reference))`.
- As in the manuscript runs, the trainer selects `best.pt` by the flow loss on the held-out cases. The manuscript's elasticity checkpoint stores its raw weights as `model` (its exponential moving average under `ema`), and its airfoil checkpoint stores the averaged weights; with `training.ema_decay`, this trainer validates and saves the averaged weights as `model` and the raw ones as `model_raw`.
- The stored elasticity stress has no recorded unit; this package reports it as stored.
