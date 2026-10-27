# Paper-to-code map

The manuscript and its Supplementary Information specify the scientific model and evaluation protocols. Their authoring sources are not part of this package.

| Paper item | Release implementation | Evidence and limit |
| --- | --- | --- |
| Marked observations `(xi, channel, value)` and independent query coordinates | `dmf_gen.data.ObservationBatch`; GL-RBF model inputs | Main Results, direct measurement-to-field generation, Eqs. 1 and 3–5 |
| Sensor → latent → sensor conditioning | `dmf_gen.models` GL-RBF family | Main Eq. 3 and architecture methods |
| Local RBF and global query readout | `dmf_gen.models` GL-RBF family | Main Eqs. 4–5 describe the enhanced path; the base example uses local RBF and a pooled global summary without query-to-latent readout or the learned importance bias |
| Smooth Gaussian source and straight rectified-flow loss | `dmf_gen.priors`, `dmf_gen.models.PointCloudFFM` | Main Eqs. 2 and 6; SI model section |
| Euler/Heun generation and observed-entry replacement | `dmf_gen.models.PointCloudFFM.sample` | Main direct-generation section; SI evaluation sections |
| L/M/H mixed-resolution CFD | `dmf_gen.data.PDEBenchCFDDataset` adapter | SI mixed-resolution data and training; same query interface at every scale |
| Five-channel combustion with subset observations | `dmf_gen.data.CombustionH5Dataset` adapter | SI combustion data/training; target remains five channels |
| Changing geometry | `dmf_gen.data.ElasticityDataset` and `AirfoilDataset` through `SupportMaskAdapter`; `dmf_gen.metrics` | SI elasticity and airfoil sections; see the [validation record](VALIDATION.md) and [dataset guide](../dataset/README.md) |

## Model identity

`GL_rbf` is the earlier global/local backbone, not the backbone used for the manuscript's combustion results. The base example reads the sensor set into latents once, returns latent context to sensor anchors, and combines local RBF evidence with a pooled global summary. It does not enable repeated sensor reads, coordinate-based query-to-latent readout, or the `topk_rbf_glres` importance-bias and coarse-residual paths. Its RBF width can be fixed or learned according to the config. The main manuscript's full query-readout and fusion equations describe the enhanced implementation; they should not be read as a list of mandatory features for every earlier profile.

`GL_rbf_ENH` enables Fourier sensor tokenization, repeated latent sensor reads, query-to-latent readout, and normalized fusion. The manuscript's combustion models and H-only/H-limited/Mixed-HML CFD models use this enhanced family. The two zero-H CFD recipes use an earlier global/local backbone. Every included YAML names its model explicitly; choose the matching profile when loading a checkpoint.

`GL_rbf_ENH_CQ` is an optional compact-query variant. The SI describes an additive query head for the airfoil setting, and [the airfoil profile](../configs/airfoil_gl_rbf_cq.yaml) uses it. It is not a substitute for the enhanced combustion or mixed-resolution checkpoints. The manuscript's elasticity model is `GL_rbf_ENH` with a fixed RBF width ([elasticity profile](../configs/elasticity_gl_rbf_enh.yaml)); its airfoil model is this compact-query variant with a learned width. Each profile reproduces its manuscript checkpoint's architecture, so that checkpoint loads with strict state-dictionary checks.

## Data and evaluation contracts

- Combustion target fields are `CH4, CO, T, U_1, p`, in that order; the configured observations select a subset of those fields.
- Mixed-resolution CFD uses `Vx` (processed channel zero), with aligned case/frame indices across resolutions.
- Elasticity and airfoil use seed-42 80/20 case splits: 1,600/400 unit cells and 1,992/498 airfoils. Evaluate pretrained checkpoints with their matching normalization statistics.
- Elasticity uses 16 stress sensors from material cells within three grid steps of the void, eight averaged draws, and four Euler steps. The runnable example is in the [README](../README.md#demo-elasticity-reconstruction).
- Airfoil uses 32 pressure sensors from a fixed elliptic ring, eight averaged draws, and 32 Euler steps. Density and pressure are modeled in log space; normalized draw averaging corresponds to a geometric mean for those fields.
- Airfoil Mach number is derived from density, velocity, and pressure. Drag is integrated on the reference surface. Task metrics are implemented in `dmf_gen.metrics`.

For input formats and compatible user data, see the [dataset guide](../dataset/README.md).
