# Quantum Hall Evaluation

Configuration reference for `jaqmc hall evaluate`.
This page shows the effective defaults for the evaluation workflow preset. Use
`--dry-run` to see the resolved config for your run, or add
`workflow.config.verbose=true` to include field descriptions. Evaluation keys
for `run.*`, `sampler.*`, and `writers.*` live at the config root. Defaults are
resolved in this order: schema defaults, workflow preset, YAML config, then CLI
overrides. For training config, see <project:train.md>.

Root-level runtime keys such as `logging.*`, `jax.*`, and `distributed.*` are
shared by all commands. See <project:../../guide/runtime-configuration.md>.

## Workflow (`workflow.*`)

These keys control evaluation-wide settings and checkpoint loading.

`workflow.source_path` selects the trained parameters and sampler state to
evaluate. It is required for MHPO and other trainable wavefunctions. The
analytic `laughlin` and `free` wavefunctions have no parameters, so they can
start evaluation from fresh walkers without a source path.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
   :prefix: workflow
```

## System (`system.*`)

When evaluating a training checkpoint, these settings must match the training
run. For direct `laughlin` or `free` evaluation, they define the target system.
The effective defaults are identical to the [training system config](#hall-train-system).

## Wavefunction (`wf.*`)

When evaluating a training checkpoint, these settings must match the training
run. For direct `laughlin` or `free` evaluation, select the analytic module
with `wf.module`. The effective defaults and built-in module choices are
identical to the [training wavefunction config](#hall-train-wf).

## Run Options (`run.*`)

Evaluation reuses the same checkpointing and sampling controls as training, but
adds `digest_step_interval` for previewing accumulated statistics.

```{eval-rst}
.. config-defaults:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
   :prefix: run
```

## Sampler (`sampler.*`)

Hall evaluation uses adaptive Metropolis-Hastings sampling with a spherical
proposal.

```{eval-rst}
.. config-defaults:: jaqmc.sampler.mcmc.MCMCSampler
   :prefix: sampler
```

## Writers (`writers.*`)

The evaluation HDF5 writer is always enabled because its per-step statistics
are required for digest computation. It writes to `evaluation_stats.h5`; other
root-level writer keys enable additional outputs.

### Console writer (`writers.console.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.console.ConsoleWriter
   :prefix: writers.console
```

### CSV writer (`writers.csv.*`)

```{eval-rst}
.. config-defaults:: jaqmc.writer.csv.CSVWriter
   :prefix: writers.csv
```

(hall-estimators)=
## Estimators (`estimators.*`)

Energy estimator definitions match training, with additional evaluation-only
estimators enabled through boolean flags.

- `TotalEnergy` is added automatically by the workflow and is not configurable
  via a config key.
- When `system.lz_penalty` or `system.l2_penalty` are nonzero, a
  `SpherePenalizedLoss` estimator is added automatically.
- `estimators.enabled.energy` defaults to `true`.
- `estimators.enabled.density` defaults to `false`.
- `estimators.enabled.pair_correlation` defaults to `false`.
- `estimators.enabled.one_rdm` defaults to `false`.

### Kinetic energy (`estimators.energy.kinetic.*`)

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.SphericalKinetic
   :prefix: estimators.energy.kinetic
```

### Coulomb potential (`estimators.energy.potential.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.hamiltonian.SpherePotential
   :prefix: estimators.energy.potential
```

### Density (`estimators.density.*`)

On the sphere, this accumulates a histogram of the polar angle $\theta$:

```{eval-rst}
.. config-defaults:: jaqmc.estimator.density.spherical.SphericalDensity
   :prefix: estimators.density
```

On the torus, it accumulates a 2-D histogram in the fractional coordinates
$(u,v)\in[0,1)^2$:

```{eval-rst}
.. config-defaults:: jaqmc.estimator.density.torus.TorusDensity
   :prefix: estimators.density
```

### Pair correlation (`estimators.pair_correlation.*`)

On the sphere, this computes $g(\theta)$ from geodesic pair angles:

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.sphere.pair_correlation.SpherePairCorrelation
   :prefix: estimators.pair_correlation
```

On the torus, it computes the two-dimensional
$g(\Delta u,\Delta v)$ from directed fractional-coordinate displacements,
periodically wrapped into the centered fundamental domain
$[-1/2,1/2)\times[-1/2,1/2)$. This places the origin at the center of the
heatmap, so points separated across a periodic boundary remain visually close
and all four directions around the origin are shown. The normalized matrix and
its bin-center axes are written to the digest as `pair_correlation`,
`pair_correlation:u`, and `pair_correlation:v`:

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.torus.pair_correlation.TorusPairCorrelation
   :prefix: estimators.pair_correlation
```

The matrix can be plotted directly as a heatmap:

```python
import matplotlib.pyplot as plt
import numpy as np

digest = np.load("evaluation_digest.npz")
g_uv = digest["pair_correlation"]
u = digest["pair_correlation:u"]
v = digest["pair_correlation:v"]

plt.pcolormesh(u, v, g_uv.T, shading="auto")
plt.xlabel(r"$\Delta u$")
plt.ylabel(r"$\Delta v$")
```

### One-body RDM (`estimators.one_rdm.*`)

On the sphere, this uses the lowest-Landau-level monopole-harmonic basis:

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.sphere.one_rdm.SphereOneRDM
   :prefix: estimators.one_rdm
```

On the torus, it uses the `flux` normalized lowest-Landau-level
guiding-centre orbitals defined by the same theta-function basis as the MHPO
wavefunction:

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.torus.one_rdm.TorusOneRDM
   :prefix: estimators.one_rdm
```
