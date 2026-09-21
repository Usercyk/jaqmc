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

When evaluating a training checkpoint, these settings must match the
training run. For direct `laughlin` or `free` evaluation, select the
analytic module with `wf.module`. MHPO options match the
[training wavefunction config](#hall-train-wf).

### Laughlin options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.laughlin.Laughlin
   :prefix: wf
   :scope: Laughlin
```

### Free options (`wf.*`)

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.free.Free
   :prefix: wf
   :scope: Free
```

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
- `estimators.enabled.fubini` defaults to `false`.
- When `system.lz_penalty` or `system.l2_penalty` are nonzero, a
  `SpherePenalizedLoss` estimator is added automatically. That requires both
  energy and angular momentum to be enabled. Reported energy remains
  `total_energy`.
- `estimators.enabled.energy` defaults to `true`.
- `estimators.enabled.angular_momentum` defaults to `true`. Setting it to
  `false` while an angular-momentum penalty is active raises a configuration
  error.
- `estimators.enabled.density` defaults to `false`.
- `estimators.enabled.pair_correlation` defaults to `false`.
- `estimators.enabled.one_rdm` defaults to `false`.

### Kinetic energy (`estimators.energy.kinetic.*`)

Covariant kinetic energy on the Haldane sphere. See
[Kinetic energy](../../guide/estimators/kinetic.md#spherical-kinetic-energy).

```{eval-rst}
.. config-defaults:: jaqmc.estimator.kinetic.SphericalKinetic
   :prefix: estimators.energy.kinetic
```

### Angular momentum (`estimators.angular_momentum.*`)

Computes `angular_momentum_z`, `angular_momentum_z_square`, and
`angular_momentum_square` on the Haldane sphere. See
[Angular momentum](../../guide/estimators/angular-momentum.md).

```{eval-rst}
.. config-defaults:: jaqmc.estimator.angular_momentum.SphericalAngularMomentum
   :prefix: estimators.angular_momentum
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

On the sphere, this accumulates a histogram of geodesic pair angles
$\theta_{ij}$ weighted by $1/\sin\theta_{ij}$. The digest holds the raw
weighted counts; multiply by
$4b / (\pi N^2 n_{\text{walkers}} n_{\text{steps}})$ — with $b$ the
bin count, $N$ the electron number, $n_{\text{walkers}}$ the global walker
count (`workflow.batch_size`), and $n_{\text{steps}}$ the step count from
the digest key `pair_correlation:n_steps` — to obtain $g(\theta)$:

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

On the sphere, this uses the lowest-Landau-level monopole-harmonic basis.
Its trace is the number of electrons on the lowest Landau level,
$N_\text{LLL}$:

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.sphere.one_rdm.SphereOneRDM
   :prefix: estimators.one_rdm
```

On the torus, it uses the normalized `n=0,1,2` Landau-level orbitals
defined by the same theta-function basis as the MHPO wavefunction. The basis is
ordered first by Landau-level index and then by guiding-centre index, so the
matrix shape is `(3 * flux, 3 * flux)`:

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.estimator.torus.one_rdm.TorusOneRDM
   :prefix: estimators.one_rdm
```

For both geometries, the digest contains `one_rdm`,
`one_rdm:diagonal`, and `one_rdm:trace`. The torus digest additionally
contains the diagonal Landau-level blocks `one_rdm:n0`, `one_rdm:n1`, and
`one_rdm:n2`; the trace of each block is the occupation of that level.

### Fubini--Study distance (`estimators.fubini.*`)

This estimator compares every training checkpoint in a directory with one
reference checkpoint. By default the checkpoint with the largest saved step is
the reference; set `reference_step` to select another one. The directory
defaults to `workflow.source_path`, or it can be set explicitly with
`checkpoint_path`.

```{eval-rst}
.. config-defaults:: jaqmc.estimator.fubini.FubiniStudyDistance
   :prefix: estimators.fubini
```

For checkpoints $\psi_i$, a reference $\psi_r$, and samples from the network
$\psi_s$ selected by `workflow.source_path`, the estimator uses

$$
\Delta_i(x)=\log\psi_i(x)-\log\psi_s(x)
$$

and computes

$$
\cos\gamma_i =
\frac{\left|\mathbb E_s\left[
e^{\Delta_i+\Delta_r^*}\right]\right|}
{\sqrt{\mathbb E_s[e^{2\operatorname{Re}\Delta_i}]
       \mathbb E_s[e^{2\operatorname{Re}\Delta_r}]}}.
$$

Thus the sampling checkpoint does not have to be the reference, although using
the last checkpoint for both is normally the most efficient choice. The
distance is $\gamma_i=\arccos(\cos\gamma_i)$ in radians.

For example, to compare all checkpoints in a Hall training directory with its
last checkpoint:

```console
jaqmc hall evaluate --yml runs/hall/train_config.yaml \
  workflow.source_path=runs/hall \
  workflow.save_path=runs/hall/fubini-eval \
  workflow.config.ignore_extra=true \
  estimators.enabled.energy=false \
  estimators.enabled.fubini=true
```

To use a particular saved step instead, add for example
`estimators.fubini.reference_step=50000`. The digest
`fubini-eval/evaluation_digest.npz` contains aligned arrays:

- `fubini:step`: saved checkpoint steps;
- `fubini:distance`: Fubini--Study angles in radians;
- `fubini:cosine` and `fubini:fidelity`;
- `fubini:modulus_cosine` and `fubini:phase_factor`, whose product is
  `fubini:cosine` up to Monte Carlo error;
- `fubini:reference_step`: the selected reference step.

All checkpoint parameter sets are loaded together and evaluated for each
walker. Consequently, device memory and evaluation cost grow approximately
linearly with the number of saved checkpoints.
