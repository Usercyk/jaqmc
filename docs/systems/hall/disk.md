# Quantum Hall on a disk

The disk geometry uses Cartesian electron coordinates `(x, y)` in magnetic-length
units, with complex coordinate $z=x-iy$. Select it with
`system.module=jaqmc.app.hall.config:HallDiskConfig`:

```bash
jaqmc hall train system.module=jaqmc.app.hall.config:HallDiskConfig \
  system.nspins='[6,0]' system.flux=18 \
  workflow.save_path=runs/disk

jaqmc hall evaluate system.module=jaqmc.app.hall.config:HallDiskConfig \
  system.nspins='[6,0]' system.flux=18 \
  workflow.source_path=runs/disk workflow.save_path=runs/disk-eval
```

A configuration check runs through the same entry point with `--dry-run`.
Training and evaluation share the system and wavefunction settings. See
<project:../../guide/running-workflows.md> for iteration limits, resuming, and
checkpoint handling.

## System and energy

The default disk has six spin-up electrons, flux 18, and background separation
`system.d=0.1`. `system.radius` defaults to $\sqrt{2\,\mathtt{flux}}$.
The radius sets the initial sampling region and the uniform positive background
disk. Electron motion has no hard radial boundary.

The Hamiltonian combines the planar magnetic kinetic energy, pairwise Coulomb
repulsion, and attraction to the background disk. `system.interaction_strength`
scales both interaction terms. The reported potential omits the constant
background self-energy. Background potentials are integrated before sampling
and interpolated on a radial table; `estimators.energy.potential.radial_grid_size`
controls the table resolution.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.config.HallDiskConfig
   :prefix: system
   :scope: Disk

.. config-defaults:: jaqmc.estimator.kinetic.disk.DiskKinetic
   :prefix: estimators.energy.kinetic
   :scope: Disk

.. config-defaults:: jaqmc.app.hall.hamiltonian.disk.DiskPotential
   :prefix: estimators.energy.potential
   :scope: Disk
```

Disk energy evaluation also reports `observable:angular_momentum_z` and
`observable:angular_momentum_z_square`. With $z=x-iy$, an analytic factor $z^m$
has angular momentum $-m$. Training uses `system.lz_penalty`, `system.l2_penalty`,
and `system.lz_var_penalty` to select angular-momentum sectors. On the disk,
`l2_penalty` multiplies $L_z^2$; `lz_var_penalty` penalizes its variance.
Penalties require energy and angular-momentum evaluation to remain enabled.

## Wavefunctions

The default neural ansatz is `jaqmc.app.hall.wavefunction.disk.mhpo:DiskMHPO`.
It uses a Psiformer, disk orbitals, and a trainable Jastrow. Its envelope choices
are `no_m_learnable`, `m_learnable`, `m_learnable2` (default), and
`m_learnable2_old`. `system.num_orbitals` overrides the default orbital count,
which equals `system.flux`.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.wavefunction.disk.mhpo.DiskMHPO
   :prefix: wf
   :scope: Disk
```

The parameter-free `wf.module=laughlin` evaluates
$\prod_{i<j}(z_i-z_j)^m\exp(-\sum_i|z_i|^2/4)$, where
$m=\mathtt{flux}/N$. Fully polarized fermionic Laughlin states have odd integer
$m$. They can be evaluated directly without a training checkpoint:

```bash
jaqmc hall evaluate system.module=jaqmc.app.hall.config:HallDiskConfig \
  system.nspins='[2,0]' system.flux=6 wf.module=laughlin \
  workflow.save_path=runs/disk-laughlin run.iterations=1000
```

The imported fitted six-electron wavefunctions retain their original
coefficients and explicit class selections:

| Module under `jaqmc.app.hall.wavefunction.disk` | Classes |
| --- | --- |
| `symbolic` | `DiskPhi4`, `DiskPhi7`, `DiskPhi8` |
| `symbolic_20261021` | `DiskP6`, `DiskP12`, `DiskPsi5`, `DiskPsi6`, `DiskPsi7` |
| `symbolic_round13` | `DiskP05C330`, `DiskP10C459`, `DiskP11C476` |
| `symbolic_phi7_refit` | `DiskPhi7RefitC363` |

For example, `wf.module=symbolic:DiskPhi4` selects a fitted state. Its `wf.kappa`
and the Hamiltonian's `system.interaction_strength` are separate settings and
should describe the same intended coupling.

## Observables

Optional disk observables are configured under `estimators`:

| Enable flag under `estimators.enabled` | Result |
| --- | --- |
| `density` | Cartesian and radial electron counts |
| `pair_correlation` | Pair correlation relative to a fixed radius |
| `one_rdm` | Reduced density matrix in a disk Landau-level basis |
| `structure_factor` | Structure factor and connected structure factor |
| `edge_green` | Edge Green function with the first electron pinned |
| `wavefunction_scan` | Conditional Cartesian and radial wavefunction scans |
| `short_range` | Energy and density sums resolved by electron separation |

```yaml
system:
  module: jaqmc.app.hall.config:HallDiskConfig
  nspins: [6, 0]
  flux: 18
workflow:
  source_path: runs/disk
  save_path: runs/disk-observables
estimators:
  enabled:
    density: true
    pair_correlation: true
    structure_factor: true
  density:
    bins: 100
    range_max: 10
  pair_correlation:
    bins: 100
    max_r: 10
    r: 3
```

Pair-correlation bins with no density support are undefined and remain `NaN`.
`short_range` requires energy and angular momentum, and its `kappa` must match
`system.interaction_strength`. Its bin edges include an overflow bin ending at
positive infinity.

Edge Green-function evaluation uses a conditional sampler that fixes the first
electron at the configured edge radius. Equilibrium observables use a separate
evaluation with the ordinary disk sampler.

## Wavefunction overlap

`jaqmc hall overlap` samples the symmetric density
$\rho\propto|\psi|\,|\psi_{\rm ref}|$ to measure the overlap of two disk states:

```bash
jaqmc hall overlap system.nspins='[6,0]' system.flux=18 \
  workflow.source_path=runs/disk-current \
  workflow.reference_path=runs/disk-reference \
  workflow.save_path=runs/disk-overlap
```

Both checkpoints use the configured architecture and parameter structure.
Omitting `workflow.reference_path` computes self-overlap. Parameter-free states
can also omit `workflow.source_path`. The output `overlap_digest.npz` contains
`cos_gamma_modulus`, `f_phase`, `cosine`, and `fubini_distance`.
Disk overlap uses this dedicated command; the checkpoint comparison flag
`estimators.enabled.fubini` retains its existing sphere/torus behavior.

Disk code was integrated from
[`zyh1001/jaqmc_merge`, commit `45d68f1`](https://github.com/zyh1001/jaqmc_merge/commit/45d68f116307290a3ed82acd6f45a6d34abbe23b).
The original disk parameter names and envelope modes are retained. Earlier
checkpoints missing parameters such as `amplitude_imag` require explicit
conversion. Fresh evaluation directories avoid mixing accumulated states from
different versions.
