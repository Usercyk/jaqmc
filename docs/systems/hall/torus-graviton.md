# Torus Geometry and Quadrupole Spectra

Hall torus evaluation measures geometric stresses, physical density structure
factors, and a finite-subspace approximation to the quadrupole spectrum from
an existing ground-state checkpoint. It uses the complete real-space
Hamiltonian, including Landau-level mixing and the neutralized periodic
Coulomb interaction.

## Evaluate a ground-state checkpoint

This configuration enables up to six Ritz peaks and the structure factor.
`graviton` already includes both mass and interaction stresses.

```yaml
# torus-graviton.yaml
system:
  module: jaqmc.app.hall.config:HallTorusConfig
  flux: 18
  nspins: [6, 0]
  tau: [0.0, 1.0]
  interaction_strength: 1.0
  mass_shear: [0.0, 0.0]
  interaction_shear: [0.0, 0.0]
workflow:
  source_path: ./runs/torus-ground
  save_path: ./runs/torus-graviton
  batch_size: 256
estimators:
  enabled:
    energy: true
    structure_factor: true
    graviton: true
  structure_factor:
    indices: [[0, 0], [1, 0], [0, 1], [1, 1], [1, -1]]
  graviton:
    rcond: 1.0e-6
    trial_radius: 0.5
    krylov_order: 3
    trial_power: 3
    vmap_chunk_size: 1
run:
  iterations: 1000
  burn_in: 100
```

The wavefunction architecture, flux, cell shape, electron counts, and
Hamiltonian settings must match the source calculation. Additional `wf.*`
settings from that calculation belong in the evaluation configuration as well.

Both torus MHPO variants embed each electron's fractional coordinates $(u,v)$
as $(\cos 2\pi u,\sin 2\pi u,\cos 2\pi v,\sin 2\pi v)$. The backbone
appends a spin feature, so its input projection has five rows. Checkpoints
trained with the former center-of-mass inputs have nine rows and require
their original wavefunction implementation for evaluation.

Configuration inspection resolves the workflow without sampling or training:

```bash
jaqmc hall evaluate --yaml torus-graviton.yaml --dry-run
```

On a suitable compute host, the same command without `--dry-run` evaluates
the checkpoint without parameter updates. The result is
`evaluation_digest.npz`, with per-step matrices in `evaluation_stats.h5`.
The `graviton` estimator is evaluation-only. `krylov_order=1` retains the
original two-peak calculation and remains the default. Orders two and three
provide up to four and six peaks. Three blocks require up to sixth coordinate
derivatives and can be substantially more expensive than the two-peak
calculation. `vmap_chunk_size=1` limits simultaneous walker evaluations.
`jax.enable_x64=true` enables double precision when resolving small Gram
eigenvalues, with additional computational cost.

For static geometry measurements, `estimators.enabled.stress=true` selects
{class}`~jaqmc.app.hall.estimator.torus.TorusStress` without constructing the
excitation matrices. The `stress` and `structure_factor` estimators are also
available during existing ground-state optimization workflows.

## Independent metrics at fixed magnetic boundary conditions

Fractional electron coordinates satisfy $r=L\xi$, where

$$
L=L_1\begin{pmatrix}1&\tau_1\\0&\tau_2\end{pmatrix},\qquad
L_1^2\tau_2=2\pi N_\phi\ell_B^2.
$$

`system.tau` fixes the simulation cell. The two additional Cartesian shears
define independent metrics:

$$
h(s)=\begin{pmatrix}s_1&s_2\\s_2&-s_1\end{pmatrix},\qquad
g_m^{-1}=e^{h(s_m)},\qquad \widetilde g_V=e^{h(s_V)}.
$$

`system.mass_shear` specifies $s_m$, the **inverse mass** metric parameters.
`system.interaction_shear` specifies $s_V$, the **covariant real-space
interaction** metric parameters. Both matrices have unit determinant, and
neither shear changes the stored fractional coordinates or magnetic boundary
conditions. Their zero values recover the isotropic Hamiltonian.

The kinetic estimator contracts the covariant second derivatives with
$M=L^{-1}g_m^{-1}L^{-T}$. The interaction evaluates
$1/\sqrt{r^T\widetilde g_Vr}$ by transforming positions and all periodic
images with $C=e^{h(s_V)/2}$ and reciprocal vectors with $C^{-T}$.
The Ewald reference cell is built at the configured interaction shear.
Geometry derivatives use relative transformations about that reference,
with fixed reciprocal lists and a fixed Ewald separation parameter.
The self and background terms are included, and remain constant for this
area-preserving deformation.

Finite Ewald lists require convergence checks under deformation.
`estimators.energy.potential.ewald_gmax` and `.nlatvec` control ordinary energy
evaluation. The corresponding `estimators.stress.*` or
`estimators.graviton.*` keys control geometric evaluation. Matching cutoffs
and increasing both provides a consistent convergence check. Large shears
can require larger cutoffs than the undeformed cell.

```{eval-rst}
.. config-defaults:: jaqmc.app.hall.config.HallTorusConfig
   :prefix: system
   :scope: HallTorusConfig
```

## Static density and shear output

For each integer reciprocal label $n$, the density mode and physical
momentum are

$$
\rho_n=\sum_j e^{2\pi i n\cdot\xi_j},\qquad k_n=2\pi L^{-T}n.
$$

{class}`~jaqmc.app.hall.estimator.torus.TorusStructureFactor` preserves
directional information. Its digest contains:

| Key | Meaning |
| --- | --- |
| `structure_factor` | Connected $(\langle|\rho_n|^2\rangle-|\langle\rho_n\rangle|^2)/N$ |
| `structure_factor:raw` | Unsubtracted $\langle|\rho_n|^2\rangle/N$ |
| `structure_factor:rho` | Complex mean density mode |
| `structure_factor:indices` | Integer reciprocal labels |
| `structure_factor:k` | Cartesian wavevectors in $\ell_B^{-1}$ |
| `stress:mass` | Two derivatives of kinetic energy with respect to `mass_shear` |
| `stress:interaction` | Two derivatives of periodic potential energy with respect to `interaction_shear` |

The connected subtraction uses averages over all walkers, devices, and
evaluation steps. The zero mode has raw value $N$ and connected value zero.
This is a physical electron-density structure factor. Guiding-center
structure factors require a justified Landau-level projection.

The stresses are derivatives at the configured shears. At zero shear,
$O_\pm=T_1^V\pm iT_2^V$ identify the two quadrupole channels. At finite
background anisotropy they remain defined derivative combinations, while
continuous rotational helicity is no longer an exact quantum number.
Bare mass stress and interaction stress correspond to different physical
perturbations. Their weights do not automatically equal an experimental
Raman response.

## Excitation matrices and spectral interpretation

{class}`~jaqmc.app.hall.estimator.torus.TorusGraviton` constructs a
Hamiltonian Krylov basis from two interaction-stress seed states.
The smooth periodic pair factor $R$ suppresses short-distance coincidences:

$$
d_{ij}^2=\left|\frac{(e^{2\pi i(\xi_i-\xi_j)}-1)L^T}{2\pi}\right|^2,
\qquad R=\prod_{i<j}\left(1-e^{-d_{ij}^2/(2a^2)}\right).
$$

Here the exponential acts componentwise, and the complex squared norm
includes both real and imaginary parts. Near coincidence $d_{ij}$ approaches
the Cartesian pair distance. The radius $a$ is
`estimators.graviton.trial_radius`, defaulting to $0.5\ell_B$.
The two seeds and full input basis are

$$
s_a=R^pT_a^V\Psi_0,\qquad
\Phi=(\Psi_0,s_1,s_2,Hs_1,Hs_2,\ldots,H^{m-1}s_1,H^{m-1}s_2),
$$

where $m$ is `estimators.graviton.krylov_order`, from one to three.
`estimators.graviton.trial_power` sets the integer $p\ge m$ and defaults
to $m$ when unset. For $m=p=1$, the factors are the original
$F=(1,RT_1^V,RT_2^V)$. With $m=p=3$, the seven input functions yield at most
six excited Ritz states after reference projection. Krylov blocks use the
complete real-space Hamiltonian, rather than powers of its small projected
matrix. Reference projection is performed after constructing all blocks.

At a linear fermionic pair zero, $\Psi_0\sim r$ and $T_a^V\sim1/r$.
The seeds behave as $r^{2p}$. Each Hamiltonian application can lower the
radial power by two, so $H^m s_a$ behaves no worse than $r^{2p-2m}$
in the leading pair-collision power count. Requiring $p\ge m$ keeps the
actions needed for the full residual square integrable. Repeatedly applying
$H$ to the original $p=1$ seeds would not satisfy that requirement at
higher orders. This condition addresses pair-collision integrability, while
numerical and sampling convergence still need verification.

Holding `trial_power=3` and `trial_radius` fixed while increasing
`krylov_order` from one to three gives nested spaces for basis convergence
comparisons. With automatic `trial_power`, the seeds change between orders.
The physical Hamiltonian and transition sources $O_\pm$ remain unchanged.
Varying `trial_radius` and `trial_power` tests seed dependence.

With
$q_{i\mu}=D_{i\mu}\Psi_0/\Psi_0$, the Hamiltonian action is

$$
\frac{H(F_a\Psi_0)}{\Psi_0}
=F_aE_{\rm loc}-\frac12\sum_iM^{\mu\nu}
\left(\partial_{i\mu}\partial_{i\nu}F_a
+2q_{i\mu}\partial_{i\nu}F_a\right).
$$

This expression keeps the complete local energy of the approximate reference.
The implementation applies the covariant Hamiltonian directly to scaled
wavefunctions using nested directional derivatives. It avoids constructing
full high-order derivative tensors and avoids differentiating $\log T_a$ or
the logarithm of a Krylov state, which can have additional zeros. The
reference amplitude is divided out using a fixed local log-amplitude anchor.
Computing all basis actions and residuals requires up to $2m$ coordinate
derivatives of the reference wavefunction and the seed factors.

The estimator averages

$$
G_{ab}=\langle F_a^*F_b\rangle_0,\quad
H_{ab}=\left\langle F_a^*H(F_b\Psi_0)/\Psi_0\right\rangle_0,\quad
D_{a,\pm}=\langle F_a^*O_\pm\rangle_0.
$$

These matrices are saved as `graviton:gram`, `graviton:hamiltonian`, and
`graviton:transitions`. Digest processing projects out the sampled ground
reference, truncates small Gram eigenvalues, and solves $Hc=EGc$ on the
remaining space. Input matrices are symmetrized after recording their
relative anti-Hermitian norms. Trial columns are normalized by their
unprojected norms before Gram diagonalization, so the different scales of
$s_a$, $Hs_a$, and $H^2s_a$ do not by themselves remove valid directions.
Returned coefficients are transformed back to the original input basis.
Significant negative Gram eigenvalues cause
an error rather than producing invalid spectral weights.
The unprojected trial-block norm provides a floor for the Gram cutoff scale,
so subtraction noise in nearly constant operators does not create false modes.

| Key | Meaning |
| --- | --- |
| `graviton:excitation_energies` | Ritz energies relative to the sampled reference energy |
| `graviton:weights` | Per-electron discrete peak weights, columns **plus, minus** |
| `graviton:coefficients` | Normalized Ritz eigenvectors in the reference plus $2m$ Krylov basis, in block order |
| `graviton:energy_variances` | Full-Hamiltonian energy variances per Ritz state, in $(\hbar\omega_c)^2$ |
| `graviton:residual_norms` | Square roots of those variances, in $\hbar\omega_c$ |
| `graviton:ground_energy` | Real reference Rayleigh energy |
| `graviton:gram_eigenvalues` | Projected Gram eigenvalues after trial-column normalization, before truncation |
| `graviton:retained_gram_eigenvalues` | Eigenvalues retained with `rcond` |
| `graviton:hermiticity_error` | Relative anti-Hermitian norms of raw $G,H$ |
| `graviton:subspace_weight` | Sum of retained peak weights per channel |
| `graviton:subspace_first_moment` | Sum of energy times weight per channel |
| `graviton:subspace_centroid` | First moment divided by weight, NaN for zero weight |
| `graviton:unprojected_norm` | Full unprojected $\langle|O_\pm|^2\rangle/N$ |
| `graviton:n_ground_references` | Number of supplied ground references, currently one |
| `graviton:krylov_order` | Number of two-direction Krylov blocks |
| `graviton:trial_power` | Resolved pair-suppression exponent $p$ |

The $2m$ Krylov trial functions provide at most $2m$ Ritz peaks. Small Gram
eigenvalues can reduce the number of retained peaks. Their energy
differences and weights approximate a finite subspace and need basis, sampling,
system-size, and Ewald convergence checks. Because the physical source is
not restricted to the trial basis, the retained moments describe captured
subspace weight rather than complete source-state moments. The centroid is a spectral average
and can differ from every individual peak energy. Excitation energy differences
have no automatic variational upper-bound property. Negative differences are
preserved for diagnosing reference quality and sampling errors.

The default projection removes only $\Psi_0$. A torus at filling $1/3$ can
have other ground partners, so the default digest does not establish
projection of the entire ground manifold. Explicit multi-reference matrices
can be analyzed through
{func}`~jaqmc.app.hall.graviton.solve_graviton_subspace` with
`ground_indices=(...)` and an explicit `ground_energy`. Reference states and
trial states belong to the same full input basis. Their cross overlaps and
all Hamiltonian blocks must be included, including approximate-reference
residuals. The routine projects using the reference Gram pseudoinverse.

Real multiplication stresses commute, so their two unprojected full norms
are equal. Low-energy chirality needs energy-resolved weights and a specified
energy window. On a square torus, $+2$ and $-2$ also have the same $C_4$
rotation label. Identification of a low-energy FQH graviton further requires
ground-manifold control and separation from high-energy cyclotron response.
Independent excited-state optimization and strict LLL projection are separate
capabilities beyond this estimator.

## Full-Hamiltonian Ritz residuals

Each evaluation also saves the Hamiltonian-action overlap matrix

$$
B_{ab}=\left\langle h_a^*h_b\right\rangle_0,
\qquad h_a=H\phi_a/\Psi_0,
$$

as `graviton:hamiltonian_square`. This is $\langle H\phi_a|H\phi_b\rangle$
in the same normalization as $G$. It is not the square of the projected
Hamiltonian matrix. Actions through $H^m s_a$ provide this statistic
without applying $H$ twice to each basis state. For three blocks, the
highest action is $H^3s_a$, with up to sixth coordinate derivatives.

For a Ritz vector $c_n$, its total Rayleigh energy is
$E_n=\texttt{ground\_energy}+\Delta_n$. The residual obeys

$$
\epsilon_n^2
=\frac{\|(H-E_n)\sum_a c_{an}\phi_a\|^2}
       {\|\sum_a c_{an}\phi_a\|^2}
=\frac{c_n^\dagger Bc_n}{c_n^\dagger Gc_n}-E_n^2
=\operatorname{Var}_{\psi_n}(H).
$$

`graviton:energy_variances` and `graviton:residual_norms` report
$\epsilon_n^2$ and $\epsilon_n$. They include residual components along the
removed approximate reference and outside the input basis. In contrast, the
projected Ritz equation has zero residual by construction and cannot measure
the quality of the trial subspace. `graviton:hermiticity_error` measures
matrix symmetry rather than eigenstate quality.

The host computes these contractions in double precision. Subtracting the
second moment and squared energy limits the resolution near an exact
eigenstate, especially for matrices accumulated in float32. Negative
variances within roundoff tolerance are clipped to zero, while larger
negative values raise an error. A reported zero within this resolution does
not establish an exact eigenstate. Independent sampling or block resampling
is needed to assess statistical uncertainty, and basis convergence remains
necessary to identify physical modes.

## Analyze a saved digest

The host-side solver can recompute a spectrum with a different Gram cutoff.
All energies below are in $\hbar\omega_c$, and weights use the electron count
supplied to the solver.

```python
import numpy as np
from jaqmc.app.hall.graviton import solve_graviton_subspace

with np.load("runs/torus-graviton/evaluation_digest.npz") as digest:
    spectrum = solve_graviton_subspace(
        digest["graviton:gram"],
        digest["graviton:hamiltonian"],
        digest["graviton:transitions"],
        n_electrons=6,
        hamiltonian_square=digest["graviton:hamiltonian_square"],
        rcond=1e-6,
    )

energy_grid = np.linspace(0.0, 1.0, 1001)
intensity = spectrum.broaden(energy_grid, eta=0.01)
low_weights = spectrum.low_energy_weights(cutoff=0.5)
chirality = spectrum.chirality(cutoff=0.5)
residual_norms = spectrum.residual_norms
energy_variances = spectrum.energy_variances
```

Older digests without `graviton:hamiltonian_square` can still be analyzed by
omitting that argument. Their residual norms and energy variances are
unavailable and returned as `None`. A new evaluation of the checkpoint
collects the missing matrix without retraining.

`broaden` applies a Lorentzian display half-width, which is not a predicted
lifetime. `low_energy_weights` sums unbroadened peaks with
$0<\Delta\le E_{\rm cut}$ to avoid Lorentzian tails changing the selected
window. `chirality` returns $(W_--W_+)/(W_-+W_+)$, with NaN for a window
containing zero weight. Statistical errors require block or independent-run
resampling of the averaged matrices and repeated diagonalization. A single
digest does not estimate those errors automatically.

The physical motivation follows
[Haldane (2011)](https://arxiv.org/abs/1106.3375v2) and
[Liou et al. (2019)](https://arxiv.org/abs/1904.12231).
The implementation and derivation are recorded in
{ghsrc}`develop/fqh_geometry/torus_graviton_design_zh.md`.
