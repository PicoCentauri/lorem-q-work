# cpmace

Constant-potential DFT (VASPsol++) on a single 207-atom system,
**C70 H89 N Ni O46** — a Ni complex with water, `config_type=H2O`, fully
periodic (`pbc = T T T`), cell 14.80 × 12.82 × 25.0 Å.

| | |
|---|---|
| frames | 1093 (984 train / 109 valid, `split.py`, seed 0) |
| `energy` | −1334.38 … −1319.16 eV (spread 15.2) |
| `electron` | 661.26 … 662.31 (spread 1.05) |
| `potential` | −3.746 … −2.846 eV (spread 0.90) |
| forces | max \|F\| 5.65 eV/Å |

Frames are independently sampled, not a trajectory: consecutive frames differ
by ~1.8 Å RMSD. So the 90/10 split is per frame, with no grouping to preserve
(unlike razor, where one geometry appears at three bias charges).

No frame exceeds the 10 eV/Å max-force screen used in `../razor/`, so that
screen is a genuine no-op here.

## Reading the fields: two conversions, one of them not obvious

The fields are named differently from razor (`potential`/`electron` rather
than `work_function`/`bias_charge`), and the mapping is **not** a rename. Both
conversions below were checked against the data rather than assumed.

### 1. Electrons → charge: subtract 660

`q = −(electron − 660)`, the minus sign because electrons carry −e.

660 is the sum of standard VASP PAW valences (H 1, C 4, N 5, O 6, Ni 10) for
this composition. That is an assumption about which POTCARs were used, so it
was checked independently — see below, where thermodynamic consistency alone
implies **N₀ = 660.2**, agreeing to 0.2 electrons.

This gives `q ∈ [−2.31, −1.26]`, mean −1.90: the system is always ~1.9
electrons negative, i.e. a negatively charged interface throughout. Comparable
in magnitude to razor's `q ∈ [−1.75, 1.75]`.

Note that an error in N₀ would be a *constant offset* in q. For a
single-composition dataset the FiLM conditioning absorbs it, so training is
insensitive to it; it matters only for transferring a model between systems.

### 2. `energy` is the grand potential, not the total energy

**This is the one that bites.** The relation being tested is the standard
`μ = ∂E/∂N` (the Fermi level *is* the electron chemical potential), which
would give `∂E/∂q = −potential`.

Testing it needs care, because geometry dominates the energy and consecutive
frames are uncorrelated, so raw finite differences are useless (median
`dE/dN` = −3.05 with an IQR of [−11.1, +5.9]). Instead: build a
species-resolved pair-distance histogram per frame as a geometry descriptor,
then use Frisch–Waugh–Lovell — strip the geometry-predictable part from both
`energy` and `electron`, and regress the residuals. 37% of the electron-count
variance survives that projection, which is ample.

The result, stable across four orders of magnitude of ridge parameter:

```
d(energy)/dN = -1.58 eV        mean potential = -3.36 eV      ratio 0.47
```

A factor of ~2, far too stable to be noise. The explanation is that the
reported `energy` is the **grand potential** Ω = E − μ(N − N₀). Reconstructing
the total energy and repeating the regression:

| candidate | dE/dN | ratio to mean μ |
|---|---|---|
| `energy` as reported | −1.581 | 0.47 |
| `energy + potential·(electron − 660)` | −3.185 | **0.95** |
| `energy + potential·electron` | +609 | −181 |
| `energy − potential·(electron − 660)` | +0.02 | 0.00 |

And solving for the N₀ that makes it exact — requiring
`dΩ/dN + (N−N₀)·dμ/dN = 0` — gives **N₀ = 660.20**, which is the independent
confirmation of the valence sum quoted above. At that N₀ the ratio is 1.003.

So:

```
E_total      = energy + potential × (electron − 660)
q            = −(electron − 660)
∂E_total/∂q  = −potential          (≈ +2.85 … +3.75 eV/e)
```

The last line is the analogue of razor's `work_function`, and lands in the
same range razor's did (2.5–7 V), which is a further sanity check.

**Consequence for training.** `experiments/cpmace/prepare.py` must train on
`E_total`, not on the reported `energy`. Using the reported energy while
supervising `dE/dq = −potential` would be internally inconsistent: the
derivative of the grand potential is −1.58 eV/electron, not the −3.36 the
label asserts, so the energy and work-function targets would be pulling the
model in incompatible directions.

**Forces are unaffected** and need no conversion. By the envelope theorem,
`∂Ω/∂R|_μ = ∂E/∂R|_N` — the extra term carries `∂E/∂N − μ`, which vanishes at
the self-consistent electron count. So the reported forces are the gradient of
the total energy either way.

## Loss weights differ sharply from razor

Weights are set from label variances, and this dataset's are nothing like
razor's — one composition with a narrow energy spread, and a potential window
of 0.9 V rather than razor's 1.36 V *standard deviation*:

| target | cpmace variance | razor variance |
|---|---|---|
| energy (per atom) | 5.80e−5 | 1.40e−3 |
| forces | 0.708 | 0.473 |
| work function | 2.27e−2 | 1.688 |

Carrying razor's `100 : 1 : 0.05` over unchanged would give shares of
**E 0.8% / F 99.0% / W 0.2%** — energy and the work function would be
effectively untrained. What transfers between datasets is the *share*, not the
weight. Reproducing razor's tuned E 20.1 / F 67.8 / W 12.1 needs

```
energy 3620 : forces 1 : work_function 5.6
```

which is what `experiments/cpmace/` uses.

## Files

- `data.xyz` — as delivered, the source of truth. Raw VASP fields.
- `cpmace_train.xyz`, `cpmace_val.xyz` — 90/10 split, **verbatim subsets** with
  fields unconverted. The conversions above live in
  `experiments/cpmace/prepare.py` so the dataset files stay faithful to VASP.
- `split.py` — regenerates the split deterministically (seed 0).
