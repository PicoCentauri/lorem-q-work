# natcomm2025

Grand-canonical DFT (ABACUS + compensating-charge plate) on a Pt(111)/water
interface, from Nat. Commun. 2025 (`s41467-025-58871-7`). The published DP-Nₑ
model feeds the electron number straight into the network's input layer.

| | |
|---|---|
| frames | 16,821 (19,538 source, 2,717 exact duplicates dropped) |
| distinct geometries | 2,750 (each at 4–18 electron counts, median 6) |
| composition | Pt₁₀₀ H₉₇ O₃₆ — 233 atoms |
| cell | slab normal **x** (30.888 Å); surface area \|b×c\| = 171.20 Å² |
| pbc | T T T |
| energy | −347307.50 … −347297.27 eV (spread 10.23) |
| `fparam` | 0.0011 … 1.1975 |
| forces | eV/Å, no conversion needed (DeepMD is eV/Å throughout) |

27 DeepMD systems: six named `0.60`…`1.10` hold one fixed `fparam` each and
share geometries with one another (the stencil), plus 21 `data.NNN-M`
exploration sets from DP-GEN.

## `fparam` is the electron number — confirmed against the paper

The training input declares `numb_fparam: 1` and never names it. It was first
identified from the data, then confirmed by the paper:

> *"We incorporate the total number of electrons Nₑ directly into the input
> layer of the neural network"*

and, decisively for the sign:

> *"the work function of an instantaneous configuration will fluctuate around
> −μₑ"*, with μₑ = ∂E/∂Nₑ

The data-only argument reached the same place and is worth keeping, since it
also fixes the conventions:

- **Not a potential.** E(`fparam`) curves *upward*, d²E/d`fparam`² = **+7.68 eV**
  (IQR [7.44, 7.73]). A grand potential in U would give d²Ω/dU² = −C < 0.
- **Charge-like.** That curvature is near-constant across all 2750 geometries —
  the signature of a capacitive q²/2C term.
- **Electrons, not charge.** d E/d`fparam` is negative (median −2.53 eV). Read as
  a positive charge that would make the work function negative; read as excess
  electrons it gives +2.53 V, inside razor's 2.51–6.99 V range.

So:

```
total_charge  = -fparam                     (electrons carry -e)
work_function = dE/dq = -dE/dfparam         ( = -mu_e, the paper's W)
```

`fparam` is the *excess* electron count, not the absolute one — 233 atoms would
carry ~2000 electrons. Whether `fparam = 0` is exactly neutral is not resolvable
from the data, and does not matter for training: a constant shift in q is
absorbed by the FiLM charge conditioning on a single-composition dataset. It
would matter for transferring a model to another system.

## Where the work function comes from

There is no per-frame label, and none is needed: **every geometry appears at
4–84 `fparam` values**, so ∂E/∂`fparam` is available by finite difference at
fixed geometry — the same construction as razor's `dEdq_fd`. `convert.py` fits a
quadratic to E(`fparam`) per geometry and differentiates at each frame's own
`fparam`, which respects the curvature instead of assuming linearity.

The fits are excellent: **median residual 0.23 meV**, p95 15 meV, over groups
with a median of 6 distinct `fparam` values spanning 0.8.

### Read the work-function range with care

`work_function` spans **−4.49 … +8.09 V (std 2.36)**, which is far wider than
the paper's quoted work-function *fluctuation* of ~±0.5 eV. Both are correct
and they measure different things:

- The paper's ±0.5 eV is the fluctuation **at fixed μₑ**, where Nₑ sits near its
  grand-canonical equilibrium and W stays near −μₑ.
- This dataset deliberately evaluates the **same geometry off-equilibrium**, at
  5–6 electron counts spanning ~0.8 e. Since dW/dq = d²E/dq² = 7.68 V/e, that
  alone sweeps W by ~6 V.

Removing the charge dependence leaves a residual std of **1.82 V**, which is the
geometry-driven part.

Note also that 7.68 V/e corresponds to **1.22 μF/cm²** over this cell's 171 Å²,
roughly ten times smaller than a physical double-layer capacitance (10–20
μF/cm²). That is expected: the SI describes *"placing a compensating charge
plate in the vacuum region right above the water solvation layer"*, so the
curvature is dominated by the supercell's compensating capacitor, not by the
electrochemical double layer. The labels are faithful to the calculations; they
are not a physical electrode capacitance.

## Duplicates are dropped

**2,717 of the 19,538 source frames (13.9%) were exact duplicates** — identical
geometry, `fparam`, energy *and* forces. They carry no information and only
reweight whatever they repeat, so `convert.py` drops them
(`DROP_DUPLICATES = True`; set False to keep them, flagged `duplicate=True`).
16,821 frames remain, and the largest geometry group shrinks from 84 to 18.

Groups are recomputed after the drop, so the E(`fparam`) fits see only
surviving frames.

## The energy baseline is essential, and marathon fits it

Raw energies are ~−347,303 eV. float32 spacing at that magnitude is **31.2 meV**,
so the energy signal (residual std 1.28 eV, 5.5 meV/atom) would be destroyed by
rounding if it were used directly. `marathon.grain.prepare()` fits a per-species
elemental baseline and the model learns the residual, whose float32 spacing is
0.48 µeV. Nothing needs doing by hand -- but this is load-bearing here, not a
convenience.

Composition is constant, so the per-species split is degenerate: only the total
offset is identifiable. That is harmless, and identical to cpmace.

Energy-per-atom variance after the baseline is **3.03e-05**, against cpmace's
5.80e-05 and razor's 1.40e-03 — so loss weights must be recomputed from these
variances rather than carried over, as they were for cpmace.

## Splitting: use `group`, never split within it

`group` is the geometry id. The 4–84 charge states of one geometry share atomic
positions, so a per-frame split would put near-identical structures in both
train and validation. This is razor's `struc_pk` situation exactly. Split on
`group`.

## Files

- `natcomm2025.xyz` — 417 MB, the only tracked artefact (git-lfs). Fields:
  `total_charge`, `work_function`, `fparam`, `group`, `n_in_group`, `duplicate`,
  `system`; energy and forces on the calculator.
- `convert.py` — regenerates the xyz from `data_set/`. Pure numpy + ase;
  **dpdata is not needed**, DeepMD's on-disk format is `.npy` plus two text
  files, so no extra dependency or venv was introduced.
- **Not tracked** (see `.gitignore`): `data_set/`, `OH-strus/`, `data.tar.gz`,
  `OH-strus.zip`, `compressed-dEdN.pb` (93 MB frozen TF model), and the paper
  PDFs.

`system` is written with a `sys/` prefix on purpose: six directories are named
`0.60`…`1.10`, and extxyz round-trips a bare `"0.60"` as the float `0.6`, which
would leave the field typed float for those and str for the `data.*` ones.

`OH-strus/` is a separate 233-atom system in the same format, not converted here.
