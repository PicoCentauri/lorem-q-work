# razor

This dataset tests grand-canonical charge response directly: each Pt(111)/
water interface geometry is evaluated at several bias charges, so the model
can be trained to treat charge as an input and get the work function
(`∂E/∂q`) as an autograd derivative, rather than as a separate fitted label.
It's the closest dataset here in spirit to `beastdb`, but with dense
per-geometry charge sampling instead of one calculation per (structure,
potential) pair.

## Source

N. Bergmann, K. Reuter & N. G. Hörmann, *J. Chem. Phys.* **164**, 174110
(2026), campaign `RAZOR-qQ`. Provided directly by a colleague, not fetched by
us — `raw/razor/DATASET_README.md` is their original documentation and is
worth reading in full; it covers several non-obvious points (sign
conventions, dispersion parameters, why `bec_z` is damped) not repeated here.

## Computational details

Pt(111)/water interface: 36 Pt (3-layer slab) + 24 H₂O = 108 atoms, periodic
in-plane, non-periodic along the slab normal (`pbc="T T F"`, fixed in-plane
cell area 82.31 Å², out-of-plane vector varies per structure). Each of 5,989
geometries was evaluated at a 3-point bias-charge stencil around its native
charge (`q_MD ± 0.25 e`). Energies and forces are PBE-D3(BJ) (GPAW,
Solvated Jellium Method for the constant-charge electrostatics).

`bias_charge` is deliberately not called `tot_charge` like in the other
datasets here — it's the model input, not a passive label. `work_function`
(`∂E/∂q`) and `bec_z` (`∂F/∂q`, a Born-effective-charge-like response) are
the derivative targets. `train`/`val` are split by `struc_pk` (never by
frame — the 3 charges of one geometry must stay together), 90/10, our own
choice.

## Physical test cases

**Work function response — this works today.** Compare the model's autograd
`dE/dq` against the DFT `work_function` label, on every frame of
`razor_train.xyz`/`razor_val.xyz`. Sanity check: `∂²E/∂q²` should come out
close to 9 V/e and roughly structure-independent inside the `polarizable`
window.

**Wide charge-range extrapolation — this works today.** `razor_test.xyz`
holds dense 13-point charge sweeps (`q ∈ [−1.5, 1.5]`) per structure, well
beyond the ±0.25 e stencil width seen in training. Compare the predicted
`E(q)`/work-function curve against DFT across the full sweep — a real
extrapolation test, not just a held-out sample of the training distribution.

**Polarizable vs. non-polarizable structures.** Split by the `polarizable`
flag (67.6% of `razor_train.xyz`/`razor_val.xyz` pass) and compare error
between the two groups. Frames outside the window sit near dielectric
breakdown / Fermi pinning, where the physics gets harder — a similar theme
to the overpolarization pathology in the water datasets.

**Born effective charge (`bec_z`) — optional, use with caution.** The
colleague's own recommendation is not to supervise on `bec_z` by default: it
is a finite-difference estimate damped by at least 15% relative to the
converged value. If used, mask on `polarizable`, weight it low, and validate
against `razor_test.xyz`, whose `bec_z` comes from a 13-point spline (still
an estimate, but the least-damped one available). `razor_centre.xyz` (one
frame per training structure, the centre of each stencil) carries `bec_z`
too, but overlaps `razor_train.xyz`/`razor_val.xyz` — same (r, q) points,
extra columns — so don't add it as extra training data.

## How to read the data

```python
from ase.io import read

frames = read("razor/razor_train.xyz", index=":")
atoms = frames[0]
energy = atoms.get_potential_energy()        # eV, PBE-D3(BJ)
forces = atoms.get_forces()                  # eV/Å
bias_charge = atoms.info["bias_charge"]      # e, model input
work_function = atoms.info["work_function"]  # V, dE/dq target
struc_pk = atoms.info["struc_pk"]            # split key -- never split within one struc_pk
```
