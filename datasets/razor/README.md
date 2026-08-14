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

## The publication subset (`razor_centre_paper_{train,test}.xyz`)

`Publication_data_for_ploche/{train,test}.xyz` is **`razor_centre` restricted
to |q| ≤ 1.0 e**, not a separate dataset. Verified rather than assumed:

- all 5113 of its frames match ours on `(struc_pk, charge)`, 0 unmatched
- positions, cell, pbc and species order are **bit-identical**
- `DFT_wf` == our `work_function` exactly (corr 1.000000, max diff 0.0000)
- `DFT_d2Edq2` == the curvature of our 3-point stencil exactly (corr 1.0000)
- **100%** of its frames have `bias_charge == q_MD`, i.e. every one is a
  stencil centre, and 100% of its `struc_pk` are in `razor_centre.xyz`

So it is the same calculations, sampled one-per-structure at the centre.

**What differs is the reference charge of the energy and force labels.**
`DFT_E0` / `DFT_F0` are back-extrapolated to q = 0; our `energy` / `forces`
are at the frame's own charge. `DFT_E0` sits within −0.119 ± 0.375 eV of our
E extrapolated to q = 0, against +1.893 ± 3.555 eV when compared at the
frame's own charge.

**What the |q| ≤ 1 cap removes.** Of `razor_centre`'s 5989 structures, 5113
are in the publication set and 876 are not:

| | n | polarizable | \|q\| > 1.0 | max_force p99 |
|---|---|---|---|---|
| in the paper set | 5113 | 99.3% | 0 | 6.04 |
| excluded | 876 | 36.5% | 665 (76%) | 18.79 |

Capping the charge does most of the `polarizable` filtering for free — the
excluded set is where dielectric breakdown lives.

`make_paper_split.py` rebuilds the publication's **selection** with **our
labels** into `razor_centre_paper_train.xyz` (4598) and
`razor_centre_paper_test.xyz` (515), taking each structure's stencil centre
straight out of `razor_centre.xyz`.

**It survives deletion of the source folder.** Because every selected frame is
a stencil centre, the selection is fully determined by `struc_pk` alone — no
charges need recording — so the script writes
`razor_centre_paper_{train,test}_struc_pk.txt` (plain text, 4598 and 515 ids)
and falls back to them when `Publication_data_for_ploche/` is gone. Verified:
re-running with the folder hidden reproduces both xyz files byte-identically.
The script asserts every frame really is a centre before relying on that.

**Do not mix the two splits.** 452 of the publication's 515 test structures
sit in our `razor_train`, so training on our split and testing on theirs
would leak. Their split is internally clean (0 shared `struc_pk`), so use it
end to end.
