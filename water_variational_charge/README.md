# water_variational_charge

This dataset tests whether a model can get the total charge of a system
right without relying on local, short-range cues. A cutoff-based potential
can guess charge near equilibrium just by counting H atoms nearby. That
guess breaks down once a cluster pulls apart beyond the cutoff. This is also
where QEq-style models tend to fail: they push too much charge between
molecules, and the energy shows small unphysical wiggles as the cluster
separates.

## Source

Vondrák, Reuter & Margraf, ["Pushing charge equilibration-based machine
learning potentials to their limits"](https://doi.org/10.1038/s41524-025-01791-3),
*npj Comput. Mater.* **11**, 288 (2025).

## Computational details

Water clusters of 3–10 molecules, taken from Xantheas' atlas of known
minima. Each cluster appears in three charge states: neutral, protonated
(+1, an extra H⁺), and deprotonated (−1, an OH removed). Non-equilibrium
geometries come from GFN2-xTB MD at 300 K. Clusters are also isotropically
expanded or contracted, from 0.9× to 5.0× their equilibrium size, to push
molecules apart beyond any reasonable cutoff.

Energies, forces, and Hirshfeld charges come from FHI-aims, using the hybrid
PBE0 functional with tight basis and integration settings. Structures are
isolated molecules, not periodic (`pbc="F F F"`, no lattice).

## Physical test cases

**Exp. 1 — mixed charge states near equilibrium.** Evaluate on the test (or
validation) set, grouped by `tot_charge`. Target: about 2 meV/atom, 49 meV/Å
RMSE, per charge state.

**Exp. 2 — dissociation, binned version.** Bin energy and force error by
`exp_coef` (e.g. {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}), separately for each
charge state, using the test or validation set. This averages over many
different clusters rather than tracking one cluster continuously.

**Exp. 3 — single-cluster dissociation curve.** Use
`water_variational_charge_dissociation_curves.xyz` — 17 clusters (4 neutral, 4 cation, 9
anion), each scaled through the full 0.9×–5.0× range (16 points), Group by the `series`
field.

## How to read the data

```python
from ase.io import read

frames = read("water_variational_charge/water_variational_charge_train.xyz", index=":")
atoms = frames[0]
forces = atoms.arrays["forces"]      # eV/Å
charges = atoms.arrays["charges"]    # Hirshfeld, per atom
energy = atoms.info["energy"]        # eV
tot_charge = atoms.info["tot_charge"]  # e, in {-1, 0, +1}
```
