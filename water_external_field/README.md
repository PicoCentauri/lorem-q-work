# water_external_field

Water clusters, each subjected to a randomly oriented external electric
field spanning the realistic range at electrode interfaces — a direct probe
of field-induced polarization and charge redistribution. The field response
is formally an additive, "ML-free" perturbation on top of the trained
short-range part, yet the paper finds QEq-based models still get its
*shape* wrong compared to DFT.

## Source

Vondrák, Reuter & Margraf, ["Pushing charge equilibration-based machine
learning potentials to their limits"](https://doi.org/10.1038/s41524-025-01791-3),
*npj Comput. Mater.* **11**, 288 (2025).

## Computational details

Neutral, non-equilibrium water clusters (same generation method as
`water_variational_charge`: GFN2-xTB MD at 300 K, no local minima), each
subjected to a uniform external electric field — direction sampled
uniformly on the unit sphere, magnitude sampled uniformly in 0.01–0.2 V/Å
(the realistic range at electrode interfaces). Energies, forces and
Hirshfeld charges from FHI-aims, hybrid PBE0, *tight* settings. Isolated (no
lattice); `ext_field` (V/Å, xyz vector) given per frame, zero vector for
field-free structures.

## How to read the data

```python
from ase.io import read

frames = read("water_external_field/water_external_field_train.xyz", index=":")
atoms = frames[0]
forces = atoms.arrays["forces"]        # eV/Å
charges = atoms.arrays["charges"]      # Hirshfeld, per atom
energy = atoms.info["energy"]          # eV
ext_field = atoms.info["ext_field"]    # V/Å, xyz vector
```
