# water_external_field

This dataset tests how a model responds to an external electric field. Each
water cluster is subjected to a random field, at a strength typical of an
electrode interface. It's a direct probe of field-induced polarization and
charge redistribution. The field enters the QEq energy as a simple additive
term, so in principle its effect is "free" for any model to learn. Even so,
the paper finds that QEq-based models still get the *shape* of the field
response wrong compared to DFT.

## Source

Vondrák, Reuter & Margraf, ["Pushing charge equilibration-based machine
learning potentials to their limits"](https://doi.org/10.1038/s41524-025-01791-3),
*npj Comput. Mater.* **11**, 288 (2025).

## Computational details

Neutral, non-equilibrium water clusters, generated the same way as in
`water_variational_charge` (GFN2-xTB MD at 300 K, no equilibrium minima).
Each cluster gets a uniform external field: the direction is sampled
uniformly on the unit sphere, and the magnitude is sampled uniformly between
0.01 and 0.2 V/Å (a realistic range at electrode interfaces).

Energies, forces, and Hirshfeld charges come from FHI-aims, hybrid PBE0,
tight settings. Structures are isolated, not periodic (no lattice). Every
frame carries `ext_field` (V/Å, xyz vector), a zero vector for field-free
structures.

## Physical test cases

**Perturbed vs. field-free MAE — this works today.**
Split into field-free (`ext_field` key) and perturbed
groups and compare energy/force error between them.

**Matched field-free/field-on pairs.** For a stricter same-geometry test,
use `water_external_field_paired_free.xyz` and
`water_external_field_paired_perturbed.xyz` — they line up 1:1, same
geometries, same order, index by index. Compare each field-free structure
directly against its field-on counterpart. These two files overlap with
`_train`/`_val`/`_test.xyz` (same underlying structures, just kept in their
original paired order here instead of being reshuffled), so use them only
for this specific test, not as extra training data.

**Field-magnitude sweep (Fig. 5 style).** Not possible with this dataset —
each geometry here has exactly one field value, not a range. To build this
test case: pick one geometry and one field direction, then compute energy
and forces across several field magnitudes (FHI-aims, PBE0, tight settings).
Compare the resulting E(ε) curve to DFT — QEq predicts an exact parabola.

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
