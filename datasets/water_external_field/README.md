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

## Field sign convention

`ext_field` is the electric field **E** as actually seen by the DFT
calculation: force on an atom is `q * E` (O carries negative partial
charge, H positive, so the field-induced force on O projects opposite to
`ext_field` and on H projects along it), and the energy of a dipole in the
field is `-mu . E`. All files (`train`/`val`/`test`/`paired_free`/
`paired_perturbed`/`magnitude_sweep`) are consistent with this convention.

(2026-08-03: `val`, `test`, `paired_perturbed`, and `magnitude_sweep` were
found to have `ext_field` sign-flipped relative to `train` -- a pure
file-generation inconsistency, not noise, caught by projecting the
field-induced force change onto `ext_field` per element and checking the
sign against `F = qE`. Fixed in place; `check_field_conventions.py` in this
repo's tooling verifies the convention and should stay green for all files.)

## Physical test cases

**Exp 1: Perturbed vs. field-free MAE — this works today.**
Split into field-free (`ext_field` key) and perturbed
groups and compare energy/force error between them.

**Exp 2: Matched field-free/field-on pairs.** For a stricter same-geometry test,
use `water_external_field_paired_free.xyz` and
`water_external_field_paired_perturbed.xyz` — they line up 1:1, same
geometries, same order, index by index. Compare each field-free structure
directly against its field-on counterpart. These two files overlap with
`_train`/`_val`/`_test.xyz` (same underlying structures, just kept in their
original paired order here instead of being reshuffled), so use them only
for this specific test, not as extra training data.

**Exp 3: Field-magnitude sweep.** Use
`water_external_field_magnitude_sweep.xyz` — 50 neutral clusters (4–11
water molecules), each held at a fixed geometry and swept through 21 field
magnitudes from −0.2 to 0.2 V/Å (0.02 step) along a fixed direction (the
z-axis). Group by the `series` field and sort by `field_magnitude` to trace
out E(ε) per cluster, following Fig. 5 of the paper. Compare the resulting
curve to DFT — QEq predicts an exact parabola, but the paper finds the
QEq-based MLIP gets the shape wrong.

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

For the field-magnitude sweep, group by `series` (one per cluster) and sort
by `field_magnitude` (V/Å, signed component along the fixed field direction):

```python
frames = read("water_external_field/water_external_field_magnitude_sweep.xyz", index=":")
cluster = [a for a in frames if a.info["series"] == "STRUC000"]
cluster.sort(key=lambda a: a.info["field_magnitude"])
```
