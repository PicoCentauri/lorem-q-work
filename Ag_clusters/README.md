# Ag_clusters

Example of a system where a purely local, cutoff-based
potential cannot distinguish two structures that differ only in total
charge: at this size there are no long-range effects, and the whole system
sits inside any reasonable descriptor cutoff.

## Source

Ko, Finkler, Goedecker & Behler, ["A fourth-generation high-dimensional neural
network potential with accurate electrostatics including non-local charge
transfer"](https://doi.org/10.1038/s41467-020-20427-2), *Nat. Commun.* **12**,
398 (2021). Reference data archived at the
[Materials Cloud Archive](https://doi.org/10.24435/materialscloud:f3-yh).

## Computational details

Small Ag₃ trimer in two ionic charge states, Ag₃⁺ and Ag₃⁻ (both present,
mixed, in `train.xyz`/`test.xyz`). Structures generated per charge state via
Born-Oppenheimer MD at 300 K (Nosé-Hoover thermostat, NVT, effective mass
1700 cm⁻¹), 5000 steps at 0.5 fs, plus the trajectory of geometry
relaxations run to a force convergence of 0.001 eV/Å (0.0015 eV/Å for
Ag₃⁻). Energies, forces and Hirshfeld charges computed with the all-electron
code FHI-aims, PBE functional, *light* basis/integration settings,
spin-polarized. Non-periodic (`pbc="F F F"`, no lattice).

Per-atom `initial_charges` (kept under its original name from the source
file) is the converged DFT Hirshfeld charge, not an initial guess — a
naming quirk inherited from the published data.

**Split:** 90%/10% train/test by random selection, exactly as done in the
paper ("the remaining 10% of the data points was used as an independent test
set to confirm the reliability of PESs and detect possible over-fitting") —
9,912 train / 1,101 test here. No separate validation set is defined, in the
paper or here. Note: the paper's text states 10,019 total reference points
for Ag₃ (Results, "Metal clusters: Ag₃"); the archived files actually total
11,013 — reported as measured rather than repeating the paper's figure.

## How to read the data

```python
from ase.io import read

frames = read("Ag_clusters/Ag_clusters_train.xyz", index=":")
atoms = frames[0]
forces = atoms.arrays["forces"]           # eV/Å
charges = atoms.arrays["initial_charges"] # Hirshfeld, per atom
energy = atoms.info["energy"]             # eV
tot_charge = atoms.info["tot_charge"]     # e, +1 or -1
```
