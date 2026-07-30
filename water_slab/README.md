# water_slab

This dataset is the periodic version of the field-response test. A slab of
water molecules sits under a range of electric fields, applied along the
slab normal. It's a condensed-phase probe of the same question as
`water_external_field`: does the model polarize correctly? QEq-based models
tend to get this wrong in a specific way: they predict charge spreading
linearly through the slab, as if it were transferring from one surface to
the other. DFT instead shows the charge staying near the two surfaces, not
moving through the bulk.

## Source

Vondrák, Reuter & Margraf, ["Pushing charge equilibration-based machine
learning potentials to their limits"](https://doi.org/10.1038/s41524-025-01791-3),
*npj Comput. Mater.* **11**, 288 (2025), Supplementary Note 3
("Polarization of water slabs").

## Computational details

A periodic slab of 32 water molecules (96 atoms). Starting configurations
come from classical MD (SPC/E water model, LAMMPS): a 1 ns run, with the
first 0.2 ns discarded as equilibration, and 160 snapshots taken from the
rest. Each snapshot is then evaluated by DFT under 7 field values along z:
0, ±0.1, ±0.2, and ±0.3 V/Å.

Energies, forces, and Hirshfeld charges come from FHI-aims, using plain PBE
(a GGA functional, not the hybrid PBE0 used for the cluster datasets), with
intermediate basis and integration settings, and a 5×5×1 k-point mesh. All
structures are neutral. The cell is 8×8×60 Å, and `pbc="T T T"` — this uses
full 3D Ewald electrostatics. The 60 Å vacuum gap, plus a single k-point
along z, is what keeps periodic images along the slab normal from
interacting.

## How to read the data

```python
from ase.io import read

frames = read("water_slab/water_slab_train.xyz", index=":")
atoms = frames[0]
forces = atoms.arrays["forces"]      # eV/Å
charges = atoms.arrays["charges"]    # Hirshfeld, per atom
energy = atoms.info["energy"]        # eV
ext_field = atoms.info["ext_field"]  # V/Å, xyz vector (z-directed here)
```
