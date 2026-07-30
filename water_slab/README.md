# water_slab

A periodic slab of water molecules subjected to a range of external
electric fields along the slab normal — a condensed-phase probe of
field-induced polarization. QEq-based models predict an
unphysical, linearly varying charge distribution through the slab under
field, effectively transferring charge from one surface to the other,
where DFT instead shows localized charge rearrangement confined to the two
surfaces.

## Source

Vondrák, Reuter & Margraf, ["Pushing charge equilibration-based machine
learning potentials to their limits"](https://doi.org/10.1038/s41524-025-01791-3),
*npj Comput. Mater.* **11**, 288 (2025), Supplementary Note 3
("Polarization of water slabs").

## Computational details

Periodic slab of 32 water molecules (96 atoms), initial configurations from
classical MD (SPC/E water model, LAMMPS: 1 ns run, 0.2 ns equilibration
discarded, 160 snapshots taken from the remaining 0.8 ns). Each snapshot
evaluated by DFT under 7 homogeneous field values along z:
0, ±0.1, ±0.2, ±0.3 V/Å. Energies, forces and Hirshfeld charges from
FHI-aims, plain PBE (GGA, not the hybrid PBE0 used for the cluster
experiments), *intermediate* basis/integration settings, 5×5×1 k-point mesh.
All neutral. Cell: 8×8×60 Å, `pbc="T T T"` — full 3D Ewald electrostatics;
the 60 Å vacuum gap and single k-point along z decouple periodic images
along the slab normal.

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
