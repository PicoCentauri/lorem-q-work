# razor

A Pt(111)-water electrochemical interface dataset for the RAZOR
response-learning scheme: 3-layer Pt slabs covered by 24 explicit water
molecules, each computed at a range of interfacial bias charges.

## Source

Method: Bergmann, Bonnet, Marzari, Reuter & Hörmann, ["Machine Learning the
Energetics of Electrified Solid-Liquid Interfaces"](https://doi.org/10.1103/lm64-m3bn),
*Phys. Rev. Lett.* **135**, 146201 (2025)

## Computational details

5,113 Pt(111)-water interface structures (4,598 train / 515 test, disjoint
by structure ID), each 36 Pt + 24 H₂O = 108 atoms: a 3-layer, 12-atom/layer
Pt slab (bottom two layers fixed at bulk positions, top layer relaxed) in a
fixed 8.443×9.749 Å² in-plane cell (82.311 Å², `pbc="T T T"`; the
out-of-plane lattice vector varies slightly per structure to accommodate
the water layer). DFT-D3 dispersion correction. Structures were filtered to
a physically reasonable double-layer capacitance range (1.0–3.5 μF/cm²),
discarding pathological cases.

Per-frame info: `struc_pk` (source database id), `charge` (the ±q
finite-difference step used to evaluate that structure's response
derivatives, ranging -1.0 to +1.0 e in increments as fine as 0.05 e),
`DFT_E0` (zero-charge total energy, eV), `DFT_wf` / `DFT_clean_wf` (work
function of this structure / of the reference clean Pt slab, eV — the
latter constant across all structures), `DFT_P` / `DFT_clean_P` /
`DFT_delta_P` (interfacial polarization, reference polarization, and their
difference), `DFT_dEdq` (first-order energy response to q, Helmholtz-
transformed from `DFT_P`). Per-atom arrays: `DFT_F0` (zero-charge force,
eV/Å), `DFT_Z` (first-order force response to q — RAZOR's Born-effective-
charge-like Z*), `DFT_d2Fdq2` (second-order force response to q).

`training_data.json` / `test_data.json` are not additional data: the same
structures exploded into a long per-atom-per-Cartesian-dimension table
(hence their large size — 652MB / 68MB), with a NequIP-based RAZOR model's
predictions (`ML_P`, `ML_Z`, `ML_dEdq`, `ML_dFdq`) added alongside the DFT
ground truth for parity-plot benchmarking (see `collect_data.py`); not
needed for training.

## How to read the data

```python
from ase.io import read

frames = read("razor/train.xyz", index=":")
atoms = frames[0]
charge = atoms.info["charge"]        # e, finite-difference charge step for this structure
energy0 = atoms.info["DFT_E0"]       # eV, zero-charge total energy
wf = atoms.info["DFT_wf"]            # eV, work function of this structure
forces0 = atoms.arrays["DFT_F0"]     # eV/Å, zero-charge forces
born_Z = atoms.arrays["DFT_Z"]       # first-order force response to charge
d2Fdq2 = atoms.arrays["DFT_d2Fdq2"]  # second-order force response to charge
```
