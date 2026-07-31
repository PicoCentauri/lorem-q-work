# beastdb

Grand-canonical DFT database of relaxed electrocatalyst surfaces, computed at both
fixed applied potential and fixed charge. Most structures carry an explicit
potential alongside the resulting non-integer DFT charge response — a
genuinely potential/charge-resolved training signal, not a single neutral
ground state per structure.

## Source

Tezak, Clary, Gerits et al., ["BEAST DB: Grand-Canonical Database of
Electrocatalyst Properties"](https://doi.org/10.1021/acs.jpcc.4c06826), *J.
Phys. Chem. C* **128**, 20165 (2024).

## Computational details

Grand-canonical DFT (GC-DFT) computed with JDFTx in implicit solvent (CANDLE/LinearPCM).
Covers four material spaces — single-metal-atom N-doped graphene, flat/stepped pure
metal surfaces, binary covalent alloys, and bimetallic single-atom alloys — across CO2R,
OER/ORR, HER/HOR, and NRR reaction pathways, spanning 50 elements. Periodic slabs,
`pbc="T T F"` (JDFTx explicitly truncates the Coulomb interaction along the surface
normal). Per-atom `charges` are DDEC6 partial charges; only relaxed-geometry energies
and charges are reported, no forces. All structures are relaxed to an unknown  force
tolerance.

84% of structures are GC-DFT at fixed applied potential (`applied_potential`,
V vs. SHE, ~10 discrete values from -1 to 1.8 V), with `net_charge_state`
(generally non-integer) responding self-consistently. The remaining 16% are
neutral, fixed-charge calculations (`net_charge_state=0`, no
`applied_potential`). Extra per-frame info: `unique_id`, `adsorption_energy`,
`catalyst_formula`, `catalyst_facet`, `adsorbate_formula`.

## How to read the data

```python
from ase.io import read

frames = read("beastdb/beastdb_train.xyz", index=":")
atoms = frames[0]
charges = atoms.arrays["charges"]
energy = atoms.info["energy"]
tot_charge = atoms.info["tot_charge"]
applied_potential = atoms.info["applied_potential"]  # str "None" for the 16% non-GC-DFT frames
```
