# omol_10K

A locally-usable, closed-shell subset of the Open Molecules 2025 (OMol25)
dataset, spanning a wide range of total charge states (-8 to +8 e) rather
than just neutral species. Small enough to load and iterate on directly,
subsampled with a rarest-element-first strategy so the full dataset's
periodic-table coverage survives the downsampling instead of collapsing to
only the common elements.

## Source

[OMol25](https://arxiv.org/abs/2505.08762) (FAIR Chemistry / Meta), CC-BY-4.0

## Computational details

Molecular DFT at the ωB97M-V/def2-TZVPD level (ORCA), spanning biomolecules,
metal complexes, electrolytes, and community datasets — the same underlying
data as `../omol/`, filtered and subsampled for local use. Isolated
molecules (no PBC).

Filtered to closed-shell structures only (spin multiplicity 1, no
radicals) — a different filter than OMol25's own "neutral" split; total
charge here still spans -8 to +8 e, only the spin state is restricted.
11,000 structures total (9,000 train / 1,000 val / 1,000 test), subsampled
with a rarest-element-first strategy to preserve periodic-table coverage
(72 / 62 / 63 distinct elements per split) rather than pure uniform
sampling. Per-atom `mulliken_charges` and `lowdin_charges` are present for
most structures; `nbo_charges` for a smaller, genuinely optional fraction —
the `Properties` column list varies frame-to-frame with what's actually
available.

Train/val/test are each drawn from a single one of OMol25's own 80 shards
per split (not the full multi-million-structure pool); the official OMol25
*test* split ships without labels, so both val and test here are drawn from
two different shards of the official *val* split instead.

## How to read the data

```python
from ase.io import read

frames = read("omol_10K/omol_small_train.xyz", index=":")
atoms = frames[0]
forces = atoms.arrays["forces"]                    # eV/Å
mulliken = atoms.arrays["mulliken_charges"]         # per atom
lowdin = atoms.arrays["lowdin_charges"]             # per atom
nbo = atoms.arrays.get("nbo_charges")               # per atom, only present for some frames
energy = atoms.info["energy"]                       # eV
tot_charge = atoms.info["tot_charge"]               # e
```
