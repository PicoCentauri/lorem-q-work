# lorem-q

Datasets for testing and benchmarking models that are aware of total charge
and an external electric field.

## How to use

Anything that's bigger than a few KB should be stored in [`git-lfs`](https://git-lfs.com). Hence, you need to install and enable it before cloning this repo. Otherwise, files stored in LFS will appear as empty, except for a hash.

You can add file types to be stored in LFS by editing `.gitattributes`.

## Datasets

The datasets are extracted from various sources, each described in its own
`README.md`. Data is stored as ASE extxyz.

- [`Ag_clusters`](./Ag_clusters): Charged Ag₃ clusters isolating long-range charge-transfer effects.
- [`beastdb`](./beastdb): Electrocatalyst surfaces at fixed potential and fixed charge.
- [`omol_10K`](./omol_10K): Small, closed-shell, charge-diverse subset of OMol25.
- [`razor`](./razor): Pt-water interfaces with fixed charge labels.
- [`water_external_field`](./water_external_field): Water clusters under randomly oriented external electric fields.
- [`water_slab`](./water_slab): Periodic water slab polarized by a z-axis field.
- [`water_variational_charge`](./water_variational_charge): Water clusters at neutral, protonated, deprotonated charge states.
