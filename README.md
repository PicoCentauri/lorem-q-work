# lorem-q

Datasets and experiments for testing and benchmarking models that are aware
of total charge and an external electric field.

## Layout

- `datasets/` -- the reference datasets
- `experiments/` -- training runs against those datasets

## How to use

Anything that's bigger than a few KB should be stored in [`git-lfs`](https://git-lfs.com). Hence, you need to install and enable it before cloning this repo. Otherwise, files stored in LFS will appear as empty, except for a hash.

You can add file types to be stored in LFS by editing `.gitattributes`.

## Datasets

- [`ag_clusters`](./datasets/ag_clusters): Charged Ag₃ clusters isolating long-range charge-transfer effects.
- [`beastdb`](./datasets/beastdb): Electrocatalyst surfaces at fixed potential and fixed charge.
- [`omol_10K`](./datasets/omol_10K): Small, closed-shell, charge-diverse subset of OMol25.
- [`razor`](./datasets/razor): Pt-water interfaces with fixed charge labels.
- [`water_external_field`](./datasets/water_external_field): Water clusters under randomly oriented external electric fields.
- [`water_slab`](./datasets/water_slab): Periodic water slab polarized by a z-axis field.
- [`water_variational_charge`](./datasets/water_variational_charge): Water clusters at neutral, protonated, deprotonated charge states.

## Experiments

- [`ag_clusters`](./experiments/ag_clusters): `Lorem` charge-conditioning ablation (`none`/`film`/`latent`) on the `ag_clusters` dataset.
- [`water_variational_charge`](./experiments/water_variational_charge): `Lorem` charge-conditioning (`film`/`latent`) crossed with long-range (`lr` on/off) on the `water_variational_charge` dataset.
- [`water_external_field`](./experiments/water_external_field): `Lorem` field-conditioning ablation (`none`/`l1`/`l1_l0`) crossed with long-range (`lr` on/off) on the `water_external_field` dataset.
- [`water_slab`](./experiments/water_slab): same field-conditioning x `lr` grid as `water_external_field`, on the periodic `water_slab` dataset.
- [`razor`](./experiments/razor): `Lorem` charge conditioning (`sr`/`lr`) on the `razor` Pt(111)/water dataset, with the work function read out as an autograd `dE/dq` rather than trained on.
