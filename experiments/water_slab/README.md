# water_slab field conditioning

Trains `Lorem` on `../../datasets/water_slab/`, crossed with `lr` in
`{true, false}` and `field_conditioning` in `{none, l1, l1_l0}` -- 6 variants
total, same axes as `../water_external_field/`. This is the periodic,
condensed-phase counterpart of that experiment: a slab of 32 water
molecules under a range of `pbc="T T T"` fields applied along the slab
normal (z), full 3D Ewald electrostatics with a large vacuum gap along z.

Unlike `water_external_field`, this dataset gets **no dedicated physical
test-case scripts** -- just the same simple test-set MAE/RMSE bar-chart
pattern used across the other experiments, grouped by the 7 discrete
applied-field values (`0, +-0.1, +-0.2, +-0.3` V/A). The dataset's own
qualitative charge-vs-depth profile-shape check (does polarization stay
near the surfaces vs. spread linearly through the bulk, per the source
paper's Supplementary Note 3) needs `LoremBEC`-specific per-atom charge
output, which is out of scope for this pass -- a future follow-up, not
blocking.

**Not to be confused with:** `LoremBEC.predict()`'s separate `electric_field`
kwarg, a purely post-hoc `F_ext = apt . E` force correction applied *after*
the network has already run -- unrelated to this mechanism, which flows
through the forward pass and affects the energy itself. This experiment only
concerns `Lorem` (E/F), not `LoremBEC`.

## Layout

- `data/` -- prepared marathon datasets (`train`/`valid`/`test`), built by
  `prepare.py` from `../../datasets/water_slab/*.xyz`. Renames `tot_charge`
  -> `total_charge` and `ext_field` -> `external_field`. Run locally and
  synced to the training machine, rather than re-run per job.
- `sr-none/`, `sr-l1/`, `sr-l1_l0/`, `lr-none/`, `lr-l1/`, `lr-l1_l0/` -- one
  experiment dir per (`lr`, `field_conditioning`) combination.
- `evaluate.py` -- simple test-set E/F MAE/RMSE, grouped by applied field
  value; bar plot per variant.

This run uses direct SSH + `nohup` on a shared GPU machine rather than
SLURM.

## Running

```bash
# prepare data locally (shared by all 6 variants), then sync
DATASETS=. python prepare.py

# launch all 6 variants (from each variant dir)
source ../../.venv/bin/activate  # or wherever the venv lives on the training machine
cd sr-none && DATASETS=.. lorem-train

# once all variants have finished, evaluate
python evaluate.py
```

## Results

_TBD -- filled in after training and evaluation._
