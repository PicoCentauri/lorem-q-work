# water_external_field field conditioning

Trains `Lorem` on `../../datasets/water_external_field/`, crossed with `lr`
in `{true, false}` and `field_conditioning` in `{none, l1, l1_l0}` -- 6
variants total. `field_conditioning` controls how the model consumes
`atoms.info["external_field"]` (a genuine l=1 vector, coupled into the
network's equivariant channels via a Clebsch-Gordan tensor product, unlike
`total_charge` which is an l=0 scalar consumed by a plain FiLM module):

- `none` -- field is read but never used (baseline; should behave
  identically to a model with no field mechanism at all).
- `l1` -- field couples into equivariant node features only.
- `l1_l0` -- `l1` plus an additional FiLM conditioning on `|field|` (a
  legitimate invariant scalar) into scalar node features.

This is the ablation: does the l=1 coupling alone let the model learn field
response, or does the model also need the explicit magnitude channel? And
does the long-range Ewald channel (`lr`) matter here the way it did for
`water_variational_charge`'s dissociation test?

**Not to be confused with:** `LoremBEC.predict()`'s separate `electric_field`
kwarg, which is a purely post-hoc `F_ext = apt . E` force correction applied
*after* the network has already run -- unrelated to this mechanism, which
flows through the forward pass and affects the energy itself. This
experiment only concerns `Lorem` (E/F), not `LoremBEC` (Born effective
charges).

## Layout

- `data/` -- prepared marathon datasets (`train`/`valid`/`test`), built by
  `prepare.py` from `../../datasets/water_external_field/*.xyz`. Renames
  `tot_charge` -> `total_charge` and `ext_field` -> `external_field`,
  matching the keys `lorem/batching.py` reads. Run locally and synced to the
  training machine, rather than re-run per job.
- `sr-none/`, `sr-l1/`, `sr-l1_l0/`, `lr-none/`, `lr-l1/`, `lr-l1_l0/` -- one
  experiment dir per (`lr`, `field_conditioning`) combination.
- `evaluate.py` -- Exp 1 (perturbed vs. field-free MAE on the held-out test
  set).
- `evaluate_paired.py` -- Exp 2 (matched field-free/field-on pairs at fixed
  geometry). Reads `water_external_field_paired_{free,perturbed}.xyz`
  directly, bypassing `prepare.py`/marathon entirely -- these files carry
  `dft_energy`/`dft_forces`/`dft_hirshfeld`, not the standard label keys, and
  have no `atoms.calc`.
- `evaluate_magnitude_sweep.py` -- Exp 3 (field-magnitude sweep per cluster,
  Fig. 5-style curves).

Unlike `water_variational_charge`, this run uses direct SSH + `nohup` on a
shared GPU machine rather than SLURM.

## Running

```bash
# prepare data locally (shared by all 6 variants), then sync
DATASETS=. python prepare.py

# launch all 6 variants (from each variant dir)
source ../../.venv/bin/activate  # or wherever the venv lives on the training machine
cd sr-none && DATASETS=.. lorem-train

# once all variants have finished, evaluate
python evaluate.py
python evaluate_paired.py
python evaluate_magnitude_sweep.py
```

## Physical test cases

From `../../datasets/water_external_field/README.md`:

**Exp 1 -- perturbed vs. field-free MAE.** Split the test set into
field-free (`external_field == 0`) and perturbed groups, compare
energy/force MAE/RMSE between them, per variant.

**Exp 2 -- matched field-free/field-on pairs.** Same geometry, field on vs.
off; compare each model's predicted Delta E against the DFT Delta E,
per variant.

**Exp 3 -- field-magnitude sweep.** 50 clusters, each swept through 21 field
magnitudes at fixed geometry. Group by `series`, sort by `field_magnitude`,
plot E(field) per cluster relative to its own field_magnitude == 0 point.
QEq predicts an exact parabola; the paper finds QEq-based MLIPs get the
shape wrong -- this is the direct visual check of whether LOREM's field
conditioning does better.

## Results

_TBD -- filled in after training and evaluation._
