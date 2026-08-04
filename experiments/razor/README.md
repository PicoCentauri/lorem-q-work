# razor charge conditioning

Trains `Lorem` on `../../datasets/razor/` (Pt(111)/water interfaces, 36 Pt +
24 H₂O = 108 atoms, `pbc="T T F"`) with energy + force targets and total-charge
conditioning. The dataset's `bias_charge` -- the DFT bias charge each frame was
evaluated at -- is renamed `total_charge` and enters the model as an input,
FiLM-conditioning the invariant node features (`ChargeEmbedding` in
`lorem/models/backbone.py`); `Lorem` always applies it, there is no
`charge_conditioning` switch any more.

Two variants, same axis as the other experiments here: `sr` (`lr: false`) vs
`lr` (`lr: true`, Ewald long-range head). `max_degree_lr: 0` keeps the
long-range head to monopoles only.

## Naming convention

Sub-experiment directories are `<range>[-<extra targets>]`:

- `<range>` is `sr` or `lr`, the `lr` model flag.
- `<extra targets>` is omitted when the run trains on energy + forces only.
  Planned suffixes: `-wf` (adds the work function as a target), `-wf-bec`
  (adds the Born-effective-charge response as well).

Anything that changes *which frames are trained on* gets its own top-level
experiment folder rather than a suffix here, because it needs its own
`prepare.py` and `data/`. The first planned one is `experiments/razor_centre/`
(see "Next experiments"). That folder reuses the same `<range>[-<extra
targets>]` sub-experiment names, so results stay directly comparable
row-by-row across folders.

## Data

`prepare.py` builds four splits from the shipped extxyz files:

| split | source | frames | note |
|---|---|---|---|
| `data/train` | `razor_train.xyz` | 10,931 | polarizable only, of 16,170 |
| `data/valid` | `razor_val.xyz` | 1,218 | polarizable only, of 1,797 |
| `data/test` | `razor_test.xyz` | 34 | polarizable only, of 260 |
| `data/test_sweep` | `razor_test.xyz` | 260 | unfiltered |

Non-polarizable frames (`polarizable=False`, ~32% of train/val) sit outside
the linear-response window, near dielectric breakdown / Fermi pinning, and
are dropped from training. They are only 13% of `razor_test.xyz` though --
filtering that file leaves 34 frames, too few to serve as the wide-charge-range
extrapolation check the test set exists for. Hence `data/test_sweep`, which
keeps all 260 frames (the full 13-point `q ∈ [−1.5, 1.5] e` sweep over 20
structures); both are reported at train time and `evaluate.py` splits its
metrics by the flag.

`work_function` (the DFT `∂E/∂q`) is persisted into `data/` alongside
`total_charge` even though nothing trains on it in this round -- `evaluate.py`
compares it against the model's autograd `dE/dq`, and the planned `-wf`
variants then need no data rebuild.

### Caveat: training frames are labelled off their own MD charge

This is the main known weakness of this first round. Each geometry was sampled
from an MD run at some bias charge `q_MD`, then re-evaluated at a 3-point
stencil `q_MD ± 0.25 e`. So two thirds of the training frames pair a geometry
with a charge that geometry never equilibrated at:

| file | frames | structures (`struc_pk`) | frames with `bias_charge == q_MD` |
|---|---|---|---|
| `razor_train.xyz` | 16,170 | 5,390 | 5,390 |
| `razor_val.xyz` | 1,797 | 599 | 599 |
| `razor_test.xyz` | 260 | 20 | 20 |

That off-stencil labelling is exactly what makes `∂E/∂q` learnable, so it is
not a mistake -- but it does mean the training distribution is not the
physical (geometry, charge) distribution an MD run would visit, and energy /
force errors reported here mix both regimes. `razor_centre.xyz` holds the
one-frame-per-structure centre of each stencil (`bias_charge == q_MD`) and is
the basis for the follow-up experiment below.

Splits are by `struc_pk`, never within one -- the 3 charges of a geometry stay
together. That split ships with the dataset; `prepare.py` does not re-split.

## Layout

- `prepare.py` -- builds `data/` from `../../datasets/razor/*.xyz`. Run
  locally and synced to the cluster, not re-run per job.
- `sr/`, `lr/`, `sr-wf/` -- one experiment dir per variant: `model.yaml` +
  `settings.yaml` + `srun.sh`.
- `evaluate.py` -- energy / force / work-function parity plots with RMSE, on
  `valid` and `test_sweep`, for both variants.
- `srun.sh` (top level) -- separate SLURM job that just runs `evaluate.py`;
  submit once both training jobs have finished.

## Settings

`model.yaml` follows the recipe that currently performs best in
`../water_external_field/lr-l1/` -- `num_features=128`, `max_degree=6`,
`num_radial=16`, `num_message_passing=1`, `num_spherical_features=4`,
`initialize_node_features: True`, `cutoff=5.0` -- with `max_degree_lr: 0`.

`settings.yaml` does *not*, since that experiment still runs the older adam /
linear-decay recipe. It uses the current house settings, shared verbatim with
`../ag_clusters/{sr,lr}`, `../water_variational_charge/`, `../beastdb/` and
`../omol_10K/`: muon, `loss_weights={"energy": 0.5, "forces": 0.5}`,
`gradient_clip: 1.0`, and a `warmup_cosine` schedule -- warming 1e-6 -> 2e-4
over the first 10 epochs, then cosine-decaying back to 1e-6 by `max_epochs`.

`batch_size: 16` in the batcher. `ToBatch` packs `batch_size - 1` real
structures and pads the last slot, so that is 15 × 108 = **1,620 atoms per
batch**, and ~728 batches per epoch on `data/train`.

`max_epochs: 200` is a wall-clock-driven guess (24 h on one A100-80), and the
one place this experiment departs from the runs listed above -- they use 1000.
It doubles as the cosine schedule's `decay_steps`, so a job that hits the time
limit stops with the LR only partly decayed -- if that happens, lower
`max_epochs` rather than just resubmitting.

## Work-function supervision (`sr-wf/`)

`sr-wf/` is `sr/` plus `work_function` as a third training target, i.e. the
first variant that supervises `dE/dq` rather than only reading it out
post-hoc. Model config is byte-identical to `sr/`; only `loss_weights` differs.

`Lorem.predict` already took `jax.value_and_grad` of the energy with respect to
the *whole batch*, so `dE/dq` is `grads.total_charge` -- it falls out of the
same backward pass that produces the forces, at no extra cost. Structures in a
batch don't interact, so the derivative of the summed energy w.r.t. the
per-structure charge vector is each structure's own `dE/dq`. The per-species
baseline is charge-independent and drops out, so no offset correction applies
(unlike for the energy). This lives on the `charge-conditioning` branch of
`lorem-jax`, hence `sr-wf/srun.sh` uses `~/venv/lorem-wf` rather than
`~/venv/lorem313`.

**Loss weights.** `energy` and `forces` stay at the house 0.5/0.5, which makes
`sr/` an exact control -- the new key is inert when it isn't in `loss_weights`
(no label, so no residual and no loss term; there's a test for this). The
work-function weight is set from validation-set variances so the new term
enters at the same magnitude as the force term at initialisation:

| target | std | var | contribution |
|---|---|---|---|
| energy | 0.033 eV/atom | 0.0011 | 0.5 × 0.0011 = 0.0006 |
| forces | 0.722 eV/Å | 0.5214 | 0.5 × 0.5214 = 0.261 |
| work function | 1.361 V | 1.8523 | **0.15** × 1.8523 = 0.278 |

At the house weight of 0.5 the work function would have contributed 0.926 --
3.6x the force term and ~1700x the energy term -- and would have dominated
training. Note `work_function` is intensive (V), so unlike `energy` it is *not*
normalised per atom: `DEFAULT_NORMALIZATION` covers only `energy` and `stress`.

**Caveat, worth watching in the first epochs.** Those weights balance the
*asymptotic* residuals, i.e. what a model that has learned the mean would see.
At initialisation the untrained `dE/dq` is not the mean -- it came out around
-2 to -18 V against labels of +3 to +7 V in a 3-frame probe, so the work-function
term starts at ~99% of the total loss. Warmup (10 epochs) and `gradient_clip: 1.0`
should absorb that, but if the learning curves show energy/force errors stalling
while the work function drops, lower the 0.15 rather than assuming it converged.

## Running

```bash
# prepare data locally (shared by both variants), then sync to the cluster
DATASETS=. python prepare.py

# on the cluster: submit both training jobs (run in parallel)
cd sr && sbatch srun.sh && cd ../lr && sbatch srun.sh

# once both have finished, evaluate
cd .. && sbatch srun.sh
```

## What `evaluate.py` reports

Per variant and per split (`valid`, `test_sweep`), and on `test_sweep` also
split by the `polarizable` flag:

- **energy** parity, meV/atom, RMSE + MAE
- **force** parity, meV/Å, over all Cartesian components
- **work function** parity, V. The model's `dE/dq` comes from backpropagating
  the forward pass with respect to the `total_charge` input -- one backward
  pass per batch gives every structure's value, since structures in a batch
  don't interact. The per-species energy baseline is charge-independent and
  drops out of the derivative, so no offset correction is applied there
  (unlike the energy parity). Verified against a central finite difference
  in `q`.

Nothing in this round trains on the work function, so that panel is a clean
test of whether charge conditioning learned the right charge *derivative*
from energies and forces alone. The dataset's own sanity check applies:
`∂²E/∂q²` should be near 9 V/e and roughly structure-independent inside the
polarizable window.

## Results so far

_TBD -- not yet trained._

## Next experiments

1. **`experiments/razor_centre/`** -- train only on centre frames
   (`bias_charge == q_MD`), i.e. a split of `razor_centre.xyz`, removing the
   off-equilibrium (geometry, charge) pairs described above. Roughly 5,390 +
   599 structures, so ~3x less data than here; the comparison against this
   folder's `sr`/`lr` is the point. Note `razor_centre.xyz` overlaps
   `razor_train.xyz`/`razor_val.xyz` on the same (r, q) points (it just carries
   extra columns), so it must not be added as extra data on top of this
   experiment -- it replaces it.
2. **`lr-wf`** -- the long-range counterpart of `sr-wf/` (see "Work-function
   supervision" above), once `sr-wf` shows whether the extra target helps.
3. **`sr-wf-bec` / `lr-wf-bec`** -- additionally supervise `bec_z` (`∂F/∂q`).
   Use with caution: per the dataset's source documentation it is a
   finite-difference estimate damped by ≥15%, so mask on `polarizable`, weight
   it low, and validate against `razor_test.xyz`, whose `bec_z` comes from a
   13-point spline. See `lorem/models/bec.py`.
