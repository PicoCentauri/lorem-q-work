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

All three variants trained to the full 200 epochs. Converged **validation**
RMSE (last validation block, epoch ~198; the test-sweep numbers come from
`evaluate.py`):

| variant | energy | forces | work function | runtime |
|---|---|---|---|---|
| `sr` | 0.855 meV/atom | 23.5 meV/Å | -- | 13h50m |
| `lr` | 0.777 meV/atom | **20.3 meV/Å** | -- | 14h15m |
| `sr-wf` | 1.606 meV/atom | 40.6 meV/Å | **0.121 V** | 13h51m |

R²: `sr` 99.93/99.89, `lr` 99.95/99.92, `sr-wf` 99.77/99.68/99.20.

**The Ewald head helps forces.** `lr` is the best force model at 20.3 meV/Å,
14% better than `sr` -- notable given `max_degree_lr: 0` means monopole-only,
i.e. the equivariant long-range messages that are Lorem's headline
contribution are switched off. Worth an `lmax_lr` ablation.

**Training on the work function costs accuracy at weight 0.15.** `sr-wf` is
~1.9x worse on energy and ~1.7x worse on forces than `sr`, and buys `dE/dq`
at 0.121 V against a 1.36 V label spread. The same trade reproduces on
`../razor_centre/`, so this is the weight, not a quirk of one dataset --
0.15 was picked to match the force term's *initial* contribution, which was
too aggressive. A sweep at 0.05 / 0.02 is the obvious follow-up.

### Work function: trained vs not (`evaluate.py`, valid split, n=1218)

`evaluate.py` computes `dE/dq` with its own `jax.grad` for every variant,
whether or not it was a training target, so this is a like-for-like probe:

| variant | trained on WF? | WF RMSE | energy | forces |
|---|---|---|---|---|
| `sr` | no | 0.236 V | 0.82 | 23.6 |
| `lr` | no | **0.181 V** | 0.76 | 20.3 |
| `sr-wf` | yes | **0.121 V** | 1.61 | 40.6 |

Charge conditioning alone already gets most of the way: against a 1.36 V
label spread, models that never saw the work function reach 0.236 V (`sr`)
and 0.181 V (`lr`). Training on it improves that by 1.9x over `sr` but only
1.5x over `lr` -- and `lr` gets there while also being the best energy and
force model. **If the work function is what you want, the Ewald head is a
better lever than the explicit target at this weight.**

On the test sweep, restricted to polarizable frames (n=34, the honest
extrapolation check), `sr-wf` is the *worst* of the three at 0.425 V against
`lr`'s 0.209 and `sr`'s 0.275: training on the target inside the +-0.25 e
stencil appears to hurt generalisation outside it. `lr` meanwhile collapses
on the non-polarizable frames (3.35 V, worst of all) while being best on
polarizable ones -- the Ewald head extrapolates well inside the
linear-response window and badly outside it.

### Born effective charges, none of them trained on it

`evaluate.py` computes `Z* = -(A ε₀) ∂²E/∂r∂q` itself, so `bec_z` is scored
here even though no variant in this folder supervises it. `razor_val.xyz`
carries no label, so this is the test sweep. RMSE in e, label spread 0.145 e:

| variant | polarizable (34) | non-pol. (226) | all (260) |
|---|---|---|---|
| `sr` | 0.0399 | 0.0466 | 0.0458 |
| `lr` | **0.0270** | 0.0744 | 0.0701 |
| `sr-wf` | 0.0462 | 0.0645 | 0.0624 |

**The Ewald head gives the best Born effective charges of any model here, at
0.0270 e, without ever training on them** -- 1.5x better than `sr`. It also
gives the best work function on the same frames (0.209 V). Both are second
derivatives of `E(r, q)`, so a long-range term that improves the charge
response evidently improves its position derivative too.

For context, `../razor_centre/sr-wf-bec/` -- the only variant anywhere that
*does* supervise `bec_z` -- reaches 0.0373 e, worse than `lr` here. That
comparison is confounded (different training set, 2.25x fewer updates), so it
is not evidence that supervision is useless; within its own folder it is 1.7x
better than the unsupervised `sr`. But it does mean **the cheapest route to
good Born effective charges in this project so far is the Ewald head, not the
explicit target.**

**Watch the warmup transient, not the mid-training numbers.** `sr-wf`'s force
R² was 1.5% at epoch 2 and 98.8% by epoch 34; its energy looked 5x *better*
than `sr` at epoch 34 and ended up 1.9x worse. Single mid-training validation
snapshots on this dataset are close to meaningless.

## Next experiments

**Done since this list was written:** `experiments/razor_centre/` exists and
its `sr`, `sr-wf` and `sr-wf-bec` variants have all trained -- see that
folder's README for the cross-folder comparison. The headline is that
centre-only training is *worse* across the board, so the off-stencil frames
earn their place.

1. **Work-function weight sweep** -- the clearest open question. 0.15 costs
   ~1.9x on energy and ~1.7x on forces here, and was chosen to match the force
   term's *initial* contribution, which turned out too aggressive. Try 0.05
   and 0.02 and find where `dE/dq` is still learned but E/F recover. Cheap on
   `../razor_centre/` at 6h a run.
2. **`lr-wf`, and an `lmax_lr` ablation** -- `lr` is the best model here on
   all three metrics *and* reaches 0.181 V on the work function without ever
   training on it, all with `max_degree_lr: 0` (monopole-only). Both the
   long-range counterpart of `sr-wf/` and simply raising `lmax_lr` to the
   paper's default of 2 look more promising than pushing the work-function
   weight.
3. **`bec_z` on the full stencil** -- `../razor_centre/sr-wf-bec/` shows it
   reaches 0.037 e on polarizable frames and helps `dE/dq` rather than
   competing with it. Worth repeating here, where the extra off-stencil data
   should make the derivative targets easier. Note this folder's
   `evaluate.py` does not score `bec_z`; `../razor_centre/evaluate.py` does,
   and that code can be lifted across.
