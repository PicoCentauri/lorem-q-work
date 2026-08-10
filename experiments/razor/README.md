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

`sr-wf/` uses `lorem.LoremQ`, the charge-aware subclass that exposes
`dE/dq` (and `bec_z`); `sr/` and `lr/` stay on plain `lorem.Lorem`, whose
`predict` returns energy and forces only. The architecture is identical --
`LoremQ` inherits `__call__` untouched -- so this is purely about which
outputs are meaningful, and checkpoints are weight-compatible either way.

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

### TODO: max-force screening (checked -- currently a no-op here)

No force-based exclusion is applied anywhere in this repo, and the dataset
README does not mention one. `atoms.info["max_force"]` is present on every
frame, so a "drop configs with max force > 20 eV/Å" rule is directly
checkable. Measured:

| file | frames | `max_force` median / p95 / max | >20 eV/Å | of which polarizable |
|---|---|---|---|---|
| `razor_train.xyz` | 16,170 | 3.44 / 5.02 / **92.3** | 21 | **2** |
| `razor_val.xyz` | 1,797 | 3.45 / 5.13 / **145.5** | 3 | **0** |
| `razor_test.xyz` | 260 | 3.72 / 4.84 / 5.14 | 0 | 0 |

**The `polarizable` filter already removes almost all of them.** Of the 21
high-force training frames only 2 survive into the 10,931-frame training
pool -- 0.018% -- and none survive into the validation pool. So adding the
cutoff would change these results not at all, and the two criteria are
evidently correlated: frames near dielectric breakdown are also the ones with
runaway forces.

Two things to note if it is added anyway:

- The 2 surviving frames sit in a single `struc_pk`. Dropping frames
  individually would break that geometry's 3-charge stencil, which the
  splits are explicitly built to keep together, so the exclusion should be
  applied per `struc_pk` (3 frames) rather than per frame.
- The value is worth having as a guard for *future* data rather than for
  this dataset -- a 145 eV/Å frame is a broken configuration, and relying on
  `polarizable` to catch it is incidental rather than by design.

## Loss-weight sweep

`sr-wf/`'s work-function weight of 0.15 puts the work function at **52% of the
entire loss** and cost ~1.9x on energy and ~1.7x on forces against `sr/`. Two
runs look for a better point.

This folder is the right place for the sweep rather than `../razor_centre/`,
for two reasons. Its 200 epochs already *is* the converged,
gradient-update-matched budget (10,931 frames -> 145,747 updates), so `sr/`
and `sr-wf/` are usable endpoints as they stand and the curve costs two jobs
rather than three. And the weight-to-share arithmetic is self-consistent here:
`razor_centre`'s training pool has 32% less work-function variance (1.147 vs
1.688) because it carries one charge per structure, so the same nominal
weight would mean a different share there.

Shares from `razor_train`'s own polarizable pool (E 0.00140, F 0.47330,
W 1.68757):

| variant | E : F : W | E% | F% | W% |
|---|---|---|---|---|
| `sr` | 0.5 : 0.5 : 0 | 0.3 | 99.7 | 0 |
| `sr-wf0.05` | 0.5 : 0.5 : 0.05 | 0.2 | 73.6 | **26.2** |
| `sr-e100-wf0.1` | 100 : 1 : 0.1 | **17.9** | 60.5 | 21.6 |
| `sr-wf` | 0.5 : 0.5 : 0.15 | 0.1 | 48.2 | **51.6** |

Two questions, one run each:

- **`sr` -> `sr-wf0.05` -> `sr-wf`** is the work-function weight on its own at
  fixed energy:force -- a 0% / 26% / 52% curve with an exact control at each
  end.
- **`sr-wf0.05` -> `sr-e100-wf0.1`** holds the work-function share roughly
  fixed (26% -> 22%) and moves energy from 0.2% to 17.9%, isolating the
  energy axis. The 0.1 rather than 0.05 is deliberate: it is what keeps the
  work-function sides comparable.

Testing the energy axis at all is motivated by an asymmetry: **forces
constrain `∂E/∂r` and say nothing about `∂E/∂q`**. In a charge-conditioned
model the energy term is the only thing besides the work function itself that
constrains the charge direction, so it may matter more here than in a plain
MLIP -- where energy at 0.2% of the loss still reaches R² 99.9% and is
evidently almost free.

Both use `lorem.LoremQ`.

## Model-size comparison at the sweep's weights

Four runs sharing **identical** `settings.yaml` -- `100 : 1 : 0.05`, all
training on energy, forces and the work function, differing only in
`max_epochs` for the last -- so the *only* things that vary are the model and
the schedule length. `sr-e100-wf0.05/` is the full-size member and therefore
the exact control for the smaller ones.

| variant | $d$ | $l_{\max}$ | $c$ | CG paths | CG ops | params |
|---|---|---|---|---|---|---|
| `sr-e100-wf0.05` | 128 | 6 | 4 | 175 | 393k | 1.16 M |
| `sr-small-l2-e100-wf0.05` | 64 | 2 | **16** | 15 | 9.8k (0.03x) | 300k (0.26x) |
| `sr-small-l3-e100-wf0.05` | 64 | 3 | 8 | 34 | 27k (0.07x) | 295k (0.25x) |
| `sr-small-l2-e100-wf0.05-300ep` | 64 | 2 | **16** | 15 | 9.8k (0.03x) | 300k (0.26x) |

("CG ops" is $\sum_{l_1l_2l_3}(2l_1{+}1)(2l_2{+}1)(2l_3{+}1)\times c$ over
valid paths -- the $m_1,m_2$ contraction that dominates the forward pass.)

The three knobs hit different costs, which is what makes the trade work:

- **$l_{\max}$ 6 -> 2 collapses the CG cost ~40x** (175 valid paths -> 15).
  This is the real speedup.
- **$d$ 128 -> 64 cuts parameters ~4x**, since `Update` and
  `RadialCoefficients` go as $d^2$.
- **$c$ 4 -> 16 costs almost nothing** -- $c$ is linear in the CG kernel, and
  it is spent in the place that just got 40x cheaper.

**Why `l2` and `l3` are a pair.** Their parameter counts are within 2% of each
other, so if the small model loses accuracy, running only one would leave
"less angular resolution" confounded with "smaller everything".
$l_{\max}=2$ is the more aggressive cut, though not unusual -- LOREM's default
of 6 is high and NequIP/MACE typically run 1-3.

On the weights: the shares (E 20.1%, F 67.8%, W 12.1%) come from label
variances and so are model-independent -- but whether a model with 3% of the
CG work can still represent `dE/dq` as well is exactly what these measure.

### Results

Converged **validation** RMSE, last validation block of each run (E meV/atom,
F meV/Å, Φ V):

| variant | epochs | E | F | Φ | wall |
|---|---|---|---|---|---|
| `sr-e100-wf0.05` | 200 | 0.769 | **31.17** | 0.1121 | 13h46m |
| `sr-small-l2-e100-wf0.05` | 200 | 0.791 | 33.90 | 0.1034 | 5h42m |
| `sr-small-l3-e100-wf0.05` | 200 | 0.867 | 34.19 | 0.1217 | 5h18m |
| `sr-small-l2-e100-wf0.05-300ep` | 300 | **0.755** | 31.21 | **0.0958** | 8h32m |

**A 4x smaller model matches the full one on energy and forces, given enough
epochs.** At its own 300-epoch budget `l2c16` ties the control on forces
(31.21 vs 31.17, inside noise) and on energy, in 62% of the wall clock. Its
15% work-function win here does *not* hold up on the test sweep -- see
"the test sweep reverses the headline" below before quoting it.

**Read the two comparisons separately.** At *matched* 200 epochs the full
model wins forces by 8% (31.17 vs 33.90), so the small model is not simply
better -- the 300-epoch run also has 1.5x the gradient updates, the same
confound as `../razor_centre/`. What is fair: the small model reaches parity
for less total compute; whether the control would also improve at 300 epochs
is untested (~20h).

**200 epochs was learning-rate-limited, not capacity-limited.** `l2c16`'s
force error per 20-epoch window over its second half was −4.6, −5.0, −2.3,
−1.2, −0.26 %, tracking the cosine almost proportionally -- 120→140 halved
the LR and still gave the best window of that half, while the final window
bought nothing at lr 5e-8. Validation forces were still monotone at the end
(best 33.90 at epoch 194, which is also the last value). Extending to 300
epochs recovered −7.9% on forces, −4.5% on energy, −7.3% on Φ. **General
warning for this repo: with `warmup_cosine`, a flattening validation curve is
the schedule ending, not convergence.** The check that distinguishes them is
whether the per-window improvement falls *faster* than the LR.

Two caveats against reading further gains into more epochs. Over the same
span `l2c16`'s train loss fell 41% but validation only 28%, and valid/train
widened monotonically 1.24 → 1.52, so part of what remains is generalisation
rather than optimisation. And these per-window numbers describe the 200-epoch
schedule; the 300-epoch cosine is a different schedule, not a continuation.

**$l_{\max}$ helps forces and hurts the work function.** Across the three
converged 200-epoch models the control has the most angular resolution and
the *worst* Φ (0.1121 vs `l2c16`'s 0.1034) while having the best forces. Φ is
a global scalar response to charge, and plausibly benefits more from
scalar/channel capacity than from angular degrees, which the $c=16$ models
have in quantity.

**The $l_{\max}$ question itself is still confounded.** `l2c16` and `l3c8`
were matched on *parameter count*, not channels, so they differ in both
`max_degree` and `num_spherical_features`. "l=3 at c=8 bought nothing over
l=2 at c=16 at equal parameters" is supported; "l=2 is enough" is not.
An `sr-small-l3c16-e100-wf0.05/` run (l=3 at c=16, one variable vs `l2c16`)
was set up to break this and cancelled after ~2 epochs in favour of the
300-epoch run; the directory has since been deleted, but the config is
recoverable from commit `333458c` and the run costs ~7h. The cheaper way to
close it is `l2c8`, the fourth corner of the 2x2, at ~4.5h.

A cost note: **CG ops did not predict wall clock at this size.** `l3c8` does
2.8x the CG work of `l2c16` and finished *faster* (5h18m vs 5h42m). Fitting
fixed-plus-proportional to the three 200-epoch runs puts the fixed cost (data
pipeline, optimiser, validation) near 5h, with CG only dominating up at the
control's 393k ops. Scale small-model runtimes off that fit, not off CG ops.

### `evaluate.py`: the test sweep reverses the headline

Validation reproduces the training logs exactly, so the interesting part is
`test_sweep`. Quoting the polarizable subset, the honest extrapolation check:

| variant | E | F | Φ | `bec_z` |
|---|---|---|---|---|
| `sr-e100-wf0.05` | 1.40 | **33.64** | **0.3115** | 0.0399 |
| `sr-small-l2-e100-wf0.05` | **1.26** | 37.77 | 0.3320 | 0.0368 |
| `sr-small-l3-e100-wf0.05` | 1.56 | 38.70 | 0.3359 | 0.0414 |
| `sr-small-l2-e100-wf0.05-300ep` | 1.39 | 34.64 | *0.3621* | **0.0364** |

(n=34. No variant trains on `bec_z`; `evaluate.py` computes it via its own
`jvp`.)

**The 300-epoch run's work-function win does not survive.** It is the *best*
Φ model on validation (0.0955) and the *worst* here (0.3621 against the
control's 0.3115) -- the ordering inverts. That is the generalisation gap seen
during training (valid/train 1.24 → 1.52) showing up as a real cost outside
the ±0.25 e stencil. **The extra 100 epochs bought in-distribution Φ by
overfitting the charge direction.**

So the honest summary is narrower than the validation table alone suggests:

- **Energy and forces: the small model genuinely reaches parity** (F 34.64 vs
  33.64, E 1.39 vs 1.40), consistent with validation, at 62% of the wall clock.
- **Work function: the full model is better where it matters** -- it wins on
  the sweep despite losing on validation, and longer training makes the small
  model worse.
- `bec_z` separates almost nothing (0.0364-0.0414), as expected since nothing
  supervises it.

Two smaller observations. `l3c8` is much the best model on the
*non-polarizable* frames (Φ 0.844 vs 1.11-1.49, E 7.00 vs 9.46-12.43), the
regime no variant trains on and where the labels are least trustworthy; worth
noting, not worth weighting. And from the weight-sweep round, `lr/` reached
**Φ 0.209 and `bec_z` 0.0270** on these same 34 frames -- better than every
model in this section. The Ewald head remains the strongest lever on the
charge response, which is what makes the deferred `lr-e100-wf0.05/` the most
valuable run still outstanding.

### RMSE resolved by charge, and one frame that distorts everything

`figures/rmse_vs_charge_{valid,test_sweep}.pdf` bin the same rows by
`total_charge` (0.25 e bins) instead of pooling them -- a grid of variants x
targets, keeping the polarizable split, with bins under 20 structures hatched.
Parity plots say whether a model is biased; these say *where in charge space*
the error lives, which is the question the ±0.25 e stencil raises. Use the
`valid` figure: `test_sweep` has only 1-5 polarizable frames per charge, so
nearly every bar there is hatched and it is not readable.

First, error is **U-shaped in q** for all four models -- lowest around
q ∈ [−0.75, 0], rising toward both extremes and more steeply on the positive
side for Φ (0.13-0.17 V above q = +0.75 against ~0.09 mid-range). That tracks
training density, and means the pooled RMSE understates the error at the
charges a potentiostat sweep would actually visit.

Second, and more actionable: **a single frame accounts for the whole
force-error spike at q = −1.25, and for 10% of the reported validation force
RMSE.** It is the worst frame for every variant -- `struc_pk=301216`,
`q = −1.25`, `max_force = 12.66 eV/Å`:

| | control | l2c16 200ep | l3c8 | l2c16 300ep |
|---|---|---|---|---|
| that frame's F RMSE | 481 | 445 | 371 | 474 |

The next-worst frame in its bin is 41 meV/Å. Dropping it takes the bin from
79.8 to 27.0 meV/Å, in line with its neighbours (23.9), and takes the **whole
split** from **31.17 to 27.97 meV/Å**.

- Every force number in this README is inflated ~10% by one frame. It hits
  all four variants nearly equally, so the *comparisons* stand, but the
  absolute values are pessimistic.
- **This revises the max-force screening TODO above.** That section measured
  a >20 eV/Å rule and correctly found it a no-op. At 12.66 eV/Å this frame is
  under that threshold and still pathological -- the threshold was wrong, not
  the idea. A cut near 10 eV/Å would catch it. It is in *validation*, so it
  never corrupted training; it corrupts the metric.

## Layout

- `prepare.py` -- builds `data/` from `../../datasets/razor/*.xyz`. Run
  locally and synced to the cluster, not re-run per job.
- One experiment dir per variant, each `model.yaml` + `settings.yaml` +
  `srun.sh`:
  - weight sweep: `sr/`, `lr/`, `sr-wf/`, `sr-wf0.05/`, `sr-e100-wf0.1/`
  - size comparison at 100:1:0.05: `sr-e100-wf0.05/`,
    `sr-small-l2-e100-wf0.05/`, `sr-small-l3-e100-wf0.05/`,
    `sr-small-l2-e100-wf0.05-300ep/`
  - set up but not run: `lr-e100-wf0.05/` (deferred; configs only, no `run/`)

Directories for runs that were cancelled or superseded are deleted rather
than left lying around -- a stale `run/` looks like a trained model to
`evaluate.py`, and `lorem-train` will happily restore from a partial one.
Configs stay recoverable from git history.
- `evaluate.py` -- two figures per split: `parity_<split>.pdf` (energy /
  force / work function / `bec_z` parity with RMSE) and
  `rmse_vs_charge_<split>.pdf` (the same RMSEs binned by `total_charge`).
  `VARIANTS` at the top selects which runs appear; it holds the four
  size-comparison runs, since mixing in the weight-sweep runs would put four
  different loss weights in one figure.
- `srun.sh` (top level) -- SLURM job that runs `evaluate.py`. Uses
  `~/venv/lorem-wf`: every variant it evaluates was trained as `lorem.LoremQ`
  and `load_checkpoint` deserialises that class name out of the checkpoint's
  own `model.yaml`, so `lorem313` fails on the first `from_dict`.

Two traps in the evaluate workflow, both hit once:

- `evaluate.py` caches to `evaluate_rows.json` and **loads that cache in
  preference to re-evaluating**. Move it aside after retraining, or the
  figures will silently be the previous round's. Re-plotting off an existing
  cache needs no GPU and runs locally in seconds -- `scp` the cache down and
  run `DATASETS=. python evaluate.py` rather than queueing a job.
- `make pull` syncs the *whole* experiment tree remote→local and will
  overwrite local edits to tracked files (it silently reverted this README
  once). Commit before pulling, or `scp` the specific files you want.

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

1. ~~**Work-function weight sweep**~~ -- **done**; the outcome is
   `100 : 1 : 0.05`, which is what every run in the model-size section above
   uses. The follow-on size/length comparison is also done.
2. **`lr-wf`, and an `lmax_lr` ablation** -- now the highest-value run
   outstanding. `lr/` still has the best charge response of anything in this
   folder (Φ 0.209, `bec_z` 0.0270 on the test sweep's polarizable frames),
   beating every model in the size comparison, and it does so with
   `max_degree_lr: 0`. `lr-e100-wf0.05/` is set up and deferred, not run.
   Note also that the equivariant Ewald features currently update only the
   scalar features `P` -- `nodes_spherical` is never reassigned in the
   long-range block, so the potentials enter as invariants via `Norm` and
   `S` is read-only there. At `max_degree_lr: 0` that path degenerates to a
   per-channel scalar rescaling of `S`, so raising `lmax_lr` is what switches
   the equivariant long-range machinery on at all. On validation `lr` also
   reaches 0.181 V on the work function without ever training on it. Both the
   long-range counterpart of `sr-wf/` and raising `lmax_lr` to the paper's
   default of 2 look more promising than pushing the work-function weight, or
   than shrinking the model further.
3. **A ~10 eV/Å max-force screen** -- see "one frame that distorts
   everything" above. A single validation frame at 12.66 eV/Å costs 10% of
   the reported force RMSE across every variant, and the >20 eV/Å rule
   measured in the TODO section misses it entirely. Cheap to apply, and it
   should be applied per `struc_pk` so the 3-charge stencil stays intact.
4. **`bec_z` on the full stencil** -- `../razor_centre/sr-wf-bec/` shows it
   reaches 0.037 e on polarizable frames and helps `dE/dq` rather than
   competing with it. Worth repeating here, where the extra off-stencil data
   should make the derivative targets easier. Note this folder's
   `evaluate.py` does not score `bec_z`; `../razor_centre/evaluate.py` does,
   and that code can be lifted across.
