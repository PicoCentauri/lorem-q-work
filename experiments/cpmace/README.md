# cpmace charge conditioning

Trains the `d64 l2 c8` sr/lr pair on `../../datasets/cpmace/` — constant-potential
VASPsol++ on a 207-atom Ni complex in water, `C70 H89 N Ni O46`, fully periodic.
Same architecture as the last `../razor/` pair, so results read across.

| dir | model | loss weights |
|---|---|---|
| `sr-small-l2c8-1000ep/` | d64 l2 c8, `lr: false` | 1000 : 1 : 5 |
| `lr-small-l2c8-1000ep/` | same, `lr: true` (`max_degree_lr: 0`) | 1000 : 1 : 5 |
| `sr-small-l2c8-cpmace-weights-1000ep/` | d64 l2 c8, `lr: false` | **1 : 100 : 10** (cp-MACE's) |

All train on energy + forces + work function. The first two share a
byte-identical `settings.yaml` and differ only on the `lr` line of
`model.yaml`, so that pair isolates the Ewald head exactly as in `../razor/`.
The third differs from the first *only* in `loss_weights`, so it isolates the
weighting.

### The cp-MACE-weights run

`sr-small-l2c8-cpmace-weights-1000ep/` uses the weights from the cp-MACE
paper's SI verbatim — "100.0 for atomic forces, 1.0 for energy, and 10.0 for
the Fermi level predictions" — on what is essentially the same system and
sampling, so it doubles as a direct comparison against their published
**16.5 meV/Å forces** and **0.03–0.04 eV Fermi level**.

On this dataset's variances those weights give **E 0.0001% / F 99.68% /
W 0.32%**: energy effectively untrained, work function a rounding error. That
is not obviously wrong — they reached 0.03 eV on E_F at that share — but it
may not transfer, because **their Fermi level is a direct output head and ours
is `dE/dq` from autograd**. The SI compares "node augmentation" against
"global feature", both ways of *injecting* the electron number with E_F
predicted as an output. A target that is cheap to fit as an independent head
is not necessarily cheap to fit as a derivative that must reshape E(q), and
`../razor/`'s sweep found a large work-function share costs ~1.7× on forces
precisely because of that coupling.

Which way it goes is the question the run answers.

## The fields are not a rename of razor's

`potential`/`electron` map onto razor's `work_function`/`total_charge`, but
**not** by renaming. Both conversions are derived and checked in
`../../datasets/cpmace/README.md`; `prepare.py` applies them:

```
q             = -(electron - 660)
E_total       = energy + potential x (electron - 660)
work_function = -potential                    ( = dE_total/dq )
```

Two things to know before using this data:

**`energy` is the grand potential**, Ω = E − μ(N − N₀), not the total energy.
Regressing it on electron count while controlling for geometry gives
dΩ/dN = −1.58 eV against a reported potential of −3.36 eV; adding μ(N−N₀) back
recovers dE/dN = μ to within 0.3%. Training on the reported energy while
supervising `dE/dq = -potential` would set the two targets against each other.
Forces need no conversion (envelope theorem).

**660 is the VASP PAW valence sum** (H 1, C 4, N 5, O 6, Ni 10). It is an
assumption about the POTCARs, so it was checked: requiring thermodynamic
consistency alone implies N₀ = 660.20. Any residual error is a constant offset
in q, which the FiLM conditioning absorbs for a single-composition dataset.

## Loss weights are *not* razor's

`1000 : 1 : 5`, not `100 : 1 : 0.05`. Weights are set from label variances,
and this dataset's differ by one to two orders of magnitude:

| target | cpmace var | razor var |
|---|---|---|
| energy (per atom) | 5.80e−5 | 1.40e−3 |
| forces | 0.708 | 0.473 |
| work function | 2.27e−2 | 1.688 |

Razor's weights carried over unchanged would give **E 0.8% / F 99.0% / W 0.2%** —
energy and the work function effectively untrained. What transfers between
datasets is the loss *share*, not the weight.

| weights | E% | F% | W% | |
|---|---|---|---|---|
| `100 : 1 : 0.05` | 0.8 | 99.0 | 0.2 | razor's, carried over naively |
| `10 : 1 : 1` | 0.08 | 96.8 | 3.1 | looks conservative, actually deletes the energy term |
| **`1000 : 1 : 5`** | **6.6** | **80.5** | **12.9** | **used here** |
| `3620 : 1 : 5.6` | 20.1 | 67.8 | 12.1 | exactly razor's tuned shares |

`1000 : 1 : 5` is deliberately lighter on energy than razor's tuned share — a
conservative first setting on the axis this dataset has not been swept on —
while keeping the work-function share at essentially razor's value, since that
is the one the charge conditioning depends on.

**The large energy weight is not a hazard.** A randomly-initialised model
predicts ~0 on top of the *fitted per-species baseline*, so the initial energy
residual is the label spread itself (7.6 meV/atom) and these shares hold from
step 0 — there is no transient where the energy term explodes. The weight is
large only because it divides out a small variance. The term that *can*
transient is the work function, whose untrained `dE/dq` is unconstrained;
razor saw exactly that at weight 0.15 (99% of the loss at epoch 2, recovered
by epoch 34), and warmup plus `gradient_clip: 1.0` absorbed it.

## Splits

90/10 by frame, seed 0, generated by `../../datasets/cpmace/split.py` into
`cpmace_train.xyz` / `cpmace_val.xyz`: **984 train / 109 valid**.

Splitting per frame is safe here. Frames are independently sampled, not a
trajectory — consecutive frames differ by ~1.8 Å RMSD — so there is no group
structure to preserve, unlike razor where one geometry appears at three bias
charges.

There is **no test set**. The dataset ships one file; the 10% validation split
is all the held-out data there is, and `evaluate.py` reports on it rather than
inventing a test set from the same pool.

## Budget

`max_epochs: 1000`, matched to `../razor/`'s **200-epoch** budget in gradient
updates rather than epochs. 984 frames at 7 real structures per batch = 141
batches/epoch, so 1000 epochs = **141,000 updates** against razor's
729 × 200 = 145,800 — 97%, and a round number. `warmup_epochs: 50` scales the
same way (razor's 10 epochs = 7,290 updates = 51.7 here).

Matching updates rather than epochs is what makes the two folders comparable:
this dataset is 11× smaller, so equal epochs would have meant a fifth of the
training. `../razor_centre/`'s first round was confounded by exactly that
mistake, and its 675-epoch rerun is the same correction.

Still worth checking the learning curves before concluding anything about
capacity: a flattening cosine tail is the schedule ending, not convergence
(see `../razor/README.md`).

## 1000 epochs is not the limit; data is

The force curve is still schedule-limited at the end -- improvement per
100-epoch window over the second half runs -13.4, -7.6, -5.3, -2.8, -0.68%,
tracking the cosine down to lr = 0, the same signature `../razor/` showed at
200 epochs. So a longer run *would* improve forces.

It should not be run anyway, because the model is overfitting hard:

| epoch | train | valid | valid/train |
|---|---|---|---|
| 200 | 0.0165 | 0.0244 | 1.48 |
| 500 | 0.00435 | 0.0129 | 2.98 |
| 800 | 0.00185 | 0.0111 | **6.01** |

razor's equivalent gap went 1.24 -> 1.52 over its last 100 epochs. Here it is
6.0 and climbing, on 984 frames of a single composition against razor's
10,931. And the other two targets have already turned: **the work function
bottomed near epoch 700** (0.04128) and drifted back up to 0.0421, while
energy plateaued around epoch 600. The checkpointer picking an epoch-550ish
model for `sr` is the same fact seen from the other side.

So more epochs trades a few percent of force accuracy for a worse charge
response -- the thing this dataset exists to model. **The binding constraint
is data.** cp-MACE's SI reports the same: 16.9 -> 8.67 meV/Å purely from
adding force-only structures.

## Things to watch

- **The Ewald head runs in 3D here for the first time.** cpmace is
  `pbc = T T T`; every long-range run so far in this repo has been razor's
  slab (`T T F`), which uses the 2D-mixed variant. The `lr` run exercises a
  code path this project has not tested.
- **The per-species baseline is not identifiable.** One composition means the
  elemental fit is degenerate — `prepare.py` reports −0.089 eV for both N and
  Ni, which is an artefact of there being one atom of each in every frame.
  Only the total offset is meaningful, and only the total offset is used.
- **The max-force screen is a genuine no-op.** cpmace's largest force is
  5.65 eV/Å, well under the 10 eV/Å cut inherited from `../razor/`. Kept so
  both folders screen identically.
- **q never changes sign.** The system carries 1.26–2.31 excess electrons
  throughout, so the model only ever sees a negatively charged interface. Any
  `dE/dq` it learns is local to that window, and extrapolating to positive
  charge is unsupported.

## Layout

- `prepare.py` — applies the conversions and writes `data/{train,valid}`.
- `evaluate.py` — parity and RMSE-vs-charge figures on `valid`. Derived from
  `../razor/evaluate.py` with the razor-only machinery removed (no polarizable
  flag, no `bec_z` labels, no charge sweep).
- `sr-small-l2c8-1000ep/`, `lr-small-l2c8-1000ep/` — `model.yaml` +
  `settings.yaml` + `srun.sh`.

## Results

All three ran the full 1000 epochs. Converged **validation** RMSE (last
validation block), on the 109-frame held-out split:

| run | weights | E share | E (meV/atom) | F (meV/Å) | Φ (V) | wall |
|---|---|---|---|---|---|---|
| `sr-small-l2c8-1000ep` | 1000:1:5 | 6.6% | **0.600** | 40.70 | 0.0421 | 3h47m |
| `lr-small-l2c8-1000ep` | 1000:1:5 | 6.6% | 0.631 | **40.24** | **0.0376** | 4h34m |
| `sr-...-cpmace-weights-` | 1:100:10 | 0.0001% | 1.609 | 62.66 | 0.0595 | 3h45m |

R²: 99.47/99.77/89.65, 99.42/99.77/91.78, 96.21/99.45/79.38.

### `evaluate.py` scores a different model than the last epoch

The checkpointer saves on **best summed R² (`R2_E+F+W`), not the final
epoch**, so `evaluate.py` loads whichever epoch won that, and its numbers are
not the table above:

| run | | E | F | Φ |
|---|---|---|---|---|
| `sr` 1000:1:5 | last epoch | 0.600 | **40.70** | 0.0421 |
| | evaluated checkpoint | 0.66 | **50.02** | **0.0409** |
| `lr` 1000:1:5 | last epoch | 0.631 | 40.24 | 0.0376 |
| | evaluated checkpoint | 0.62 | 41.58 | 0.0373 |
| cp-MACE weights | last epoch | 1.609 | 62.66 | 0.0595 |
| | evaluated checkpoint | 1.62 | 62.69 | 0.0581 |

`lr` and the cp-MACE-weights run barely move, but **`sr`'s checkpoint is 23%
worse on forces and better on Φ than its final epoch** — it was selected from
an earlier epoch, around 550-600 judging by the force curve. That is early
stopping working as intended on a model that overfits (see below), but it has
a consequence for reading the comparison:

- **At matched epoch (the last block), `sr` and `lr` are tied on forces**
  (40.70 vs 40.24).
- **At their selected checkpoints they are not** (50.02 vs 41.58), because
  the two were stopped at different epochs.

The last-epoch row is the fair architecture comparison; the checkpoint row is
what you would actually deploy. Quote whichever matches the question, and say
which.

### cp-MACE's weights lose on forces, the target they weight at 99.68%

This is the result worth recording. The `1:100:10` run puts **99.68% of its
loss on forces** and comes out **54% worse on forces** (62.66 vs 40.70 meV/Å)
than the run that puts 80.5% there — while also being 2.7× worse on energy and
1.4× worse on the work function. It loses on every target, including its own.

That inverts what `../razor/`'s sweep found, where force error tracked the
force share monotonically (99.7% → 23.5, 73.6% → 32.6, 60.5% → 33.7,
48.2% → 40.6 meV/Å). Here more force share gives worse forces.

Note this is not a learning-rate artefact. The `1:100:10` loss is ~81× larger
in absolute magnitude, but `gradient_clip: 1.0` normalises the update, so what
differs between the runs is the *direction* of the gradient, not its size.

The likeliest explanation is the one `../razor/README.md` already argues, seen
from the other side: **forces constrain ∂E/∂r and say nothing about ∂E/∂q.**
In a charge-conditioned model the energy and work-function terms are the only
things that pin down the charge direction. Starve both — energy at 0.0001%,
work function at 0.32% — and the FiLM conditioning becomes an effectively
unconstrained nuisance input that can absorb variance which ought to be
explained by geometry. That would degrade forces too, which is what is
observed. Plausible, not established; a run at `1:100:10` with charge
conditioning disabled would test it directly.

**Why cp-MACE can get away with it:** their Fermi level is a separate output
head, so their electron-number input carries no thermodynamic-consistency
obligation. Ours is `dE/dq`, so E(q) has to be right. The weighting is sound
for their architecture and does not transfer to a derivative formulation.

### Against cp-MACE's published numbers

Same system and sampling, so their SI figures are a meaningful reference:

| | cp-MACE (SI) | this work (best) |
|---|---|---|
| forces | 16.5 meV/Å | 40.24 meV/Å |
| Fermi level / Φ | 0.03–0.04 eV | **0.0376 V** |

**The work function matches their published accuracy.** Forces are 2.4× worse,
but not on equal terms: their MACE is `128x0e + 128x1o` against our d64 l2 c8
(~300k parameters), and their final training set is larger — the SI reports
16.9 → 8.67 meV/Å from adding force-only structures, which we do not have.
Closing the force gap is a model-size and data question, not a weighting one.

### The Ewald head helps the charge response, again

`lr` beats `sr` on the work function by 11% (0.0376 vs 0.0421) and on forces
by 1% at matched epoch — or by 17% on forces at the selected checkpoints, for
the checkpoint-selection reason above rather than an architectural one. Both
at 21% more wall clock. The Φ gain reproduces `../razor/`, where the
Ewald head was consistently the strongest lever on the charge response. The
force gain does not: razor saw 15% there against 1% here.

Worth noting this is the **first 3D-periodic** long-range run in the repo —
every previous `lr` run was razor's slab, using the 2D-mixed Ewald variant.
Whether the small force gain reflects the 3D path, the system, or the
water-dominated environment is untested.
