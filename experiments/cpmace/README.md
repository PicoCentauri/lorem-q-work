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

Not yet run.
