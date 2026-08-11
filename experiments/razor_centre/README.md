# razor centre-frame training

Trains `Lorem` on `razor_centre.xyz` -- one frame per structure, at the centre
of its bias-charge stencil, i.e. exactly the frames where
`bias_charge == q_MD`. This is the follow-up experiment 1 planned in
`../razor/README.md`.

The point is the comparison against `../razor/`. There, two thirds of the
training frames pair a geometry with a charge that geometry never equilibrated
at (the `q_MD ± 0.25 e` stencil wings). That off-stencil labelling is
deliberate -- it is what makes `∂E/∂q` learnable at all -- but it means the
training distribution is not the physical (geometry, charge) distribution an
MD run would visit. Here every (r, q) pair is physical, at the cost of ~3x
less data.

## Variants

| dir | targets | notes |
|---|---|---|
| `sr/` | energy + forces | control for this folder; direct counterpart of `../razor/sr/` |
| `sr-wf/` | + `work_function` | supervises the autograd `dE/dq` |
| `sr-wf-bec/` | + `bec_z` | additionally supervises the Born effective charge |
| `sr-450ep/` | energy + forces | `sr/` with the budget matched to `../razor/sr/` in gradient updates, not epochs -- see the caveat under Results |
| `sr-small-l2c8-e100-wf0.05-675ep/` | energy + forces + `work_function` | d64 l2 c8 at 100:1:0.05, 675 epochs = 218,700 updates, matching `../razor/sr-small-l2c8-...-300ep/` exactly |
| `lr-small-l2c8-e100-wf0.05-675ep/` | same | the same run with `lr: true` (`max_degree_lr: 0`, monopole Ewald) |
| `lr-e100-wf0.05-450ep/` | -- | set up, deferred, never run (configs only) |

The last two trained variants are the ones to trust for cross-folder claims:
they hold model, loss weights and update budget fixed against `../razor/`,
so the training distribution is the only thing left varying. The four rows
above them differ in several of those at once.

`model.yaml` is identical across `sr/` and `sr-wf/`; `sr-wf-bec/` adds
`predict_bec: True`. All three share one `data/`, so **only `loss_weights`
(and that one model flag) differ** -- `sr/` is a clean control with no data or
lorem-version confound.

`sr-wf/` and `sr-wf-bec/` use `lorem.LoremQ`, the charge-aware subclass
that exposes `dE/dq` and `bec_z`; `sr/` and `sr-450ep/` stay on plain
`lorem.Lorem`, whose `predict` returns energy and forces only. Same
architecture either way -- `LoremQ` inherits `__call__` untouched -- so the
comparisons above are unaffected and checkpoints are weight-compatible.

## Splits

`razor_centre.xyz` holds 5989 structures, which is exactly
`razor_train.xyz` (5390) + `razor_val.xyz` (599) by `struc_pk`; the 20
`razor_test.xyz` structures are absent. Verified, not assumed:

```
razor_centre.xyz: 5989 frames, 5989 unique struc_pk
razor_train.xyz: 16170 frames, 5390 unique struc_pk
razor_val.xyz:    1797 frames,  599 unique struc_pk
razor_test.xyz:    260 frames,   20 unique struc_pk
```

So rather than invent a split, `prepare.py` **inherits razor's**: a centre
frame goes to train or valid according to which razor file its `struc_pk`
appears in, and the test sets are taken from `razor_test.xyz` verbatim, as in
`../razor/prepare.py`.

| split | source | note |
|---|---|---|
| `data/train` | `razor_centre.xyz`, struc_pk ∈ razor_train | polarizable only |
| `data/valid` | `razor_centre.xyz`, struc_pk ∈ razor_val | polarizable only |
| `data/test` | `razor_test.xyz` | polarizable only, 34 frames |
| `data/test_sweep` | `razor_test.xyz` | unfiltered, 260 frames |

This means the **evaluation sets are byte-identical to `../razor/`**, so only
the training distribution differs and results compare row-by-row across the
two folders. The split key is `struc_pk` and is never split within a
structure, so there is no leakage. `prepare.py` raises if any centre frame
fails to land in a split, rather than silently dropping it.

## Max-force screening (implemented at 10 eV/Å, evaluation only)

This folder's `evaluate.py` applies the same `MAX_FORCE_CUTOFF = 10 eV/Å` as
`../razor/`, to the same evaluation splits -- both folders score on
`razor_val.xyz` and `razor_test.xyz`, so the screen is identical and the
cross-folder comparison stays like-for-like. **Training data is not
screened.**

An earlier version of this section measured a **>20 eV/Å** rule against
`razor_centre.xyz` and found one polarizable frame, concluding it was a
no-op. That was right about the rule and wrong about the threshold; see
`../razor/README.md` for the frame that motivated dropping to 10. Measured
over this file's polarizable pool:

| file | polarizable | median / p95 / max | >20 eV/Å | >10 eV/Å |
|---|---|---|---|---|
| `razor_centre.xyz` | 5,398 | 3.43 / 4.98 / **35.78** | 1 | **8** |

Since this folder is one frame per structure, those 8 frames are 8 distinct
`struc_pk` and dropping them would cost exactly 8 training structures
(0.15%). Nothing is currently dropped from training, so this is a TODO here
as it is in `../razor/`, and the same argument applies: it is unlikely to
matter at 0.15%, but train and evaluation should ideally be screened
identically, and that has not been tested.

What the screen does affect is every number reported below. It removes 5 of
1218 polarizable `razor_val` frames and none of the 34 polarizable
`razor_test` frames, and lowers force RMSE by 6-11% for every variant in both
folders. Numbers labelled "screened" post-date it; the rest are marked where
they do not.

## Born effective charges

`bec_z` in the dataset is the dimensionless Born effective charge; the model
computes the mixed second derivative. The charge sets a surface charge density
`q/A`, hence a field `q/(A ε₀)` along the slab normal, so an atom with Born
effective charge `Z*` feels `F = Z* q/(A ε₀)` and

```
Z* = (A ε₀) ∂F/∂q = -(A ε₀) ∂²E/∂r∂q
```

This was checked against the data rather than assumed. Regressing a
finite-difference `∂F/∂q` (from razor's 3-point stencil) on the `bec_z` label
over 200 structures gives

```
∂F/∂q = 2.19839 * bec_z + (-2.3e-11)     corr = 1.00000
1/(A ε₀) = 2.19839  for A = 82.3108 Å², ε₀ = 0.005526349 e²/(eV·Å)
```

i.e. the relation holds to six significant figures with zero intercept. The
in-plane area is computed from the cell inside the model
(`|c₁ × c₂|`, so the varying out-of-plane vector never enters) rather than
being passed in as a label.

**Weighting.** The dataset README warns that `bec_z` is a finite-difference
estimate damped by ≥15% and recommends masking on `polarizable` and weighting
it low. Train/valid here are already filtered to polarizable frames, so the
mask is handled by the data. On the weight: `bec_z` has var 0.0209
(std 0.145 e), so the bare number is misleading -- weight 1.0 contributes
0.021 to the loss, about **8% of the force term** (0.5 × 0.521 = 0.261), not
2x it. That is the intended "low".

## Loss weights

Same reasoning as `../razor/sr-wf/`: energy and forces stay at the house
0.5/0.5 so `sr/` is an exact control, and the derivative targets are weighted
from validation-set variances.

| target | var | weight | contribution |
|---|---|---|---|
| energy | 0.0011 | 0.5 | 0.0006 |
| forces | 0.5214 | 0.5 | 0.261 |
| work function | 1.8523 | 0.15 | 0.278 |
| bec_z | 0.0209 | 1.0 | 0.021 |

The work-function caveat from `../razor/README.md` applies here too: these
balance *asymptotic* residuals, and at initialisation the untrained `dE/dq` is
far enough off that its term briefly dominates. In `../razor/sr-wf/` that
suppressed force accuracy through warmup (force R² 1.5% at epoch 2) and then
fully recovered once warmup ended (98.8% by epoch 34), so it is expected
behaviour rather than a problem -- but worth confirming here too.

## Running

```bash
# on the cluster, once (all three variants share this data/)
DATASETS=. python prepare.py

# then submit
cd sr && sbatch srun.sh && cd ../sr-wf && sbatch srun.sh && cd ../sr-wf-bec && sbatch srun.sh
```

`prepare.py` has to run on the cluster: it needs `marathon`, which is not
installed on the local machine.

## Results

Six variants have trained here. The first three ran 200 epochs; their
converged **validation** RMSE (last validation block) on this folder's own
538-frame centre validation set:

| variant | energy | forces | work function | bec_z | runtime |
|---|---|---|---|---|---|
| `sr` | **0.746 meV/atom** | **25.2 meV/Å** | -- | -- | 6h09m |
| `sr-wf` | 1.478 meV/atom | 47.2 meV/Å | 0.123 V | -- | 6h07m |
| `sr-wf-bec` | 1.593 meV/atom | 52.5 meV/Å | **0.093 V** | **0.0253 e** | 13h23m |

The `l2c8` pair added later, same set:

| variant | energy | forces | work function | runtime |
|---|---|---|---|---|
| `sr-small-l2c8-e100-wf0.05-675ep` | 0.636 | 30.0 | 0.1158 | 7h03m |
| `lr-small-l2c8-e100-wf0.05-675ep` | **0.589** | **26.8** | **0.0868** | 7h36m |

R²: `sr` 99.95/99.88, `sr-wf` 99.82/99.57/98.09,
`sr-wf-bec` 99.79/99.47/98.90/96.96.

`sr-wf-bec`'s post-training test collation crashed (see below), but its final
validation block was written before that, so these are its own numbers on the
same 538-frame set as the other two rows.

Two things stand out even before the cross-folder comparison: `sr-wf-bec` has
the **best work function of the three** (0.093 V vs `sr-wf`'s 0.123), so
supervising `∂²E/∂r∂q` helped `∂E/∂q` rather than competing with it; and
`bec_z` itself lands at 0.0253 e against a 0.145 e label spread.

> **Do not compare the validation RMSEs above across folders.** Each run's
> own validation set is different: `../razor/` validates on 1218 full-stencil
> frames, this folder on 538 centre frames, and the centre frames are the
> easier set -- equilibrium (r, q) pairs rather than off-stencil ones. Use
> `evaluate.py`'s numbers below, which score every variant in both folders on
> the *same* held-out set.

### On the common evaluation set

`evaluate.py` scores all six variants on `razor_val.xyz` and on the wide-charge
test sweep, so the two folders are directly comparable.

**Frames with `max_force > 10 eV/Å` are screened out** of the evaluation
splits: 5 of 1218 on `razor_val` (n=1213 kept), none on `razor_test`. RMSE is
outlier-dominated, and those five cost 6-11% of the force RMSE across every
variant. The table immediately below predates the screen and is kept for
continuity with the surrounding text; the screened numbers are in "the clean
cross-folder test" further down. Validation split, RMSE:

| variant | trained on | energy | forces | work function |
|---|---|---|---|---|
| `../razor/sr` | full stencil | **0.82** | **23.6** | 0.236 |
| `../razor/lr` | full stencil | 0.76 | 20.3 | 0.181 |
| `../razor/sr-wf` | full stencil | 1.61 | 40.6 | **0.121** |
| `sr` | centre only | 1.04 | 31.9 | 0.434 |
| `sr-wf` | centre only | 1.98 | 52.1 | 0.243 |
| `sr-wf-bec` | centre only | 1.91 | 56.3 | 0.202 |
| `sr-small-l2c8-...-675ep` | centre only | 0.85 | 33.0 | 0.202 |
| `lr-small-l2c8-...-675ep` | centre only | 0.77 | 28.8 | **0.161** |

(The last two rows are the update-matched `l2c8` pair added later; see "the
clean cross-folder test" below, which is the comparison to trust. The rows
above them differ in model size, loss weights *and* update budget all at
once.)

**Centre-only training is worse across the board** at equal epochs -- by 27% on energy, 35% on
forces and 84% on the work function.

> **Caveat: this comparison is confounded by training length, and the size of
> the confound is exactly the size of the data difference.** Both folders run
> `max_epochs: 200`, but an epoch here is 2.25x smaller, so:
>
> | run | frames | steps | gradient updates | runtime |
> |---|---|---|---|---|
> | `../razor/sr` | 10,931 | 2,186,200 | 145,700 | 13h50m |
> | `sr` | 4,860 | 972,000 | 64,800 | 6h09m |
>
> So "less data" and "less training" are not separated here, and the gap
> above may be partly or wholly the latter.
>
> The flat tail of the validation curve does *not* settle it. Both runs end
> perfectly flat (loss 2.75e-04 x3 for `../razor/sr`, 3.18e-04 x3 here), but
> a cosine schedule always ends flat because the LR has annealed to 1e-6 --
> that shows the schedule finished, not that the model reached its best
> achievable error.
>
> The fix is to match gradient updates rather than epochs, the convention the
> group's own benchmarking paper uses (arXiv 2602.22931, SI §S-II: "it is
> important to train models with a similar number of gradient updates, where
> GU = epochs x dataset size / batch size"). `sr-450ep/` does exactly that:
> `max_epochs 450`, `warmup_epochs 22`, giving 145,800 updates against
> razor's 145,700 with the cosine annealing over the matched budget.
> Everything else is identical to `sr/`.

### Resolved: it was about half training length, half data

`sr-450ep/` ran to 450 epochs in 13h39m. On the common evaluation set
(`razor_val.xyz`, polarizable, n=1218):

| variant | updates | energy | forces | work function |
|---|---|---|---|---|
| `../razor/sr` | 145,700 | **0.82** | **23.56** | **0.236** |
| `sr` | 64,800 | 1.04 | 31.93 | 0.434 |
| `sr-450ep` | 145,800 | 0.92 | 26.32 | 0.313 |

Gap to `../razor/sr`, before and after matching the budget:

| | energy | forces | work function |
|---|---|---|---|
| at 200 epochs | +27% | +36% | +84% |
| at matched updates | **+12%** | **+12%** | **+33%** |

**Matching the update budget closes between half and two thirds of every
gap, and a real data effect survives underneath.** So both readings were
partly right: centre-only training genuinely is worse, but by 12% on energy
and forces rather than 27-36%, and the earlier numbers were substantially
inflated by the shorter schedule.

The work function keeps the largest residual gap (+33%), which is consistent
with the argument that the off-stencil frames are what make `∂E/∂q`
identifiable -- that part is about the data and does not go away with more
training.

`sr-450ep` also improves everything on the test sweep's polarizable frames
(E 1.04, F 26.18, WF 0.356, `bec_z` 0.0518, against `sr`'s 1.09 / 31.04 /
0.440 / 0.0620), so this is not a validation-set artefact.

Two consequences worth carrying forward: **`sr-wf` and `sr-wf-bec` are also
undertrained by the same factor**, so their comparisons against `sr` are
internally consistent but their absolute numbers are not comparable with
`../razor/`'s; and at ~30h a matched-update `sr-wf-bec` does not fit the 24h
limit, so it would need a restart path.

**The off-stencil frames matter for `∂E/∂q`, as originally argued.** Removing
them doubles the work-function error, both for the models that train on it
(0.243 vs 0.121 V) and those that do not (0.434 vs 0.236 V). `../razor/`'s
README says the off-stencil labelling "is exactly what makes `∂E/∂q`
learnable" -- that holds up quantitatively.

**Training on the work function improves it ~1.8x here** (0.434 -> 0.243 V),
matching the ~1.9x on the full stencil, at ~1.9x worse energy and ~1.6x worse
forces. Note `../razor/lr` reaches 0.181 V *without* the target while being
the best model on energy and forces too, so the Ewald head looks like a
better route to the work function than the explicit target at weight 0.15.

**Adding `bec_z` slightly helps the work function** (0.202 vs 0.243 V) and
energy (1.91 vs 1.98), at some cost in forces (56.3 vs 52.1).

### The clean cross-folder test: forces converge, the work function does not

`sr-450ep` matched the update budget but only for the old `0.5 : 0.5` weights
and the full-size model. The `l2c8` pair does it properly: **the same model
(d64 l2 c8), the same weights (100:1:0.05), and the same 218,700 gradient
updates in both folders** -- 300 epochs on `../razor/` (729 batches/epoch)
against 675 here (324). `model.yaml` is byte-identical across the four runs
except the `lr` flag. Nothing is left varying except the training
distribution and the range of the model.

Common evaluation set (`razor_val.xyz`, polarizable, screened at 10 eV/Å,
n=1213):

| model | trained on | E | F | Φ |
|---|---|---|---|---|
| `sr` l2c8 | full stencil | 0.775 | 32.9 | 0.114 |
| `sr` l2c8 | centre only | 0.846 | 33.0 | 0.202 |
| | | +9% | **+0.3%** | **+77%** |
| `lr` l2c8 | full stencil | 0.746 | 28.1 | 0.084 |
| `lr` l2c8 | centre only | 0.773 | 28.8 | 0.161 |
| | | +4% | **+2.5%** | **+91%** |

**With the budget properly matched, centre-only training costs essentially
nothing on forces and roughly doubles the work-function error.** The force
gap is 0.3% for the short-range pair and 2.5% for the long-range one; energy
is within 4-9%. (These are the 10 eV/Å-screened numbers -- the conclusion is
unchanged from the unscreened ones, which gave 0% and 4%.)

This sharpens the earlier "worse across the board" reading considerably.
That statement came from unmatched budgets; `sr-450ep` then showed matching
updates closes half to two thirds of each gap. This pair closes the force gap
**entirely** and leaves the work-function gap almost untouched.

The result is exactly what the data argument predicts and is worth stating in
those terms: **a geometry that appears at one charge constrains `∂E/∂r` just
as well as one that appears at three, and cannot constrain `∂E/∂q` at all.**
Forces are available from every frame regardless; the charge derivative needs
the stencil. So the off-stencil frames are not general-purpose extra data --
they buy one specific quantity, and they buy it at roughly 2x.

Test sweep, polarizable frames (n=34), same four runs:

| model | trained on | E | F | Φ | `bec_z` |
|---|---|---|---|---|---|
| `sr` l2c8 | full stencil | 1.35 | 38.7 | 0.332 | 0.0417 |
| `sr` l2c8 | centre only | 1.02 | 37.0 | 0.319 | 0.0499 |
| `lr` l2c8 | full stencil | 1.41 | 32.2 | **0.151** | **0.0342** |
| `lr` l2c8 | centre only | 1.13 | 32.1 | 0.177 | 0.0435 |

Outside the ±0.25 e stencil the Φ gap narrows sharply (0.332 vs 0.319 for
`sr`; 0.151 vs 0.177 for `lr`) and centre-only is actually *better* on
energy. That is not a contradiction: the sweep's charges lie outside both
training distributions, so neither model is interpolating, and the advantage
the stencil frames confer inside their window does not extend beyond it.

**The Ewald head dominates both effects.** `lr` beats `sr` by more, in both
folders, than full-stencil beats centre-only: on the sweep it halves Φ
(0.151 vs 0.332 on `../razor/`, 0.177 vs 0.319 here) where the data axis
moves it by a few percent. See `../razor/README.md` for that comparison in
full. Note it is still `max_degree_lr: 0`, monopole-only.

### Born effective charges, trained on or not

`evaluate.py` computes `Z* = -(A ε₀) ∂²E/∂r∂q` itself, so every variant is
scored whether or not it supervised `bec_z`. The label only exists on
`razor_test.xyz` and `razor_centre.xyz` (not `razor_val.xyz`), so this is the
test sweep -- which is where the dataset README wants it anyway, those labels
coming from a 13-point spline rather than the damped 3-point estimate.

Test sweep, RMSE in e (label spread 0.145 e):

| variant | trained on `bec_z`? | all (260) | polarizable (34) | non-pol. (226) |
|---|---|---|---|---|
| `sr` | no | **0.0738** | 0.0620 | **0.0754** |
| `sr-wf` | no | 0.0904 | 0.0541 | 0.0947 |
| `sr-wf-bec` | yes | 0.0915 | **0.0373** | 0.0971 |

**Supervising `bec_z` gives a 1.7x better Born effective charge exactly where
the physics is valid** -- 0.0373 e on polarizable frames vs `sr`'s 0.0620 --
and is *worse* than the untrained `sr` over the full sweep (0.0915 vs
0.0738). The full-sweep column is dominated by the 226 non-polarizable frames
near dielectric breakdown, where all three models are equally poor and the
label itself is least trustworthy. The dataset README's advice to mask on
`polarizable` is doing real work here: on the unmasked number the target
looks harmful, on the masked one it clearly helps.

Note also that `sr`, which never saw `bec_z` or `work_function`, already
reaches 0.062 e -- so as with the work function, charge conditioning plus
energy/force supervision recovers much of the derivative structure unaided.

`../razor/` now scores `bec_z` too, though nothing there trains on it. On the
same frames, `../razor/lr` reaches **0.0270 e** -- better than this folder's
supervised `sr-wf-bec` at 0.0373. That cross-folder comparison is confounded
(2.25x more data and 2.25x more gradient updates, per the caveat above), so
it does not show supervision to be useless: within each folder the respective
lever helps by a similar factor, 1.7x here from supervising `bec_z` and 1.5x
there from the Ewald head. It does suggest the Ewald head is the cheaper
route, and `sr-450ep/` plus an `lmax_lr` ablation would settle it.

**Born effective charges are learnable.** `bec_z` R² went -0.2% (epoch 2) →
82% (22) → 97.0% (200), on a finite-difference label damped by ≥15% carrying
only ~8% of the loss. All four targets converged simultaneously; supervising
`∂²E/∂r∂q` did not destabilise the others.

### `sr-wf-bec` post-training crash

Training completed all 200 epochs (13h23m, 94% GPU), the final validation
block was written, and every checkpoint was saved. The job then failed in
`predict_and_collate` on the test splits:

```
INTERNAL: RET_CHECK failure (config_assigner.cc:403) !candidates.empty()
Autotuning failed for HLO: f32[4,49,2048] ... "kind":"__triton_gemm"
... "device_type":"DEVICE_TYPE_INVALID"  No configs could be compiled.
op_name=".../Lorem._bec_z/jvp(jvp(Lorem.energy))/..."
```

This is an XLA autotuner failure, not a modelling bug: the test splits (34 and
260 frames) produce a trailing padded batch shape training never generated, so
the `_bec_z` double-JVP kernel had to be compiled fresh and Triton's GEMM
autotuner found no viable config. `evaluate.py` is unaffected in principle --
it computes `dE/dq` with its own `jax.grad` -- but the evaluate job exports
`--xla_gpu_autotune_level=0` as a guard, since it loads a `predict_bec` model.
A rerun of the training job would want the same flag.
