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

## TODO: max-force screening (checked -- currently a no-op here)

Same check as `../razor/README.md`, on this folder's source file. No
force-based exclusion is applied anywhere in the repo; `max_force` is present
on every frame, so the "drop configs with max force > 20 eV/Å" rule is
directly measurable:

| file | frames | `max_force` median / p95 / max | >20 eV/Å | of which polarizable |
|---|---|---|---|---|
| `razor_centre.xyz` | 5,989 | 3.42 / 5.02 / **145.5** | 8 | **1** |

**One frame.** The `polarizable` filter already removes 7 of the 8, so the
cutoff would drop 0.019% of the 5,398-frame polarizable pool and change
nothing in the results above. That frame sits in one `struc_pk`; since this
folder is one frame per structure, dropping it costs exactly one training
structure.

Worth adding as a guard for future data rather than for this dataset -- a
145 eV/Å configuration is broken, and it is incidental rather than by design
that `polarizable` happens to catch it.

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

All three variants trained to the full 200 epochs. Converged **validation**
RMSE (last validation block, epoch 200):

| variant | energy | forces | work function | bec_z | runtime |
|---|---|---|---|---|---|
| `sr` | **0.746 meV/atom** | **25.2 meV/Å** | -- | -- | 6h09m |
| `sr-wf` | 1.478 meV/atom | 47.2 meV/Å | 0.123 V | -- | 6h07m |
| `sr-wf-bec` | 1.593 meV/atom | 52.5 meV/Å | **0.093 V** | **0.0253 e** | 13h23m |

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

`evaluate.py` scores all six variants on `razor_val.xyz` (polarizable,
n=1218) and on the wide-charge test sweep, so the two folders are directly
comparable. Validation split, RMSE:

| variant | trained on | energy | forces | work function |
|---|---|---|---|---|
| `../razor/sr` | full stencil | **0.82** | **23.6** | 0.236 |
| `../razor/lr` | full stencil | 0.76 | 20.3 | 0.181 |
| `../razor/sr-wf` | full stencil | 1.61 | 40.6 | **0.121** |
| `sr` | centre only | 1.04 | 31.9 | 0.434 |
| `sr-wf` | centre only | 1.98 | 52.1 | 0.243 |
| `sr-wf-bec` | centre only | 1.91 | 56.3 | 0.202 |

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
