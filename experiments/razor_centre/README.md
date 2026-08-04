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

`model.yaml` is identical across `sr/` and `sr-wf/`; `sr-wf-bec/` adds
`predict_bec: True`. All three share one `data/`, so **only `loss_weights`
(and that one model flag) differ** -- `sr/` is a clean control with no data or
lorem-version confound.

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

_TBD -- not yet trained._
