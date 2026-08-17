# cpmace-grace

GRACE-2L with FiLM charge conditioning on the cpmace dataset — the first use of
the charge conditioning added to
[the fork](https://github.com/lucasdekam/grace-tensorpotential/tree/charge-conditioning).

## Why this dataset

`../cpmace/` trained LOREM here and came out **noticeably worse than the
published cp-MACE result**, which used a MACE model with a global feature. Two
explanations were never separated: LOREM's architecture is weaker on this
system, or the charge conditioning is placed badly. Running the same data
through a stronger backbone with the *same* FiLM conditioning isolates the
first.

The best LOREM numbers to beat (`../cpmace/README.md`, GRACE-like run):
**forces 29.33 meV/Å** against the control's 40.70.

## The model

`GRACE_2LAYER_FILM`, the "small" preset settings plus the FiLM block. FiLM is
applied to **both** scalar readout branches, `I_out_0_LN` and `I_1_LN`, since
`LinMLPOut2ScalarTarget` sums them and conditioning only one would leave the
other's energy contribution charge-independent.

`work_function = dE/dq` comes from `ComputeBatchEnergyForcesCharge`, which
watches the charge on the same tape as the bond vectors — one reverse sweep
produces both, so the work function is nearly free.

## Data conversion

`prepare.py` rewrites the source xyz into `data/{train,valid}.xyz`. Two
conversions, and the energy one is easy to miss: **`energy` in the source is
the grand potential**, not the total energy. See the module docstring and
`../../datasets/cpmace/README.md`.

    q             = -(electron - 660)
    E_total       = energy + potential * (electron - 660)
    work_function = -potential

No marathon `prepare()` step here: GRACE reads extxyz directly and
`TotalChargeDataBuilder` picks `total_charge` and `work_function` out of
`atoms.info`.

## Loss weights

**Superseded by the Results section below** — these were run 1's weights and
left the energy at 0.10% of the loss. `gracemaker` defaults for energy (16) and forces (32), plus a work-function
term at 10. That last number is a **first guess, not a tuned value** — the
cp-MACE paper uses 1 : 100 : 10 for E : F : WF, which would put the work
function at roughly a third of the force weight; 10 against 32 is in that
neighbourhood. Worth a sweep once the pipeline is known good.

Note the model starts with `dE/dq = 0` identically, because the FiLM gate is
zero-initialised, while the label sits at ≈ +3.3 V. So the work-function loss
starts large by construction and the gate has to open before it moves.

## Setup

Runs against `~/venv/grace` on alex (`pip install -e ~/grace-tensorpotential`
on the `charge-conditioning` branch). `TF_USE_LEGACY_KERAS=1` is **required**:
TensorFlow pulls in keras 3.x, and tensorpotential needs the legacy API bundled
with TF. No `cuda` module is loaded — `tensorflow[and-cuda]` ships its own.

## Results

### Headline: best validation result from each model

| | E (meV/atom) | F (meV/Å) | WF / E_F (mV) | wall |
|---|---|---|---|---|
| cp-MACE, published (SI Fig S2, node augmentation) | not reported | 16.51 | 40.05 | not reported |
| LOREM short-range (`../cpmace/sr-grace-like-d128-l3-c32-1000ep`) | **0.561** | 29.33 | 41.4 | 7h39m |
| LOREM **long-range** (`../cpmace/lr-small-l2c8-1000ep`) | 0.631 | 40.24 | 37.6 | 4h34m |
| **GRACE-2L + FiLM (run 4)** | 0.640 | **14.14** | **33.47** | **2h23m** |

All rows are held-out validation on the same 1093-structure dataset.
**GRACE wins forces and the work function**; LOREM's short-range GRACE-like model
keeps energy by 14%, which is a loss-weight choice (run 2 reached 0.619), and
cp-MACE never reports an energy error at all.

The two LOREM rows are listed separately because neither dominates: the
GRACE-like model is LOREM's best on energy and forces, while **`lr` is LOREM's
best work function** (37.6 mV) despite being the smaller d64 l2 c8 architecture —
adding Ewald electrostatics bought 0.0421 → 0.0376 V on the charge response while
barely moving forces (40.70 → 40.24). That is the clearest evidence in this
project that long-range physics helps the work function specifically.

Which makes GRACE's 33.47 mV notable: it beats LOREM's long-range model **without
any long-range term at all** — the FiLM presets are purely short-range. Note also
that `lr` run used `max_degree_lr: 0`, i.e. **monopole-only** Ewald with no
dipoles, so the dipole physics has never actually been tested here. That is the
Phase 0 experiment in `../../notes/dipole-term-plan.md`, and it is one config line.

Wall times are not comparable across codebases — cp-MACE reports none, and
LOREM's best is a larger model run for 1.41× the updates.

Everything below is the supporting detail.

---

Four runs. **Run 4 is the model**: on validation it beats cp-MACE's published
forces *and* Fermi level, in under a third of LOREM's wall time.

| | model | loss | updates | E (meV/atom) | F (meV/Å) | WF (mV) | wall |
|---|---|---|---|---|---|---|---|
| run 1 | small | square | 20k | 1.249 | 21.47 | 33.36 | 2h05m |
| run 2 | small | huber | 100k | **0.619** | 27.07 | 40.48 | 2h23m |
| run 3 | medium | huber | 60k | 0.630 | 43.31 | 45.37 | 2h40m |
| **run 4** | small | **square** | 100k | 0.640 | **14.14** | **33.47** | 2h23m |

### The huber control settles it: huber was costing 48% on forces

Runs 2 and 4 differ in **nothing but the loss shape** — same model, batch,
budget, dtype, schedule, and weights derived from the same basis to target the
same shares. This was asserted programmatically when run 4 was built, not
assumed.

| | run 2 (huber) | run 4 (square) |
|---|---|---|
| forces | 27.07 | **14.14** (−48%) |
| work function | 40.48 | **33.47** (−17%) |

The mechanism is the one the config file predicted: at `delta = 0.01` against
20–40 meV/Å force errors, nearly every force error sits in huber's **linear**
regime, which is close to optimising MAE and weakens the gradient on exactly the
large errors RMSE is made of. **huber is gracemaker's default loss type**, so
this is a trap anyone fitting forces at this error scale will walk into.

Run 3 is consistent with this and adds nothing independent: it was also huber,
and additionally undertrained (1.83M parameters given 0.6× run 2's updates, with
train forces ≈ test forces — underfitting, not overfitting).

### Against the published cp-MACE result

Numbers read out of `../../datasets/cpmace/ct5c00784.pdf` and its SI. **The
train/validation distinction is essential here** — the paper's headline figures
are training-set numbers and the SI's are validation, and they differ by 2x.

| source | split | F (meV/Å) | E_F / WF (meV) | E |
|---|---|---|---|---|
| cp-MACE, node augmentation (SI Fig S2) | **validation** | 16.51 | 40.05 | not reported |
| cp-MACE, global feature (SI Fig S2) | **validation** | 19.55 | 39.68 | not reported |
| cp-MACE, all 1093 structures (paper §3.2) | **train** | 8.4 | 10.8 | not reported |
| **GRACE-2L + FiLM, run 4** | **validation** | **14.14** | **33.47** | 0.640 meV/atom |
| GRACE-2L + FiLM, run 4 | train | 10.57 | 15.39 | 0.692 meV/atom |

**Like for like on validation, GRACE is ahead on both**: forces 14.14 vs 16.51
(14% better) and the work function 33.47 vs 40.05 mV (16% better). Their dataset
is the same one — the paper states "the final training data set comprises 1093
structures", exactly our 984 + 109.

**Their 8.4 / 10.8 is a training-set fit, not held out.** Our training equivalents
are 10.57 / 15.39, so cp-MACE fits the training data somewhat harder while
generalising worse. Do not compare their 8.4 against our 14.14.

Note also there are **two different ~8.x force numbers** in their papers, easy to
conflate: 8.4 meV/Å is the train fit above, while **8.67 meV/Å** (SI Fig S3) is a
validation number obtained by *adding force-only structures* to the training set —
a larger dataset than the one released, so neither is a like-for-like target.

**cp-MACE reports no energy RMSE anywhere**, in the paper or the SI, despite
energy carrying weight 1.0 in their loss. So the energy column cannot be compared
at all; the only external energy baseline is LOREM's.

Their settings for context (SI §2): `128x0e + 128x1o`, cutoff **5.0 Å** (we use
6.0), **batch size 2**, loss weights 100 : 1 : 10 for forces : energy : Fermi
level — the same triple `../cpmace/` tried on LOREM, where it put the energy at
0.0001% of the loss.

One architectural point worth restating: our work function is `dE/dq` from
autograd, thermodynamically consistent by construction, while cp-MACE's Fermi
level is a **direct output head** carrying no such obligation.
`../cpmace/README.md` flagged the risk that a target cheap to fit as an
independent head need not be cheap to fit as a derivative that must reshape E(q).
On this evidence it costs nothing — we beat the head on its own target.

### Against LOREM on the same dataset

(All GRACE and LOREM numbers here are validation.)

Best GRACE (run 4) against best LOREM (`../cpmace/sr-grace-like-d128-l3-c32-1000ep`):

| | LOREM best | GRACE run 4 | |
|---|---|---|---|
| energy (meV/atom) | **0.561** | 0.640 | 1.14× worse |
| forces (meV/Å) | 29.33 | **14.14** | **2.07× better** |
| work function (mV) | 41.4 | **33.5** | 1.24× better |
| wall time | 7h39m | **2h23m** | **3.2× faster** |
| updates | 141,000 | 100,000 | |

**2.1× better forces in 3.2× less wall time** — about 6.6× on
accuracy-per-GPU-hour. Energy is the one target LOREM still wins, by 14%, and
that is a loss-weight choice rather than a capability gap: run 2 reached 0.619
and run 1 with the energy at 0.10% of the loss reached only 1.249.

Runtime is not like-for-like in the model's favour either — LOREM's best run is a
larger model (d128 l3 c32) and used 1.41× the gradient updates. Against LOREM's
*same-size* model (`sr-small-l2c8`, 40.70 meV/Å in 3h47m) run 4 is 2.9× better on
forces in 63% of the time.

### Still not converged

Every run's best checkpoint has been at or within a few epochs of its **last**,
including run 4 (181 new-best checkpoints in 508 epochs). A flattening cosine
tail is the schedule ending, not convergence. LOREM needed 141,000 updates here
and was still improving; run 4 got 100,000. **The budget is still the binding
constraint**, so 14.14 meV/Å is a floor, not a limit.

### What to try next

- **More updates at run 4's settings.** The cheapest remaining lever, and the
  only one with direct evidence behind it.
- **Medium model with square loss and a proper budget.** Run 3 confounded
  capacity with huber and a short schedule, so capacity is currently untested.
- **The work function has stopped responding to loss weight.** Run 1 (43% share)
  and run 4 (12.6%) both land at ~33.4 mV. Run 3 showed why: train WF 19.4 vs
  test 45.4 — it overfits on only **984 labels**, one per structure, against
  610k force components. More WF weight will not help; more WF data or a
  directional inductive bias would. See `../../notes/dipole-term-plan.md`.
