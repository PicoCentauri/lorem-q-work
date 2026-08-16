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

`gracemaker` defaults for energy (16) and forces (32), plus a work-function
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

`GRACE_2LAYER_FILM` small, 323 epochs (20,000 updates), 2h05m on one A100.
Test split (109 frames), best checkpoint:

| | E (meV/atom) | F (meV/Å) | WF (V) |
|---|---|---|---|
| `../cpmace/` LOREM best (`sr` GRACE-like) | **0.561** | 29.33 | 0.0414 |
| **GRACE-2L + FiLM** | 1.25 | **21.47** | **0.0334** |
| cp-MACE published | — | **16.5** | 0.03–0.04 |

**Forces improve 27% over LOREM's best** (21.47 vs 29.33 meV/Å), so the
backbone was a real part of the gap this experiment was built to test. It does
not close it: cp-MACE's published 16.5 is still 30% ahead. (Their 8.67 comes
from adding force-only structures we do not have, so it is not a like-for-like
target.)

**The work function matches cp-MACE** — 0.0334 V against their 0.03–0.04 eV,
and 19% better than LOREM's 0.0414.

That last number is the one worth keeping. `../cpmace/README.md` flagged the
risk that "a target that is cheap to fit as an independent head is not
necessarily cheap to fit as a derivative that must reshape E(q)": cp-MACE's
Fermi level is a **direct output head** carrying no thermodynamic-consistency
obligation, ours is `dE/dq` from autograd. On this evidence that constraint
costs approximately nothing — we match them on the target their architecture
is free to fit directly. One data point, not a settled result, but it is the
argument for keeping the derivative formulation, which is what makes
constant-potential MD well-posed (forces = −∇Ω, with Ω conserved).

### The energy is undertrained, and it is the loss weights

Energy is 2.2x worse than LOREM's (1.25 vs 0.561 meV/atom). Decomposing the
loss with the reported RMSEs reconstructs the logged total to 0.6%, so the
shares are trustworthy:

| target | weight | share |
|---|---|---|
| energy | 16 | **0.10%** |
| forces | 32 | 56.9% |
| work function | 10 | 43.0% |

**The energy is effectively untrained.** This is the same failure `../cpmace/`
diagnosed for cp-MACE's own 1:100:10 weights (energy at 0.0001%), reproduced
here by accident: gracemaker's 16/32 defaults are tuned for plain E/F fitting
and never anticipated a third target large enough to take 43% of the loss.

Shares transfer between codes, weights do not. `2000 : 32 : 1` reproduces
razor's tuned triple on this data (E 16.0 / F 78.2 / W 5.8 against razor's
16.4 / 78.6 / 5.0) and is the natural next run. Raising the work-function
share instead costs forces -- razor's sweep found ~1.7x -- and forces are
currently the only place this model beats LOREM, so that trade should be
measured rather than assumed.
