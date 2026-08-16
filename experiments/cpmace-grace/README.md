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

Not yet run.
