# Directional charge response: test LOREM's dipole Ewald first, then GRACE

**Headline: start with Phase 0 — a one-line LOREM config change that tests the
same physics for free. The GRACE implementation below is only worth building if
that shows a gain.**


*(The FiLM charge-conditioning plan this file previously held is implemented and
merged on the fork's `charge-conditioning` branch — see `tensorpotential/extra/charge/`.
This plan builds on it.)*

## Context

The work function is physically tied to the interfacial **z-dipole** through the
Helmholtz relation, Φ ⊃ μ_n/(A ε₀). Our model has no route to that quantity:
FiLM acts on the **invariant** (l=0) features only, and γ(Q), β(Q) are identical
for every atom in a structure. GRACE is translation-invariant with no global
frame, so the entire spatial charge response has to be inferred from differences
between invariant local environments — and for a *symmetric* slab the two faces
have identical descriptors, so the net dipole cancels by construction and the
physical asymmetry is unlearnable.

There is already empirical evidence this limitation costs accuracy:
`razor_centre_paper`'s `sr-bec` run, which supervises `bec_z` (a *directional*
charge response), got the **best work function of the three** — 0.0845 V against
`sr`'s 0.1239, a 32% improvement — despite being short-range and carrying a
*smaller* work-function share. Handing the model directional information helped.

**The identity that ties it together.** From the Maxwell relation on E(r, q):

```
∂F_iα/∂q = −∂²E/∂r_iα∂q = −∂Φ/∂r_iα = −(1/(A ε₀)) ∂μ_n/∂r_iα
```

so, up to sign convention, `Z* ≡ (A ε₀) ∂F/∂q = ∂μ_n/∂r` — our BEC definition and
the textbook one (derivative of the dipole w.r.t. position) are the same object.
Unit check: 1/(A ε₀) = **2.19839 V per e·Å** for razor's cell, exactly the
constant razor's README already quotes for `∂F/∂q = 2.19839 × bec_z`.

**Two consequences.** The energy term gives the model a directional handle it
structurally lacks, recovering Helmholtz from autograd rather than fitting it.
And `Z* = ∂μ_n/∂r` is a **first** derivative of a forward-pass output, so BEC
*training* needs only second-order autodiff — the same order force training
already uses — instead of the third order that supervising `∂²E/∂r∂q` requires.

**Decisions made:** build and validate on **razor first with `bec_z` supervision**
(the only dataset with labels that constrain μ), then port to cpmace. Keep μ
**charge-independent**, so the new term is strictly linear in q and FiLM continues
to supply the rest of the charge response including the q² capacitive part.

## Honest assessment of the chances

**Moderate, and much better supervised than not.** Two things temper it:

1. **This adds no expressive power.** FiLM can already represent arbitrary
   Q-dependence. The gain would come purely from a better inductive bias and
   easier optimisation, not from a function the model previously could not fit.
2. **Unsupervised, μ is unconstrained.** Fitted only through its effect on E,
   nothing forces μ_n to be the physical dipole — the model can learn whatever
   value of μ_n minimises the loss. That is why razor-with-`bec_z` comes first:
   supervision turns an inductive bias into a constraint.

### PHASE 0 (do this first): LOREM `max_degree_lr: 1` — one config line, no code

Reading LOREM's source changed the ordering of this plan.

`mlip.py:252-258` builds the long-range charges as
`scalar_charges` (monopoles) **concatenated with** `spherical_charges =
TensorDense(max_degree=max_degree_lr)(nodes_spherical)`. With
`max_degree_lr ≥ 1` those are atomic **dipoles**, fed into
`jaxpme.batched_mixed.Ewald` — a *mixed 2D/3D* lattice sum, i.e. the
slab-appropriate one.

The Ewald energy of a charged slab carrying atomic dipoles contains the
monopole–dipole cross term ∝ Q·μ_z/A. **That is the Helmholtz term.** So the
long-range head already subsumes the energy term this plan proposes, as one
contribution to a correct lattice sum — with no explicit `slab_normal_axis`, no
area computation, and no origin convention for the monopole piece. Every
difficulty catalogued below is handled by construction there.

`max_degree_lr` **defaults to 2** (`mlip.py:30`), but every run in this project
set it to **0** — monopole-only — including all three `razor_centre_paper`
model.yaml files. So `lr`'s 29% work-function win over `sr` was achieved
*without* dipoles, and the dipole hypothesis has never actually been tested.

**Therefore: first run `razor_centre_paper` with `lr: true, max_degree_lr: 1`
against the existing `lr` (`max_degree_lr: 0`) baseline.** One line in
model.yaml, no code, and it answers the physics question directly. Only if that
helps is it worth building anything in GRACE — and then the right GRACE move is
to wire up `extra/gen_tensor/ewald.py` (already accepts per-atom dipoles of shape
`(n_atoms, 3)`, exactly what an l=1 head emits, currently imported by nothing)
rather than the hand-built term below.

The hand-built `SlabDipoleEnergy` is retained in this plan as the **fallback**:
simpler, self-contained, and appropriate if wiring a full Ewald into GRACE proves
too invasive.

### A discrepancy worth fixing independently

**LOREM applies FiLM *early*; our GRACE implementation applies it pre-readout.**
`mlip.py:134` runs `ChargeEmbedding` immediately after the first `Update`, before
the message-passing layers, so the charge propagates through the whole network
*and* into the long-range charges at line 252. GRACE's `FiLMChargeScalar` sits on
the readout branches and reaches nothing else.

That was the correct choice for GRACE-2L in isolation — `I_out_0` is a readout
branch, not the layer-2 input — but it means the "directly comparable to LOREM"
claim made when the FiLM work was planned does **not** hold. Worth either fixing
(condition the equivariant layer-2 message, which needs γ per-(channel,l) and β
restricted to l=0) or documenting.

### Relation to LOREM's long-range head

LOREM's Ewald term already does **part** of this, and the numbers say so: on
`razor_centre_paper`, `lr` reached Φ = 0.0875 V against `sr`'s 0.1239 — a 29%
work-function improvement from adding electrostatics alone. An Ewald sum over
learned partial charges contains a surface-dipole contribution, so it reaches the
same physics by a different route.

But two things make this still worth doing:

- **`sr-bec` (0.0845 V) beat `lr` (0.0875 V).** Directional supervision edged out
  the Ewald head on razor, so the two routes are not equivalent and the explicit
  dipole is not strictly dominated.
- **GRACE has no long-range term at all.** The FiLM presets are purely
  short-range; `extra/gen_tensor/ewald.py` exists but is wired into nothing. So
  for GRACE this is partly a *substitute* for the missing `lr` head, which is the
  single largest architectural gap against LOREM's best configuration.

For LOREM the honest expectation is that this would help `sr` and add little to
`lr`. For GRACE it has more room, because there is nothing there yet.

### Why `Σ qᵢ rᵢ` is deliberately omitted (and the diagnostic for it)

A full dipole is `Σ qᵢ rᵢ + Σ pᵢ`; this plan uses **`Σ pᵢ` only**. The monopole
term is not omitted for convenience — it is genuinely ill-posed here:

- `Σ qᵢ rᵢ` needs absolute positions. `ATOMIC_POS` / `PositionsDataBuilder` do
  exist, so it is *available*, but for a **charged** cell (`Σ qᵢ = Q ≠ 0`) the sum
  shifts by `Q·d` under a translation `d`. μ itself is then origin-dependent, so
  the energy term `q·μ_n/(A ε₀)` would be ill-defined without fixing a convention.
- `Σ pᵢ` alone is translation-invariant by construction, since each `pᵢ` depends
  only on a local environment.

Worth knowing that the **derivative** is better behaved than μ:
`∂μ_n/∂rᵢ = qᵢ n̂ + Σⱼ(∂qⱼ/∂rᵢ)rⱼ + …`, and the origin-dependent piece cancels
because `Σⱼ ∂qⱼ/∂rᵢ = 0` when total charge is conserved. So `Z*` stays well-defined
even where μ is not — which is the usual acoustic-sum-rule situation.

That matters because `Z*ᵢ ≈ qᵢ` to leading order for a rigid ion: the monopole is
the *dominant* part of a Born effective charge. Omitting it means the model must
reproduce all of `Z*` through environment-dependent atomic dipoles, which is
harder. **The diagnostic is concrete:** if the fitted `Z*` is systematically off
by roughly a per-species constant times `n̂`, that is the missing monopole, and the
fix is a scalar charge head with a `Σ qᵢ = Q` constraint plus an explicit origin
convention. Check this before concluding the approach failed.

Rough expectation for the **hand-built GRACE term**: ~50-60% of a measurable
work-function improvement on razor with `bec_z` supervision (where the 32% result
says directional information helps); ~30% on cpmace unsupervised.

But **Phase 0 below supersedes this as the first thing to run**: LOREM's existing
long-range head already contains the same physics and has never been tested with
dipoles enabled, so it tests the hypothesis for one config line instead of a
package of new code. Do that before writing anything.

## Implementation

**On a new branch `dipole-term`, cut from `charge-conditioning`.** The FiLM work is
finished, tested and feeding live experiments (runs 1-4 on cpmace), so this stays
off it until it earns a merge. Note the cluster checkout at `~/grace-tensorpotential`
is an **editable install** on `charge-conditioning`: switching branches there
changes what any queued or running job imports. Either wait for the queue to drain
or use a second checkout for dipole runs.

All new code in `tensorpotential/extra/charge/`, reusing the existing equivariant
machinery rather than adding any.

### 1. The dipole head (preset wiring, no new class)

Uses existing instructions. `FunctionReduceParticular` (`compute.py:4571`), *not*
`FunctionReduceN` — it selects exactly one (l, p) block:

```python
dip = FunctionReduceParticular(
    instructions=instructions,      # the same product-basis list the energy uses
    name="dipole_l1", selected_l=1,
    selected_p=-1,                  # true polar vector; +1 would be a pseudovector
    n_out=1, is_central_atom_type_dependent=True,
    number_of_atom_types=num_elements)
out_dip = CreateOutputTarget(name=cc.PREDICT_ATOMIC_DIPOLE, l=1)
LinearOut2EquivarTarget(origin=[dip], l=1, target=out_dip, name="atomic_dipole")
```

`LinearOut2EquivarTarget.frwrd` does `tf.roll(tensor, shift=1, axis=2)`, which is
the **(y,z,x) → (x,y,z) conversion** — GRACE's internal l=1 order is real
spherical harmonics m = −1,0,+1. Reading `out_dip` therefore gives a **Cartesian**
`[n_atoms, 1, 3]`. Exposing the dipole as its own target (rather than contracting
`dip` directly) is deliberate: it makes μ available for evaluation, for `∂μ_n/∂r`,
and for supervision.

### 2. New instruction `SlabDipoleEnergy(TPOutputInstruction)`

In `extra/charge/instructions.py`, modelled on `TrainableShiftTarget`
(`output.py:866`) for the padding mask and on `FiLMChargeScalar` for the
structure→atom gather.

```python
input_tensor_spec = {
    constants.TOTAL_CHARGE:           {"shape": [None, 1], "dtype": "float"},
    constants.CELL_VECTORS:           {"shape": [None, 3, 3], "dtype": "float"},
    constants.ATOMS_TO_STRUCTURE_MAP: {"shape": [None], "dtype": "int"},
    constants.ATOMIC_MU_I:            {"shape": [None], "dtype": "int"},
    constants.N_ATOMS_BATCH_REAL:     {"shape": [], "dtype": "int"},
}
def __init__(self, dipole, target, slab_normal_axis, name=...)
```

`frwrd` adds, **per atom**:

```
contrib_i = q_s · (p_i · n̂_s) / (A_s ε₀)          shape [n_atoms, 1]
```

with `n̂_s` the normalised `cell[:, slab_normal_axis, :]` and `A_s` the
perpendicular face area (reuse the `FiLMChargeScalar` `per_area` cross-product,
`instructions.py:143`).

**Distributing per atom is the point, not an implementation detail.** Σᵢ contrib_i
= q·μ_n/(A ε₀) exactly. Adding the structure-level value to every atom instead
would multiply it by N.

Mask padded atoms with the `r_map < N_ATOMS_BATCH_REAL` pattern, and declare the
instruction **after** `ConstantScaleShiftTarget`/`TrainableShiftTarget`, since the
term is in physical eV — the ZBL pattern (`presets.py:518`).

### 3. New presets

`GRACE_2LAYER_FILM_DIPOLE` in `extra/charge/model.py`, a copy of
`GRACE_2LAYER_FILM` with the head and term added and `slab_normal_axis` exposed.
FiLM stays exactly as it is. Writing the energy out in full — `LinMLPOut2ScalarTarget`
computes `E_i = ρ_0 + MLP(ρ_{1:})`, so the **linear ACE passthrough** is a separate
term and must not be folded into the MLP:

```
E_i = (1+γ₀(Q))·ρ_{i,0} + β₀(Q)          <- linear ACE, charge-modulated
    + MLP( (1+γ(Q))⊙ρ_{i,1:} + β(Q) )    <- nonlinear readout
    + Q·(p_i·n̂)/(A ε₀)                   <- NEW: Helmholtz / dipole
```

(2L sums both readout branches, `I_out_0_film` and `I_1_film`, into the same
`lin_origin` and `transformed_origin` before the single MLP.)

Written this way it is clear the three q-dependent terms are doing **different**
jobs, which is the argument that the new one is not redundant:

- `β₀(Q)` summed over atoms gives `N·β₀(Q)` — a size-extensive, purely
  charge-dependent offset. This is the model's existing route to the **capacitive
  q²/2C** energy, and it is geometry-independent.
- `γ₀(Q)·ρ_{i,0}` and the MLP branch make the *bonding* charge-dependent, but only
  through **invariant** descriptors — no direction.
- `Q·(p_i·n̂)/(A ε₀)` is the only term carrying a **direction**, and the only one
  whose q-derivative is a geometric quantity rather than a fitted function of Q.

So `dE/dQ` picks up the explicit Helmholtz term on top of what FiLM learns, and
the new term supplies the one thing the others structurally cannot.

### 4. `Z*` output and loss (razor phase)

`ComputeBatchEnergyForcesChargeDipole` in `extra/charge/model.py`: after the
existing tape, one extra reverse sweep of the scalar `μ_n` w.r.t. `BOND_VECTOR`,
assembled into per-atom `∂μ_n/∂r` the same way `total_f` is. Emit as
`cc.PREDICT_DF_DQ`. Add `WeightedWorkFunctionLoss`-style loss against razor's
`bec_z`, per-atom `[n_atoms, 3]` like the force loss.

Keep the `(A ε₀)` conversion **out of the graph** — the existing docstrings record
why (one conversion site, next to the data; an inferred in-plane area caused a
silent 2.54× error in natcomm2025).

### 5. Plumbing

- `extra/charge/constants.py`: `PREDICT_ATOMIC_DIPOLE`, `PREDICT_MU_N`.
- **`CellDataBuilder` is not exported** from `extra/extra_data_builders.py` (only
  `ReferenceTensorDataBuilder` and `TotalChargeDataBuilder` are). Add it, or
  `data: extra_components:` cannot reference it. The ASE calculator already
  auto-appends it when `CELL_VECTORS` is in `compute_specs`
  (`asecalculator.py:946`).
- Existing razor `prepare.py` already writes `bec_z`; no data work needed there.

## Verification

1. **Rotation equivariance.** Rotate positions *and* cell together: E, μ_n and Φ
   must all be invariant. Rotating positions alone must change μ_n. There is no
   existing equivariance test pinning the l=1 axis convention — this adds one.
2. **The axis is right.** Deliberately set `slab_normal_axis` wrong and confirm the
   answer changes materially. Guards the 2.54× class of bug.
3. **Helmholtz.** Check `dE/dq` contains `μ_n/(A ε₀)`. With FiLM's gate zeroed,
   γ = β = 0 exactly, so *all* of FiLM's q-dependence vanishes — including the
   `β₀(Q)` linear-channel offset — and Φ must equal `μ_n/(A ε₀)` alone, to float
   tolerance. `1/(A ε₀)` must come out at 2.19839 V/(e·Å) on razor's cell. This
   isolation only works because the gate is zero-initialised; do it before any
   training step.
4. **The two routes to Z\* agree** — the key check, and nearly free:
   `∂μ_n/∂r` from the dipole head vs `(A ε₀)·∂F/∂q` from forward-over-reverse.
   They must match; disagreement means the dipole head is an unconstrained
   auxiliary output rather than the physical dipole, which is exactly the failure
   mode this design is trying to avoid.
5. **Monopole diagnostic.** If `Z*` is systematically off by ~a per-species
   constant times `n̂`, that is the missing `Σ qᵢ rᵢ` term, not a failure of the
   approach — see the section above before abandoning it.
6. **Backwards compatibility.** No dipole instruction in the graph ⇒ no
   `CELL_VECTORS` in `compute_specs`, existing models untouched. Re-run
   `tests/test_charge_conditioning.py` unchanged.
7. **End to end, razor.** Train on `razor_centre_paper` against the three existing
   runs (`sr` 0.1239 V, `lr` 0.0875, `sr-bec` **0.0845**; Z\* 0.0398 / 0.0345 /
   **0.0193**). The bar is `sr-bec`. Note the 3-point stencil labels carry
   ~0.024 e of their own error, so do not chase Z\* below that.
8. **Then cpmace.** Port and compare against runs 1-4 (best so far: forces 21.47
   meV/Å, WF 33.4 mV, both from run 1).

## Notes

- Do the queued `WeightedSSEWorkFunctionLoss` → `WeightedWorkFunctionLoss` rename
  first (recorded in `experiments/cpmace-grace/README.md`), so the new loss lands
  with consistent naming.
- `ewald.py` in `extra/gen_tensor/` already has a reciprocal-space energy taking
  per-atom dipoles of shape `(n_atoms, 3)` — the exact tensor this head produces.
  It is wired into nothing. Out of scope here, but it is the natural next step if
  an explicit long-range term is ever wanted.
