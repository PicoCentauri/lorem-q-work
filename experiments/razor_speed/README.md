# razor inference speed

How fast is LOREM in an ASE MD loop, and where does the time actually go?

The motivating worry: with many MLIPs, ASE-driven MD is far slower than the
same model under LAMMPS, and the usual explanation is per-step CPU↔GPU
traffic — LAMMPS keeps the model resident on the device, ASE round-trips
positions and forces through the host every step. This experiment measures
whether that applies here, on the two models trained last in `../razor/`.

## What is measured

`benchmark.py` times the same model four ways per system size. Each layer adds
one source of overhead, so the differences localise the cost rather than just
reporting a single number:

| layer | what it adds | what it represents |
|---|---|---|
| `device_async` | nothing | pure device throughput. The jitted `predict_fn` on one fixed batch, dispatched asynchronously with a single `block_until_ready` at the end, so calls pipeline. **The floor** — roughly what a resident-on-GPU driver could reach. |
| `device_sync` | a `block_until_ready` per call | dispatch + synchronisation latency. **Not ASE's fault**: any step-by-step integrator needs the forces on the host before it can propose the next positions. |
| `calculator` | `Calculator.calculate` | H2D of positions/cell, D2H of energy/forces, `atoms.copy()`, and the per-step species-offset sum. |
| `md` | `ase` `VelocityVerlet` | the integrator, plus whatever neighbour-list rebuilds the dynamics triggers. |

`md / device_async` is the honest "what does the Python driver cost me" ratio.
`device_sync / device_async` is the part of that gap no MD driver can remove.

Sizes are in-plane supercells of the 108-atom razor slab — 108, 432, 972 atoms
(`pbc = T T F`, so only x and y repeat). Host overhead per step is roughly
constant while device work grows with N, **so the trend across sizes matters
more than any single number.** If the ratio falls steeply with N, the model is
launch-bound at small sizes and the fix is a bigger system, not a different
driver.

## The charge input does not transfer across cell sizes

Found while setting this up, and it matters well beyond the benchmark.
`mlip.py` builds the conditioning input as

```python
Q_i = Q[atom_to_structure] * atom_mask
```

— the **raw total charge, broadcast unchanged to every atom**. Not `q/A`, not
`q/N`. The FiLM MLP (`ChargeEmbedding`) therefore consumes an absolute number
whose meaning depends entirely on the cell it was trained in.

Every razor model has only ever seen 108-atom cells with q ∈ [−1.75, 1.75].
So **a model trained here cannot be applied to a different cell size at
matched surface charge density**: a 3×3 supercell at the same density carries
−4.5 e, 2.6× outside the trained range, and the conditioning MLP has no reason
to extrapolate sensibly there. The two requirements are in direct conflict —
physical charge density and in-distribution conditioning cannot both be
satisfied at a new cell size.

### Future work: condition on surface charge density, not total charge

The model should take **σ = q/A** (e/Å², A the in-plane cell area `|c₁ × c₂|`)
rather than q. This is not just a normalisation convenience — σ is the
physically meaningful variable for a charged slab, and this repo already
contains a numerical demonstration of that. From
`../razor_centre/README.md`, the Born-effective-charge relation

```
F = Z* q/(A ε₀)      i.e.   Z* = (A ε₀) ∂F/∂q
```

was checked against the data by regressing a finite-difference `∂F/∂q` on the
`bec_z` label over 200 structures:

```
∂F/∂q = 2.19839 * bec_z + (-2.3e-11)     corr = 1.00000
1/(A ε₀) = 2.19839  for A = 82.3108 Å²
```

**The forces depend on q through q/A, to six significant figures.** The area
is already computed inside the model (`_bec_z` takes `|c₁ × c₂|` from the
cell), so feeding σ instead of q costs nothing and would make the conditioning
cell-size transferable by construction — a 3×3 supercell at the same physical
state would present the *same* input, which is exactly the invariance wanted.

`q/N` would be the wrong choice: it makes the input depend on how much water
is in the cell, which the electrostatics does not care about.

This is an architecture change and needs a retrain, so it is future work
rather than something to patch in here. Worth raising with the lorem
developers before anyone runs large-cell MD with a charge-conditioned model.

### What the benchmark does instead

It holds `total_charge` at the source frame's −0.5 e for all sizes: the
conditioning stays in distribution and the dynamics stay stable, at the cost
of the supercells sitting at 1/4 and 1/9 the physical charge density. That is
acceptable *here only* because q affects neither tensor shapes nor FLOP
counts, so the timings are unaffected — and because out-of-distribution
conditioning would actively corrupt them, producing garbage forces that blow
the trajectory apart and force a neighbour-list rebuild every step.

It is not acceptable for anything but timing.

## Why the ASE calculator here is better placed than the usual case

Worth reading `lorem/calculator.py` before assuming the LAMMPS folklore
applies. Two things it already does that naive ASE calculators do not:

- **A Verlet skin cache** (`NeighborListCache`, `skin=0.25 Å`). The neighbour
  list is rebuilt only when accumulated displacement exceeds the skin, not
  every step.
- **In-place geometry updates** (`_update_geometry`). On a non-rebuild step
  only `positions` and `cell` are re-uploaded into the *existing* padded
  batch. The padded shapes never change, so there is no XLA recompile — which
  is the failure mode the `evaluate.py` docstring warns about for
  per-structure calculator use.

So the per-step host traffic is two small H2D arrays and two small D2H reads —
for 108 atoms, a few kB. That is unlikely to be the bottleneck; kernel-launch
and Python-interpreter overhead are the more plausible suspects, which is
exactly what the four-layer split separates.

`benchmark.py` counts neighbour-list rebuilds during both the calculator and
MD loops, since a rebuild is the one host-side operation that is *not* O(1)
per step: it re-runs `to_sample` and re-uploads the whole batch, and can force
a recompile if the padded pair count changes.

## If ASE does turn out to be the bottleneck

There is already an i-PI driver — `lorem/ipi.py`, installed with
`lorem-install-ipi-driver`. i-PI keeps the calculator process alive and talks
to it over a socket, so the model and its compiled kernels stay resident while
i-PI does the integration. That is the closest available analogue to the
LAMMPS arrangement and the natural next thing to measure, but it is only worth
the trouble if `md` sits well above `device_sync` — i.e. if the loss is in the
Python driver rather than in synchronisation.

## Layout

- `benchmark.py` — the measurement. Writes `results.json`.
- `srun.sh` — SLURM job, a100_80 to match the training hardware.

Two caveats on the numbers, both recorded because a timing result is
meaningless without them:

- `srun.sh` exports `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`. That is a
  correctness workaround — the Triton GEMM autotuner cannot identify the GPU
  on this jaxlib build and aborts — but it is also a performance knob, routing
  those fusions to cuBLAS. **Every number here is "with Triton GEMM off".**
  Worth revisiting if the jaxlib build is fixed.
- Timings are card-specific; the job prints `nvidia-smi` and `jax.devices()`.

## Results

Not yet run.
