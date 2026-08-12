"""How fast is LOREM in an ASE MD loop, and where does the time go?

The question this answers: with other MLIPs, ASE-driven MD is often far slower
than the same model in LAMMPS, and the usual explanation is per-step CPU<->GPU
traffic. Is that true here?

To separate the causes, each model/size is timed four ways, cheapest first:

1. `device_async`  -- the jitted predict_fn called in a loop on ONE fixed
   batch, with a single block_until_ready at the end. JAX dispatch is async,
   so this pipelines: it is the pure device throughput, the number a
   LAMMPS-style resident-on-GPU driver could approach.
2. `device_sync`   -- the same call blocking every iteration. The difference
   against (1) is dispatch + synchronisation latency, i.e. what you pay for
   needing the answer on the host before taking the next step. *No* MD
   integrator can avoid this, ASE or otherwise, unless the integrator itself
   runs on the device.
3. `calculator`    -- calc.calculate(atoms) per step with a displacement small
   enough to stay inside the neighbour-list skin. Adds the host-side work in
   Calculator.calculate: H2D of positions/cell, D2H of energy/forces, the
   atoms.copy(), and the per-step species-offset sum.
4. `md`            -- an actual ase VelocityVerlet run. Adds the integrator and
   whatever neighbour-list rebuilds the dynamics triggers.

(4)/(1) is the honest "how much am I losing to ASE" ratio. (2)/(1) is the part
that is inherent to step-by-step MD rather than ASE's fault.

The neighbour-list rebuild count is reported too, since a rebuild is the one
host-side operation that is not O(1) per step -- it re-runs to_sample and
re-uploads the whole batch, and can force an XLA recompile if the padded pair
count changes.

Run from experiments/razor_speed/:

    python benchmark.py
"""

import json
import time
from pathlib import Path

import numpy as np

import jax

from ase import units
from ase.io import iread
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from lorem.calculator import Calculator

# The two models trained last on razor/, same architecture (d64 l2 c8, 300
# epochs, 100:1:0.05) differing only in the lr flag -- so any timing gap
# between them is the Ewald head and nothing else.
MODELS = {
    "sr-l2c8": "../razor/sr-small-l2c8-e100-wf0.05-300ep",
    "lr-l2c8": "../razor/lr-small-l2c8-e100-wf0.05-300ep",
}
CHECKPOINT_GLOB = "R2_*"

# In-plane supercell repeats. razor is a 108-atom slab with pbc = T T F, so
# only x and y may be repeated. Host overhead per step is roughly constant
# while device work grows with N, so the ratio (4)/(1) should fall with size --
# that trend is the actual answer to "is ASE too slow", not any single number.
REPEATS = [(1, 1), (2, 2), (3, 3)]

STRUCTURE = "../../datasets/razor/razor_test.xyz"

N_WARMUP = 5
N_TIMED = 50
N_MD = 50
TIMESTEP_FS = 0.5
TEMPERATURE_K = 300.0

OUT = Path("results.json")


def load_checkpoint_dir(variant):
    candidates = sorted(
        p
        for p in (Path(variant) / "run" / "checkpoints").glob(CHECKPOINT_GLOB)
        if p.is_dir() and not p.name.endswith(".backup")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"{variant}: found {[p.name for p in candidates]}")
    return candidates[0]


def make_atoms(repeat):
    atoms = next(
        a for a in iread(STRUCTURE, index=":") if a.info.get("polarizable", True)
    )
    q = float(atoms.info.pop("bias_charge", 0.0))
    atoms = atoms.repeat((repeat[0], repeat[1], 1))
    # total_charge is deliberately NOT scaled by the number of cells, even
    # though that is what the physics says. mlip.py does
    #
    #     Q_i = Q[atom_to_structure] * atom_mask
    #
    # i.e. it broadcasts the raw total charge to every atom -- not q/N, not
    # q/A. The FiLM conditioning MLP therefore only ever saw the training
    # range q in [-1.75, 1.75] (108-atom cells). Feeding the physically
    # correct -4.5 e for a 3x3 supercell would put every atom 2.6x outside
    # that range, and the MLP has no reason to behave sensibly there.
    #
    # For a timing run that is not a cosmetic issue: out-of-distribution
    # conditioning gives garbage forces, the dynamics blow apart, and the
    # neighbour list then rebuilds every step -- which is precisely the
    # quantity being measured. Holding q fixed keeps the conditioning in
    # distribution and the trajectory stable. The cost is that the supercells
    # sit at 1/4 and 1/9 the surface charge density of the source frame,
    # which is physically odd but irrelevant to cost: q changes no shape and
    # no FLOP count.
    #
    # The real conclusion is in the README: this model is not size
    # transferable in its charge input at all.
    atoms.info["total_charge"] = q
    atoms.calc = None
    return atoms


def time_device(calc, block_every_step):
    """Pure predict_fn timing on one fixed batch, no host round trip."""
    params, batch = calc.params, calc.batch

    for _ in range(N_WARMUP):
        out = calc.predict_fn(params, batch)
    jax.block_until_ready(out)

    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        out = calc.predict_fn(params, batch)
        if block_every_step:
            jax.block_until_ready(out)
    if not block_every_step:
        jax.block_until_ready(out)
    return (time.perf_counter() - t0) / N_TIMED


def time_calculator(calc, atoms):
    """calc.calculate per step, displacements kept inside the skin so the
    neighbour list is never rebuilt -- isolates the per-step host cost."""
    atoms = atoms.copy()
    atoms.calc = calc
    rng = np.random.default_rng(0)
    # skin is 0.25 A; 1e-4 A per step over N_TIMED steps stays far inside it
    step = 1e-4

    for _ in range(N_WARMUP):
        atoms.positions += rng.normal(0, step, atoms.positions.shape)
        calc.calculate(atoms)

    rebuilds_before = calc._n_setup
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        atoms.positions += rng.normal(0, step, atoms.positions.shape)
        calc.calculate(atoms)
    dt = (time.perf_counter() - t0) / N_TIMED
    return dt, calc._n_setup - rebuilds_before


def time_md(calc, atoms):
    """A real ase MD loop: integrator + whatever rebuilds the dynamics needs."""
    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=TEMPERATURE_K, rng=np.random.default_rng(0)
    )
    dyn = VelocityVerlet(atoms, timestep=TIMESTEP_FS * units.fs)

    dyn.run(N_WARMUP)
    rebuilds_before = calc._n_setup
    t0 = time.perf_counter()
    dyn.run(N_MD)
    dt = (time.perf_counter() - t0) / N_MD
    return dt, calc._n_setup - rebuilds_before


def instrument(calc):
    """Count neighbour-list rebuilds without touching lorem itself."""
    calc._n_setup = 0
    original = calc.setup

    def counted(atoms):
        calc._n_setup += 1
        return original(atoms)

    calc.setup = counted
    return calc


def ns_per_day(ms_per_step, timestep_fs):
    steps_per_day = 86400.0 / (ms_per_step * 1e-3)
    return steps_per_day * timestep_fs * 1e-6


def main():
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    rows = []

    for name, variant in MODELS.items():
        folder = load_checkpoint_dir(variant)
        for repeat in REPEATS:
            atoms = make_atoms(repeat)
            n = len(atoms)
            print(f"\n=== {name}  {repeat[0]}x{repeat[1]}  n={n} ===", flush=True)

            calc = instrument(Calculator.from_checkpoint(folder))
            calc.calculate(atoms)  # triggers setup + the first XLA compile

            dev_async = time_device(calc, block_every_step=False)
            dev_sync = time_device(calc, block_every_step=True)
            calc_dt, calc_rebuilds = time_calculator(calc, atoms)
            md_dt, md_rebuilds = time_md(calc, atoms)

            row = {
                "model": name,
                "repeat": list(repeat),
                "n_atoms": n,
                "device_async_ms": dev_async * 1e3,
                "device_sync_ms": dev_sync * 1e3,
                "calculator_ms": calc_dt * 1e3,
                "md_ms": md_dt * 1e3,
                "calc_rebuilds": calc_rebuilds,
                "md_rebuilds": md_rebuilds,
                "md_steps": N_MD,
                "ns_per_day": ns_per_day(md_dt * 1e3, TIMESTEP_FS),
                "overhead_ratio": md_dt / dev_async,
            }
            rows.append(row)
            print(
                f"  device async {row['device_async_ms']:7.2f} ms/step\n"
                f"  device sync  {row['device_sync_ms']:7.2f} ms/step\n"
                f"  calculator   {row['calculator_ms']:7.2f} ms/step  "
                f"({calc_rebuilds} rebuilds)\n"
                f"  ase MD       {row['md_ms']:7.2f} ms/step  "
                f"({md_rebuilds} rebuilds in {N_MD} steps)\n"
                f"  -> {row['ns_per_day']:.3f} ns/day, "
                f"{row['overhead_ratio']:.2f}x the device floor",
                flush=True,
            )

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {OUT}")

    print(f"\n{'model':<10}{'n':>6}{'async':>9}{'sync':>9}{'calc':>9}{'MD':>9}"
          f"{'ns/day':>9}{'MD/async':>10}")
    for r in rows:
        print(
            f"{r['model']:<10}{r['n_atoms']:>6}{r['device_async_ms']:>9.2f}"
            f"{r['device_sync_ms']:>9.2f}{r['calculator_ms']:>9.2f}"
            f"{r['md_ms']:>9.2f}{r['ns_per_day']:>9.3f}{r['overhead_ratio']:>10.2f}"
        )


if __name__ == "__main__":
    main()
