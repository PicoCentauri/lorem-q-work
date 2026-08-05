"""Parity plots + RMSE for the razor-centre `sr`/`sr-wf`/`sr-wf-bec` variants.

Three quantities per variant, on both the validation split and the
wide-charge-range test sweep:

- energy (eV/atom)
- forces (eV/A, all Cartesian components)
- work function (V) -- the model's autograd `dE/dq`, i.e. backprop through
  the forward pass with respect to the `total_charge` input, against the
  DFT `work_function` label. For `sr`/`lr` nothing trains on it, so it is a
  pure "did charge conditioning learn the right charge derivative?" probe;
  for `sr-wf` it is a fitted target and this panel is a training diagnostic
  instead. The per-species energy baseline is charge-independent and
  therefore drops out of `dE/dq` entirely -- no offset correction needed
  there, unlike for the energy parity.

The `dE/dq` here is computed by this script's own `jax.grad`, so it works
against checkpoints from either lorem version -- it does not depend on
`Lorem.predict` exposing `work_function`.

Reads the raw extxyz rather than `data/`, so it also gets `polarizable`
and `q_MD` for the split-out plots. Loads each trained checkpoint directly
(model + params + per-species baseline) instead of going through the
per-structure ASE Calculator, which triggers a fresh XLA compile per
distinct padded shape.

Run from experiments/razor/ once both variants have trained:

    DATASETS=. python evaluate.py
"""

import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpltex

import jax

from ase.io import read

from marathon.grain import Record, RecordMetadata
from marathon.io import from_dict, read_msgpack, read_yaml

DATA = "../../datasets/razor"
VARIANTS = ["sr", "sr-wf", "sr-wf-bec"]
# the summed-R2 checkpoint is named after the targets it covers -- "R2_E+F"
# for energy+forces, "R2_E+F+W" once the work function is trained on,
# "R2_E+F+W+B" with the Born effective charges on top. Resolve it per variant
# instead of hardcoding one name, which only ever matched the E+F runs.
CHECKPOINT_GLOB = "R2_*"

# (split name, xyz file, polarizable-only). "valid" mirrors what the model
# actually trained on; "test_sweep" is the full 13-point q in [-1.5, 1.5]
# sweep, only 13% of which is polarizable -- the extrapolation case, kept
# unfiltered on purpose (see prepare.py).
SPLITS = [
    ("valid", "razor_val", True),
    ("test_sweep", "razor_test", False),
]

COLORS = {"sr": "steelblue", "sr-wf": "seagreen", "sr-wf-bec": "rebeccapurple"}
POLARIZABLE_COLORS = {True: "steelblue", False: "darkorange"}

# 8, not razor's 32: sr-wf-bec's model carries predict_bec, and the
# jvp(jvp(energy)) path for d2E/dr dq holds far more live intermediates than a
# plain forward+backward. At 32 it reached 36.7 GB and OOM'd on an A40 (48 GB).
# Evaluation is not throughput-critical, so trade batch size for headroom.
BATCH_SIZE = 8

# Points drawn in the force parity scatter (RMSE is always over all of
# them). ~400k components for valid would make an unopenable PDF.
MAX_SCATTER_POINTS = 20000

# None = full split; set to an int for a quick smoke test.
SAMPLE_SIZE = None

ROWS_CACHE = Path("evaluate_rows.json")


def load_frames(name, polarizable_only, n=None):
    frames = read(f"{DATA}/{name}.xyz", index=":")
    for atoms in frames:
        atoms.info["total_charge"] = float(atoms.info.pop("bias_charge"))
    if polarizable_only:
        frames = [a for a in frames if a.info["polarizable"]]
    if n:
        frames = frames[:n]
    return frames


def load_checkpoint(variant):
    candidates = sorted(
        p
        for p in (Path(variant) / "run" / "checkpoints").glob(CHECKPOINT_GLOB)
        if p.is_dir() and not p.name.endswith(".backup")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"{variant}: expected exactly one {CHECKPOINT_GLOB} checkpoint, "
            f"found {[p.name for p in candidates]}"
        )
    folder = candidates[0] / "model"
    model = from_dict(read_yaml(folder / "model.yaml"))
    params = read_msgpack(folder / "model.msgpack")
    species_weights = read_yaml(folder / "baseline.yaml")["elemental"]
    return model, params, species_weights


def make_predict_fn(model):
    def energy_of_charge(params, batch, q):
        # model.energy returns (sum over the batch, per-atom energies); the
        # sum is over structures that don't interact, so d(sum)/dq_s is
        # exactly structure s's own dE/dq -- one backward pass gives the
        # whole batch's work functions.
        total, _ = model.energy(params, batch._replace(total_charge=q))
        return total

    @jax.jit
    def predict_fn(params, batch):
        preds = model.predict(params, batch)
        dEdq = jax.grad(energy_of_charge, argnums=2)(params, batch, batch.total_charge)
        return preds, dEdq

    return predict_fn


def evaluate_variant(model, params, species_weights, frames):
    to_sample = model.to_sample(cutoff=model.cutoff, keys=["energy", "forces"])
    batcher = model.to_batch(
        batch_size=BATCH_SIZE,
        coarse_strategy="powers_of_4",
        size_strategy="powers_of_4",
        fine_strategy="powers_of_4",
        drop_remainder=False,
    )

    def it():
        for i, atoms in enumerate(frames):
            yield Record(
                data=to_sample.map(atoms), metadata=RecordMetadata(index=i, record_key=i)
            )

    predict_fn = make_predict_fn(model)

    rows = []
    cursor = 0
    for i, record in enumerate(batcher(it())):
        batch = record.data
        preds, dEdq = predict_fn(params, batch)
        e_pred = np.array(preds["energy"])
        f_pred = np.array(preds["forces"])
        wf_pred = np.array(dEdq)
        atom_to_structure = np.array(batch.sr.atom_to_structure)
        n_real = int(np.array(batch.sr.structure_mask).sum())

        batch_frames = frames[cursor : cursor + n_real]
        cursor += n_real

        for local_i, atoms in enumerate(batch_frames):
            offset = sum(species_weights[int(Z)] for Z in atoms.get_atomic_numbers())
            n_atoms = len(atoms)

            e_pred_i = (float(e_pred[local_i]) + offset) / n_atoms
            e_ref_i = atoms.get_potential_energy() / n_atoms
            f_pred_i = f_pred[atom_to_structure == local_i]
            f_ref_i = atoms.get_forces()

            rows.append(
                {
                    "total_charge": float(atoms.info["total_charge"]),
                    "q_MD": float(atoms.info["q_MD"]),
                    "polarizable": bool(atoms.info["polarizable"]),
                    "e_pred": e_pred_i,
                    "e_ref": e_ref_i,
                    "f_pred": f_pred_i.ravel().tolist(),
                    "f_ref": f_ref_i.ravel().tolist(),
                    "wf_pred": float(wf_pred[local_i]),
                    "wf_ref": float(atoms.info["work_function"]),
                }
            )
        if (i + 1) % 10 == 0:
            print(f"  ...{cursor}/{len(frames)} structures", flush=True)
    return rows


# -- metrics --


def rmse(pred, ref):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(ref)) ** 2)))


def mae(pred, ref):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(ref))))


def collect(rows):
    """Flatten a row list into arrays, in the units used for reporting."""
    polarizable = np.array([r["polarizable"] for r in rows])
    return {
        "e_pred": np.array([r["e_pred"] for r in rows]) * 1000,  # meV/atom
        "e_ref": np.array([r["e_ref"] for r in rows]) * 1000,
        "f_pred": np.concatenate([r["f_pred"] for r in rows]) * 1000,  # meV/A
        "f_ref": np.concatenate([r["f_ref"] for r in rows]) * 1000,
        "wf_pred": np.array([r["wf_pred"] for r in rows]),  # V
        "wf_ref": np.array([r["wf_ref"] for r in rows]),
        # per-structure flag, plus the per-force-component broadcast of it
        "e_polarizable": polarizable,
        "wf_polarizable": polarizable,
        "f_polarizable": np.repeat(
            polarizable, [len(r["f_ref"]) for r in rows]
        ),
    }


def metrics(d):
    return {
        "e_rmse": rmse(d["e_pred"], d["e_ref"]),
        "e_mae": mae(d["e_pred"], d["e_ref"]),
        "f_rmse": rmse(d["f_pred"], d["f_ref"]),
        "f_mae": mae(d["f_pred"], d["f_ref"]),
        "wf_rmse": rmse(d["wf_pred"], d["wf_ref"]),
        "wf_mae": mae(d["wf_pred"], d["wf_ref"]),
        "n": len(d["e_pred"]),
    }


def print_table(all_rows):
    header = (
        f"{'split':<12}{'variant':<8}{'subset':<14}"
        f"{'E RMSE':>10}{'E MAE':>10}{'F RMSE':>10}{'F MAE':>10}"
        f"{'WF RMSE':>10}{'WF MAE':>10}{'n':>8}"
    )
    print("\nE in meV/atom, F in meV/A, WF (dE/dq) in V")
    print(header)
    print("-" * len(header))
    for split, _, _ in SPLITS:
        for v in VARIANTS:
            rows = [r for r in all_rows if r["split"] == split and r["variant"] == v]
            if not rows:
                continue
            subsets = [("all", rows)]
            pol = [r for r in rows if r["polarizable"]]
            nonpol = [r for r in rows if not r["polarizable"]]
            if pol and nonpol:
                subsets += [("polarizable", pol), ("non-polarizable", nonpol)]
            for label, rs in subsets:
                m = metrics(collect(rs))
                print(
                    f"{split:<12}{v:<8}{label:<14}"
                    f"{m['e_rmse']:>10.2f}{m['e_mae']:>10.2f}"
                    f"{m['f_rmse']:>10.2f}{m['f_mae']:>10.2f}"
                    f"{m['wf_rmse']:>10.4f}{m['wf_mae']:>10.4f}{m['n']:>8d}"
                )


# -- plots --


def _parity(ax, ref, pred, color, label, unit, polarizable=None, legend=False, seed=0):
    lo = min(ref.min(), pred.min())
    hi = max(ref.max(), pred.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lims = (lo - pad, hi + pad)

    if len(ref) > MAX_SCATTER_POINTS:
        idx = np.random.default_rng(seed).choice(
            len(ref), MAX_SCATTER_POINTS, replace=False
        )
    else:
        idx = np.arange(len(ref))

    ax.plot(lims, lims, "-", color="0.4", lw=0.6, zorder=1)
    if polarizable is None or len(set(polarizable.tolist())) < 2:
        ax.scatter(ref[idx], pred[idx], s=2, lw=0, color=color, zorder=2)
    else:
        # non-polarizable frames sit near dielectric breakdown / Fermi
        # pinning and were never trained on -- worth seeing separately
        for flag, c in POLARIZABLE_COLORS.items():
            m = polarizable[idx] == flag
            ax.scatter(
                ref[idx][m],
                pred[idx][m],
                s=2,
                lw=0,
                color=c,
                zorder=2,
                label="polarizable" if flag else "non-polarizable",
            )
        if legend:
            ax.legend(fontsize=5, markerscale=3, loc="lower right", framealpha=0.9)

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel(f"DFT {label} ({unit})")
    ax.set_ylabel(f"LOREM {label} ({unit})")
    ax.text(
        0.04,
        0.96,
        f"RMSE = {rmse(pred, ref):.3g} {unit}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6,
    )


@mpltex.acs_decorator
def plot_split(all_rows, split, name):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine

    quantities = [
        ("e", "energy", "meV/atom"),
        ("f", "force", "meV/Å"),
        ("wf", "work function", "V"),
    ]

    fig, axes = plt.subplots(
        len(VARIANTS), len(quantities), squeeze=False, constrained_layout=True
    )
    fig.set_figwidth(2.6 * fig.get_figwidth())
    fig.set_figheight(1.0 * len(VARIANTS) * fig.get_figwidth() / len(quantities))

    for row, v in enumerate(VARIANTS):
        rows = [r for r in all_rows if r["split"] == split and r["variant"] == v]
        if not rows:
            continue
        d = collect(rows)
        for col, (key, label, unit) in enumerate(quantities):
            ax = axes[row][col]
            _parity(
                ax,
                d[f"{key}_ref"],
                d[f"{key}_pred"],
                COLORS.get(v, "grey"),
                label,
                unit,
                polarizable=d[f"{key}_polarizable"],
                legend=(row == 0 and col == 0),
            )
        # variant name once per row, outside the axes, so it can't collide
        # with the row above's x-label
        axes[row][0].annotate(
            v,
            xy=(-0.42, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    fig.suptitle(f"razor -- {split}")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


def main():
    if ROWS_CACHE.exists():
        print(f"loading cached rows from {ROWS_CACHE}")
        with open(ROWS_CACHE) as f:
            all_rows = json.load(f)
    else:
        all_rows = []
        for split, xyz, polarizable_only in SPLITS:
            frames = load_frames(xyz, polarizable_only, SAMPLE_SIZE)
            print(f"=== {split}: {len(frames)} structures ===", flush=True)
            for v in VARIANTS:
                print(f"--- {v} ---", flush=True)
                model, params, species_weights = load_checkpoint(v)
                rows = evaluate_variant(model, params, species_weights, frames)
                for r in rows:
                    r["variant"] = v
                    r["split"] = split
                all_rows.extend(rows)
                jax.clear_caches()

        with open(ROWS_CACHE, "w") as f:
            json.dump(all_rows, f)
        print(f"cached rows to {ROWS_CACHE}")

    print_table(all_rows)
    for split, _, _ in SPLITS:
        plot_split(all_rows, split, f"parity_{split}.pdf")


if __name__ == "__main__":
    main()
