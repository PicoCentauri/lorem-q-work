"""Exp 1 (mixed charge states near equilibrium) and Exp 2 (dissociation,
binned by exp_coef) for the water_variational_charge sr vs lr variants.
Loads each trained checkpoint directly (model + params + per-species
baseline) and evaluates in batches -- not via the per-structure ASE Calculator
interface, which triggers a fresh XLA compile per distinct padded shape and
blows up GPU memory across ~15k structures of varying size.

Run from experiments/water_variational_charge/ after both variants have
been trained:

    DATASETS=. python evaluate.py
"""

import json
from collections import defaultdict
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

VARIANTS = ["sr", "lr"]
CHECKPOINT = "R2_E+F"
TEST_FILE = "../../datasets/water_variational_charge/water_variational_charge_test.xyz"
EXP_COEF_BINS = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
CHARGE_STATES = [-1.0, 0.0, 1.0]
CHARGE_LABELS = {-1.0: "Q=-1", 0.0: "Q=0", 1.0: "Q=+1"}

COLORS = {"sr": "steelblue", "lr": "tomato"}

BATCH_SIZE = 64

# None = full test set (15,000 structures x 2 models); set to an int for a
# quick smoke test before committing to the full evaluation.
SAMPLE_SIZE = None


def nearest_bin(x, bins=EXP_COEF_BINS):
    return float(bins[np.argmin(np.abs(bins - x))])


def load_test_atoms(n=None):
    frames = read(TEST_FILE, index=":")
    if n:
        frames = frames[:n]
    for atoms in frames:
        atoms.info["total_charge"] = float(atoms.info.pop("tot_charge"))
    return frames


def load_checkpoint(variant):
    folder = Path(variant) / "run" / "checkpoints" / CHECKPOINT / "model"
    model = from_dict(read_yaml(folder / "model.yaml"))
    params = read_msgpack(folder / "model.msgpack")
    species_weights = read_yaml(folder / "baseline.yaml")["elemental"]
    return model, params, species_weights


def evaluate_variant(model, params, species_weights, frames):
    # sort by atom count so consecutive batches share the same padded shape --
    # otherwise batch-to-batch size swings (this dataset spans ~7-30+ atoms,
    # unsorted in the raw file) force a fresh XLA compile almost every batch,
    # and JAX never releases those, blowing up GPU memory across ~15k structures
    frames = sorted(frames, key=len)

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

    predict_fn = jax.jit(model.predict)

    rows = []
    cursor = 0
    for i, record in enumerate(batcher(it())):
        batch = record.data
        preds = predict_fn(params, batch)
        e_pred = np.array(preds["energy"])
        f_pred = np.array(preds["forces"])
        atom_to_structure = np.array(batch.sr.atom_to_structure)
        n_real = int(np.array(batch.sr.structure_mask).sum())

        batch_frames = frames[cursor : cursor + n_real]
        cursor += n_real

        for local_i, atoms in enumerate(batch_frames):
            offset = sum(species_weights[int(Z)] for Z in atoms.get_atomic_numbers())
            e_pred_i = float(e_pred[local_i]) + offset
            f_pred_i = f_pred[atom_to_structure == local_i]

            e_ref = atoms.get_potential_energy()
            f_ref = atoms.get_forces()
            n_atoms = len(atoms)
            q = atoms.info["total_charge"]
            exp_coef = nearest_bin(atoms.info["exp_coef"])

            e_err = (e_pred_i - e_ref) / n_atoms * 1000  # meV/atom
            f_err = (f_pred_i - f_ref) * 1000  # meV/A

            rows.append(
                {
                    "tot_charge": float(q),
                    "exp_coef": float(exp_coef),
                    "e_abs_err": float(abs(e_err)),
                    "e_sq_err": float(e_err**2),
                    "f_abs_err": float(np.mean(np.abs(f_err))),
                    "f_sq_err": float(np.mean(f_err**2)),
                }
            )
        if (i + 1) % 20 == 0:
            print(f"  ...{cursor}/{len(frames)} structures", flush=True)
    return rows


def aggregate(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)

    result = {}
    for key, rs in groups.items():
        result[key] = {
            "e_mae": np.mean([r["e_abs_err"] for r in rs]),
            "e_rmse": np.sqrt(np.mean([r["e_sq_err"] for r in rs])),
            "f_mae": np.mean([r["f_abs_err"] for r in rs]),
            "f_rmse": np.sqrt(np.mean([r["f_sq_err"] for r in rs])),
            "n": len(rs),
        }
    return result


def print_exp1_table(agg1):
    print("\nExp 1 -- mixed charge states near equilibrium")
    print(f"{'variant':<12}{'charge':<8}{'E MAE':>10}{'E RMSE':>10}{'F MAE':>10}{'F RMSE':>10}{'n':>8}")
    for v in VARIANTS:
        for c in CHARGE_STATES:
            if (v, c) not in agg1:
                continue
            m = agg1[(v, c)]
            print(
                f"{v:<12}{c:<8.0f}{m['e_mae']:>10.3f}{m['e_rmse']:>10.3f}"
                f"{m['f_mae']:>10.3f}{m['f_rmse']:>10.3f}{m['n']:>8d}"
            )


@mpltex.acs_decorator
def plot_exp1(agg1, name="exp1_charge_states.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine
    x = np.arange(len(CHARGE_STATES))
    width = 0.35

    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(1.8 * fig.get_figwidth())
    for i, v in enumerate(VARIANTS):
        offset = (i - 0.5) * width
        e_vals = [agg1[(v, c)]["e_mae"] for c in CHARGE_STATES]
        f_vals = [agg1[(v, c)]["f_mae"] for c in CHARGE_STATES]
        axes[0].bar(x + offset, e_vals, width, label=v, color=COLORS[v])
        axes[1].bar(x + offset, f_vals, width, label=v, color=COLORS[v])

    titles = ["Exp 1 -- energy", "Exp 1 -- forces"]
    ylabels = ["Energy MAE (meV/atom)", "Force MAE (meV/A)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xticks(x)
        ax.set_xticklabels([CHARGE_LABELS[c] for c in CHARGE_STATES])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(fontsize=6)
    fig.align_labels()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    print(f"saved figures/{name}")


@mpltex.acs_decorator
def plot_exp2(agg2, name="exp2_dissociation.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine
    fig, axes = plt.subplots(2, 3, sharex=True)
    fig.set_figwidth(2.2 * fig.get_figwidth())
    fig.set_figheight(1.6 * fig.get_figheight())

    for col, c in enumerate(CHARGE_STATES):
        for v in VARIANTS:
            xs, e_ys, f_ys = [], [], []
            for b in EXP_COEF_BINS:
                key = (v, c, b)
                if key in agg2:
                    xs.append(b)
                    e_ys.append(agg2[key]["e_mae"])
                    f_ys.append(agg2[key]["f_mae"])
            axes[0, col].plot(xs, e_ys, color=COLORS[v], marker="o", label=v)
            axes[1, col].plot(xs, f_ys, color=COLORS[v], marker="o", label=v)
        axes[0, col].set_title(CHARGE_LABELS[c])
        axes[1, col].set_xlabel("expansion coefficient")

    axes[0, 0].set_ylabel("Energy MAE (meV/atom)")
    axes[1, 0].set_ylabel("Force MAE (meV/A)")
    for ax in axes.flat:
        ax.set_yscale("log")
    axes[0, -1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.align_labels()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    print(f"saved figures/{name}")


ROWS_CACHE = Path("evaluate_rows.json")


def main():
    if ROWS_CACHE.exists():
        print(f"loading cached rows from {ROWS_CACHE}")
        with open(ROWS_CACHE) as f:
            all_rows = json.load(f)
    else:
        print("loading test set...")
        frames = load_test_atoms(SAMPLE_SIZE)
        print(f"evaluating {len(frames)} structures x {len(VARIANTS)} models...")

        all_rows = []
        for v in VARIANTS:
            print(f"--- {v} ---", flush=True)
            model, params, species_weights = load_checkpoint(v)
            rows = evaluate_variant(model, params, species_weights, frames)
            for r in rows:
                r["variant"] = v
            all_rows.extend(rows)
            jax.clear_caches()

        with open(ROWS_CACHE, "w") as f:
            json.dump(all_rows, f)
        print(f"cached rows to {ROWS_CACHE}")

    agg1 = aggregate(all_rows, ["variant", "tot_charge"])
    print_exp1_table(agg1)
    plot_exp1(agg1)

    agg2 = aggregate(all_rows, ["variant", "tot_charge", "exp_coef"])
    plot_exp2(agg2)


if __name__ == "__main__":
    main()
