"""Energy/force error, grouped by charge state, for the ag_clusters sr vs lr
variants, plus sr-energyonly (loss_weights={"energy": 1.0}, no force
supervision) as a reference-quality check -- the published Ag3+/Ag3-
errors (Ko, Finkler, Goedecker & Behler, Nat. Commun. 12, 398 (2021)) may
be from an energy-only fit, so this isolates whether joint E+F training is
what's holding sr/lr back from that reference accuracy. Loads each trained
checkpoint directly (model + params + per-species baseline) and evaluates
in batches on the held-out test set.

Run from experiments/ag_clusters/ after all variants have been trained:

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

VARIANTS = ["sr", "lr", "sr-energyonly"]
TEST_FILE = "../../datasets/ag_clusters/Ag_clusters_test.xyz"
CHARGE_STATES = [-1.0, 1.0]
CHARGE_LABELS = {-1.0: "Ag3-", 1.0: "Ag3+"}

# sr/lr train on energy+forces, so their combined-metric checkpoint is
# named "R2_E+F"; sr-energyonly trains on energy only
# (loss_weights={"energy": 1.0}), so its best checkpoint is "R2_E" instead.
CHECKPOINTS = {
    "sr": "R2_E+F",
    "lr": "R2_E+F",
    "sr-energyonly": "R2_E",
}

COLORS = {"sr": "steelblue", "lr": "tomato", "sr-energyonly": "seagreen"}

BATCH_SIZE = 64

# None = full test set (1101 structures x 2 models); set to an int for a
# quick smoke test before committing to the full evaluation.
SAMPLE_SIZE = None


def load_test_atoms(n=None):
    frames = read(TEST_FILE, index=":")
    if n:
        frames = frames[:n]
    for atoms in frames:
        atoms.info["total_charge"] = float(atoms.info.pop("tot_charge"))
    return frames


def load_checkpoint(variant):
    folder = Path(variant) / "run" / "checkpoints" / CHECKPOINTS[variant] / "model"
    model = from_dict(read_yaml(folder / "model.yaml"))
    params = read_msgpack(folder / "model.msgpack")
    species_weights = read_yaml(folder / "baseline.yaml")["elemental"]
    return model, params, species_weights


def evaluate_variant(model, params, species_weights, frames):
    # sort by atom count so consecutive batches share the same padded shape
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

            e_err = (e_pred_i - e_ref) / n_atoms * 1000  # meV/atom
            f_err = (f_pred_i - f_ref) * 1000  # meV/A

            rows.append(
                {
                    "tot_charge": float(q),
                    "e_abs_err": float(abs(e_err)),
                    "e_sq_err": float(e_err**2),
                    "f_abs_err": float(np.mean(np.abs(f_err))),
                    "f_sq_err": float(np.mean(f_err**2)),
                }
            )
        if (i + 1) % 5 == 0:
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


def print_table(agg):
    print("\nEnergy/force error by charge state")
    print(f"{'variant':<10}{'charge':<8}{'E MAE':>10}{'E RMSE':>10}{'F MAE':>10}{'F RMSE':>10}{'n':>8}")
    for v in VARIANTS:
        for c in CHARGE_STATES:
            if (v, c) not in agg:
                continue
            m = agg[(v, c)]
            print(
                f"{v:<10}{c:<8.0f}{m['e_mae']:>10.3f}{m['e_rmse']:>10.3f}"
                f"{m['f_mae']:>10.3f}{m['f_rmse']:>10.3f}{m['n']:>8d}"
            )


@mpltex.acs_decorator
def plot_bars(agg, name="error_by_charge_state.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine
    x = np.arange(len(CHARGE_STATES))
    width = 0.8 / len(VARIANTS)

    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(1.8 * fig.get_figwidth())
    for i, v in enumerate(VARIANTS):
        offset = (i - (len(VARIANTS) - 1) / 2) * width
        e_vals = [agg[(v, c)]["e_mae"] for c in CHARGE_STATES]
        f_vals = [agg[(v, c)]["f_mae"] for c in CHARGE_STATES]
        axes[0].bar(x + offset, e_vals, width, label=v, color=COLORS[v])
        axes[1].bar(x + offset, f_vals, width, label=v, color=COLORS[v])

    titles = ["energy", "forces"]
    ylabels = ["Energy MAE (meV/atom)", "Force MAE (meV/A)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xticks(x)
        ax.set_xticklabels([CHARGE_LABELS[c] for c in CHARGE_STATES])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")
    axes[0].legend(fontsize=6)
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

    agg = aggregate(all_rows, ["variant", "tot_charge"])
    print_table(agg)
    plot_bars(agg)


if __name__ == "__main__":
    main()
