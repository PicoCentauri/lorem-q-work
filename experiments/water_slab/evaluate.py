"""Simple RMSE/MAE check on the water_slab test set, grouped by the 7
discrete applied field values (0, +-0.1, +-0.2, +-0.3 V/A along z), for the
6 variants (sr/lr x none/l1/l1_l0). Loads each trained checkpoint directly
(model + params + per-species baseline) and evaluates in batches -- not via
the per-structure ASE Calculator interface, which triggers a fresh XLA
compile per distinct padded shape.

No dedicated physical-test-case scripts for this dataset -- just the
standard bar-chart pattern used across the other experiments.

Run from experiments/water_slab/ after all variants have been trained:

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

VARIANTS = ["sr-none", "sr-l1", "sr-l1_l0", "lr-none", "lr-l1", "lr-l1_l0"]
CHECKPOINT = "R2_E+F"
TEST_FILE = "../../datasets/water_slab/water_slab_test.xyz"
FIELD_VALUES = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

COLORS = {
    "sr-none": "steelblue",
    "sr-l1": "cornflowerblue",
    "sr-l1_l0": "navy",
    "lr-none": "tomato",
    "lr-l1": "salmon",
    "lr-l1_l0": "firebrick",
}

BATCH_SIZE = 32

SAMPLE_SIZE = None


def nearest_field(atoms, bins=FIELD_VALUES):
    z = float(np.asarray(atoms.info["external_field"])[2])
    return min(bins, key=lambda b: abs(b - z))


def load_test_atoms(n=None):
    frames = read(TEST_FILE, index=":")
    if n:
        frames = frames[:n]
    for atoms in frames:
        atoms.info["total_charge"] = float(atoms.info.pop("tot_charge"))
        atoms.info["external_field"] = atoms.info.pop("ext_field")
    return frames


def load_checkpoint(variant):
    folder = Path(variant) / "run" / "checkpoints" / CHECKPOINT / "model"
    model = from_dict(read_yaml(folder / "model.yaml"))
    params = read_msgpack(folder / "model.msgpack")
    species_weights = read_yaml(folder / "baseline.yaml")["elemental"]
    return model, params, species_weights


def evaluate_variant(model, params, species_weights, frames):
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

            e_err = (e_pred_i - e_ref) / n_atoms * 1000  # meV/atom
            f_err = (f_pred_i - f_ref) * 1000  # meV/A

            rows.append(
                {
                    "field": nearest_field(atoms),
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


def print_table(agg):
    print("\nwater_slab -- E/F error by applied field (V/A)")
    print(f"{'variant':<12}{'field':<8}{'E MAE':>10}{'E RMSE':>10}{'F MAE':>10}{'F RMSE':>10}{'n':>8}")
    for v in VARIANTS:
        for f in FIELD_VALUES:
            if (v, f) not in agg:
                continue
            m = agg[(v, f)]
            print(
                f"{v:<12}{f:<8.1f}{m['e_mae']:>10.3f}{m['e_rmse']:>10.3f}"
                f"{m['f_mae']:>10.3f}{m['f_rmse']:>10.3f}{m['n']:>8d}"
            )


@mpltex.acs_decorator
def plot_bar(agg, name="water_slab_field_sweep.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine
    x = np.arange(len(FIELD_VALUES))
    width = 0.13

    fig, axes = plt.subplots(1, 2)
    fig.set_figwidth(1.8 * fig.get_figwidth())
    for i, v in enumerate(VARIANTS):
        offset = (i - (len(VARIANTS) - 1) / 2) * width
        e_vals = [agg[(v, f)]["e_mae"] for f in FIELD_VALUES]
        f_vals = [agg[(v, f)]["f_mae"] for f in FIELD_VALUES]
        axes[0].bar(x + offset, e_vals, width, label=v, color=COLORS[v])
        axes[1].bar(x + offset, f_vals, width, label=v, color=COLORS[v])

    titles = ["energy", "forces"]
    ylabels = ["Energy MAE (meV/atom)", "Force MAE (meV/A)"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f:+.1f}" for f in FIELD_VALUES])
        ax.set_xlabel("applied field (V/A)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(fontsize=5)
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

    agg = aggregate(all_rows, ["variant", "field"])
    print_table(agg)
    plot_bar(agg)


if __name__ == "__main__":
    main()
