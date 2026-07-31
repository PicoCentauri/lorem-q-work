"""Exp 3 -- single-cluster dissociation curves (Vondrak et al. Fig. 2 style).

17 individual clusters (4 neutral, 4 cation, 9 anion), each scaled through
0.9x-5.0x (16 points), grouped by the `series` field. For each series,
plots relative energy per atom (referenced to that series' own 1.0x point)
vs. scaling factor: DFT as points, each trained variant as a line.

Run from experiments/water_variational_charge/ after both variants have
been trained:

    DATASETS=. python evaluate_dissociation.py
"""

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
DATA_FILE = "../../datasets/water_variational_charge/water_variational_charge_dissociation_curves.xyz"
BATCH_SIZE = 64

COLORS = {"sr": "steelblue", "lr": "tomato"}
CHARGE_ORDER = {-1.0: 0, 0.0: 1, 1.0: 2}


def load_checkpoint(variant):
    folder = Path(variant) / "run" / "checkpoints" / CHECKPOINT / "model"
    model = from_dict(read_yaml(folder / "model.yaml"))
    params = read_msgpack(folder / "model.msgpack")
    species_weights = read_yaml(folder / "baseline.yaml")["elemental"]
    return model, params, species_weights


def load_frames():
    frames = read(DATA_FILE, index=":")
    for atoms in frames:
        atoms.info["total_charge"] = float(atoms.info["tot_charge"])
    return frames


def predict_energies(model, params, species_weights, frames):
    # sort by atom count so consecutive batches share the same padded shape
    order = sorted(range(len(frames)), key=lambda i: len(frames[i]))
    sorted_frames = [frames[i] for i in order]

    to_sample = model.to_sample(cutoff=model.cutoff, keys=["energy", "forces"])
    batcher = model.to_batch(
        batch_size=BATCH_SIZE,
        coarse_strategy="powers_of_4",
        size_strategy="powers_of_4",
        fine_strategy="powers_of_4",
        drop_remainder=False,
    )

    def it():
        for i, atoms in enumerate(sorted_frames):
            yield Record(
                data=to_sample.map(atoms), metadata=RecordMetadata(index=i, record_key=i)
            )

    predict_fn = jax.jit(model.predict)

    energies = [None] * len(frames)
    cursor = 0
    for record in batcher(it()):
        batch = record.data
        preds = predict_fn(params, batch)
        e_pred = np.array(preds["energy"])
        n_real = int(np.array(batch.sr.structure_mask).sum())
        batch_frames = sorted_frames[cursor : cursor + n_real]
        for local_i, atoms in enumerate(batch_frames):
            offset = sum(species_weights[int(Z)] for Z in atoms.get_atomic_numbers())
            energies[order[cursor + local_i]] = float(e_pred[local_i]) + offset
        cursor += n_real
    return energies


@mpltex.acs_decorator
def plot_curves(frames, series_indices, dft_energy, all_model_energies, name="exp3_dissociation_curves.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine

    def sort_key(s):
        c = frames[series_indices[s][0]].info["tot_charge"]
        return (CHARGE_ORDER[float(c)], s)

    series_names = sorted(series_indices.keys(), key=sort_key)

    n_cols = 5
    n_rows = -(-len(series_names) // n_cols)  # ceil div
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True)
    fig.set_figwidth(2.6 * fig.get_figwidth())
    fig.set_figheight(0.9 * n_rows * fig.get_figheight())
    axes_flat = axes.flatten()

    for ax_i, s in enumerate(series_names):
        ax = axes_flat[ax_i]
        idxs = series_indices[s]
        exp_coefs = [frames[i].info["exp_coef"] for i in idxs]
        n_atoms = len(frames[idxs[0]])
        ref_i = idxs[exp_coefs.index(1.0)]

        dft_rel = [(dft_energy[i] - dft_energy[ref_i]) / n_atoms * 1000 for i in idxs]
        ax.scatter(exp_coefs, dft_rel, color="black", s=8, zorder=5, label="DFT")

        for v in VARIANTS:
            e = all_model_energies[v]
            model_rel = [(e[i] - e[ref_i]) / n_atoms * 1000 for i in idxs]
            ax.plot(exp_coefs, model_rel, color=COLORS[v], label=v)

        c = int(frames[idxs[0]].info["tot_charge"])
        ax.set_title(f"{s} (Q={c:+d})", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("scaling factor", fontsize=7)

    for ax_i in range(len(series_names), len(axes_flat)):
        axes_flat[ax_i].axis("off")

    for row in range(n_rows):
        axes[row, 0].set_ylabel("rel. E (meV/atom)", fontsize=7)

    axes_flat[0].legend(fontsize=5, loc="best")
    fig.align_labels()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    print(f"saved figures/{name}")


def main():
    print("loading dissociation curve data...")
    frames = load_frames()
    print(f"{len(frames)} structures")

    series_indices = defaultdict(list)
    for i, atoms in enumerate(frames):
        series_indices[atoms.info["series"]].append(i)
    for s in series_indices:
        series_indices[s].sort(key=lambda i: frames[i].info["exp_coef"])

    dft_energy = [f.get_potential_energy() for f in frames]

    all_model_energies = {}
    for v in VARIANTS:
        print(f"--- {v} ---", flush=True)
        model, params, species_weights = load_checkpoint(v)
        all_model_energies[v] = predict_energies(model, params, species_weights, frames)
        jax.clear_caches()

    plot_curves(frames, series_indices, dft_energy, all_model_energies)


if __name__ == "__main__":
    main()
