"""Exp 3 -- field-magnitude sweep (Vondrak et al. Fig. 5 style).

50 neutral clusters, each held at a fixed geometry and swept through 21
field magnitudes from -0.2 to 0.2 V/A along a fixed direction (z-axis).
Grouped by the `series` field, sorted by `field_magnitude`. For each series,
plots relative energy per atom (referenced to that series' own
field_magnitude == 0.0 point) vs. field magnitude: DFT as points, each
trained variant as a line. QEq predicts an exact parabola; the paper finds
QEq-based MLIPs get the shape wrong.

Also plots the mean per-atom force projected along the field direction (with
error bars = std across atoms) vs. field magnitude -- a second, independent
check of the same field response, directly on forces rather than on energy
differences.

Structural clone of evaluate_dissociation.py in ../water_variational_charge/.

Run from experiments/water_external_field/ after all variants have been
trained:

    DATASETS=. python evaluate_magnitude_sweep.py
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

VARIANTS = ["sr-none", "sr-l1", "sr-l1_l0"]  # lr-* stale post dipole-field fix, retraining
CHECKPOINT = "R2_E+F"
DATA_FILE = "../../datasets/water_external_field/water_external_field_magnitude_sweep.xyz"
BATCH_SIZE = 64

COLORS = {
    "sr-none": "steelblue",
    "sr-l1": "cornflowerblue",
    "sr-l1_l0": "navy",
    "lr-none": "tomato",
    "lr-l1": "salmon",
    "lr-l1_l0": "firebrick",
}


def load_checkpoint(variant):
    folder = Path(variant) / "run" / "checkpoints" / CHECKPOINT / "model"
    model = from_dict(read_yaml(folder / "model.yaml"))
    params = read_msgpack(folder / "model.msgpack")
    species_weights = read_yaml(folder / "baseline.yaml")["elemental"]
    return model, params, species_weights


def load_frames():
    frames = read(DATA_FILE, index=":")
    for atoms in frames:
        atoms.info["total_charge"] = float(atoms.info.pop("tot_charge"))
        atoms.info["external_field"] = atoms.info.pop("ext_field")
    return frames


def predict_energies_and_forces(model, params, species_weights, frames):
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
    forces = [None] * len(frames)
    cursor = 0
    for record in batcher(it()):
        batch = record.data
        preds = predict_fn(params, batch)
        e_pred = np.array(preds["energy"])
        f_pred = np.array(preds["forces"])
        atom_to_structure = np.array(batch.sr.atom_to_structure)
        n_real = int(np.array(batch.sr.structure_mask).sum())
        batch_frames = sorted_frames[cursor : cursor + n_real]
        for local_i, atoms in enumerate(batch_frames):
            offset = sum(species_weights[int(Z)] for Z in atoms.get_atomic_numbers())
            global_i = order[cursor + local_i]
            energies[global_i] = float(e_pred[local_i]) + offset
            forces[global_i] = f_pred[atom_to_structure == local_i]
        cursor += n_real
    return energies, forces


def field_direction(frames, idxs):
    # fixed direction per series -- use the frame with the largest field
    # magnitude, normalized (falls back to z-axis if all-zero)
    i_max = max(idxs, key=lambda i: abs(frames[i].info["field_magnitude"]))
    field = np.asarray(frames[i_max].info["external_field"], dtype=float)
    norm = np.linalg.norm(field)
    return field / norm if norm > 0 else np.array([0.0, 0.0, 1.0])


def force_along_field(forces_i, direction):
    # forces_i: [n_atoms, 3] -> (mean, std) of the per-atom force component
    # along `direction`, in meV/A, across atoms
    proj = forces_i @ direction * 1000
    return float(np.mean(proj)), float(np.std(proj))


@mpltex.acs_decorator
def plot_curves(frames, series_indices, dft_energy, all_model_energies, name="exp3_magnitude_sweep.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine

    series_names = sorted(series_indices.keys())

    n_cols = 5
    n_rows = -(-len(series_names) // n_cols)  # ceil div
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True)
    fig.set_figwidth(2.6 * fig.get_figwidth())
    fig.set_figheight(0.9 * n_rows * fig.get_figheight())
    axes_flat = axes.flatten()

    for ax_i, s in enumerate(series_names):
        ax = axes_flat[ax_i]
        idxs = series_indices[s]
        field_mags = [frames[i].info["field_magnitude"] for i in idxs]
        n_atoms = len(frames[idxs[0]])
        ref_i = idxs[field_mags.index(min(field_mags, key=abs))]

        dft_rel = [(dft_energy[i] - dft_energy[ref_i]) / n_atoms * 1000 for i in idxs]
        ax.scatter(field_mags, dft_rel, color="black", s=8, zorder=5, label="DFT")

        for v in VARIANTS:
            e = all_model_energies[v]
            model_rel = [(e[i] - e[ref_i]) / n_atoms * 1000 for i in idxs]
            ax.plot(field_mags, model_rel, color=COLORS[v], label=v, linewidth=1)

        ax.set_title(s, fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("field magnitude (V/A)", fontsize=7)

    for ax_i in range(len(series_names), len(axes_flat)):
        axes_flat[ax_i].axis("off")

    for row in range(n_rows):
        axes[row, 0].set_ylabel("rel. E (meV/atom)", fontsize=7)

    axes_flat[0].legend(fontsize=5, loc="best")
    fig.align_labels()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    print(f"saved figures/{name}")


@mpltex.acs_decorator
def plot_force_curves(
    frames, series_indices, dft_forces, all_model_forces, name="exp3_force_along_field.pdf"
):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine

    series_names = sorted(series_indices.keys())

    n_cols = 5
    n_rows = -(-len(series_names) // n_cols)  # ceil div
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True)
    fig.set_figwidth(2.6 * fig.get_figwidth())
    fig.set_figheight(0.9 * n_rows * fig.get_figheight())
    axes_flat = axes.flatten()

    for ax_i, s in enumerate(series_names):
        ax = axes_flat[ax_i]
        idxs = series_indices[s]
        direction = field_direction(frames, idxs)
        field_mags = [frames[i].info["field_magnitude"] for i in idxs]

        dft_mean, dft_std = zip(*[force_along_field(dft_forces[i], direction) for i in idxs])
        ax.errorbar(
            field_mags,
            dft_mean,
            yerr=dft_std,
            color="black",
            marker="o",
            markersize=3,
            linewidth=0.8,
            capsize=2,
            zorder=5,
            label="DFT",
        )

        for v in VARIANTS:
            f = all_model_forces[v]
            model_mean, model_std = zip(*[force_along_field(f[i], direction) for i in idxs])
            ax.errorbar(
                field_mags,
                model_mean,
                yerr=model_std,
                color=COLORS[v],
                linewidth=1,
                capsize=1.5,
                alpha=0.8,
                label=v,
            )

        ax.set_title(s, fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("field magnitude (V/A)", fontsize=7)

    for ax_i in range(len(series_names), len(axes_flat)):
        axes_flat[ax_i].axis("off")

    for row in range(n_rows):
        axes[row, 0].set_ylabel("mean F . field_hat (meV/A)", fontsize=7)

    axes_flat[0].legend(fontsize=5, loc="best")
    fig.align_labels()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    print(f"saved figures/{name}")


def main():
    print("loading field-magnitude sweep data...")
    frames = load_frames()
    print(f"{len(frames)} structures")

    series_indices = defaultdict(list)
    for i, atoms in enumerate(frames):
        series_indices[atoms.info["series"]].append(i)
    for s in series_indices:
        series_indices[s].sort(key=lambda i: frames[i].info["field_magnitude"])

    dft_energy = [f.get_potential_energy() for f in frames]
    dft_forces = [f.get_forces() for f in frames]

    all_model_energies = {}
    all_model_forces = {}
    for v in VARIANTS:
        print(f"--- {v} ---", flush=True)
        model, params, species_weights = load_checkpoint(v)
        energies, forces = predict_energies_and_forces(model, params, species_weights, frames)
        all_model_energies[v] = energies
        all_model_forces[v] = forces
        jax.clear_caches()

    plot_curves(frames, series_indices, dft_energy, all_model_energies)
    plot_force_curves(frames, series_indices, dft_forces, all_model_forces)


if __name__ == "__main__":
    main()
