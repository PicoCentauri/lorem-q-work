"""Exp 2 -- matched field-free/field-on pairs, same geometry. Reads
water_external_field_paired_{free,perturbed}.xyz directly (they bypass
prepare.py/marathon entirely -- these files use dft_energy/dft_forces/
dft_hirshfeld keys, never went through to_labels, and have no atoms.calc),
lines them up 1:1 by index, and compares each model's predicted
Delta E = E(field-on) - E(field-free) against the DFT reference.

Run from experiments/water_external_field/ after all variants have been
trained:

    DATASETS=. python evaluate_paired.py
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

VARIANTS = ["sr-none", "sr-l1", "sr-l1_l0"]  # lr-* stale post dipole-field fix, retraining
CHECKPOINT = "R2_E+F"
FREE_FILE = "../../datasets/water_external_field/water_external_field_paired_free.xyz"
PERTURBED_FILE = "../../datasets/water_external_field/water_external_field_paired_perturbed.xyz"
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


def load_pairs():
    free = read(FREE_FILE, index=":")
    perturbed = read(PERTURBED_FILE, index=":")
    assert len(free) == len(perturbed), "paired files must line up 1:1"

    for atoms in free:
        atoms.info["external_field"] = [0.0, 0.0, 0.0]
    for atoms in perturbed:
        atoms.info["external_field"] = atoms.info.pop("ext_field")

    return free, perturbed


def predict_energies(model, params, species_weights, frames):
    # sort by atom count so consecutive batches share the same padded shape
    order = sorted(range(len(frames)), key=lambda i: len(frames[i]))
    sorted_frames = [frames[i] for i in order]

    # keys=[] / keys=() -- these files carry no atoms.calc, so skip the
    # standard energy/forces label extraction entirely; we only need the
    # model's forward pass, and compare against dft_energy ourselves below.
    to_sample = model.to_sample(cutoff=model.cutoff, keys=[])
    batcher = model.to_batch(
        batch_size=BATCH_SIZE,
        keys=(),
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


def aggregate_deltas(dft_free, dft_perturbed, model_free, model_perturbed):
    dft_delta = np.array(dft_perturbed) - np.array(dft_free)
    model_delta = np.array(model_perturbed) - np.array(model_free)
    err = (model_delta - dft_delta) * 1000  # meV
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "n": len(err),
    }


def print_table(agg):
    print("\nExp 2 -- matched field-free/field-on pairs (Delta E)")
    print(f"{'variant':<12}{'MAE (meV)':>12}{'RMSE (meV)':>12}{'n':>8}")
    for v in VARIANTS:
        m = agg[v]
        print(f"{v:<12}{m['mae']:>12.3f}{m['rmse']:>12.3f}{m['n']:>8d}")


@mpltex.acs_decorator
def plot_bar(agg, name="exp2_paired_delta_energy.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine
    fig, ax = plt.subplots()
    x = np.arange(len(VARIANTS))
    vals = [agg[v]["mae"] for v in VARIANTS]
    colors = [COLORS[v] for v in VARIANTS]
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(VARIANTS, rotation=45, ha="right")
    ax.set_ylabel("Delta E MAE (meV)")
    ax.set_title("Exp 2 -- paired field-free/field-on Delta E")
    fig.align_labels()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    print(f"saved figures/{name}")


ROWS_CACHE = Path("evaluate_paired_energies.json")


def main():
    print("loading paired files...")
    free, perturbed = load_pairs()
    print(f"{len(free)} pairs")

    dft_free = [a.info["dft_energy"] for a in free]
    dft_perturbed = [a.info["dft_energy"] for a in perturbed]

    if ROWS_CACHE.exists():
        print(f"loading cached energies from {ROWS_CACHE}")
        with open(ROWS_CACHE) as f:
            all_energies = json.load(f)
    else:
        all_energies = {}
        for v in VARIANTS:
            print(f"--- {v} ---", flush=True)
            model, params, species_weights = load_checkpoint(v)
            e_free = predict_energies(model, params, species_weights, free)
            e_perturbed = predict_energies(model, params, species_weights, perturbed)
            all_energies[v] = {"free": e_free, "perturbed": e_perturbed}
            jax.clear_caches()

        with open(ROWS_CACHE, "w") as f:
            json.dump(all_energies, f)
        print(f"cached energies to {ROWS_CACHE}")

    agg = {}
    for v in VARIANTS:
        agg[v] = aggregate_deltas(
            dft_free, dft_perturbed, all_energies[v]["free"], all_energies[v]["perturbed"]
        )

    print_table(agg)
    plot_bar(agg)


if __name__ == "__main__":
    main()
