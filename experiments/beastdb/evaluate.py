"""Energy error, grouped by (binned) total charge, for the beastdb sr vs
lr variants. beastdb's tot_charge is continuous and
mostly non-integer (grand-canonical DFT, self-consistent response to an
applied potential), unlike ag_clusters/omol_10K's integer charge states, so
structures are grouped into charge bins rather than exact values. Loads each
trained checkpoint directly (model + params + per-species baseline) and
evaluates in batches. beastdb ships no held-out test split (only train/val),
so this evaluates on data/valid, same set reported on during training.

Run from experiments/beastdb/ after both variants have been trained:

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
CHECKPOINT = "R2_E"
VALID_FILE = "../../datasets/beastdb/beastdb_val.xyz"

COLORS = {"sr": "steelblue", "lr": "tomato"}

BATCH_SIZE = 32

# None = full validation set (2107 structures x 2 models); set to an int for
# a quick smoke test before committing to the full evaluation.
SAMPLE_SIZE = None

# charge bins: (label, lower bound exclusive, upper bound inclusive); q==0
# (the fixed-charge, non-GC-DFT subset) gets its own bin, checked first
BIN_EDGES = [
    ("q<-2", -np.inf, -2.0),
    ("-2..-0.5", -2.0, -0.5),
    ("-0.5..0", -0.5, 0.0),
    ("0..0.5", 0.0, 0.5),
    ("0.5..2", 0.5, 2.0),
    ("q>2", 2.0, np.inf),
]


def charge_bin(q):
    if q == 0.0:
        return "q=0 (fixed)"
    for label, lo, hi in BIN_EDGES:
        if lo < q <= hi:
            return label
    return "q=0 (fixed)"  # unreachable given BIN_EDGES spans (-inf, inf)


BIN_ORDER = ["q<-2", "-2..-0.5", "-0.5..0", "q=0 (fixed)", "0..0.5", "0.5..2", "q>2"]

# beastdb has no forces; ToSample/ToBatch default to a properties schema that
# includes forces, so it must be overridden explicitly here to match what
# prepare.py declared, or batching fails looking for a "forces" label that
# was never populated.
PROPERTIES = {
    "energy": {"shape": (1,), "storage": "atoms.calc"},
    "total_charge": {"shape": (1,), "storage": "atoms.info"},
}


def load_valid_atoms(n=None):
    frames = read(VALID_FILE, index=":")
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
    # sort by atom count so consecutive batches share the same padded shape
    frames = sorted(frames, key=len)

    to_sample = model.to_sample(cutoff=model.cutoff, keys=["energy"], properties=PROPERTIES)
    batcher = model.to_batch(
        batch_size=BATCH_SIZE,
        keys=("energy",),
        coarse_strategy="powers_of_4",
        size_strategy="powers_of_4",
        fine_strategy="powers_of_4",
        drop_remainder=False,
        properties=PROPERTIES,
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
        n_real = int(np.array(batch.sr.structure_mask).sum())

        batch_frames = frames[cursor : cursor + n_real]
        cursor += n_real

        for local_i, atoms in enumerate(batch_frames):
            offset = sum(species_weights[int(Z)] for Z in atoms.get_atomic_numbers())
            e_pred_i = float(e_pred[local_i]) + offset

            e_ref = atoms.get_potential_energy()
            n_atoms = len(atoms)
            q = atoms.info["total_charge"]

            e_err = (e_pred_i - e_ref) / n_atoms * 1000  # meV/atom

            rows.append(
                {
                    "tot_charge": float(q),
                    "charge_bin": charge_bin(float(q)),
                    "e_abs_err": float(abs(e_err)),
                    "e_sq_err": float(e_err**2),
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
            "n": len(rs),
        }
    return result


def print_table(agg):
    print("\nEnergy error by charge bin")
    print(f"{'variant':<10}{'charge bin':<14}{'E MAE':>10}{'E RMSE':>10}{'n':>8}")
    for v in VARIANTS:
        for b in BIN_ORDER:
            if (v, b) not in agg:
                continue
            m = agg[(v, b)]
            print(f"{v:<10}{b:<14}{m['e_mae']:>10.3f}{m['e_rmse']:>10.3f}{m['n']:>8d}")


@mpltex.acs_decorator
def plot_bars(agg, name="error_by_charge_bin.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine
    bins_present = [b for b in BIN_ORDER if any((v, b) in agg for v in VARIANTS)]
    x = np.arange(len(bins_present))
    width = 0.8 / len(VARIANTS)

    fig, ax = plt.subplots()
    for i, v in enumerate(VARIANTS):
        offset = (i - (len(VARIANTS) - 1) / 2) * width
        e_vals = [agg.get((v, b), {}).get("e_mae", np.nan) for b in bins_present]
        ax.bar(x + offset, e_vals, width, label=v, color=COLORS[v])

    ax.set_xticks(x)
    ax.set_xticklabels(bins_present, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Energy MAE (meV/atom)")
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    fig.tight_layout()
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
        print("loading validation set...")
        frames = load_valid_atoms(SAMPLE_SIZE)
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

    agg = aggregate(all_rows, ["variant", "charge_bin"])
    print_table(agg)
    plot_bars(agg)


if __name__ == "__main__":
    main()
