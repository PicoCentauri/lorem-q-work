"""Training-dynamics comparison across field_conditioning modes (none/l1/l1_l0)
for water_external_field's sr variants, from each run's valid.txt log.

Run from experiments/water_external_field/ against local copies of the logs
(pulled via rsync from run/logs/valid.txt on the training machine, since the
figure itself needs no GPU/checkpoint access):

    python plot_learning_curves.py
"""

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["sr-none", "sr-l1", "sr-l1_l0"]
LOG_FILES = {v: f"{v}_valid.txt" for v in VARIANTS}

COLORS = {
    "sr-none": "steelblue",
    "sr-l1": "cornflowerblue",
    "sr-l1_l0": "navy",
}


# R2 can go negative (down to ~-800% early in training), which a log axis
# can't show directly -- plot 100 - R2 instead (distance from a perfect
# score, always >= 0) so it can share the log-log treatment with everything
# else. A tiny epsilon avoids log(0) if a checkpoint ever hits R2 == 100
# exactly.
_R2_EPS = 1e-6


def parse_valid_log(path):
    # lines look like:
    #    Step |      Loss |   E R2 |     E MAE |    E RMSE |   F R2 |     F MAE |    F RMSE
    # pipe-delimited with surrounding whitespace; skip the two header lines.
    rows = []
    with open(path) as f:
        lines = f.readlines()[2:]
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 8:
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    rows = np.array(rows)
    e_r2 = rows[:, 2]
    f_r2 = rows[:, 5]
    return {
        "step": rows[:, 0],
        "loss": rows[:, 1],
        "e_r2": e_r2,
        "e_r2_gap": 100.0 - e_r2 + _R2_EPS,
        "e_mae": rows[:, 3],
        "e_rmse": rows[:, 4],
        "f_r2": f_r2,
        "f_r2_gap": 100.0 - f_r2 + _R2_EPS,
        "f_mae": rows[:, 6],
        "f_rmse": rows[:, 7],
    }


def plot_learning_curves(logs, name="learning_curves.pdf"):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine

    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(2.2 * fig.get_figwidth())
    fig.set_figheight(1.3 * fig.get_figheight())

    panels = [
        ("loss", "training loss", axes[0, 0]),
        ("e_r2_gap", "100 - energy R2 (%)", axes[0, 1]),
        ("f_r2_gap", "100 - force R2 (%)", axes[0, 2]),
        ("e_mae", "energy MAE (meV/atom)", axes[1, 0]),
        ("f_mae", "force MAE (meV/A)", axes[1, 1]),
    ]

    for key, ylabel, ax in panels:
        for v in VARIANTS:
            if v not in logs:
                continue
            d = logs[v]
            y = d[key]
            ax.plot(d["step"], y, color=COLORS[v], linewidth=1, alpha=0.85, label=v)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[1, 2].axis("off")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.align_labels()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    print(f"saved figures/{name}")


def main():
    logs = {}
    for v in VARIANTS:
        path = Path(LOG_FILES[v])
        if not path.exists():
            print(f"skipping {v}: {path} not found")
            continue
        logs[v] = parse_valid_log(path)
        print(f"{v}: {len(logs[v]['step'])} validation checkpoints")

    plot_learning_curves(logs)


if __name__ == "__main__":
    main()
