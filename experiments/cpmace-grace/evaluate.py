"""Parity plots for the trained GRACE model, matching `../cpmace/`'s figures.

Same three quantities, same layout, same axis conventions as the LOREM figures
so the two can be put side by side. The plotting helpers are lifted from
`../cpmace/evaluate.py`; only the prediction backend differs -- a GRACE
SavedModel driven through `TPCalculator` instead of a marathon checkpoint.

Quantities:
  - energy (meV/atom) on the *total* energy, converted from the grand potential
    by `prepare.py` exactly as in `../cpmace/`
  - forces (meV/A, all Cartesian components)
  - work function (V), the model's autograd dE/dq against -potential

cpmace has no `polarizable` flag and no bec_z labels, so every panel is
single-coloured and there is no fourth column.

    python evaluate.py            # from experiments/cpmace-grace/
"""

import ast
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpltex

from ase.io import read

# The run to plot, and the row label. A list so a second run can be stacked
# the way ../cpmace/ stacks its variants; run 4 alone is the "best model"
# figure, which is what this was written for.
VARIANTS = ["square-control-w500-10-0.7"]
LABELS = {
    "square-control-w500-10-0.7": "GRACE-2L + FiLM\nsmall, square, 100k",
}
COLORS = {"square-control-w500-10-0.7": "steelblue"}

DATA = "data"
SPLIT = "valid"

# Points drawn in the force parity scatter; RMSE is always over all of them.
# ~68k components would make an unopenable PDF.
MAX_SCATTER_POINTS = 20000

ROWS_CACHE = Path("evaluate_rows.json")


# -- prediction ---------------------------------------------------------------


def atomic_shift(variant):
    """Per-element E0 map that `reference_energy: auto` subtracted from the labels.

    NOT a cosmetic correction. `reference_energy: auto` fits per-element E0 and
    trains the model on `energy - sum(E0)`, but the offset is applied to the
    DATA and is never baked into the graph -- the saved model here contains only
    a `TrainableShiftTarget`, no `ConstantScaleShiftTarget`. So **the saved model
    returns energies on the shifted scale, not raw DFT energies**, and anything
    comparing against raw labels (this script, MD, a published number) must add
    sum(E0) back. Skipping it here produced a 6441.659 meV/atom "error" that was
    entirely this offset -- identical RMSE and MAE, the signature of a constant.

    Parsed from the run's own log rather than hardcoded, so it stays correct if
    the model is retrained on different data.
    """
    log = Path(variant) / "seed" / "1" / "log.txt"
    marker = "single-atom energies (eV):"
    for line in log.read_text().splitlines():
        if marker in line:
            return ast.literal_eval(line.split(marker, 1)[1].strip())
    raise RuntimeError(
        f"no '{marker}' line in {log} -- was this run trained with "
        f"reference_energy: auto? If reference_energy was 0, return {{}} here."
    )


def predict(variant, frames):
    """Energy, forces and dE/dq for every frame, via the saved GRACE model."""
    from tensorpotential.calculator import TPCalculator
    from tensorpotential.extra.charge import constants as cc

    e0 = atomic_shift(variant)
    print(f"  per-element E0 (added back to predictions): {e0}")

    model_path = Path(variant) / "seed" / "1" / "final_model"
    calc = TPCalculator(model=str(model_path))

    rows = []
    for i, atoms in enumerate(frames):
        a = atoms.copy()
        a.calc = calc
        # ASE's check_state ignores atoms.info, so a structure differing from
        # its predecessor only in total_charge would silently return a cached
        # result. Every frame here has distinct positions, but reset anyway --
        # the failure mode is invisible if it ever does apply.
        calc.reset()
        # add back the per-element E0 the model was trained without
        e = a.get_potential_energy() + sum(e0[s] for s in a.get_chemical_symbols())
        f = a.get_forces()
        wf = float(
            calc.outputs[0][cc.PREDICT_WORK_FUNCTION].numpy().ravel()[0]
        )
        rows.append(
            dict(
                variant=variant,
                e_pred=e / len(a),
                e_ref=atoms.get_potential_energy() / len(a),
                f_pred=f.ravel().tolist(),
                f_ref=atoms.get_forces().ravel().tolist(),
                wf_pred=wf,
                wf_ref=float(atoms.info["work_function"]),
                q=float(atoms.info["total_charge"]),
            )
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(frames)}", flush=True)
    return rows


# -- metrics ------------------------------------------------------------------


def rmse(pred, ref):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(ref)) ** 2)))


def mae(pred, ref):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(ref))))


def collect(rows):
    """Flatten rows into the per-quantity arrays the plotters expect.

    Energies and work functions are per structure; forces are per component.
    Units match ../cpmace/: meV/atom, meV/A, V.
    """
    out = {
        "e_ref": np.array([r["e_ref"] for r in rows]) * 1000,
        "e_pred": np.array([r["e_pred"] for r in rows]) * 1000,
        "wf_ref": np.array([r["wf_ref"] for r in rows]),
        "wf_pred": np.array([r["wf_pred"] for r in rows]),
        "f_ref": np.concatenate([r["f_ref"] for r in rows]) * 1000,
        "f_pred": np.concatenate([r["f_pred"] for r in rows]) * 1000,
        "q": np.array([r["q"] for r in rows]),
    }
    # cpmace has no polarizable flag; the plotters accept None for it
    for k in ("e", "f", "wf"):
        out[f"{k}_polarizable"] = None
    return out


def print_table(all_rows):
    print("\nE in meV/atom, F in meV/A, WF (dE/dq) in V")
    print(f"{'variant':<34}{'E RMSE':>9}{'E MAE':>9}{'F RMSE':>9}{'F MAE':>9}"
          f"{'WF RMSE':>10}{'WF MAE':>9}{'n':>6}")
    print("-" * 95)
    for v in VARIANTS:
        rows = [r for r in all_rows if r["variant"] == v]
        if not rows:
            continue
        d = collect(rows)
        print(
            f"{v:<34}{rmse(d['e_pred'], d['e_ref']):>9.3f}"
            f"{mae(d['e_pred'], d['e_ref']):>9.3f}"
            f"{rmse(d['f_pred'], d['f_ref']):>9.2f}"
            f"{mae(d['f_pred'], d['f_ref']):>9.2f}"
            f"{rmse(d['wf_pred'], d['wf_ref']):>10.4f}"
            f"{mae(d['wf_pred'], d['wf_ref']):>9.4f}{len(rows):>6}"
        )


# -- plots (lifted from ../cpmace/evaluate.py so the figures match) ------------


def _parity(ax, ref, pred, color, label, unit, legend=False, seed=0, lims=None):
    if lims is None:
        lo, hi = min(ref.min(), pred.min()), max(ref.max(), pred.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lims = (lo - pad, hi + pad)

    if len(ref) > MAX_SCATTER_POINTS:
        idx = np.random.default_rng(seed).choice(
            len(ref), MAX_SCATTER_POINTS, replace=False
        )
    else:
        idx = np.arange(len(ref))

    ax.plot(lims, lims, "-", color="0.4", lw=0.6, zorder=1)
    ax.scatter(ref[idx], pred[idx], s=2, lw=0, color=color, zorder=2)

    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel(f"DFT {label} ({unit})")
    ax.set_ylabel(f"GRACE {label} ({unit})")
    ax.text(
        0.04, 0.96, f"RMSE = {rmse(pred, ref):.3g} {unit}",
        transform=ax.transAxes, va="top", ha="left", fontsize=6,
    )


CHARGE_BIN_WIDTH = 0.25
MIN_BIN_N = 20


@mpltex.acs_decorator
def plot_split(all_rows, name):
    plt.rcParams["text.usetex"] = False

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

    per_variant = {}
    for v in VARIANTS:
        rows = [r for r in all_rows if r["variant"] == v]
        if rows:
            per_variant[v] = collect(rows)

    # Axis anchored on the LABELS, with predictions widening it only via their
    # 1-99 percentile -- a few wild predictions clip instead of rescaling the
    # plot and squashing every real point into a dot. The RMSE annotation still
    # reports the true error, so the clipping hides nothing.
    col_lims = {}
    for key, _, _ in quantities:
        refs = [d[f"{key}_ref"] for d in per_variant.values()]
        preds = [d[f"{key}_pred"] for d in per_variant.values()]
        lo = min(float(np.min(a)) for a in refs)
        hi = max(float(np.max(a)) for a in refs)
        lo = min(lo, min(float(np.percentile(a, 1)) for a in preds))
        hi = max(hi, max(float(np.percentile(a, 99)) for a in preds))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        col_lims[key] = (lo - pad, hi + pad)

    for row, v in enumerate(VARIANTS):
        d = per_variant.get(v)
        if d is None:
            continue
        for col, (key, label, unit) in enumerate(quantities):
            _parity(
                axes[row][col], d[f"{key}_ref"], d[f"{key}_pred"],
                COLORS.get(v, "grey"), label, unit,
                legend=(row == 0 and col == 0), lims=col_lims.get(key),
            )
        axes[row][0].annotate(
            LABELS.get(v, v), xy=(-0.42, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center", fontsize=7,
            fontweight="bold", linespacing=1.4,
        )

    fig.suptitle(f"cpmace-grace -- {SPLIT}")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


@mpltex.acs_decorator
def plot_rmse_vs_charge(all_rows, name):
    """RMSE per target resolved by total charge.

    cpmace's charges are continuous (the electron count is a constant-potential
    DFT output, not a set point), so 0.25 e bins over the validation split's
    109 frames give ~4 usable bins. Bins under MIN_BIN_N are drawn hatched with
    their count rather than hidden -- dropping them would misrepresent the
    charge range actually tested.
    """
    plt.rcParams["text.usetex"] = False
    quantities = [("e", "energy", "meV/atom"), ("f", "force", "meV/Å"),
                  ("wf", "work function", "V")]

    q_all = np.array([r["q"] for r in all_rows])
    edges = np.arange(
        np.floor(q_all.min() / CHARGE_BIN_WIDTH) * CHARGE_BIN_WIDTH,
        q_all.max() + CHARGE_BIN_WIDTH, CHARGE_BIN_WIDTH,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, axes = plt.subplots(
        len(VARIANTS), len(quantities), squeeze=False, constrained_layout=True
    )
    fig.set_figwidth(2.6 * fig.get_figwidth())
    fig.set_figheight(1.0 * len(VARIANTS) * fig.get_figwidth() / len(quantities))

    for row, v in enumerate(VARIANTS):
        rows = [r for r in all_rows if r["variant"] == v]
        if not rows:
            continue
        for col, (key, label, unit) in enumerate(quantities):
            ax = axes[row][col]
            vals, ns = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sub = [r for r in rows if lo <= r["q"] < hi]
                ns.append(len(sub))
                if not sub:
                    vals.append(np.nan); continue
                d = collect(sub)
                vals.append(rmse(d[f"{key}_pred"], d[f"{key}_ref"]))
            for c, val, n in zip(centers, vals, ns):
                if np.isnan(val):
                    continue
                thin = n < MIN_BIN_N
                ax.bar(c, val, width=CHARGE_BIN_WIDTH * 0.85,
                       color=COLORS.get(v, "grey"), alpha=0.35 if thin else 1.0,
                       hatch="//" if thin else None, edgecolor="0.3", lw=0.4)
                if thin:
                    ax.text(c, val, f"n={n}", ha="center", va="bottom", fontsize=5)
            ax.set_xlabel("total charge (e)")
            ax.set_ylabel(f"{label} RMSE ({unit})")
        axes[row][0].annotate(
            LABELS.get(v, v), xy=(-0.42, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center", fontsize=7,
            fontweight="bold", linespacing=1.4,
        )

    hatched = f"hatched: n < {MIN_BIN_N}"
    fig.suptitle(f"cpmace-grace -- {SPLIT} -- RMSE vs charge ({hatched})")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


def main():
    if ROWS_CACHE.exists():
        print(f"loading cached rows from {ROWS_CACHE}")
        all_rows = json.loads(ROWS_CACHE.read_text())
    else:
        frames = read(f"{DATA}/{SPLIT}.xyz", index=":")
        print(f"=== {SPLIT}: {len(frames)} structures ===", flush=True)
        all_rows = []
        for v in VARIANTS:
            print(f"--- {v} ---", flush=True)
            all_rows.extend(predict(v, frames))
        ROWS_CACHE.write_text(json.dumps(all_rows))
        print(f"cached rows to {ROWS_CACHE}")

    print_table(all_rows)
    plot_split(all_rows, f"parity_{SPLIT}.pdf")
    plot_rmse_vs_charge(all_rows, f"rmse_vs_charge_{SPLIT}.pdf")


if __name__ == "__main__":
    main()
