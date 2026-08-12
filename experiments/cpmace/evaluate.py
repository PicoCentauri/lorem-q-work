"""Parity plots + RMSE for the cpmace sr/lr pair.

Three quantities per variant on the validation split:

- energy (meV/atom), on the *total* energy -- the raw `energy` field is the
  grand potential and is converted here exactly as in prepare.py. See
  datasets/cpmace/README.md.
- forces (meV/A, all Cartesian components)
- work function (V) -- the model's autograd dE/dq against -potential.

Derived from ../razor/evaluate.py, with razor-only machinery removed: cpmace
has no polarizable flag (all frames are treated as one subset), no bec_z
labels, and no charge-sweep test set.

Run from experiments/cpmace/ once both variants have trained:

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
import jax.numpy as jnp

from ase.io import read

from marathon.grain import Record, RecordMetadata
from marathon.io import from_dict, read_msgpack, read_yaml

# defined here rather than imported from lorem.models.mlip: this script is
# meant to run against checkpoints from either lorem version, and the older
# build (~/venv/lorem313, which razor/ trained on) has no EPSILON_0.
# vacuum permittivity in e^2 / (eV * Angstrom)
EPSILON_0 = 0.005526349406

DATA = "../../datasets/cpmace"

# Standard VASP PAW valence sum for C70 H89 N Ni O46. Must match
# prepare.py -- see datasets/cpmace/README.md, where it is confirmed against
# the thermodynamics rather than assumed.
NOMINAL_ELECTRONS = 660.0
# The model-size / training-length / range comparison, all at loss weights
# 100:1:0.05 with settings.yaml differing only in max_epochs -- so the only
# things varying across these six are the architecture, the schedule length
# and (for the last one) the Ewald head. The earlier weight-sweep runs (sr,
# lr, sr-wf, sr-wf0.05, sr-e100-wf0.1) are deliberately left out: they answer
# a question that is already settled, and mixing them in would put four
# different loss weights in one figure.
VARIANTS = [
    "sr-small-l2c8-300ep",
    "lr-small-l2c8-300ep",
]

# Row labels for the figure: the directory names encode the loss weights,
# which are constant here, so they are all prefix and no signal. Name the
# axis that actually varies instead.
LABELS = {
    "sr-small-l2c8-300ep": "d64 l2 c8 sr\n300 ep",
    "lr-small-l2c8-300ep": "d64 l2 c8 LR\n300 ep",
}
# the summed-R2 checkpoint is named after the targets it covers -- "R2_E+F"
# for energy+forces, "R2_E+F+W" once the work function is trained on,
# "R2_E+F+W+B" with the Born effective charges on top. Resolve it per variant
# instead of hardcoding one name, which only ever matched the E+F runs.
CHECKPOINT_GLOB = "R2_*"

# One split. cpmace ships a single file of 1093 frames; the 90/10 train/valid
# split is all the held-out data there is, so nothing here pretends to be a
# test set. The third element is the polarizable-only flag razor needs; cpmace
# has no such flag, so it is always False and every plot is single-coloured.
SPLITS = [
    ("valid", "cpmace_val", False),
]

COLORS = {
    "sr-small-l2c8-300ep": "steelblue",
    "lr-small-l2c8-300ep": "crimson",
}
POLARIZABLE_COLORS = {True: "steelblue", False: "darkorange"}

# 8, not 32: this script now computes bec_z via jvp(jvp(energy)) for every
# variant, which holds far more live intermediates than a plain
# forward+backward and OOM'd at 32 on an A40's 48 GB -- the same reason
# razor_centre/evaluate.py uses 8. Evaluation is not throughput-critical.
BATCH_SIZE = 8

# Points drawn in the force parity scatter (RMSE is always over all of
# them). ~400k components for valid would make an unopenable PDF.
MAX_SCATTER_POINTS = 20000

# None = full split; set to an int for a quick smoke test.
SAMPLE_SIZE = None

ROWS_CACHE = Path("evaluate_rows.json")


# Drop configurations whose DFT max force exceeds this, in eV/A. Not a
# cosmetic choice in general -- in ../razor/ one frame at 12.66 eV/A carried
# 10% of the force RMSE. Here it is a genuine no-op: cpmace's largest force is
# 5.65 eV/A, so nothing is dropped. Kept so the two folders screen identically
# and the number stays honest if the dataset is ever extended.
MAX_FORCE_CUTOFF = 10.0


def load_frames(name, polarizable_only, n=None):
    """Raw cpmace fields -> the (E, q, dE/dq) convention the model was trained
    on. Must match experiments/cpmace/prepare.py exactly; see
    datasets/cpmace/README.md for why `energy` needs converting at all.

    `polarizable_only` is ignored -- cpmace has no such flag. Every frame is
    marked polarizable=True so the polarizable/non-polarizable machinery
    inherited from razor collapses to a single subset rather than being
    special-cased throughout.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    frames = read(f"{DATA}/{name}.xyz", index=":")
    kept = []
    dropped = []
    for atoms in frames:
        n_excess = float(atoms.info["electron"]) - NOMINAL_ELECTRONS
        mu = float(atoms.info["potential"])
        forces = atoms.get_forces()
        total_energy = atoms.get_potential_energy() + mu * n_excess
        atoms.calc = SinglePointCalculator(atoms, energy=total_energy, forces=forces)
        atoms.info["total_charge"] = -n_excess
        atoms.info["work_function"] = -mu
        atoms.info["polarizable"] = True
        # cpmace carries no max_force label, unlike razor -- compute it from
        # the forces so the same screen can be applied and reported on
        mf = float(np.abs(forces).max())
        atoms.info["max_force"] = mf
        (dropped if mf > MAX_FORCE_CUTOFF else kept).append(atoms)
    if dropped:
        print(
            f"  {name}: dropped {len(dropped)}/{len(frames)} frames over "
            f"{MAX_FORCE_CUTOFF} eV/A",
            flush=True,
        )
    frames = kept
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
    model = from_dict(_drop_dead_keys(read_yaml(folder / "model.yaml")))
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

    def bec_of_batch(params, batch):
        # Z* = -(A eps0) d2E/(dr dq). Computed here rather than read off
        # model.predict, so it works on checkpoints trained without
        # predict_bec -- the whole point being to score variants that never
        # supervised it. Area from |c1 x c2|, so the varying out-of-plane
        # vector never enters (pbc = T T F).
        sr = batch[1]

        def dE_dpositions(total_charge):
            def energy_of_positions(positions):
                shifted = batch._replace(
                    sr=sr._replace(positions=positions), total_charge=total_charge
                )
                return model.energy(params, shifted)[0]

            return jax.grad(energy_of_positions)(sr.positions)

        _, d2E_drdq = jax.jvp(
            dE_dpositions,
            (batch.total_charge,),
            (jnp.ones_like(batch.total_charge),),
        )
        area = jnp.linalg.norm(jnp.cross(sr.cell[:, 0, :], sr.cell[:, 1, :]), axis=-1)
        scale = (area * EPSILON_0)[sr.atom_to_structure][:, None]
        return -d2E_drdq * scale

    @jax.jit
    def predict_fn(params, batch):
        preds = model.predict(params, batch)
        dEdq = jax.grad(energy_of_charge, argnums=2)(params, batch, batch.total_charge)
        bec = bec_of_batch(params, batch)
        return preds, dEdq, bec

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
        preds, dEdq, bec = predict_fn(params, batch)
        e_pred = np.array(preds["energy"])
        f_pred = np.array(preds["forces"])
        wf_pred = np.array(dEdq)
        bec_pred = np.array(bec)
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
                    "polarizable": bool(atoms.info["polarizable"]),
                    "e_pred": e_pred_i,
                    "e_ref": e_ref_i,
                    "f_pred": f_pred_i.ravel().tolist(),
                    "f_ref": f_ref_i.ravel().tolist(),
                    "wf_pred": float(wf_pred[local_i]),
                    "wf_ref": float(atoms.info["work_function"]),
                }
            )
            # cpmace ships no bec_z labels, so this never fires. Left in place
            # so the script stays a drop-in sibling of ../razor/evaluate.py.
            if "bec_z" in atoms.arrays:
                rows[-1]["bec_pred"] = (
                    bec_pred[atom_to_structure == local_i].ravel().tolist()
                )
                rows[-1]["bec_ref"] = atoms.arrays["bec_z"].ravel().tolist()
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
    out = {
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
    if all("bec_ref" in r for r in rows):
        out["bec_pred"] = np.concatenate([r["bec_pred"] for r in rows])  # e
        out["bec_ref"] = np.concatenate([r["bec_ref"] for r in rows])
        out["bec_polarizable"] = np.repeat(
            polarizable, [len(r["bec_ref"]) for r in rows]
        )
    return out


def metrics(d):
    m = {
        "e_rmse": rmse(d["e_pred"], d["e_ref"]),
        "e_mae": mae(d["e_pred"], d["e_ref"]),
        "f_rmse": rmse(d["f_pred"], d["f_ref"]),
        "f_mae": mae(d["f_pred"], d["f_ref"]),
        "wf_rmse": rmse(d["wf_pred"], d["wf_ref"]),
        "wf_mae": mae(d["wf_pred"], d["wf_ref"]),
        "n": len(d["e_pred"]),
    }
    if "bec_ref" in d:
        m["bec_rmse"] = rmse(d["bec_pred"], d["bec_ref"])
        m["bec_mae"] = mae(d["bec_pred"], d["bec_ref"])
    return m


def print_table(all_rows):
    header = (
        f"{'split':<12}{'variant':<32}{'subset':<14}"
        f"{'E RMSE':>10}{'E MAE':>10}{'F RMSE':>10}{'F MAE':>10}"
        f"{'WF RMSE':>10}{'WF MAE':>10}{'Z RMSE':>10}{'Z MAE':>10}{'n':>8}"
    )
    print("\nE in meV/atom, F in meV/A, WF (dE/dq) in V, Z (bec_z) in e")
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
                    f"{split:<12}{v:<32}{label:<14}"
                    f"{m['e_rmse']:>10.2f}{m['e_mae']:>10.2f}"
                    f"{m['f_rmse']:>10.2f}{m['f_mae']:>10.2f}"
                    f"{m['wf_rmse']:>10.4f}{m['wf_mae']:>10.4f}"
                    + (
                        f"{m['bec_rmse']:>10.4f}{m['bec_mae']:>10.4f}"
                        if "bec_rmse" in m
                        else f"{'--':>10}{'--':>10}"
                    )
                    + f"{m['n']:>8d}"
                )


# -- plots --


def _parity(
    ax, ref, pred, color, label, unit, polarizable=None, legend=False, seed=0,
    lims=None,
):
    # lims is passed in by plot_split so every row of a column shares one
    # scale -- otherwise each panel autoscales and the rows cannot be compared
    # by eye, which is the whole point of stacking them.
    if lims is None:
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
    # On mixed splits the non-polarizable frames dominate the count and sit
    # outside the linear-response window, where both the model and the
    # reference labels are least trustworthy -- a pooled RMSE is then mostly
    # a statement about those. Quote the polarizable-only figure instead, and
    # say so, rather than a number that averages the two regimes.
    if polarizable is not None and polarizable.any() and not polarizable.all():
        text = f"RMSE (pol.) = {rmse(pred[polarizable], ref[polarizable]):.3g} {unit}"
    else:
        text = f"RMSE = {rmse(pred, ref):.3g} {unit}"
    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6,
    )


# -- RMSE resolved by charge --

# cpmace's charges are continuous (1036 distinct values over q in
# [-2.31, -1.26], a span of ~1.05 e) because the electron count is a
# constant-potential DFT output rather than a set point. The validation split
# is only 109 frames, so 0.25 e bins give ~4 bins of ~27 frames -- coarse, but
# above MIN_BIN_N. Finer bins would fall below it and be hatched out.
CHARGE_BIN_WIDTH = 0.25

# Below this many structures a per-bin RMSE is too noisy to read. Bins under
# it are still drawn -- hiding them would misrepresent the charge range that
# was actually tested -- but hatched and annotated with their count.
MIN_BIN_N = 20


def _bin_centers(rows):
    q = np.array([r["total_charge"] for r in rows])
    lo = np.round(q.min() / CHARGE_BIN_WIDTH) * CHARGE_BIN_WIDTH
    hi = np.round(q.max() / CHARGE_BIN_WIDTH) * CHARGE_BIN_WIDTH
    n = int(round((hi - lo) / CHARGE_BIN_WIDTH))
    return lo + CHARGE_BIN_WIDTH * np.arange(n + 1)


def _bin_of(row, centers):
    return int(np.argmin(np.abs(centers - row["total_charge"])))


def _subset_rmse(rows, key):
    """RMSE for one target over a row subset, or None if it has no labels."""
    if key == "bec":
        rows = [r for r in rows if "bec_ref" in r]
    if not rows:
        return None, 0
    d = collect(rows)
    if f"{key}_ref" not in d:
        return None, 0
    return rmse(d[f"{key}_pred"], d[f"{key}_ref"]), len(rows)


@mpltex.acs_decorator
def plot_rmse_vs_charge(all_rows, split, name):
    """Grid of RMSE-vs-charge bar charts: one row per variant, one column per
    target. Companion to the parity figure -- parity shows whether a model is
    biased, this shows *where in charge space* the error lives, which is the
    question the +-0.25 e stencil raises."""
    plt.rcParams["text.usetex"] = False

    split_rows = _split_rows(all_rows, split)
    if not split_rows:
        return

    # short symbols rather than the parity figure's full words: four columns
    # of "Born effective charge RMSE (e)" would collide with the row labels
    quantities = [
        ("e", "$E$", "meV/atom"),
        ("f", "$F$", "meV/Å"),
        ("wf", r"$\Phi$", "V"),
    ]
    if any("bec_ref" in r for r in split_rows):
        quantities.append(("bec", "$Z^*$", "e"))

    centers = _bin_centers(split_rows)
    # keep the polarizable split wherever the split carries both, as
    # everywhere else in this script -- the two regimes have different error
    # scales and pooling them hides that
    flags = sorted({r["polarizable"] for r in split_rows}, reverse=True)
    width = 0.8 * CHARGE_BIN_WIDTH / len(flags)

    fig, axes = plt.subplots(
        len(VARIANTS), len(quantities), squeeze=False, constrained_layout=True
    )
    fig.set_figwidth(2.6 * fig.get_figwidth())
    fig.set_figheight(0.62 * len(VARIANTS) * fig.get_figwidth() / len(quantities))
    col_max = {}

    for row, v in enumerate(VARIANTS):
        rows_v = [r for r in split_rows if r["variant"] == v]
        if not rows_v:
            continue
        binned = {}
        for r in rows_v:
            binned.setdefault(_bin_of(r, centers), []).append(r)

        for col, (key, label, unit) in enumerate(quantities):
            ax = axes[row][col]
            for k, flag in enumerate(flags):
                offset = (k - (len(flags) - 1) / 2) * width
                xs, ys, thin = [], [], []
                for b, rs in sorted(binned.items()):
                    sub = [r for r in rs if r["polarizable"] == flag]
                    val, n = _subset_rmse(sub, key)
                    if val is None:
                        continue
                    xs.append(centers[b] + offset)
                    ys.append(val)
                    thin.append(n < MIN_BIN_N)
                if not xs:
                    continue
                ax.bar(
                    xs,
                    ys,
                    width=width,
                    color=POLARIZABLE_COLORS[flag] if len(flags) > 1 else COLORS.get(v, "grey"),
                    edgecolor="none",
                    label=("polarizable" if flag else "non-polarizable")
                    if len(flags) > 1
                    else None,
                    zorder=2,
                )
                # mark the under-populated bins rather than dropping them
                for x, y, t in zip(xs, ys, thin):
                    if t:
                        ax.bar(
                            x, y, width=width, color="none", edgecolor="0.25",
                            lw=0.4, hatch="///", zorder=3,
                        )
            ax.set_xlabel("total charge $q$ (e)", fontsize=7)
            ax.set_ylabel(f"{label} RMSE ({unit})", fontsize=7)
            ax.set_xticks(centers[::2])
            ax.tick_params(labelsize=6)
            ax.margins(x=0.02)
            col_max[col] = max(col_max.get(col, 0.0), ax.get_ylim()[1])
            if row == 0 and col == 0 and len(flags) > 1:
                ax.legend(fontsize=5, loc="upper center", framealpha=0.9)

        axes[row][0].annotate(
            LABELS.get(v, v),
            xy=(-0.38, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            linespacing=1.4,
        )

    # one y-scale per column, as in the parity figure, so bar heights can be
    # compared across rows rather than each panel rescaling to its own worst bin
    for col in col_max:
        for row in range(len(VARIANTS)):
            axes[row][col].set_ylim(0, col_max[col])

    hatched = "hatched bars: n < %d structures in that bin" % MIN_BIN_N
    fig.suptitle(f"razor -- {split} -- RMSE vs charge ({hatched})")
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures") / name, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


# The test sweep is only 13% polarizable, and the non-polarizable frames sit
# outside the linear-response window where neither the model nor the reference
# labels are trustworthy. They dominate every pooled number and stretch every
# axis, so the figures show the polarizable subset only. The printed table
# still reports all three subsets -- nothing is discarded, only not plotted.
PLOT_POLARIZABLE_ONLY = set()


def _split_rows(all_rows, split):
    rows = [r for r in all_rows if r["split"] == split]
    if split in PLOT_POLARIZABLE_ONLY:
        rows = [r for r in rows if r["polarizable"]]
    return rows


@mpltex.acs_decorator
def plot_split(all_rows, split, name):
    plt.rcParams["text.usetex"] = False  # no LaTeX install on this machine

    quantities = [
        ("e", "energy", "meV/atom"),
        ("f", "force", "meV/Å"),
        ("wf", "work function", "V"),
    ]
    if any("bec_ref" in r for r in _split_rows(all_rows, split)):
        quantities.append(("bec", "Born effective charge", "e"))

    fig, axes = plt.subplots(
        len(VARIANTS), len(quantities), squeeze=False, constrained_layout=True
    )
    fig.set_figwidth(2.6 * fig.get_figwidth())
    fig.set_figheight(1.0 * len(VARIANTS) * fig.get_figwidth() / len(quantities))

    # one scale per column, over every variant, so rows are comparable by eye
    per_variant = {}
    for v in VARIANTS:
        rows = [r for r in _split_rows(all_rows, split) if r["variant"] == v]
        if rows:
            per_variant[v] = collect(rows)
    col_lims = {}
    for key, _, _ in quantities:
        vals = [
            d[f"{key}_{which}"]
            for d in per_variant.values()
            for which in ("ref", "pred")
            if f"{key}_{which}" in d
        ]
        if not vals:
            continue
        lo = min(float(np.min(a)) for a in vals)
        hi = max(float(np.max(a)) for a in vals)
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        col_lims[key] = (lo - pad, hi + pad)

    for row, v in enumerate(VARIANTS):
        d = per_variant.get(v)
        if d is None:
            continue
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
                lims=col_lims.get(key),
            )
        # variant name once per row, outside the axes, so it can't collide
        # with the row above's x-label
        axes[row][0].annotate(
            LABELS.get(v, v),
            xy=(-0.42, 0.5),
            xycoords="axes fraction",
            rotation=90,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            linespacing=1.4,
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
        plot_rmse_vs_charge(all_rows, split, f"rmse_vs_charge_{split}.pdf")


if __name__ == "__main__":
    main()
