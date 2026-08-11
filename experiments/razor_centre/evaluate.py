"""Parity plots + RMSE for the razor-centre `sr`/`sr-wf`/`sr-wf-bec` variants.

Three quantities per variant, on both the validation split and the
wide-charge-range test sweep:

- energy (eV/atom)
- forces (eV/A, all Cartesian components)
- work function (V) -- the model's autograd `dE/dq`, i.e. backprop through
  the forward pass with respect to the `total_charge` input, against the
  DFT `work_function` label. For `sr`/`lr` nothing trains on it, so it is a
  pure "did charge conditioning learn the right charge derivative?" probe;
  for `sr-wf` it is a fitted target and this panel is a training diagnostic
  instead. The per-species energy baseline is charge-independent and
  therefore drops out of `dE/dq` entirely -- no offset correction needed
  there, unlike for the energy parity.

The `dE/dq` here is computed by this script's own `jax.grad`, so it works
against checkpoints from either lorem version -- it does not depend on
`Lorem.predict` exposing `work_function`.

Reads the raw extxyz rather than `data/`, so it also gets `polarizable`
and `q_MD` for the split-out plots. Loads each trained checkpoint directly
(model + params + per-species baseline) instead of going through the
per-structure ASE Calculator, which triggers a fresh XLA compile per
distinct padded shape.

Run from experiments/razor/ once both variants have trained:

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

DATA = "../../datasets/razor"
VARIANTS = [
    "sr",
    "sr-450ep",
    "sr-wf",
    "sr-wf-bec",
    "sr-small-l2c8-e100-wf0.05-675ep",
    "lr-small-l2c8-e100-wf0.05-675ep",
]

# Row labels. The first four are already short; the l2c8 pair carries the
# loss weights in its directory name, which say nothing here since they are
# fixed, so label the axes that vary. 675 epochs is the gradient-update match
# to ../razor/'s 300 (729 vs 324 batches/epoch -> 218,700 updates either way).
LABELS = {
    "sr": "d128 l6 c4 sr\n200 ep",
    "sr-450ep": "d128 l6 c4 sr\n450 ep",
    "sr-wf": "d128 l6 c4 sr\n+wf, 200 ep",
    "sr-wf-bec": "d128 l6 c4 sr\n+wf+bec, 200 ep",
    "sr-small-l2c8-e100-wf0.05-675ep": "d64 l2 c8 sr\n675 ep",
    "lr-small-l2c8-e100-wf0.05-675ep": "d64 l2 c8 LR\n675 ep",
}
# the summed-R2 checkpoint is named after the targets it covers -- "R2_E+F"
# for energy+forces, "R2_E+F+W" once the work function is trained on,
# "R2_E+F+W+B" with the Born effective charges on top. Resolve it per variant
# instead of hardcoding one name, which only ever matched the E+F runs.
CHECKPOINT_GLOB = "R2_*"

# (split name, xyz file, polarizable-only). "valid" mirrors what the model
# actually trained on; "test_sweep" is the full 13-point q in [-1.5, 1.5]
# sweep, only 13% of which is polarizable -- the extrapolation case, kept
# unfiltered on purpose (see prepare.py).
SPLITS = [
    ("valid", "razor_val", True),
    ("test_sweep", "razor_test", False),
]

COLORS = {
    "sr": "steelblue",
    "sr-450ep": "darkorange",
    "sr-wf": "seagreen",
    "sr-wf-bec": "rebeccapurple",
    "sr-small-l2c8-e100-wf0.05-675ep": "goldenrod",
    "lr-small-l2c8-e100-wf0.05-675ep": "crimson",
}
POLARIZABLE_COLORS = {True: "steelblue", False: "darkorange"}

# 8, not razor's 32: sr-wf-bec's model carries predict_bec, and the
# jvp(jvp(energy)) path for d2E/dr dq holds far more live intermediates than a
# plain forward+backward. At 32 it reached 36.7 GB and OOM'd on an A40 (48 GB).
# Evaluation is not throughput-critical, so trade batch size for headroom.
BATCH_SIZE = 8

# Points drawn in the force parity scatter (RMSE is always over all of
# them). ~400k components for valid would make an unopenable PDF.
MAX_SCATTER_POINTS = 20000

# None = full split; set to an int for a quick smoke test.
SAMPLE_SIZE = None

ROWS_CACHE = Path("evaluate_rows.json")


# Drop configurations whose DFT max force exceeds this, in eV/A. Not a
# cosmetic choice: a single razor_val frame (struc_pk 301216, q = -1.25,
# max_force 12.66) was the worst frame for every variant by an order of
# magnitude -- 481 meV/A against 41 for the next worst in its charge bin --
# and on its own inflated the pooled validation force RMSE from 27.97 to
# 31.17 meV/A. The >20 eV/A rule the READMEs measured earlier is too loose to
# catch it. Applied to the evaluation splits only; training data is untouched,
# so this changes what is reported, not what was learned.
MAX_FORCE_CUTOFF = 10.0


def load_frames(name, polarizable_only, n=None):
    frames = read(f"{DATA}/{name}.xyz", index=":")
    for atoms in frames:
        atoms.info["total_charge"] = float(atoms.info.pop("bias_charge"))
    if polarizable_only:
        frames = [a for a in frames if a.info["polarizable"]]
    over = [a.info["max_force"] for a in frames if a.info["max_force"] > MAX_FORCE_CUTOFF]
    if over:
        print(
            f"  {name}: dropped {len(over)}/{len(frames)} frames over "
            f"{MAX_FORCE_CUTOFF} eV/A (max_force "
            f"{[round(x, 2) for x in sorted(over, reverse=True)]})",
            flush=True,
        )
    frames = [a for a in frames if a.info["max_force"] <= MAX_FORCE_CUTOFF]
    if n:
        frames = frames[:n]
    return frames


# Attributes that used to live on Lorem and have since moved to the LoremQ
# subclass. A checkpoint written before that move records the attribute
# against the base class, and Lorem.__init__ now rejects the keyword -- even
# when the value is false, which is the case for every plain-Lorem run here.
# Strip those keys rather than editing the checkpoints: they are generated
# artifacts that can only be reproduced by retraining, and this script is
# meant to read checkpoints from either lorem version.
LOREMQ_ONLY_KEYS = ("predict_bec",)


def _drop_dead_keys(cfg):
    for cls, kwargs in cfg.items():
        if cls.rsplit(".", 1)[-1] == "LoremQ" or not isinstance(kwargs, dict):
            continue
        for key in LOREMQ_ONLY_KEYS:
            if kwargs.pop(key, None):
                # only false is safe to drop silently -- a true here would
                # mean the checkpoint really wants behaviour the base class
                # cannot provide, and quietly ignoring that would be wrong
                raise RuntimeError(
                    f"{cls} sets {key}=True but only LoremQ implements it"
                )
    return cfg


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
                    "q_MD": float(atoms.info["q_MD"]),
                    "polarizable": bool(atoms.info["polarizable"]),
                    "e_pred": e_pred_i,
                    "e_ref": e_ref_i,
                    "f_pred": f_pred_i.ravel().tolist(),
                    "f_ref": f_ref_i.ravel().tolist(),
                    "wf_pred": float(wf_pred[local_i]),
                    "wf_ref": float(atoms.info["work_function"]),
                }
            )
            # razor_val.xyz carries no bec_z; razor_test.xyz and
            # razor_centre.xyz do. Only score it where there is a label.
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
        f"{'split':<12}{'variant':<12}{'subset':<14}"
        f"{'E RMSE':>10}{'E MAE':>10}{'F RMSE':>10}{'F MAE':>10}"
        f"{'WF RMSE':>10}{'WF MAE':>10}{'Z RMSE':>10}{'Z MAE':>10}{'n':>8}"
    )
    print("\nE in meV/atom, F in meV/A, WF (dE/dq) in V, Z (bec_z) in e")
    print("Z is blank where the split's xyz carries no bec_z label (razor_val).")
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
                    f"{split:<12}{v:<12}{label:<14}"
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

# The dataset's charges sit on a 0.05 e grid, not 0.25, because the stencil is
# q_MD +- 0.25 about a continuously-varying q_MD -- razor_val has 52 distinct
# polarizable charges over [-1.75, 1.75], so a bar per exact value would be
# ~23 frames each and very ragged. 0.25 is the coarsest binning that still
# resolves the stencil spacing, and puts 11 of 15 bins above MIN_BIN_N.
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
PLOT_POLARIZABLE_ONLY = {"test_sweep"}


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
    # bec_z only exists for some splits (razor_val.xyz has none), so the
    # column appears or not depending on the data rather than being hardcoded
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
            fontsize=8,
            fontweight="bold",
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
