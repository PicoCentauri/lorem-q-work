"""Prepare the natcomm2025 Pt(111)/water dataset for `LoremQ` training.

The xyz already carries everything the model needs -- `total_charge` and
`work_function` were derived in `../../datasets/natcomm2025/convert.py` from
the DeepMD `fparam` (the electron number Nₑ), so nothing is converted here:

    total_charge  = -fparam
    work_function = -dE/dfparam   (per-geometry quadratic fit, = -mu_e)

See that folder's README for why, and for the confirmation against the paper.

ALL FRAMES ARE TRAINED ON, INCLUDING THE REACTIVE ONES
------------------------------------------------------
No screening of any kind is applied here -- deliberately. Two candidates were
considered and both rejected:

  * max force. The largest force in the dataset is 6.42 eV/A, well under the
    10 eV/A cut used in ../razor/. A screen would be a no-op.
  * anomalous d2E/dq2. 10.6% of frames have a capacitance more than 20% off
    the 7.676 V/e median. In ../razor/ the analogous frames are dielectric
    breakdown and are dropped; here they are the Volmer step -- they carry an
    extra adsorbed hydrogen, and added electrons form a Pt-H bond instead of
    charging the interface. Those are reaction events, which is precisely what
    a constant-potential model is wanted for, so they are kept.

The consequence is worth stating plainly: **on ~10% of frames the work
function is set by bond formation rather than by capacitive charging.** A
model can fit the capacitive majority well and still do badly there, and a
pooled RMSE will not show it. `d2E_dq2` is carried through the xyz, so
evaluate.py can split the metrics on it.

THE SPLIT IS BY GEOMETRY, NOT BY FRAME
--------------------------------------
Each of the 2750 distinct geometries appears at 4-18 electron counts sharing
identical atomic positions. Splitting per frame would put the same structure
in train and validation at a different charge, which is leakage -- the model
would be scored on geometries it had already fitted. `group` in the xyz is the
geometry id and the split is made on it, exactly as razor splits on `struc_pk`.

The consequence is a validation set of whole charge stencils, so `evaluate.py`
can look at dE/dq across a geometry's own charge range rather than only at
scattered points.

Run from experiments/natcomm2025/:

    DATASETS=. python prepare.py
"""

import numpy as np

from ase.io import read
from marathon import comms
from marathon.data import datasets
from marathon.grain import prepare

DATA = "../../datasets/natcomm2025/natcomm2025.xyz"

SEED = 0
VALID_FRACTION = 0.10

# 233 atoms per frame, one composition throughout. ToBatch packs
# batch_size - 1 real structures and pads the last slot, so 8 here is
# 7 x 233 = 1631 atoms per batch -- close to razor's 1620 and cpmace's 1449,
# which keeps the per-step cost in a familiar range.
BATCH_SIZE = 8


def main():
    frames = read(DATA, index=":")
    groups = np.array([a.info["group"] for a in frames])
    uniq = np.unique(groups)
    comms.talk(f"{len(frames)} frames, {len(uniq)} distinct geometries")

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(uniq))
    n_valid = int(round(VALID_FRACTION * len(uniq)))
    valid_groups = set(uniq[order[:n_valid]].tolist())

    train = [a for a in frames if a.info["group"] not in valid_groups]
    valid = [a for a in frames if a.info["group"] in valid_groups]

    # the split must not put one geometry on both sides
    gt = {a.info["group"] for a in train}
    gv = {a.info["group"] for a in valid}
    assert not (gt & gv), f"{len(gt & gv)} geometries leaked across the split"
    comms.talk(
        f"train {len(train)} frames / {len(gt)} geometries, "
        f"valid {len(valid)} frames / {len(gv)} geometries "
        f"({100 * len(valid) / len(frames):.1f}% of frames)"
    )

    for name, fr in (("train", train), ("valid", valid)):
        q = np.array([a.info["total_charge"] for a in fr])
        w = np.array([a.info["work_function"] for a in fr])
        e = np.array([a.get_potential_energy() / len(a) for a in fr])
        f = np.concatenate([a.get_forces().ravel() for a in fr])
        comms.talk(
            f"  {name:<6} q {q.min():+.3f}..{q.max():+.3f}  "
            f"W {w.min():+.2f}..{w.max():+.2f} (std {w.std():.3f})  "
            f"max|F| {np.abs(f).max():.2f}"
        )
        if name == "train":
            # loss weights follow label variances and this dataset's are not
            # cpmace's -- see README. Reported here so settings.yaml can be
            # justified from the data rather than carried over.
            vE, vF, vW = np.var(e), np.var(f), np.var(w)
            comms.talk(f"  variances: E/atom {vE:.4e}  F {vF:.4e}  W {vW:.4e}")
            tE, tF, tW = 0.201, 0.678, 0.121  # razor's tuned shares
            total = vF / tF
            comms.talk(
                f"  weights matching razor's shares: "
                f"E {tE * total / vE:.0f} : F 1 : W {tW * total / vW:.2f}"
            )

    reporter = comms.reporter()
    reporter.start("processing")

    # total_charge is a model *input*, not a label, but marathon.grain.prepare
    # only persists atoms.info entries listed here -- anything omitted is
    # silently dropped when the dataset is serialised.
    PROPERTIES = {
        "energy": {
            "shape": (1,),
            "storage": "atoms.calc",
            "report_unit": (1000, "meV"),
            "symbol": "E",
        },
        "forces": {
            "shape": ("atom", 3),
            "storage": "atoms.calc",
            "report_unit": (1000, "meV/Å"),
            "symbol": "F",
        },
        "total_charge": {"shape": (1,), "storage": "atoms.info"},
        "work_function": {"shape": (1,), "storage": "atoms.info"},
    }

    for fr, folder in ((train, "data/train"), (valid, "data/valid")):
        comms.talk(f"{folder}: {len(fr)} frames")
        prepare(
            fr,
            folder=datasets / folder,
            reporter=reporter,
            batch_size=BATCH_SIZE,
            samples_per_composition=100,
            properties=PROPERTIES,
        )

    reporter.done()


if __name__ == "__main__":
    main()
