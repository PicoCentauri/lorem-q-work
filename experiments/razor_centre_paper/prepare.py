"""Prepare the publication's subset of razor_centre for `LoremQ` training.

Data is `../../datasets/razor/razor_centre_paper_{train,test}.xyz`: the
publication's structure selection carrying *our* energy, force, work-function
and `bec_z` labels. See `../../datasets/razor/README.md` -- the selection is
`razor_centre` restricted to |q| <= 1.0 e, one frame per structure at the
stencil centre, and the publication's own `DFT_E0`/`DFT_F0` are
back-extrapolated to q = 0 rather than being at the frame's charge.

SPLITS
------
The publication ships train (4598) and test (515) only, so a validation set is
carved out of *their train*, leaving *their test* untouched as a clean
comparison point against the published numbers.

A per-frame split is safe here, unlike in `../razor/` or `../natcomm2025/`:
this data is one frame per `struc_pk`, so there is no charge stencil to keep
together and no way for a geometry to appear on both sides. The publication's
own train/test split shares no `struc_pk` either (asserted when the files were
built).

    DATASETS=. python prepare.py
"""

import numpy as np

from ase.io import read
from marathon import comms
from marathon.data import datasets
from marathon.grain import prepare

DATA = "../../datasets/razor"

SEED = 0
VALID_FRACTION = 0.10

# 108 atoms per frame, as in ../razor/. ToBatch packs batch_size - 1 real
# structures, so 16 is 15 x 108 = 1620 atoms per batch.
BATCH_SIZE = 16


def rename_charge(atoms):
    """`bias_charge` -> `total_charge`, the name the model reads.

    Not cosmetic. marathon writes NaN for any declared atoms.info key that is
    absent, and total_charge is a model *input*, so omitting this rename feeds
    the charge conditioning NaN from step 0 and the loss is NaN forever. That
    is exactly what happened on the first attempt here: both runs died at
    "loss became NaN at step=8280" and it was misdiagnosed as a loss-weight
    problem. ../razor/prepare.py and ../razor_centre/prepare.py both do this.
    """
    atoms.info["total_charge"] = float(atoms.info.pop("bias_charge"))
    return atoms


def main():
    train_all = [rename_charge(a) for a in read(f"{DATA}/razor_centre_paper_train.xyz", index=":")]
    test = [rename_charge(a) for a in read(f"{DATA}/razor_centre_paper_test.xyz", index=":")]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train_all))
    n_valid = int(round(VALID_FRACTION * len(train_all)))
    valid = [train_all[i] for i in sorted(order[:n_valid])]
    train = [train_all[i] for i in sorted(order[n_valid:])]

    # one frame per struc_pk is what makes a per-frame split safe -- check it
    pks = [int(a.info["struc_pk"]) for a in train_all]
    assert len(pks) == len(set(pks)), "struc_pk repeats; a per-frame split would leak"
    tp = {int(a.info["struc_pk"]) for a in train}
    vp = {int(a.info["struc_pk"]) for a in valid}
    xp = {int(a.info["struc_pk"]) for a in test}
    assert not (tp & vp) and not (tp & xp) and not (vp & xp), "splits share struc_pk"

    comms.talk(f"train {len(train)} / valid {len(valid)} / test {len(test)} frames")

    e = np.array([a.get_potential_energy() / len(a) for a in train])
    f = np.concatenate([a.get_forces().ravel() for a in train])
    w = np.array([a.info["work_function"] for a in train])
    b = np.concatenate([np.asarray(a.arrays["bec_z"]).ravel() for a in train])
    comms.talk(
        f"variances  E/atom {np.var(e):.4e}  F {np.var(f):.4e}  "
        f"W {np.var(w):.4e}  bec_z {np.var(b):.4e}"
    )
    q = np.array([a.info["total_charge"] for a in train])
    pol = np.array([bool(a.info["polarizable"]) for a in train])
    comms.talk(f"q {q.min():+.2f}..{q.max():+.2f}   polarizable {100*pol.mean():.1f}%")

    reporter = comms.reporter()
    reporter.start("processing")

    # total_charge is a model input rather than a label, but marathon only
    # persists atoms.info entries listed here. work_function and bec_z are the
    # derivative targets, both written unconditionally so the sr, lr and
    # sr-bec variants share one data/ directory and differ only in weights.
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
        "bec_z": {
            "shape": ("atom", 3),
            "storage": "atoms.arrays",
            "report_unit": (1, "e"),
            "symbol": "Z",
        },
    }

    for frames, folder in (
        (train, "data/train"),
        (valid, "data/valid"),
        (test, "data/test"),
    ):
        comms.talk(f"{folder}: {len(frames)} frames")
        prepare(
            frames,
            folder=datasets / folder,
            reporter=reporter,
            batch_size=BATCH_SIZE,
            samples_per_composition=100,
            properties=PROPERTIES,
        )

    reporter.done()


if __name__ == "__main__":
    main()
