"""Prepare the razor *centre* dataset: one frame per structure, at its own
MD charge.

`razor_centre.xyz` holds the centre of each 3-point bias-charge stencil, i.e.
exactly the frames where `bias_charge == q_MD`. Training on it removes the
off-equilibrium (geometry, charge) pairs that dominate `../razor/` -- there,
two thirds of the frames pair a geometry with a charge it never equilibrated
at. That off-stencil labelling is what makes `dE/dq` learnable in the first
place, so this is a genuine trade, not a strict improvement: less data, but
every (r, q) pair is physical.

Splits
------
`razor_centre.xyz` contains 5989 structures = 5390 + 599, which is exactly
`razor_train.xyz` + `razor_val.xyz` by `struc_pk` (verified; the 20
`razor_test.xyz` structures are absent). So rather than invent a split, this
inherits razor's:

    train  <- centre frames whose struc_pk is in razor_train.xyz   (5390)
    valid  <- centre frames whose struc_pk is in razor_val.xyz     (599)

and takes the test sets from `razor_test.xyz` verbatim, exactly as
`../razor/prepare.py` does:

    test        <- razor_test.xyz, polarizable only                (34)
    test_sweep  <- razor_test.xyz, unfiltered 13-point sweep       (260)

That keeps the evaluation sets byte-identical between this folder and
`../razor/`, so only the *training distribution* differs and the two are
comparable row-by-row. It also guarantees no leakage: the split key is
struc_pk and it is never split within one structure.

Targets
-------
`work_function` (dE/dq) and `bec_z` (the Born effective charge) are persisted
alongside `total_charge` so the `sr`, `sr-wf` and `sr-wf-bec` variants can all
read the same `data/` without a rebuild -- only their `loss_weights` differ.

Run from experiments/razor_centre/:

    DATASETS=. python prepare.py
"""

from ase.io import iread, read

from marathon import comms
from marathon.data import datasets
from marathon.grain import prepare

DATA = "../../datasets/razor"

# 108 atoms per frame (36 Pt + 24 H2O); see ../razor/README.md for why the
# batcher's batch_size is the same number.
BATCH_SIZE = 16


def rename_charge(atoms):
    atoms.info["total_charge"] = float(atoms.info.pop("bias_charge"))
    return atoms


def struc_pks(name):
    # streamed: razor_train.xyz is 16170 frames of 108 atoms and we only want
    # the split keys, so holding every Atoms object would waste GBs for nothing
    return {a.info["struc_pk"] for a in iread(f"{DATA}/{name}.xyz")}


def load(name, polarizable_only=True, keep=None):
    frames = [rename_charge(a) for a in read(f"{DATA}/{name}.xyz", index=":")]
    if keep is not None:
        frames = [a for a in frames if a.info["struc_pk"] in keep]
    if polarizable_only:
        frames = [a for a in frames if a.info["polarizable"]]
    return frames


train_pks = struc_pks("razor_train")
valid_pks = struc_pks("razor_val")
comms.talk(f"struc_pk: {len(train_pks)} train, {len(valid_pks)} valid")

centre = [rename_charge(a) for a in read(f"{DATA}/razor_centre.xyz", index=":")]
comms.talk(f"razor_centre.xyz: {len(centre)} frames")

# every centre frame must land in exactly one split -- fail loudly otherwise,
# because a silent drop here would quietly shrink the training set
unassigned = [a for a in centre if a.info["struc_pk"] not in train_pks | valid_pks]
if unassigned:
    raise RuntimeError(
        f"{len(unassigned)} centre frames have a struc_pk in neither "
        "razor_train.xyz nor razor_val.xyz; the split assumption is wrong"
    )

train_data = [a for a in centre if a.info["struc_pk"] in train_pks and a.info["polarizable"]]
valid_data = [a for a in centre if a.info["struc_pk"] in valid_pks and a.info["polarizable"]]

# unchanged from ../razor/, so the evaluation sets match exactly
test_data = load("razor_test")
test_sweep_data = load("razor_test", polarizable_only=False)

reporter = comms.reporter()
reporter.start("processing")

# total_charge is a model *input*, read straight from atoms.info by
# lorem/batching.py rather than through the keys/loss-weight mechanism -- but
# marathon.grain.prepare() only persists atoms.info entries declared here, so
# it still has to be listed or it is silently dropped.
#
# work_function (dE/dq) and bec_z (the Born effective charge, an atoms.arrays
# quantity) are the derivative targets. Both are persisted unconditionally so
# sr/, sr-wf/ and sr-wf-bec/ share one data/ directory.
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
    "total_charge": {
        "shape": (1,),
        "storage": "atoms.info",
    },
    "work_function": {
        "shape": (1,),
        "storage": "atoms.info",
        "report_unit": (1, "V"),
        "symbol": "W",
    },
    "bec_z": {
        "shape": ("atom", 3),
        "storage": "atoms.arrays",
        "report_unit": (1, "e"),
        "symbol": "Z",
    },
}

for frames, folder in [
    (train_data, "data/train"),
    (valid_data, "data/valid"),
    (test_data, "data/test"),
    (test_sweep_data, "data/test_sweep"),
]:
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
