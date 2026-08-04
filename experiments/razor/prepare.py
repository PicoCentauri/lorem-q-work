"""Prepare the razor Pt(111)/water dataset for `Lorem` charge-conditioned
energy+force training.

`bias_charge` (the DFT bias charge the frame was evaluated at) becomes
`total_charge`, the model input. Non-polarizable frames -- those flagged
`polarizable=False`, i.e. sitting outside the linear-response window near
dielectric breakdown / Fermi pinning -- are dropped from train/valid/test.

`data/test_sweep` keeps the *unfiltered* test frames: `razor_test.xyz` is
a dense 13-point charge sweep per structure (q in [-1.5, 1.5] e) and only
13% of it is polarizable, so filtering it leaves too little to serve as
the wide-charge-range extrapolation check the dataset is built for. Both
splits are reported at train time (see settings.yaml).

Run from experiments/razor/:

    DATASETS=. python prepare.py
"""

from ase.io import read
from marathon import comms
from marathon.data import datasets
from marathon.grain import prepare

DATA = "../../datasets/razor"

# 108 atoms per frame (36 Pt + 24 H2O), fixed across the whole dataset, so
# this is also the batcher's batch size in settings.yaml. Note ToBatch packs
# batch_size - 1 real structures and pads the last slot, so 16 here means
# 15 x 108 = 1620 atoms per batch.
BATCH_SIZE = 16


def rename_charge(atoms):
    # bias_charge is the model input here, deliberately not called
    # tot_charge like in the other datasets -- see datasets/razor/README.md
    atoms.info["total_charge"] = float(atoms.info.pop("bias_charge"))
    return atoms


def load(name, polarizable_only=True):
    frames = [rename_charge(a) for a in read(f"{DATA}/{name}.xyz", index=":")]
    if polarizable_only:
        frames = [a for a in frames if a.info["polarizable"]]
    return frames


train_data = load("razor_train")
valid_data = load("razor_val")
test_data = load("razor_test")
test_sweep_data = load("razor_test", polarizable_only=False)

reporter = comms.reporter()
reporter.start("processing")

# total_charge is a model input, not a training label, so lorem/batching.py
# reads it directly from atoms.info rather than through the keys/loss-weight
# mechanism below. But it still needs to be declared here with
# storage="atoms.info" -- marathon.grain.prepare() only persists atoms.info
# entries that are listed in this properties dict; anything else is silently
# dropped when the dataset is serialized to disk.
#
# work_function (dE/dq) is carried along the same way. Nothing trains on it
# in this round -- evaluate.py compares it against the model's autograd
# dE/dq -- but persisting it now means the planned work-function-target
# variants don't need the data rebuilt.
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
