import jax

from ase.io import read
from marathon import comms
from marathon.data import datasets, get_splits
from marathon.grain import prepare


def rename_charge(atoms):
    atoms.info["total_charge"] = float(atoms.info.pop("tot_charge"))
    return atoms


train_data = read("../../datasets/ag_clusters/Ag_clusters_train.xyz", format="extxyz", index=":")
test_data = read("../../datasets/ag_clusters/Ag_clusters_test.xyz", format="extxyz", index=":")

train_data = [rename_charge(a) for a in train_data]
test_data = [rename_charge(a) for a in test_data]

seed = 0
len_train = int(len(train_data) * 0.9)
len_valid = len(train_data) - len_train
idx_train, idx_valid, idx_test = get_splits(
    len(train_data), len_train, len_valid, 0, jax.random.key(seed)
)

reporter = comms.reporter()
reporter.start("processing")

# `total_charge` is a model input, not a training label, so lorem/batching.py
# reads it directly from atoms.info rather than through the keys/loss-weight
# mechanism below. But it still needs to be declared here with
# storage="atoms.info" -- marathon.grain.prepare() only persists atoms.info
# entries that are listed in this properties dict; anything else is silently
# dropped when the dataset is serialized to disk.
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
}

prepare(
    [train_data[i] for i in idx_train],
    folder=datasets / "data/train",
    reporter=reporter,
    batch_size=64,
    samples_per_composition=100,
    properties=PROPERTIES,
)

prepare(
    [train_data[i] for i in idx_valid],
    folder=datasets / "data/valid",
    reporter=reporter,
    batch_size=64,
    samples_per_composition=100,
    properties=PROPERTIES,
)

prepare(
    test_data,
    folder=datasets / "data/test",
    reporter=reporter,
    batch_size=64,
    samples_per_composition=100,
    properties=PROPERTIES,
)

reporter.done()
