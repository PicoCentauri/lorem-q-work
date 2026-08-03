import jax

from ase.io import read
from marathon import comms
from marathon.data import datasets
from marathon.grain import prepare

DATA = "../../datasets/water_slab"


def rename_keys(atoms):
    atoms.info["total_charge"] = float(atoms.info.pop("tot_charge"))
    atoms.info["external_field"] = atoms.info.pop("ext_field")
    return atoms


train_data = [rename_keys(a) for a in read(f"{DATA}/water_slab_train.xyz", index=":")]
valid_data = [rename_keys(a) for a in read(f"{DATA}/water_slab_val.xyz", index=":")]
test_data = [rename_keys(a) for a in read(f"{DATA}/water_slab_test.xyz", index=":")]

reporter = comms.reporter()
reporter.start("processing")

# total_charge/external_field are model inputs, not training labels, so
# lorem/batching.py reads them directly from atoms.info rather than through
# the keys/loss-weight mechanism below. But they still need to be declared
# here with storage="atoms.info" -- marathon.grain.prepare() only persists
# atoms.info entries that are listed in this properties dict; anything else
# is silently dropped when the dataset is serialized to disk.
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
    "external_field": {
        "shape": (3,),
        "storage": "atoms.info",
    },
}

prepare(
    train_data,
    folder=datasets / "data/train",
    reporter=reporter,
    batch_size=64,
    samples_per_composition=100,
    properties=PROPERTIES,
)

prepare(
    valid_data,
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
