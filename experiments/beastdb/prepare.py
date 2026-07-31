from ase.io import read
from marathon import comms
from marathon.data import datasets
from marathon.grain import prepare

DATA = "../../datasets/beastdb"


def rename_charge(atoms):
    atoms.info["total_charge"] = float(atoms.info.pop("tot_charge"))
    return atoms


train_data = [rename_charge(a) for a in read(f"{DATA}/beastdb_train.xyz", index=":")]
valid_data = [rename_charge(a) for a in read(f"{DATA}/beastdb_val.xyz", index=":")]

reporter = comms.reporter()
reporter.start("processing")

# beastdb has no forces (relaxed-geometry energies only) -- energy + total_charge
# is the full label set. total_charge is a model input, not a training label, so
# lorem/batching.py reads it directly from atoms.info rather than through the
# keys/loss-weight mechanism below. But it still needs to be declared here with
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
    "total_charge": {
        "shape": (1,),
        "storage": "atoms.info",
    },
}

prepare(
    train_data,
    folder=datasets / "data/train",
    reporter=reporter,
    batch_size=32,
    samples_per_composition=100,
    properties=PROPERTIES,
)

prepare(
    valid_data,
    folder=datasets / "data/valid",
    reporter=reporter,
    batch_size=32,
    samples_per_composition=100,
    properties=PROPERTIES,
)

reporter.done()
