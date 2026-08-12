"""Prepare the cpmace dataset for `LoremQ` charge-conditioned training.

Two conversions, both derived and checked in `datasets/cpmace/README.md`
rather than assumed from the field names:

    q            = -(electron - 660)
    E_total      = energy + potential * (electron - 660)
    work_function = -potential                 ( = dE_total/dq )

660 is the standard VASP PAW valence sum for C70 H89 N Ni O46 (H 1, C 4,
N 5, O 6, Ni 10); a thermodynamic-consistency fit on the data independently
implies 660.2, so the value is confirmed rather than trusted.

The second conversion is the one that matters. The reported `energy` is the
grand potential Omega = E - mu(N - N0), not the total energy: regressing it
against electron count while controlling for geometry gives dOmega/dN =
-1.58 eV against a reported potential of -3.36 eV, and adding mu(N - N0)
back recovers the expected dE/dN = mu to within 0.3%. Training on the
reported energy while supervising dE/dq = -potential would set the two
targets against each other.

Forces need no conversion: by the envelope theorem dOmega/dR|_mu =
dE/dR|_N, since the difference carries dE/dN - mu, which vanishes at the
self-consistent electron count.

Run from experiments/cpmace/:

    DATASETS=. python prepare.py
"""

import numpy as np

from ase.io import read
from marathon import comms
from marathon.data import datasets
from marathon.grain import prepare

DATA = "../../datasets/cpmace"

# Nominal valence electron count for C70 H89 N Ni O46 with standard VASP
# PAW potentials. See datasets/cpmace/README.md -- this is confirmed against
# the thermodynamics, not just assumed from the composition.
NOMINAL_ELECTRONS = 660.0

# 207 atoms per frame, one composition throughout. ToBatch packs
# batch_size - 1 real structures and pads the last slot, so 8 here is
# 7 x 207 = 1449 atoms per batch -- close to razor's 1620 despite the larger
# cell, which keeps the per-step cost in the same range.
BATCH_SIZE = 8


def convert(atoms):
    """Raw VASP fields -> the (E, q, dE/dq) the model is trained on."""
    n_excess = float(atoms.info["electron"]) - NOMINAL_ELECTRONS
    mu = float(atoms.info["potential"])

    grand_potential = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Omega -> E. Do this before touching the calculator, since
    # get_potential_energy() reads from it.
    total_energy = grand_potential + mu * n_excess

    from ase.calculators.singlepoint import SinglePointCalculator

    atoms.calc = SinglePointCalculator(atoms, energy=total_energy, forces=forces)

    atoms.info["total_charge"] = -n_excess
    atoms.info["work_function"] = -mu
    # keep the raw fields out of the prepared data: they are only meaningful
    # together with the conversion above, and PROPERTIES would drop them anyway
    atoms.info.pop("electron", None)
    atoms.info.pop("potential", None)
    return atoms


def load(name):
    return [convert(a) for a in read(f"{DATA}/{name}.xyz", index=":")]


train_data = load("cpmace_train")
valid_data = load("cpmace_val")

# no test split: the dataset ships one file and 1093 frames of a single
# composition. The 90/10 train/valid split is all the held-out data there is,
# so evaluate.py reports on `valid` and nothing pretends to be a test set.

q = np.array([a.info["total_charge"] for a in train_data])
w = np.array([a.info["work_function"] for a in train_data])
comms.talk(f"train q         {q.min():+.3f} .. {q.max():+.3f}  (mean {q.mean():+.3f})")
comms.talk(f"train dE/dq     {w.min():+.3f} .. {w.max():+.3f}  (mean {w.mean():+.3f})")

reporter = comms.reporter()
reporter.start("processing")

# total_charge is a model *input*, not a label, but marathon.grain.prepare()
# only persists atoms.info entries listed here -- anything omitted is silently
# dropped when the dataset is serialised.
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
