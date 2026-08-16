"""Convert cpmace to the field names GRACE's charge conditioning reads.

Same two conversions `../cpmace/prepare.py` does for LOREM, and for the same
reasons -- see `../../datasets/cpmace/README.md`:

    q             = -(electron - 660)
    E_total       = energy + potential * (electron - 660)
    work_function = -potential                  ( = dE_total/dq )

The energy conversion is the one that is easy to miss. The `energy` field is
the **grand potential** Omega = E - mu(N - N0), not the total energy. Training
on the reported energy while supervising dE/dq = -potential would set the two
targets against each other, because dOmega/dq and dE/dq differ by exactly the
mu(N - N0) term.

660 is the standard VASP PAW valence sum for C70 H89 N Ni O46 (H 1, C 4, N 5,
O 6, Ni 10). It is not merely assumed: fitting the curvature of the reported
energies implies N0 = 660.2 independently, agreeing to 0.2 electrons.

Unlike the LOREM version there is no marathon `prepare()` step -- GRACE reads
extxyz directly (`cli/data.py::load_extxyz`), keeps the whole ASE object, and
`TotalChargeDataBuilder` reads `total_charge` / `work_function` straight out of
`atoms.info`. So this script only rewrites the xyz.

    python prepare.py
"""

from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

DATA = Path("../../datasets/cpmace")
OUT = Path("data")

# Nominal valence electron count for C70 H89 N Ni O46 with standard VASP PAW
# potentials. See ../../datasets/cpmace/README.md.
NOMINAL_ELECTRONS = 660.0


def convert(atoms):
    n_excess = float(atoms.info["electron"]) - NOMINAL_ELECTRONS
    mu = float(atoms.info["potential"])

    # atoms.get_potential_energy() reads the grand potential off the
    # SinglePointCalculator attached by the extxyz reader
    grand_potential = atoms.get_potential_energy()
    total_energy = grand_potential + mu * n_excess
    forces = atoms.get_forces()

    atoms.calc = SinglePointCalculator(atoms, energy=total_energy, forces=forces)
    atoms.info["total_charge"] = -n_excess
    atoms.info["work_function"] = -mu
    # drop the source fields so nothing downstream can read the grand potential
    # convention by accident
    atoms.info.pop("electron", None)
    atoms.info.pop("potential", None)
    atoms.info.pop("energy", None)
    return atoms


def main():
    OUT.mkdir(exist_ok=True)
    for split, src in (("train", "cpmace_train.xyz"), ("valid", "cpmace_val.xyz")):
        frames = [convert(a) for a in read(DATA / src, index=":")]
        q = np.array([a.info["total_charge"] for a in frames])
        w = np.array([a.info["work_function"] for a in frames])
        e = np.array([a.get_potential_energy() / len(a) for a in frames])
        f = np.concatenate([a.get_forces().ravel() for a in frames])
        print(
            f"{split}: {len(frames)} frames, {len(frames[0])} atoms each\n"
            f"  total_charge   {q.min():+.3f} .. {q.max():+.3f}  (std {q.std():.4f})\n"
            f"  work_function  {w.min():+.3f} .. {w.max():+.3f}  (std {w.std():.4f})\n"
            f"  E/atom         {e.min():.4f} .. {e.max():.4f}  (std {e.std():.5f})\n"
            f"  |F| max        {np.abs(f).max():.3f} eV/A"
        )
        write(OUT / f"{split}.xyz", frames, format="extxyz")
        print(f"  wrote {OUT / f'{split}.xyz'}")


if __name__ == "__main__":
    main()
