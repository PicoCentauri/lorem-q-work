"""Convert the DeepMD-format natcomm2025 data to a single extxyz.

No dpdata required: DeepMD's on-disk format is plain .npy plus two text
files, so numpy + ase is enough and avoids a dependency and a venv.

    <system>/type.raw       atom type indices, one per atom
    <system>/type_map.raw   element symbols, indexed by the above
    <system>/set.NNN/coord.npy   (nframes, natoms*3)   A
    <system>/set.NNN/box.npy     (nframes, 9)          A, rows = cell vectors
    <system>/set.NNN/energy.npy  (nframes,)            eV
    <system>/set.NNN/force.npy   (nframes, natoms*3)   eV/A
    <system>/set.NNN/fparam.npy  (nframes, 1)          see below

DeepMD uses eV and Angstrom, so no unit conversion is needed anywhere.

WHAT fparam IS
--------------
The training input declares `numb_fparam: 1` without naming it, so it was
identified from the data (see README.md for the full argument):

  * it is not an electrode potential -- E(fparam) curves *upward*
    (d2E/dfparam2 = +7.68 eV, IQR [7.44, 7.73]), whereas a grand potential
    would give d2/dU2 = -C < 0;
  * that curvature is near-constant over all 2750 geometries, the signature
    of a capacitive q^2/2C term, so fparam is charge-like;
  * dE/dfparam is negative (median -2.53 eV). Read as a charge that would
    make the work function negative; read as an excess *electron* count it
    gives dE/dq = -dE/dfparam = +2.53 V, inside razor's 2.51-6.99 V range.

So fparam is the number of excess electrons, and

    total_charge  = -fparam
    work_function =  dE/dq = -dE/dfparam

Any constant offset in the definition of fparam (whether fparam = 0 is
exactly neutral) is unresolved from the data alone. It does not affect
training: a shift in q is absorbed by the FiLM charge conditioning for a
single-composition dataset. It would matter for transferring a model to
another system.

WHERE THE WORK FUNCTION COMES FROM
----------------------------------
There is no per-frame work-function label. There does not need to be: every
one of the 2750 distinct geometries appears at 5-6 different fparam values,
so dE/dfparam is available by finite difference on a fixed geometry -- the
same trick razor's `dEdq_fd` uses. A quadratic is fitted to E(fparam) per
geometry and differentiated at each frame's own fparam, which handles the
curvature rather than assuming E is linear in the charge.

`group` in the output is the geometry id. **Splits must be made on it, never
within it** -- the 5-6 charge states of one geometry share atomic positions,
so splitting per frame would leak. This mirrors razor's `struc_pk`.

    python convert.py
"""

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

ROOT = Path("data_set")
OUT = Path("natcomm2025.xyz")

# Exact duplicates -- identical geometry, fparam, energy AND forces -- are
# 13.9% of the source. They carry no information and only reweight whatever
# they repeat, so they are dropped. Set False to keep them (flagged
# `duplicate=True`) if fidelity to the source matters more than size.
DROP_DUPLICATES = True

# DeepMD data is fully periodic; the slab normal is x (the long cell vector).
PBC = (True, True, True)


def load_system(sysdir):
    types = np.loadtxt(sysdir / "type.raw", dtype=int)
    symbols_map = (sysdir / "type_map.raw").read_text().split()
    symbols = [symbols_map[t] for t in types]
    n = len(types)

    frames = []
    for setdir in sorted(sysdir.glob("set.*")):
        coord = np.load(setdir / "coord.npy").reshape(-1, n, 3)
        box = np.load(setdir / "box.npy").reshape(-1, 3, 3)
        energy = np.load(setdir / "energy.npy").ravel()
        force = np.load(setdir / "force.npy").reshape(-1, n, 3)
        fparam = np.load(setdir / "fparam.npy").ravel()
        for i in range(len(energy)):
            frames.append((coord[i], box[i], energy[i], force[i], fparam[i]))
    return symbols, frames


def main():
    systems = sorted(d for d in ROOT.iterdir() if d.is_dir())
    print(f"{len(systems)} systems")

    records = []
    for sysdir in systems:
        symbols, frames = load_system(sysdir)
        for coord, box, energy, force, fparam in frames:
            records.append(
                dict(
                    symbols=symbols,
                    coord=coord,
                    box=box,
                    energy=float(energy),
                    force=force,
                    fparam=float(fparam),
                    system=sysdir.name,
                )
            )
    print(f"{len(records)} frames, {len(records[0]['symbols'])} atoms each")

    # group by exact geometry -- this is what makes dE/dfparam obtainable
    groups = defaultdict(list)
    for i, r in enumerate(records):
        key = hashlib.md5(np.ascontiguousarray(r["coord"]).tobytes()).hexdigest()
        groups[key].append(i)
    print(f"{len(groups)} distinct geometries")

    sizes = np.array([len(v) for v in groups.values()])
    print(f"  frames per geometry: min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}")

    # Exact duplicates: same geometry AND fparam AND energy AND forces. 13.9%
    # of this dataset. They carry no information and only reweight whatever
    # they duplicate, but they are left in so the file stays faithful to the
    # source -- `duplicate=True` makes dropping them a one-liner downstream.
    # They cannot leak across a split: a duplicate shares its geometry, so the
    # group key already keeps it on one side.
    seen = set()
    n_dup = 0
    for r in records:
        h = hashlib.md5(
            np.ascontiguousarray(
                np.concatenate([r["coord"].ravel(), r["force"].ravel(),
                                [r["fparam"], r["energy"]]])
            ).tobytes()
        ).hexdigest()
        r["duplicate"] = h in seen
        n_dup += r["duplicate"]
        seen.add(h)
    print(f"  {n_dup} exact duplicate frames ({100*n_dup/len(records):.1f}%)"
          f"{' -- dropped' if DROP_DUPLICATES else ' -- flagged, kept'}")
    if DROP_DUPLICATES:
        records = [r for r in records if not r["duplicate"]]
        # regroup: dropping frames changes group membership and sizes, and the
        # E(fparam) fits below must see only the surviving frames
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[hashlib.md5(np.ascontiguousarray(r["coord"]).tobytes()).hexdigest()].append(i)
        sizes = np.array([len(v) for v in groups.values()])
        print(f"  after dropping: {len(records)} frames, {len(groups)} geometries, "
              f"frames per geometry min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}")

    n_lin = 0
    for gid, (key, idx) in enumerate(sorted(groups.items())):
        f = np.array([records[i]["fparam"] for i in idx])
        e = np.array([records[i]["energy"] for i in idx])
        # quadratic where there is enough spread to support it, linear
        # otherwise; never extrapolate a curvature from two points
        if len(f) >= 3 and np.ptp(f) > 0.05:
            c2, c1, _ = np.polyfit(f, e, 2)
            dEdf = c1 + 2 * c2 * f
        else:
            c1 = np.polyfit(f, e, 1)[0] if np.ptp(f) > 0 else np.nan
            dEdf = np.full_like(f, c1)
            n_lin += 1
        for j, i in enumerate(idx):
            records[i]["group"] = gid
            records[i]["work_function"] = -float(dEdf[j])
            records[i]["n_in_group"] = len(idx)
    if n_lin:
        print(f"  {n_lin} geometries fell back to a linear fit")

    wf = np.array([r["work_function"] for r in records])
    q = np.array([-r["fparam"] for r in records])
    print(f"\n  total_charge  {q.min():+.4f} .. {q.max():+.4f}")
    print(f"  work_function {wf.min():+.3f} .. {wf.max():+.3f}  (mean {wf.mean():+.3f}, std {wf.std():.3f})")
    bad = ~np.isfinite(wf)
    if bad.any():
        raise RuntimeError(f"{bad.sum()} frames have no usable dE/dfparam")

    atoms_list = []
    for r in records:
        a = Atoms(
            symbols=r["symbols"],
            positions=r["coord"],
            cell=r["box"],
            pbc=PBC,
        )
        a.calc = SinglePointCalculator(a, energy=r["energy"], forces=r["force"])
        a.info["total_charge"] = -r["fparam"]
        a.info["work_function"] = r["work_function"]
        # provenance, and the key any split must be grouped on
        a.info["fparam"] = r["fparam"]
        a.info["group"] = r["group"]
        a.info["n_in_group"] = r["n_in_group"]
        # "sys/" prefix is not decoration: six of the directories are named
        # "0.60".."1.10", and extxyz round-trips a bare "0.60" as the float
        # 0.6, which would leave `system` typed float for those and str for
        # the data.* ones. The prefix forces a string both ways.
        a.info["system"] = "sys/" + r["system"]
        a.info["duplicate"] = bool(r["duplicate"])
        atoms_list.append(a)

    write(OUT, atoms_list, format="extxyz")
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
