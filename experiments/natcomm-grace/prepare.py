"""Split natcomm2025 into the train/valid xyz files GRACE reads.

No field rewriting: `../../datasets/natcomm2025/convert.py` already wrote
`total_charge` and `work_function` into atoms.info, which is exactly what
`TotalChargeDataBuilder` reads by default.

SPLITS -- identical seed, fraction and grouping to
`../natcomm2025/prepare.py`, so the GRACE and LOREM runs are held out on the
same structures.

**The split is on `group`, never within it.** The 4-18 charge states of one
geometry share atomic positions, so a per-frame split would put the same
geometry on both sides and leak. This is the analogue of razor's `struc_pk`.

Note this trains on ALL frames including reaction (Volmer-step) configurations;
`../natcomm2025/README.md` records why the low-curvature tail is chemistry
rather than dielectric breakdown, and therefore is not screened out.

    python prepare.py
"""

from pathlib import Path

import numpy as np
from ase.io import read, write

DATA = Path("../../datasets/natcomm2025")
OUT = Path("data")
SPLIT_SEED = 0
VALID_FRACTION = 0.10


def main():
    OUT.mkdir(exist_ok=True)
    frames = read(DATA / "natcomm2025.xyz", index=":")

    groups = np.array([a.info["group"] for a in frames])
    uniq = np.unique(groups)
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(uniq))
    valid_groups = set(uniq[order[: int(round(VALID_FRACTION * len(uniq)))]].tolist())

    train = [a for a in frames if a.info["group"] not in valid_groups]
    valid = [a for a in frames if a.info["group"] in valid_groups]

    tg = {a.info["group"] for a in train}
    vg = {a.info["group"] for a in valid}
    assert not (tg & vg), "a geometry appears in both splits -- positions would leak"

    for split, fr in (("train", train), ("valid", valid)):
        q = np.array([a.info["total_charge"] for a in fr])
        w = np.array([a.info["work_function"] for a in fr])
        e = np.array([a.get_potential_energy() / len(a) for a in fr])
        f = np.concatenate([a.get_forces().ravel() for a in fr])
        print(
            f"{split}: {len(fr)} frames / {len({a.info['group'] for a in fr})} geometries, "
            f"{len(fr[0])} atoms each\n"
            f"  total_charge  {q.min():+.3f} .. {q.max():+.3f}  (std {q.std():.4f})\n"
            f"  work_function {w.min():+.3f} .. {w.max():+.3f}  (std {w.std():.4f})\n"
            f"  E/atom        {e.min():.4f} .. {e.max():.4f}  (std {e.std():.5f})\n"
            f"  |F| max       {np.abs(f).max():.3f} eV/A"
        )
        write(OUT / f"{split}.xyz", fr, format="extxyz")
        print(f"  wrote {OUT / f'{split}.xyz'}")


if __name__ == "__main__":
    main()
