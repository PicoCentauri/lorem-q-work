"""Split razor_centre_paper into the train/valid/test xyz files GRACE reads.

Deliberately does NOT rewrite any field. The source already carries
`work_function` in atoms.info and `bias_charge` as the charge, so
`TotalChargeDataBuilder(charge_key="bias_charge")` reads it directly -- see
input.yaml. Not renaming is the safer choice: a rename that silently fails
leaves the charge defaulting to 0.0, which is exactly the failure that cost
`../razor_centre_paper/` two training runs (marathon wrote NaN for the absent
key). Here the builder logs `N/N structures` and warns on any default, so a
misconfiguration is loud.

SPLITS -- identical seed and fraction to `../razor_centre_paper/prepare.py`, so
the GRACE and LOREM runs are held out on the same structures:

  * 90/10 carve-out of the publication's 4598-frame train set
  * the publication's own 515-frame test set, untouched

A per-frame split is safe here: this is one frame per `struc_pk`, so there is
no charge stencil to keep together. Asserted below, not assumed.

    python prepare.py
"""

from pathlib import Path

import numpy as np
from ase.io import read, write

DATA = Path("../../datasets/razor")
OUT = Path("data")
SEED = 0
VALID_FRACTION = 0.10


def main():
    OUT.mkdir(exist_ok=True)
    train_all = read(DATA / "razor_centre_paper_train.xyz", index=":")
    test = read(DATA / "razor_centre_paper_test.xyz", index=":")

    pks = [int(a.info["struc_pk"]) for a in train_all]
    assert len(pks) == len(set(pks)), "struc_pk repeats; a per-frame split would leak"

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(train_all))
    n_valid = int(round(VALID_FRACTION * len(train_all)))
    valid = [train_all[i] for i in sorted(order[:n_valid])]
    train = [train_all[i] for i in sorted(order[n_valid:])]

    tp = {int(a.info["struc_pk"]) for a in train}
    vp = {int(a.info["struc_pk"]) for a in valid}
    xp = {int(a.info["struc_pk"]) for a in test}
    assert not (tp & vp) and not (tp & xp) and not (vp & xp), "splits share struc_pk"

    for split, frames in (("train", train), ("valid", valid), ("test", test)):
        q = np.array([a.info["bias_charge"] for a in frames])
        w = np.array([a.info["work_function"] for a in frames])
        e = np.array([a.get_potential_energy() / len(a) for a in frames])
        f = np.concatenate([a.get_forces().ravel() for a in frames])
        print(
            f"{split}: {len(frames)} frames, {len(frames[0])} atoms each\n"
            f"  bias_charge   {q.min():+.3f} .. {q.max():+.3f}  (std {q.std():.4f})\n"
            f"  work_function {w.min():+.3f} .. {w.max():+.3f}  (std {w.std():.4f})\n"
            f"  E/atom        {e.min():.4f} .. {e.max():.4f}  (std {e.std():.5f})\n"
            f"  |F| max       {np.abs(f).max():.3f} eV/A"
        )
        write(OUT / f"{split}.xyz", frames, format="extxyz")
        print(f"  wrote {OUT / f'{split}.xyz'}")


if __name__ == "__main__":
    main()
