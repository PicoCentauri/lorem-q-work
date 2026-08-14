"""Rebuild the publication's train/test selection using *our* labels.

`Publication_data_for_ploche/{train,test}.xyz` selects a subset of the same
DFT calculations we already have, but reports different quantities. Verified
against our files on all 5113 frames:

  * every paper frame matches one of ours on (struc_pk, charge) -- 0 unmatched
  * positions, cell, pbc and species order are bit-identical (max diff 0.0)
  * `DFT_wf` == our `work_function` exactly (corr 1.000000, max diff 0.0000)
  * `DFT_d2Edq2` == the curvature of our 3-point stencil exactly (corr 1.0000)

So the underlying calculations are the same. What differs is the reference
charge of the energy and force labels: `DFT_E0` / `DFT_F0` are
back-extrapolated to q = 0, whereas our `energy` / `forces` are at the frame's
own charge. Numerically, DFT_E0 minus our E extrapolated to q=0 is
-0.119 +- 0.375 eV, against +1.893 +- 3.555 eV when compared at the frame's
own charge.

This script therefore keeps the publication's *selection* and uses our
*labels*, which are the ones the models here are trained on.

WHAT THE PUBLICATION'S SELECTION IS
-----------------------------------
One frame per structure -- 4598 train frames over 4598 struc_pk, 515 test over
515 -- with a single charge each, drawn from ~40 distinct values in [-1, +1].
Our own files instead carry a 3-point stencil (q_MD +- 0.25) per structure.
So this is a genuinely different sampling of the same calculations, not a
re-split of ours, and it is the one to use when comparing against the paper.

Note the publication's split does not respect ours: 452 of its 515 test
structures fall in our `razor_train`. That is why the two cannot be mixed --
training on our split and testing on theirs would leak. Using their split
end-to-end, as here, is self-consistent.

    python make_zausi_split.py
"""

import collections
from pathlib import Path

import numpy as np
from ase.io import iread, write

PAPER = Path("Publication_data_for_ploche")
OURS = ("razor_train", "razor_val", "razor_test")
OUT = {"train": "train_zausi.xyz", "test": "test_zausi.xyz"}


def index_ours():
    """(struc_pk, charge) -> our Atoms, from every split we have."""
    idx = {}
    for f in OURS:
        for a in iread(f"{f}.xyz", index=":"):
            key = (int(a.info["struc_pk"]), round(float(a.info["bias_charge"]), 4))
            a.info["source_split"] = f
            idx[key] = a
    return idx


def main():
    idx = index_ours()
    print(f"indexed {len(idx)} (struc_pk, charge) frames from {len(OURS)} of our files")

    picked = {}
    for name in ("train", "test"):
        out, missing = [], []
        for a in iread(PAPER / f"{name}.xyz", index=":"):
            key = (int(a.info["struc_pk"]), round(float(a.info["charge"]), 4))
            if key in idx:
                frame = idx[key]
                # geometry is bit-identical, so this is a label swap, not a
                # re-computation -- assert rather than trust
                assert np.allclose(frame.get_positions(), a.get_positions(), atol=1e-8)
                assert np.allclose(frame.get_cell()[:], a.get_cell()[:], atol=1e-8)
                out.append(frame)
            else:
                missing.append(key)
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} paper frames absent from ours, e.g. {missing[:3]}")
        picked[name] = out
        print(f"  paper {name}: {len(out)} frames matched, 0 missing")

    # the publication's own split must not share structures between train and
    # test, or the comparison it supports would be meaningless
    pk = {k: {int(a.info["struc_pk"]) for a in v} for k, v in picked.items()}
    shared = pk["train"] & pk["test"]
    assert not shared, f"{len(shared)} struc_pk appear in both paper splits"
    print(f"  struc_pk disjoint: {len(pk['train'])} train / {len(pk['test'])} test, 0 shared")

    for name, frames in picked.items():
        q = np.array([a.info["bias_charge"] for a in frames])
        w = np.array([a.info["work_function"] for a in frames])
        pol = np.array([bool(a.info["polarizable"]) for a in frames])
        src = collections.Counter(a.info["source_split"] for a in frames)
        print(
            f"\n{OUT[name]}: {len(frames)} frames\n"
            f"  charge {q.min():+.2f}..{q.max():+.2f} ({len(np.unique(np.round(q,4)))} distinct)\n"
            f"  work_function {w.min():.2f}..{w.max():.2f} (std {w.std():.3f})\n"
            f"  polarizable {pol.sum()}/{len(pol)} ({100*pol.mean():.1f}%)\n"
            f"  drawn from our splits: {dict(src)}"
        )
        write(OUT[name], frames, format="extxyz")

    print("\nwrote " + ", ".join(OUT.values()))


if __name__ == "__main__":
    main()
