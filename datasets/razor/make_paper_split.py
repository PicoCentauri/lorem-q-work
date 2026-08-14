"""Rebuild the publication's train/test selection with *our* energy, force and
work-function labels.

The publication data (`Publication_data_for_ploche/`) turns out to be
`razor_centre` restricted to |q| <= 1.0 e. Verified against our files on all
5113 frames, not assumed:

  * every paper frame matches one of ours on (struc_pk, charge), 0 unmatched
  * positions, cell, pbc and species order are bit-identical (max diff 0.0)
  * `DFT_wf` == our `work_function` exactly (corr 1.000000, max diff 0.0000)
  * `DFT_d2Edq2` == the curvature of our 3-point stencil exactly (corr 1.0000)
  * 100% of its frames sit at `bias_charge == q_MD`, i.e. every one is a
    stencil centre, and every struc_pk is present in `razor_centre.xyz`

What differs is only the reference charge of the energy and force labels:
`DFT_E0` / `DFT_F0` are back-extrapolated to q = 0, while ours are at the
frame's own charge. `DFT_E0` sits within -0.119 +- 0.375 eV of our E
extrapolated to q = 0, against +1.893 +- 3.555 eV compared at the frame's own
charge. This script keeps their *selection* and our *labels*.

SURVIVING DELETION OF THE SOURCE FOLDER
---------------------------------------
Because every selected frame is a stencil centre, the selection is fully
determined by `struc_pk` alone -- no charges need recording. The two id lists
are therefore written alongside the xyz files, and this script falls back to
them when `Publication_data_for_ploche/` is gone, so the split stays
reproducible after the 850 MB source is deleted.

    python make_paper_split.py
"""

from pathlib import Path

import numpy as np
from ase.io import iread, write

PAPER = Path("Publication_data_for_ploche")
OURS = ("razor_train", "razor_val", "razor_test")
SPLITS = {
    "train": ("razor_centre_paper_train.xyz", "razor_centre_paper_train_struc_pk.txt"),
    "test": ("razor_centre_paper_test.xyz", "razor_centre_paper_test_struc_pk.txt"),
}


def selection_from_source():
    """(name -> set of struc_pk) read from the publication folder, and checked
    frame-by-frame against ours before being trusted."""
    idx = {}
    for f in OURS:
        for a in iread(f"{f}.xyz", index=":"):
            idx[(int(a.info["struc_pk"]), round(float(a.info["bias_charge"]), 4))] = a

    sel = {}
    for name in SPLITS:
        pks, off_centre = [], 0
        for a in iread(PAPER / f"{name}.xyz", index=":"):
            pk = int(a.info["struc_pk"])
            key = (pk, round(float(a.info["charge"]), 4))
            if key not in idx:
                raise RuntimeError(f"{name}: {key} not present in our data")
            ours = idx[key]
            # a label swap only makes sense if the structure is the same one
            assert np.allclose(ours.get_positions(), a.get_positions(), atol=1e-8)
            assert np.allclose(ours.get_cell()[:], a.get_cell()[:], atol=1e-8)
            # the selection is recorded as struc_pk alone, which is only valid
            # because every frame is a stencil centre -- check, do not assume
            if not np.isclose(float(ours.info["bias_charge"]), float(ours.info["q_MD"]), atol=1e-6):
                off_centre += 1
            pks.append(pk)
        if off_centre:
            raise RuntimeError(
                f"{name}: {off_centre} frames are not stencil centres, so struc_pk "
                "alone does not determine the selection -- record charges too"
            )
        assert len(pks) == len(set(pks)), f"{name}: duplicate struc_pk"
        sel[name] = set(pks)
        print(f"  {name}: {len(pks)} frames verified against ours, all stencil centres")
    return sel


def selection_from_lists():
    sel = {}
    for name, (_, listfile) in SPLITS.items():
        sel[name] = {int(x) for x in Path(listfile).read_text().split()}
        print(f"  {name}: {len(sel[name])} struc_pk read from {listfile}")
    return sel


def main():
    if PAPER.exists():
        print(f"reading the selection from {PAPER}/")
        sel = selection_from_source()
        for name, (_, listfile) in SPLITS.items():
            Path(listfile).write_text("\n".join(str(p) for p in sorted(sel[name])) + "\n")
            print(f"  wrote {listfile}")
    else:
        print(f"{PAPER}/ is gone -- rebuilding from the stored struc_pk lists")
        sel = selection_from_lists()

    shared = sel["train"] & sel["test"]
    assert not shared, f"{len(shared)} struc_pk appear in both splits"

    # take the stencil centre of each selected structure out of razor_centre,
    # which is one frame per struc_pk at bias_charge == q_MD by construction
    centres = {}
    for a in iread("razor_centre.xyz", index=":"):
        centres[int(a.info["struc_pk"])] = a

    for name, (outfile, _) in SPLITS.items():
        missing = sel[name] - set(centres)
        if missing:
            raise RuntimeError(f"{name}: {len(missing)} struc_pk absent from razor_centre.xyz")
        frames = [centres[pk] for pk in sorted(sel[name])]
        q = np.array([a.info["bias_charge"] for a in frames])
        w = np.array([a.info["work_function"] for a in frames])
        pol = np.array([bool(a.info["polarizable"]) for a in frames])
        print(
            f"\n{outfile}: {len(frames)} frames\n"
            f"  charge {q.min():+.2f}..{q.max():+.2f} ({len(np.unique(np.round(q, 4)))} distinct)\n"
            f"  work_function {w.min():.2f}..{w.max():.2f} (std {w.std():.3f})\n"
            f"  polarizable {pol.sum()}/{len(pol)} ({100 * pol.mean():.1f}%)"
        )
        write(outfile, frames, format="extxyz")

    print("\nwrote " + ", ".join(v[0] for v in SPLITS.values()))


if __name__ == "__main__":
    main()
