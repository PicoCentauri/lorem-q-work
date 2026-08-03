"""Field sign-convention checker for the water_external_field dataset files.

Physics anchor: an external E-field exerts force q_i * E on atom i (net force
on nucleus + its share of the electron density). In water, O carries negative
partial charge and H positive, so the field-induced force change at fixed
geometry, projected on the stored field direction, must be NEGATIVE for O and
POSITIVE for H -- if the stored `ext_field` is the field the DFT actually saw,
in the F = qE convention. A perfect 100%/0% split the other way means the
stored field vector is the negative of the true one.

Three independent tests:
  1. train/val/test: internal matched pairs (same geometry, field-free +
     perturbed) -> per-species sign of dF . f_hat.
  2. paired_{free,perturbed}: same test across the two files, plus an
     energy-based check corr(dE, mu_hirshfeld . F_stored); physical
     convention gives dE ~ -mu.F, so a POSITIVE correlation means flipped.
  3. magnitude_sweep: per-series linear slope of F_i . d_hat vs. signed
     magnitude, with d_hat defined from the largest POSITIVE-magnitude frame
     (using the max-|m| frame is ambiguous: it can be the -0.2 one).

Usage:
    python check_field_conventions.py [path/to/datasets/water_external_field]

Result on 2026-08-03 (as shipped to us): train.xyz PHYSICAL, all five other
files FLIPPED. The model trains on train.xyz, so every raw-file evaluation
(Exp 1/2/3, and validation during training via val.xyz) fought the flip.
"""

import sys

import numpy as np
from ase.io import read


def verdict(frac_O_pos):
    if frac_O_pos < 0.05:
        return "PHYSICAL (F = qE)"
    if frac_O_pos > 0.95:
        return "FLIPPED"
    return "AMBIGUOUS -- investigate!"


def check_internal_pairs(path, name):
    frames = read(path, ":")
    groups = {}
    for idx, a in enumerate(frames):
        key = (len(a), a.positions.round(6).tobytes())
        groups.setdefault(key, []).append(idx)

    proj_O, proj_H = [], []
    n_pairs = 0
    for v in groups.values():
        if len(v) != 2:
            continue
        a, b = frames[v[0]], frames[v[1]]
        fa = np.asarray(a.info["ext_field"], float)
        fb = np.asarray(b.info["ext_field"], float)
        if np.linalg.norm(fa) > np.linalg.norm(fb):
            a, b, fa, fb = b, a, fb, fa
        if np.linalg.norm(fa) > 1e-12 or np.linalg.norm(fb) < 1e-12:
            continue
        n_pairs += 1
        d = fb / np.linalg.norm(fb)
        dF = (b.get_forces() - a.get_forces()) @ d * 1000  # meV/A
        Z = a.get_atomic_numbers()
        proj_O.extend(dF[Z == 8])
        proj_H.extend(dF[Z == 1])
    if n_pairs == 0:
        print(f"{name}: no internal matched pairs, cannot check")
        return
    pO, pH = np.array(proj_O), np.array(proj_H)
    frac = (pO > 0).mean()
    print(f"{name}: {n_pairs} pairs | O {pO.mean():7.1f} meV/A (frac>0 {frac:.2f}) | "
          f"H {pH.mean():7.1f} (frac>0 {(pH > 0).mean():.2f}) | {verdict(frac)}")


def check_paired_files(free_path, pert_path):
    free = read(free_path, ":")
    pert = read(pert_path, ":")
    proj_O, proj_H, dE, muF = [], [], [], []
    for a, b in zip(free, pert):
        f = np.asarray(b.info["ext_field"], float)
        n = np.linalg.norm(f)
        if n < 1e-12:
            continue
        d = f / n
        dF = (b.arrays["dft_forces"] - a.arrays["dft_forces"]) @ d * 1000
        Z = a.get_atomic_numbers()
        proj_O.extend(dF[Z == 8])
        proj_H.extend(dF[Z == 1])
        q = np.asarray(a.arrays["dft_hirshfeld"], float).reshape(len(a))
        com = a.positions.mean(axis=0)
        mu = ((a.positions - com) * q[:, None]).sum(axis=0)  # e*A
        dE.append(b.info["dft_energy"] - a.info["dft_energy"])  # eV
        muF.append(mu @ f)  # eV
    pO, pH = np.array(proj_O), np.array(proj_H)
    frac = (pO > 0).mean()
    print(f"paired files: {len(dE)} pairs | O {pO.mean():7.1f} meV/A (frac>0 {frac:.2f}) | "
          f"H {pH.mean():7.1f} (frac>0 {(pH > 0).mean():.2f}) | {verdict(frac)}")
    r = np.corrcoef(dE, muF)[0, 1]
    slope = np.polyfit(muF, dE, 1)[0]
    print(f"  energy check: corr(dE, mu_hirshfeld . F_stored) = {r:+.3f}, "
          f"slope {slope:+.2f} (physical: negative; positive => FLIPPED)")


def check_sweep(path):
    sweep = read(path, ":")
    series = {}
    for idx, a in enumerate(sweep):
        series.setdefault(a.info["series"], []).append(idx)

    slope_O, slope_H = [], []
    bad = 0
    for idxs in series.values():
        idxs.sort(key=lambda i: sweep[i].info["field_magnitude"])
        mags = np.array([sweep[i].info["field_magnitude"] for i in idxs])
        pos = [i for i in idxs if sweep[i].info["field_magnitude"] > 0]
        i_ref = max(pos, key=lambda i: sweep[i].info["field_magnitude"])
        f_ref = np.asarray(sweep[i_ref].info["ext_field"], float)
        d = f_ref / np.linalg.norm(f_ref)
        for i in idxs:
            f = np.asarray(sweep[i].info["ext_field"], float)
            if not np.allclose(f, sweep[i].info["field_magnitude"] * d, atol=1e-6):
                bad += 1
        F = np.stack([sweep[i].get_forces() for i in idxs])
        proj = F @ d * 1000
        Z = sweep[idxs[0]].get_atomic_numbers()
        slopes = np.polyfit(mags, proj, 1)[0]
        slope_O.extend(slopes[Z == 8])
        slope_H.extend(slopes[Z == 1])
    sO, sH = np.array(slope_O), np.array(slope_H)
    frac = (sO > 0).mean()
    print(f"sweep: {len(series)} series, {bad} frames where ext_field != m*d_hat | "
          f"dO/dm {sO.mean():7.1f} (frac>0 {frac:.2f}) | dH/dm {sH.mean():7.1f} | "
          f"{verdict(frac)}")


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    base = base.rstrip("/") + "/"
    p = "water_external_field_"
    check_internal_pairs(base + p + "train.xyz", "train")
    check_internal_pairs(base + p + "val.xyz", "val")
    check_internal_pairs(base + p + "test.xyz", "test")
    check_paired_files(base + p + "paired_free.xyz", base + p + "paired_perturbed.xyz")
    check_sweep(base + p + "magnitude_sweep.xyz")
