"""Split data.xyz into a 90/10 train/validation set.

Deterministic (seed 0), so the split can be regenerated rather than trusted.
Frames are split individually: unlike razor, where a geometry appears at three
bias charges and the splits must keep those together, every frame here is an
independently sampled configuration -- consecutive frames differ by ~1.8 A
RMSD -- so there is no group leakage to guard against.

The written files are verbatim subsets of data.xyz: raw `energy`, `potential`
and `electron` fields, unconverted. The physical interpretation (which is not
trivial -- see README.md) is applied in experiments/cpmace/prepare.py, so the
dataset files stay faithful to what came out of VASP.

    python split.py
"""

import numpy as np
from ase.io import read, write

SEED = 0
VALID_FRACTION = 0.10

frames = read("data.xyz", index=":")
n = len(frames)

rng = np.random.default_rng(SEED)
order = rng.permutation(n)
n_valid = int(round(VALID_FRACTION * n))
valid_idx = sorted(order[:n_valid].tolist())
train_idx = sorted(order[n_valid:].tolist())

write("cpmace_train.xyz", [frames[i] for i in train_idx])
write("cpmace_val.xyz", [frames[i] for i in valid_idx])

print(f"{n} frames -> {len(train_idx)} train / {len(valid_idx)} valid")
for name, idx in (("train", train_idx), ("valid", valid_idx)):
    q = np.array([-(frames[i].info["electron"] - 660.0) for i in idx])
    mu = np.array([frames[i].info["potential"] for i in idx])
    print(
        f"  {name:<6} q {q.min():+.3f}..{q.max():+.3f} (mean {q.mean():+.3f})   "
        f"potential {mu.min():+.3f}..{mu.max():+.3f}"
    )
