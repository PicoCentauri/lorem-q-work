# natcomm2025 charge conditioning

Pt(111)/water under grand-canonical DFT, from `../../datasets/natcomm2025/`
(Nat. Commun. 2025, `s41467-025-58871-7`). 233 atoms, Pt₁₀₀H₉₇O₃₆, fully
periodic, slab normal along **x**.

| dir | model | loss weights | epochs |
|---|---|---|---|
| `sr-l2c8-100ep/` | d64 l2 c8, `lr: false` | 1000 : 1 : 0.01 | 100 |
| `lr-l2c8-100ep/` | same, `lr: true` (`max_degree_lr: 0`) | 1000 : 1 : 0.01 | 100 |

`model.yaml` is byte-identical to `../cpmace/`'s l2c8 pair, so the two datasets
read across at fixed model; `settings.yaml` is identical between the two here,
differing only on the `lr` line of `model.yaml`, so the pair isolates the Ewald
head as in the other folders.

## Data

Targets come straight from the xyz — the conversion from DeepMD `fparam`
(the electron number Nₑ) happened in `../../datasets/natcomm2025/convert.py`:

```
total_charge  = -fparam
work_function = -dE/dfparam      (per-geometry quadratic fit, = -mu_e)
```

That folder's README has the derivation, its confirmation against the paper,
and the caveats. Two matter here:

- **The work-function range is dominated by off-equilibrium charge sampling.**
  W spans −4.49…+8.08 V, far wider than the paper's ±0.5 eV *fluctuation*,
  because the dataset evaluates each geometry at 4–18 electron counts and
  dW/dq = 7.68 V/e. This is real signal, not noise, but it is not the spread
  a constant-potential trajectory would show.
- **7.68 V/e is the supercell's compensating-charge plate** (1.22 μF/cm² over
  171 Å²), not a physical double-layer capacitance.

## The split is by geometry

| | frames | geometries |
|---|---|---|
| `data/train` | 15,107 | 2,475 |
| `data/valid` | 1,714 | 275 |

90/10 **on `group`**, seed 0, with an assertion that no geometry lands on both
sides. Each geometry appears at 4–18 electron counts sharing identical atomic
positions, so a per-frame split would score the model on structures it had
already fitted — razor's `struc_pk` situation. `evaluate.py` reconstructs the
same split from the same seed rather than storing it.

A useful side effect: validation contains whole charge stencils, so `dE/dq`
can be examined across a geometry's own charge range rather than at scattered
points.

There is **no test set**. The source ships none, and carving one out of the
same DP-GEN trajectories would not be independent.

## All frames are trained on, reactive ones included

No screening is applied. The max-force cut inherited from `../razor/` is a
no-op here (largest force 6.42 eV/Å), and the anomalous-capacitance frames are
kept **on purpose**: 10.6% of frames have d²E/dq² more than 20% off the
7.676 V/e median, and those are Volmer-step configurations carrying an extra
adsorbed hydrogen, not dielectric breakdown. See
`../../datasets/natcomm2025/README.md`. Reactions are the point of this
dataset, so they stay in.

**What to watch for because of that:** on those ~10% of frames the work
function is governed by bond formation rather than by the capacitor. A model
can fit the capacitive majority well and still be poor there, and a pooled
work-function RMSE will hide it. `d2E_dq2` rides along in the xyz precisely so
the metric can be split on it — worth doing before drawing conclusions about
`dE/dq` accuracy on this dataset.

## Loss weights must be recomputed — cpmace's are badly wrong here

Weights follow label variances, and this dataset's are unlike either previous
one:

| target | natcomm2025 | cpmace | razor |
|---|---|---|---|
| energy (per atom) | 3.02e−5 | 5.80e−5 | 1.40e−3 |
| forces | 0.362 | 0.708 | 0.473 |
| **work function** | **5.32** | 2.27e−2 | 1.69 |

The work-function variance is **233× cpmace's**, for the off-equilibrium
sampling reason above. Carrying weights across would be a disaster in the
opposite direction to cpmace's:

| weights | E% | F% | W% | |
|---|---|---|---|---|
| `1000 : 1 : 5` | 0.11 | 1.34 | **98.55** | cpmace's — would train almost nothing but W |
| `100 : 1 : 0.05` | 0.48 | 57.36 | 42.16 | razor's |
| `1 : 100 : 10` | 0.00 | 40.49 | 59.51 | cp-MACE's |
| **`3550 : 1 : 0.01`** | **20.1** | **67.8** | **12.1** | reproduces razor's tuned shares |

So the work function needs a **much smaller** weight here, not a larger one —
the opposite of the intuition that carried over from cpmace.

The runs use **`1000 : 1 : 0.01`** → E 6.8% / F 81.3% / W 11.9%, essentially
`../cpmace/`'s shares rather than razor's heavier energy term. Whether razor's
20% energy share would do better here is untested.

## Results

Both runs completed 100 epochs (15h16m / 11h53m). Converged **validation**
RMSE, last block:

| run | E (meV/atom) | F (meV/Å) | Φ (V) |
|---|---|---|---|
| `sr-l2c8-100ep` | 0.249 | 26.73 | 0.0912 |
| **`lr-l2c8-100ep`** | **0.155** | **23.85** | **0.0705** |

`evaluate.py` on the checkpointed models, 1714 validation frames:

| run | E | F | Φ | **Z\*** |
|---|---|---|---|---|
| `sr` | 0.25 | 26.73 | 0.0912 | 0.2757 |
| `lr` | **0.15** | **23.88** | **0.0699** | 0.2954 |

**The Ewald head wins on every target** — energy 38%, forces 11%, work
function 23%. That now holds on razor, razor_centre, cpmace and here.

Wall times are **not** comparable: both jobs landed on node a0933 and shared
its CPU and I/O, which is why `lr` came out *faster* than `sr`, backwards from
every other folder.

### The Born effective charges are bad, and that is the interesting result

`evaluate.py` derives `bec_z` by finite difference — Z\* = (A ε₀) dF/dq at
fixed geometry, which the validation split supports because it holds complete
charge stencils (275 geometries, `bec_z` derived for **1714/1714** frames).
F(q) is linear to ~4 meV/Å against a 602 meV/Å force scale, so the reference
is sound.

Against a label std of ~0.145 e:

| | Z\* RMSE | vs label std |
|---|---|---|
| `sr` | 0.2757 | **1.9×** |
| `lr` | 0.2954 | **2.0×** |

**Both are worse than predicting the mean.** Neither run supervises `bec_z`
here, so this is an unsupervised probe — but `../razor/`'s *unsupervised* `sr`
reached 0.0399 on the same kind of quantity, and `../razor_centre_paper/`'s
supervised run reaches 0.0192. So this is not simply "unsupervised is hard".

And `lr` is *worse* than `sr` on it, inverting the pattern it wins on
everywhere else. Unexplained. The most likely candidates are the 3D Ewald path
(this and cpmace are `pbc = T T T`, unlike razor's slab) and the compensating
plate making dF/dq an artefact of the supercell rather than a local property —
but neither is tested.

**One caveat on Z\***: it is one *column* of the 3×3 tensor, not the whole
thing. Charging a slab makes a field along the normal only, so `dF/dq` gives
Z\*_{i,α,x} for α ∈ {x,y,z}. The transverse entries are real off-diagonal
elements (std 0.054, 0.057 against Z\*_xx's 0.239), not zeros; the other two
columns need an in-plane field and are unobtainable here.

## Notes for when training starts

- **The energy baseline is load-bearing.** Raw energies are ~−347,303 eV where
  float32 spacing is 31.2 meV, against an energy signal of 5.5 meV/atom.
  `marathon.grain.prepare()` fits a per-species baseline automatically and the
  model learns the residual (spacing 0.48 µeV). Nothing to do by hand, but do
  not bypass it. Composition is constant so the per-species split is
  degenerate — only the total offset is identifiable.
- **The max-force screen is a no-op.** Largest force in the dataset is
  6.42 eV/Å, well under the 10 eV/Å cut inherited from `../razor/`. Kept so
  every folder screens identically.
- **q never changes sign** (−1.198 … −0.001), so the model only ever sees added
  electrons. Any `dE/dq` it learns is local to that window.
- **Budget.** 15,107 frames at 7 real structures per batch = 2,159
  batches/epoch. razor's 200-epoch reference is 145,800 updates, which is only
  **68 epochs** here — this is by far the largest dataset in the repo, so match
  on gradient updates, not epochs.
- `batch_size: 8` gives 7 × 233 = 1,631 atoms per batch, close to razor's 1,620
  and cpmace's 1,449.

## Layout

- `prepare.py` — group-wise split and `data/{train,valid}`. Already run.
- `evaluate.py` — parity and RMSE-vs-charge on `valid`; derived from
  `../cpmace/evaluate.py`. Charge bins are 0.1 e (q spans only 1.2 e here
  against razor's 3.5). It also derives `bec_z` by finite difference (below).
- `sr-l2c8-100ep/`, `lr-l2c8-100ep/`.
