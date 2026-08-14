# razor_centre_paper

The publication's subset of razor, with **our** labels. Data is
`../../datasets/razor/razor_centre_paper_{train,test}.xyz`.

| dir | model | loss weights | shares |
|---|---|---|---|
| `sr-l2c8-800ep/` | d64 l2 c8, `lr: false` | 150 : 1 : 0.15 | E 20.9 / F 66.5 / W 12.6 |
| `lr-l2c8-800ep/` | same, `lr: true` (`max_degree_lr: 0`) | 150 : 1 : 0.15 | same |
| `sr-l2c8-bec-800ep/` | same as `sr`, `predict_bec: True` | + `bec_z: 1.0` | E 20.3 / F 64.7 / W 12.3 / **Z 2.7** |

`model.yaml` is byte-identical to `../razor/`'s `l2c8`, and to `../cpmace/`'s
and `../natcomm2025/`'s, so all four datasets read across at fixed model. The
`sr`/`lr` pair differs on the `lr` line alone; `sr-bec` differs from `sr` only
by `predict_bec` and the extra loss term.

## What this dataset is

`razor_centre` restricted to **|q| ≤ 1.0 e**, one frame per structure at the
stencil centre — verified, not assumed, in `../../datasets/razor/README.md`.
The publication's own `DFT_E0`/`DFT_F0` are back-extrapolated to q = 0; ours
are at the frame's own charge, and its `DFT_wf` equals our `work_function`
exactly.

The charge cap does most of the `polarizable` filtering for free: 99.3% of
these frames are polarizable, against 36.5% of the 876 razor_centre structures
it leaves out.

## Splits

| | frames | source |
|---|---|---|
| `data/train` | 4,138 | 90% of the publication's train |
| `data/valid` | 460 | 10% of the publication's train |
| `data/test` | 515 | the publication's test, untouched |

The validation set is carved out of *their train* so *their test* stays a
clean comparison point against the published numbers.

**A per-frame split is safe here**, unlike in `../razor/` or
`../natcomm2025/`: this is one frame per `struc_pk`, so there is no charge
stencil to keep together. `prepare.py` asserts that, and that no `struc_pk`
appears in two splits.

**Do not mix this with `../razor/`'s split.** 452 of these 515 test structures
sit in `razor_train`, so a model trained there and tested here would leak.

## Loss weights

`150 : 1 : 0.15`, not razor's raw `100 : 1 : 0.05`. Variances against razor's:

| target | here | razor |
|---|---|---|
| energy (per atom) | 1.05e−3 | 1.40e−3 |
| forces | 5.04e−1 | 4.73e−1 |
| **work function** | **6.36e−1** | 1.69 |
| `bec_z` | 2.12e−2 | — |

Energy and forces are close to razor's, so only the work-function weight
really moves: the |q| ≤ 1 cap cuts W's variance to a third, and razor's
nominal 0.05 would buy only a 5.0% share instead of 12.1%. These weights
reproduce razor's tuned split.

`bec_z: 1.0` is deliberately low — a **2.7%** share. It is the same weight
`../razor_centre/sr-wf-bec/` used, and the dataset README warns `bec_z` is a
finite-difference estimate damped by ≥15%, so it should not be pushed hard.

## Budget

`max_epochs: 800`. 4,138 frames at 15 real structures per batch = 276
batches/epoch, so **220,800 gradient updates** — within 1% of `../razor/`'s
`l2c8` 300-epoch budget (218,700). `warmup_epochs: 26` is razor's 7,290 warmup
updates at this batch count. Epochs are not comparable across folders here;
updates are.

## Layout

- `prepare.py` — 90/10 carve-out plus their test; persists `work_function` and
  `bec_z` so all three variants share one `data/`.
- `sr-l2c8-800ep/`, `lr-l2c8-800ep/`, `sr-l2c8-bec-800ep/`.

## Results

Not yet run.
