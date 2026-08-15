# razor_centre_paper

The publication's subset of razor, with **our** labels. Data is
`../../datasets/razor/razor_centre_paper_{train,test}.xyz`.

| dir | model | loss weights | shares |
|---|---|---|---|
| `sr-l2c8-800ep/` | d64 l2 c8, `lr: false` | 100 : 1 : 0.05 | E 16.4 / F 78.6 / W 5.0 |
| `lr-l2c8-800ep/` | same, `lr: true` (`max_degree_lr: 0`) | 100 : 1 : 0.05 | same |
| `sr-l2c8-bec-800ep/` | same as `sr`, `predict_bec: True` | + `bec_z: 1.0` | E 15.9 / F 76.1 / W 4.8 / **Z 3.2** |

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

## Loss weights: 150 : 1 : 0.15 diverged, so these are razor's proven triple

The first attempt used `150 : 1 : 0.15`, chosen to reproduce razor's *tuned
shares* given this subset's smaller work-function variance. **Both runs hit
`loss became NaN at step=8280`** — about 2 epochs in, at LR 1.6e-05, still
inside warmup. SLURM reported `COMPLETED` because `lorem-train` catches the
NaN, cancels and wraps up cleanly, so the failure is invisible from `sacct`.

The data was ruled out first: zero non-finite labels across all 4598 train and
515 test frames, max|F| 14.33 eV/Å with only 2 frames over 10, `bec_z` bounded
at 1.07 e. Diverging at a *low* learning rate points at the loss scale rather
than the step size.

`150 : 1 : 0.15` was the most aggressive weighting used anywhere in this
project on both axes at once:

| | energy | work function |
|---|---|---|
| razor's proven l2c8 | 100 | **0.05** |
| razor_centre | 0.5 | 0.15 |
| **first attempt here** | **150** | **0.15** |

`../razor/README.md` warns about the 0.15 specifically: the untrained `dE/dq`
starts around −2…−18 V against labels of +3…+7, so that term begins at ~99% of
the loss, and razor's sweep later moved to 0.05 because 0.15 cost 1.9× on
energy and 1.7× on forces. Pairing it with the largest energy weight used here
was the mistake.

**Now `100 : 1 : 0.05`** — exactly razor's proven l2c8 triple. Variances for
reference:

| target | here | razor |
|---|---|---|
| energy (per atom) | 1.05e−3 | 1.40e−3 |
| forces | 5.04e−1 | 4.73e−1 |
| **work function** | **6.36e−1** | 1.69 |
| `bec_z` | 2.12e−2 | — |

**The cost is real and worth stating:** because the |q| ≤ 1 cap cuts the
work-function variance to a third of razor's, the same nominal weight buys
only a **5.0%** share here against razor's 12.1%. So the work function is
trained more weakly than in any other folder. That is the price of a weighting
proven to train on this model, and it should be raised once stability is
confirmed — not before, since the a100 queue is currently ~4 days deep and a
second divergence is expensive.

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
