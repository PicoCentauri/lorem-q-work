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

## The first two attempts died of a missing rename, not the loss weights

Both the first attempt (`150 : 1 : 0.15`) and the second (`100 : 1 : 0.05`)
hit **`loss became NaN at step=8280`** — the *same step* under very different
weightings, which should have been the tell straight away. SLURM reported
`COMPLETED` in both cases, because `lorem-train` traps the NaN, cancels and
wraps up cleanly; **the failure is invisible from `sacct`**.

The cause: `prepare.py` did not rename `bias_charge` -> `total_charge`.
marathon writes **NaN** for any declared `atoms.info` key that is absent, and
`total_charge` is a model *input*, so the charge conditioning was fed NaN from
step 0. Scanning the prepared payload found exactly **4138 non-finite values,
one per training frame**, at a fixed offset in each record.

Both `../razor/prepare.py` and `../razor_centre/prepare.py` do this rename;
this one was written fresh and omitted it. It is now done in `rename_charge()`
with a comment explaining why it is not cosmetic.

**Two lessons worth keeping.** Checking the *xyz labels* for non-finite values
proved nothing — the field that was NaN is one the xyz never had, created by
`prepare.py`. Check the prepared payload:

```python
d = np.fromfile("data/train/mmap/data.ninja", dtype=np.float64)
assert np.isfinite(d).all()
```

And a NaN at an identical step under different hyperparameters is a
determinism signature: it points at data, not at optimisation.

### Loss weights

`100 : 1 : 0.05`, razor's proven l2c8 triple. Variances:

| target | here | razor |
|---|---|---|
| energy (per atom) | 1.05e−3 | 1.40e−3 |
| forces | 5.04e−1 | 4.73e−1 |
| **work function** | **6.36e−1** | 1.69 |
| `bec_z` | 2.12e−2 | — |

Because the |q| ≤ 1 cap cuts the work-function variance to a third of razor's,
0.05 buys only a **5.0%** share here against razor's 12.1%. `150 : 1 : 0.15`
would reproduce razor's tuned shares and is **not** known to be unstable —
that divergence was the missing rename. Raising the work-function weight is a
reasonable follow-up now that the real cause is fixed.

### The Triton autotuner flag

`sr-l2c8-bec-800ep` additionally died with
`RET_CHECK ... DEVICE_TYPE_INVALID ... No configs could be compiled`, XLA's
Triton GEMM autotuner failing to identify the GPU. It is reached through
`LoremQ._bec_z`'s jvp-over-grad. No *training* `srun.sh` in this repo carried
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` — only the evaluate scripts —
which is why it had not bitten before. All three scripts here now set it.

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

## Results

All three ran the full 800 epochs with no NaN (8h36m / 8h45m / 16h50m; the
`bec` run costs ~2x per step for the forward-over-reverse pass).

`evaluate.py` on the checkpointed models. **`test` is the publication's own
515-frame test set**, untouched; `valid` is the 460-frame carve-out:

| split | run | E (meV/atom) | F (meV/Å) | Φ (V) | Z\* (e) |
|---|---|---|---|---|---|
| test | `sr` | 0.59 | 29.67 | 0.1239 | 0.0398 |
| test | **`lr`** | **0.56** | **26.90** | 0.0875 | 0.0345 |
| test | `sr-bec` | 0.60 | 30.69 | **0.0845** | **0.0193** |
| valid | `sr` | 0.61 | 29.31 | 0.1189 | 0.0401 |
| valid | `lr` | 0.56 | 26.32 | 0.0854 | 0.0518 |
| valid | `sr-bec` | 0.68 | 30.30 | 0.0765 | 0.0192 |

**valid and test agree closely** — 0.59 vs 0.61 on energy, 29.7 vs 29.3 on
forces — which is what a clean split should look like, and confirms the
publication's train/test division is not adversarial to ours.

**The Ewald head wins forces and energy**, as everywhere else: `lr` is 9-10%
better on forces than `sr` at identical settings.

### Supervising `bec_z` helps the work function

`sr-bec` has the **best work function of the three** (0.0845 on test, against
`lr`'s 0.0875 and `sr`'s 0.1239) — a 32% improvement over `sr` — despite being
short-range, and despite carrying a *smaller* work-function share (4.8% vs
5.0%, since `bec_z` takes 3.2%).

Both are charge derivatives — Φ = ∂E/∂q, Z\* = −(A ε₀) ∂²E/∂r∂q — so a term
constraining the second evidently constrains the first. `../razor_centre/`
saw the same, and this is the cleaner demonstration: `sr` and `sr-bec` differ
only by `predict_bec` and the added loss term.

It costs forces: 30.69 against `sr`'s 29.67, ~3%. Cheap for a 32% gain on Φ
and a 2× gain on Z\*.

### `bec_z` supervision is worth 2× on `bec_z` itself

**0.0193 e on test, against 0.0398 unsupervised** — and against a label std of
0.146 e, so ~7.6× better than predicting the mean. For comparison
`../razor_centre/sr-wf-bec/` reached 0.0373 and `../razor/`'s unsupervised
`lr` 0.0270, so this is the best Born-effective-charge model in the project.

Note `lr`'s Z\* is much worse on valid (0.0518) than test (0.0345) while every
other metric matches across the two splits. Unexplained; worth a look before
quoting `lr`'s Z\* anywhere.
