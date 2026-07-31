# Ag3 charge conditioning

Trains `Lorem` on the Ag3+/Ag3- dataset (`../../datasets/ag_clusters/`).
`Lorem` always conditions on total charge Q via FiLM (no
`charge_conditioning` config option any more) -- an earlier run already
confirmed the without-charge-signal baseline fails badly here (see
"Results so far" below), so this experiment now just compares `sr` vs `lr`
at `num_features=512`, and the held-out numbers get compared directly
against the published Ag3+/Ag3- results (Ko, Finkler, Goedecker & Behler,
*Nat. Commun.* **12**, 398 (2021)) rather than re-running a `none` baseline.
`lr: false` for `sr`, `lr: true` for `lr`.

## Layout

- `prepare.py` -- builds `data/` from
  `../../datasets/ag_clusters/Ag_clusters_{train,test}.xyz` (`tot_charge`
  renamed to `total_charge`). Run locally and synced to the cluster,
  rather than re-run per job.
- `sr/`, `lr/` -- one experiment dir per variant, each with `model.yaml`
  (`num_features=512`) + `settings.yaml` + its own `srun.sh` (2h
  wall-clock cap, `a100` for `sr`, `a100_80` for `lr` -- the Ewald k-space
  work needs more memory).
- `evaluate.py` -- loads both trained checkpoints directly (not via the
  per-structure ASE Calculator, which triggers a fresh XLA compile per
  distinct padded shape) and reports energy/force error broken down by
  charge state on the held-out test set, saving a bar plot.
- `srun.sh` (top level) -- separate SLURM job that just runs `evaluate.py`;
  submit once both training jobs have finished.

## Running

```bash
# prepare data locally (shared by both variants), then sync to the cluster
DATASETS=. python prepare.py

# on the cluster: submit both training jobs (run in parallel)
cd sr && sbatch srun.sh && cd ../lr && sbatch srun.sh

# once both have finished, evaluate
cd .. && sbatch srun.sh
```

## Results so far (previous run, num_features=64, none/film/latent)

300 epochs, `num_features=64, max_degree=4, num_radial=16,
num_message_passing=1, num_spherical_features=4`, muon optimizer,
LR 2e-4 with 10-epoch warmup, gradient_clip=1.0. This was the ablation that
established the without-charge-signal baseline fails and FiLM fixes it --
`none`/`film`/`latent` no longer exist as separate configs (`latent` was
dropped from the codebase entirely, FiLM is now always on), which is why
this experiment can move straight to `sr` vs `lr`.

| variant | energy R2 | energy MAE (meV/atom) | force R2 | force MAE (meV/A) |
|---|---|---|---|---|
| none | 81.2% | 481 | -85.9% | 218 |
| film | 99.997% | 4.35 | 99.84% | 5.86 |
| latent | 99.997% | 4.38 | 99.87% | 5.60 |

`none` failed badly on forces (R2 < 0, worse than predicting the mean) --
exactly the expected failure mode for a purely local model that can't tell
Ag3+ from Ag3- apart. `film` and `latent` both fixed this almost completely
and landed within noise of each other, close to chemical accuracy on
energies -- the minimal FiLM conditioning was already enough.
