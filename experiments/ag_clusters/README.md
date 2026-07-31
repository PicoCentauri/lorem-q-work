# Ag3 charge conditioning

Trains `Lorem` on the Ag3+/Ag3- dataset (`../../datasets/ag_clusters/`) with
`charge_conditioning` set to `"none"`, `"film"`, or `"latent"`, to test
whether conditioning the model on total charge Q actually helps it tell
charge states apart. Every Ag3 trimer here sits inside any reasonable
cutoff, so a purely local model (`"none"`) has no structural excuse for
failing to distinguish the two charge states -- this isolates the value of
Q-conditioning itself. `lr: false` in every model.yaml: this dataset isn't
about the Ewald long-range channel.

## Layout

- `data/` -- prepared marathon datasets (`train`/`valid`/`test`), built by
  `prepare.py` from `../../datasets/ag_clusters/Ag_clusters_{train,test}.xyz`. `tot_charge`
  in the source xyz is renamed to `total_charge`, which is the key
  `lorem/batching.py` reads to populate the model's Q input.
- `none/`, `film/`, `latent/` -- one experiment dir per `charge_conditioning`
  setting, each with `model.yaml` + `settings.yaml`.
- `evaluate.py` -- loads all three trained checkpoints directly (not via the
  per-structure ASE Calculator, which triggers a fresh XLA compile per
  distinct padded shape) and reports energy/force error broken down by
  charge state on the held-out test set, saving a bar plot.

## Running

```bash
# prepare data (shared by all three variants)
DATASETS=. python prepare.py

# run a variant
cd film && DATASETS=.. lorem-train

# once all three are trained
python evaluate.py
```

## Results so far

300 epochs, `num_features=64, max_degree=4, num_radial=16,
num_message_passing=1, num_spherical_features=4`, muon optimizer,
LR 2e-4 with 10-epoch warmup, gradient_clip=1.0.

| variant | energy R2 | energy MAE (meV/atom) | force R2 | force MAE (meV/A) |
|---|---|---|---|---|
| none | 81.2% | 481 | -85.9% | 218 |
| film | 99.997% | 4.35 | 99.84% | 5.86 |
| latent | 99.997% | 4.38 | 99.87% | 5.60 |

`none` fails badly on forces (R2 < 0, worse than predicting the mean) --
exactly the expected failure mode for a purely local model that can't tell
Ag3+ from Ag3- apart. `film` and `latent` both fix this almost completely
and land within noise of each other, close to chemical accuracy on
energies -- for this dataset, the minimal FiLM conditioning is already
enough; the extra conserving-redistribution machinery in `latent` doesn't
buy anything more.

`evaluate.py`'s held-out test-set breakdown (`figures/error_by_charge_state.pdf`)
by charge state confirms the same picture at finer resolution, and surfaces
an asymmetry not visible in the aggregate numbers above: both `film` and
`latent` do noticeably better on Ag3+ (~0.7-0.9 meV/atom, ~1.5 meV/A) than
on Ag3- (~7.7-10 meV/atom, ~9 meV/A) -- `none` is uniformly bad on both
(500-600 meV/atom, 150-320 meV/A).
