# beastdb charge conditioning

Trains `Lorem` on `../../datasets/beastdb/` with `charge_conditioning="film"`
(the arm that performed on par with or better than `latent` in the
`ag_clusters`/`water_variational_charge` ablations), crossed with `lr` in
`{true, false}` -- `film_sr` and `film_lr`. Energy-only: beastdb ships
relaxed-geometry energies and DDEC6 partial charges, no forces. Structures
are periodic slabs (`pbc="T T F"`, vacuum along the surface normal),
`tot_charge` (renamed `total_charge`) is generally non-integer since most
frames are grand-canonical (fixed applied potential, self-consistent
charge response) rather than fixed-charge.

No held-out test split ships with this dataset (only `train`/`val`), so
`settings.yaml` has no `test_datasets` block -- reported metrics are on
`data/valid`.

## Layout

- `data/` -- prepared marathon datasets (`train`/`valid`), built by
  `prepare.py` from `../../datasets/beastdb/*.xyz`.
- `film_sr/`, `film_lr/` -- one experiment dir per `lr` setting, each with
  `model.yaml` + `settings.yaml`.

## Running

```bash
# prepare data (shared by both variants)
DATASETS=. python prepare.py

# run a variant
cd film_lr && DATASETS=.. lorem-train
```
