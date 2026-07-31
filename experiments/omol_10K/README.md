# omol_10K charge conditioning

Trains `Lorem` on `../../datasets/omol_10K/` with `charge_conditioning="film"`
(the arm that performed on par with or better than `latent` in the
`ag_clusters`/`water_variational_charge` ablations), crossed with `lr` in
`{true, false}` -- `film_sr` and `film_lr`. Isolated molecules (no PBC),
energy + forces, integer `tot_charge` spanning -8 to +8 e -- by far the
widest charge range and elemental/size diversity (2-350 atoms) of any
dataset here, so this is the hardest test of whether `film` conditioning
generalizes beyond the water-cluster/silver-cluster ablations.

`batch_size` is lower than the other experiments (16 vs. 32) because of the
much wider atom-count range.

## Layout

- `data/` -- prepared marathon datasets (`train`/`valid`/`test`), built by
  `prepare.py` from `../../datasets/omol_10K/*.xyz`.
- `film_sr/`, `film_lr/` -- one experiment dir per `lr` setting, each with
  `model.yaml` + `settings.yaml`.

## Running

```bash
# prepare data (shared by both variants)
DATASETS=. python prepare.py

# run a variant
cd film_lr && DATASETS=.. lorem-train
```
