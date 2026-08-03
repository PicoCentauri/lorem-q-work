# water_variational_charge charge conditioning

Trains `Lorem` on `../../datasets/water_variational_charge/` with
`charge_conditioning` in `{"film", "latent"}` crossed with `lr` in
`{true, false}` -- four variants total (`film_lr`, `film_sr`, `latent_lr`,
`latent_sr`; `sr` = short-range only, `lr` = Ewald long-range on). Unlike
`ag_clusters`, this dataset is specifically designed to push clusters apart
beyond any reasonable cutoff (0.9x-5.0x isotropic expansion), so the `lr`
flag is the interesting axis here: does the long-range channel let the
model track dissociation the way a purely local model (`*_sr`) can't?

## Layout

- `data/` -- prepared marathon datasets (`train`/`valid`/`test`), built by
  `prepare.py` from `../../datasets/water_variational_charge/*.xyz`.
  `tot_charge` is renamed to `total_charge`, matching the key
  `lorem/batching.py` reads for the model's Q input.
- `film_lr/`, `film_sr/`, `latent_lr/`, `latent_sr/` -- one experiment
  dir per (charge_conditioning, lr) combination.
- `evaluate.py` -- loads all four trained checkpoints directly (model +
  params + per-species baseline, batched through the same `to_sample`/
  `to_batch` pipeline training uses -- not the per-structure ASE Calculator,
  which triggers a fresh XLA compile per distinct padded shape and blows up
  GPU memory across ~15k structures of varying size) and runs the two
  physical test cases below against the held-out test set, saving plots to
  `figures/`.

## Running

```bash
# prepare data (shared by all four variants)
DATASETS=. python prepare.py

# run a variant
cd film_lr && DATASETS=.. lorem-train

# once all four are trained
python evaluate.py
```

## Physical test cases

From `../../datasets/water_variational_charge/README.md`:

**Exp. 1 -- mixed charge states near equilibrium.** Test-set E/F MAE/RMSE,
grouped by `tot_charge`. Target: about 2 meV/atom, 49 meV/A RMSE, per charge
state.

**Exp. 2 -- dissociation, binned version.** Test-set E/F error binned by
`exp_coef` (`{1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}`), separately per charge
state.

## Fixed: `*_lr` variants used to diverge to NaN

`film_lr`/`latent_lr`/a `none`+`lr` isolation variant all hit NaN loss at
the identical step, regardless of charge_conditioning mode, while `*_sr`
variants trained cleanly -- pinning the bug to the Ewald long-range pathway
itself, unrelated to charge conditioning. Root cause: `spherical_norm`'s
custom gradient rule (`src/lorem/models/backbone.py`) only guarded its
forward division against a zero norm, not the second derivative through it
-- `x/||x||` is genuinely non-smooth at `x=0`, and LOREM's own training loop
needs exactly that second derivative (forces = grad(energy, positions),
then grad(loss(forces), params)). Fixed by replacing the whole custom-JVP
scheme with a smooth epsilon-regularized norm; see
`fix-spherical-norm-nan` branch / PR for the standalone fix and regression
test. All four variants now train cleanly with the fix.

## Results

300 epochs, `num_features=64, max_degree=4, num_radial=16,
num_message_passing=1, num_spherical_features=4`, muon optimizer,
LR 2e-4 with 10-epoch warmup, gradient_clip=1.0.

| variant | energy MAE (meV/atom) | energy RMSE | force MAE (meV/A) | force RMSE |
|---|---|---|---|---|
| film_lr | -- | 1.22 | 9.77 | 15.3 |
| latent_lr | -- | 1.71 | 9.02 | 14.0 |
| film_sr | 2.80 | 4.25 | 14.6 | 21.8 |
| latent_sr | 3.03 | 4.56 | 14.9 | 22.4 |

(validation-set metrics reported during training; `*_lr`'s log reported RMSE
rather than MAE at the checkpointed step)

### Exp 1 -- mixed charge states near equilibrium (`figures/exp1_charge_states.pdf`)

Held-out test set (15,000 structures), MAE in meV/atom (energy) / meV/A (forces):

| variant | Q=-1 E | Q=0 E | Q=+1 E | Q=-1 F | Q=0 F | Q=+1 F |
|---|---|---|---|---|---|---|
| film_lr | 0.91 | 0.78 | 1.00 | 10.6 | 7.6 | 11.3 |
| film_sr | 3.59 | 1.24 | 3.34 | 18.8 | 7.6 | 17.4 |
| latent_lr | 1.07 | 1.14 | 1.77 | 10.1 | 6.0 | 11.5 |
| latent_sr | 3.74 | 1.38 | 3.91 | 18.9 | 7.9 | 18.3 |

The `*_sr` variants are noticeably worse specifically on the *charged*
states (Q=-1, Q=+1) and about the same as `*_lr` on neutral (Q=0) -- exactly
the expected signature of a model that can only get charge-transfer physics
right when it has a channel to see beyond the cutoff.

### Exp 2 -- dissociation, binned by exp_coef (`figures/exp2_dissociation.pdf`)

For the charged states, `*_lr` energy error stays roughly flat (or even
improves) as `exp_coef` grows from 1 to 5, while `*_sr` error climbs by
roughly an order of magnitude over the same range (most visible for
`film_lr` vs. `film_sr`/`latent_sr` on Q=+1 energy: ~0.3-1 vs. ~10-20
meV/atom by exp_coef=5). On Q=0 the four variants track each other closely
throughout. This is the headline result: the Ewald long-range channel lets
LOREM learn charge-transfer/dissociation physics end-to-end, without it a
purely local model's error grows with separation exactly as expected.
