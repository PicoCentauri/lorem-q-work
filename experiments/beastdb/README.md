# beastdb charge conditioning

Trains `Lorem` on `../../datasets/beastdb/` (which always conditions on
total charge Q via FiLM, no config option needed) crossed with `lr` in
`{true, false}` -- `sr` and `lr`, `num_features=128`. Energy-only: beastdb
ships relaxed-geometry energies and DDEC6 partial charges, no forces.
Structures are periodic slabs (`pbc="T T F"`, vacuum along the surface
normal), `tot_charge` (renamed `total_charge`) is generally non-integer
since most frames are grand-canonical (fixed applied potential,
self-consistent charge response) rather than fixed-charge.

Per the BEAST DB authors (contacted directly): every structure is a
relaxed geometry, so the true forces at every frame are zero (to within
the relaxation's convergence tolerance) -- they just weren't written to
the released data. We tried adding `forces=0` as a synthetic regularizer
(`sr-zeroforce`/`lr-zeroforce`, `loss_weights: {"energy": 0.8, "forces":
0.2}`) but it didn't train well -- the relaxations evidently aren't
converged tightly enough for a literal zero to be a clean target, there's
still real residual structure in the true (unrecorded) forces. Dropped;
not worth chasing further without the actual force values.

No held-out test split ships with this dataset (only `train`/`val`), so
`settings.yaml` has no `test_datasets` block -- reported metrics are on
`data/valid`.

## Layout

- `data/` -- prepared marathon datasets (`train`/`valid`), built by
  `prepare.py` from `../../datasets/beastdb/*.xyz`. Run locally and synced
  to the cluster, rather than re-run per job.
- `sr/`, `lr/` -- one experiment dir per `lr` setting, each with
  `model.yaml` + `settings.yaml` + its own `srun.sh` (12h wall-clock cap,
  `a100` for `sr`, `a100_80` for `lr` -- the Ewald k-space work needs more
  memory).
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
