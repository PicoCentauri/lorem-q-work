# omol_10K charge conditioning

Trains `Lorem` (which always conditions on total charge Q via FiLM, no
config option needed) on `../../datasets/omol_10K/` crossed with `lr` in
`{true, false}` -- `sr` and `lr`, `num_features=512`. Isolated molecules
(no PBC), energy + forces, integer `tot_charge` spanning -8 to +8 e -- by
far the widest charge range and elemental/size diversity (2-350 atoms) of
any dataset here, so this is the hardest test of whether charge
conditioning generalizes beyond the water-cluster/silver-cluster ablations.

`batch_size` is lower than the other experiments (16 vs. 32) because of the
much wider atom-count range.

## Layout

- `data/` -- prepared marathon datasets (`train`/`valid`/`test`), built by
  `prepare.py` from `../../datasets/omol_10K/*.xyz`. Run locally and
  synced to the cluster, rather than re-run per job.
- `sr/`, `lr/` -- one experiment dir per `lr` setting, each with
  `model.yaml` + `settings.yaml` + its own `srun.sh` (2h wall-clock cap,
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
