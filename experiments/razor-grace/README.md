# razor-grace

GRACE-2L + FiLM on `razor_centre_paper` — the publication's subset of razor with
our labels. Direct counterpart to `../razor_centre_paper/` (LOREM).

## Settings

Copied from `../cpmace-grace/square-control-w500-10-0.7/`, the configuration
that beat cp-MACE's published forces, with only the dataset-dependent parts
re-derived.

| | value | why |
|---|---|---|
| preset | `GRACE_2LAYER_FILM`, small | as cpmace run 4 |
| loss | **square**, not huber | huber cost 48% on forces in `../cpmace-grace/`'s control |
| weights E:F:W | **2000 : 10 : 0.15** | E 7.0 / F 80.3 / W 12.7 |
| batch | 10 (1080 atoms/update) | matches cpmace's 1035 atoms/update |
| budget | 90,000 updates = 218 epochs | 97M atom-updates vs cpmace's 103.5M at 2h23m |

**Weights are re-derived, not copied.** Shares transfer between datasets, weights
do not: these come from LOREM's converged errors on *this* dataset
(`../razor_centre_paper/` `lr`: 0.56 meV/atom, 26.90 meV/Å, 0.0875 V).

## Data

`prepare.py` splits only — it rewrites no fields. The source already has
`work_function` in `atoms.info` and the charge as `bias_charge`, so
`TotalChargeDataBuilder(charge_key="bias_charge")` reads it in place.

Not renaming is deliberate: a rename that silently fails leaves the charge at
the 0.0 default, which is exactly what cost `../razor_centre_paper/` two runs
(marathon wrote NaN for the absent key, and it was misdiagnosed as a loss-weight
problem). Here the builder logs `N/N structures` and warns on any default.

Splits use the same seed and fraction as `../razor_centre_paper/prepare.py`, so
GRACE and LOREM are held out on the same structures: a 90/10 carve-out of the
publication's 4598-frame train set, plus their 515-frame test set untouched.

## Not modelled: `bec_z`

The dataset carries `bec_z`, and LOREM's `sr-bec` run used it to get the best
work function of the three (0.0845 V). GRACE has no Born-effective-charge
support — that is the deferred work in `../../notes/dipole-term-plan.md`. So
this run is the counterpart of LOREM's `sr` / `lr`, not of `sr-bec`.

## Targets

`../razor_centre_paper/` on the publication's own 515-frame test set:

| run | E (meV/atom) | F (meV/Å) | Φ (V) |
|---|---|---|---|
| `sr` | 0.59 | 29.67 | 0.1239 |
| `lr` | **0.56** | **26.90** | 0.0875 |
| `sr-bec` | 0.60 | 30.69 | **0.0845** |

## Results

Not yet run.
