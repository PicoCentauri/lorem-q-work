# natcomm-grace

GRACE-2L + FiLM on `natcomm2025`. Direct counterpart to `../natcomm2025/` (LOREM).

## Settings

Copied from `../cpmace-grace/square-control-w500-10-0.7/`, the configuration
that beat cp-MACE's published forces, with only the dataset-dependent parts
re-derived.

| | value | why |
|---|---|---|
| preset | `GRACE_2LAYER_FILM`, small | as cpmace run 4 |
| loss | **square**, not huber | huber cost 48% on forces in `../cpmace-grace/`'s control |
| weights E:F:W | **10000 : 10 : 0.2** | E 3.3 / F 82.6 / W 14.2 |
| batch | 4 (932 atoms/update) | matches cpmace's 1035 atoms/update |
| budget | 104,000 updates = 27 epochs | 97M atom-updates vs cpmace's 103.5M at 2h23m |

**Why the energy weight is 10000.** MSE scales as error², so at a fixed share the
energy weight scales as 1/error². This dataset's LOREM energy error (0.150
meV/atom) is 8.3× smaller than cpmace's (1.249), so its MSE is 69× smaller — the
weight is large because the energy here is unusually **easy**, not because it
needs attention. Matching cpmace run 4's 12.6% share exactly would take 43000;
10000 gives 3.3% instead, deliberately, because a share decays as its target
improves (cpmace run 2's energy went 12.3% → 2.7% while still reaching 0.619
meV/atom), so a high start spends early budget on a target that converges
anyway. The freed budget goes to forces.

**Only 27 epochs**, because this dataset is 15× larger than cpmace's: the same
~100k gradient updates and ~97M atom-updates, spread over more data and fewer
passes.

## Data

`prepare.py` splits only — `convert.py` already wrote `total_charge` and
`work_function` into `atoms.info`, which is what `TotalChargeDataBuilder` reads
by default.

**The split is on `group`, never within it.** The 4–18 charge states of one
geometry share atomic positions, so a per-frame split would leak. Same seed and
fraction as `../natcomm2025/prepare.py`, so GRACE and LOREM are held out on the
same geometries.

Trains on **all** frames including reaction (Volmer-step) configurations;
`../natcomm2025/README.md` records why the low-curvature tail is chemistry rather
than dielectric breakdown and is therefore not screened.

## A thing to watch

`../natcomm2025/`'s Born effective charges turned out to be a **surface-area bug**
(|a × b| where the slab normal is the *first* cell vector), not a model failure.
Nothing here computes `bec_z`, so that trap is not re-armed — but if BEC support
is ever added, this dataset's slab normal is **x**, not z.

## Targets

`../natcomm2025/` on the 1714-frame validation split:

| run | E (meV/atom) | F (meV/Å) | Φ (V) |
|---|---|---|---|
| `sr` | 0.25 | 26.73 | 0.0912 |
| `lr` | **0.15** | **23.88** | **0.0699** |

## Results

Not yet run.
