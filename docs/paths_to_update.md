# Paths To Update Before Running

Paths are **not** centralized in a config file — each script/notebook defines
its own input paths near the top of the file (or in a `CONFIG` section). Before
running anything on a new machine, edit the following:

## Sentinel-2 stack path

Search for `Sentinel2_T31UEQ_TSG.tif` / `Sentinel2_T31UDP_TSG.tif` and update to
your local location of the multi-temporal Sentinel-2 stack:

| File | What to change |
|---|---|
| `scripts/sr_inference/inference-SEN2SRLite.py` | `INPUT_TIF`, `OUTPUT_TIF` constants near the top |
| `scripts/data_preparation/audit_observation_pipeline.py` | `STACK_PATH` constant |
| `scripts/data_preparation/build_obs_dataset.py` | `stack_path` local variable in `main()` |
| `scripts/data_preparation/build_temporal_dataset.py` | `stack_path` local variable in `main()` |
| `scripts/data_preparation/prepare_s2_observation.py` | `stack_path` (per-zone) arguments |
| `scripts/run_pipeline.py` | `stack_path` local variable |
| `notebooks/01_data_preparation_validation.ipynb` | first config cell (`stack_path`) |
| `notebooks/02_sr_evaluation_unified.ipynb` | `S2_STACK_PATH` in the config cell |
| `notebooks/exploratory/pipeline-test.ipynb` | `stack_path` in the config cell |

## Parcel shapefile path

| File | What to change |
|---|---|
| `scripts/data_preparation/prepare_s2_observation.py` | `parcels_path` |
| `scripts/data_preparation/audit_observation_pipeline.py` | `PARCELS_PATH` (already repo-relative: `data/2020_buf20/2020_buf20.shp`) |

## SR output / evaluation directories

`notebooks/02_sr_evaluation_unified.ipynb` references several SR-output
directories (`sr_output_dirs`) in its config cell that currently point at
machine-specific folders (e.g. under `Desktop/results/...` or
`Downloads/...`). Point these at wherever your `inference-SEN2SRLite.py`
outputs (or `outputs/` batch results) are stored.

## Repository root folder name

Some notebooks were originally authored against a sibling folder named
`s2-super-resolution` (without ` - preso`). If you copy this repository, check
any remaining `repo_root = Path(r"...")` cells for the exact folder name and
update to match your local clone's actual folder name — or better, replace
with a relative computation such as:

```python
from pathlib import Path
repo_root = Path.cwd().resolve().parents[0]  # adjust depth as needed
```

## Model weights

`model/SEN2SRLite/` is expected to contain the cached SEN2SRLite weights. If
missing, scripts call `mlstac.download(...)` automatically from
`https://huggingface.co/tacofoundation/sen2sr` — an internet connection is
required the first time.
