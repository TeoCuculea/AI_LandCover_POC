# AI Land Cover POC

EuroSAT land-cover classification with RGB and 13-band Sentinel-2 inputs.

## Project layout

```text
data/                         Source imagery
results/                      Generated splits, metrics, charts, and models
src/
  analysis/                    Dataset exploration and experiment reports
  data/                        Dataset loading, preparation, and normalization
  geospatial/                  Coordinates, spatial splits, and leakage analysis
  models/                      Model architectures
  training/                    Training runner, metrics, and training plots
  diagnostics/                 Manual dataset and model inspections
  config.py                    Shared paths and experiment constants
```

Run commands from the project root. The top-level modules `src.train`,
`src.dataset`, `src.model`, `src.prepare_data`, and the two comparison scripts
remain as compatibility entrypoints for the existing project workflow.

## Rebuild reports from the existing baseline

Install dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

These commands read the existing baseline experiment reports. The comparison
commands write their tables and charts to `results/tables/` and
`results/figures/`.

```powershell
python -m src.analysis.compare_experiments
python -m src.analysis.compare_class_performance
```

These comparisons do not train models. `python -m src.geospatial.run_analysis`
rebuilds the spatial analysis outputs, and
`python -m src.geospatial.map_cli` regenerates the map from saved coordinate
and pair tables.

## Run the four longer baseline experiments

`run_baseline_experiments.py` runs Random RGB, Random Multispectral, Spatial
RGB, and Spatial Multispectral sequentially, then creates both comparisons.
Each batch receives one shared run ID and writes under
`results/runs/<run-id>/`; it refuses to reuse an existing run ID. The default
is 30 epochs per experiment with early stopping (patience 6, validation
macro-F1 min-delta 0.001). If all experiments reach 30 epochs, they are
estimated to take about four hours based on preliminary timings. The script
has not been launched.
Preview the commands and output path first with `--what-if`.

```powershell
.\run_baseline_experiments.cmd --what-if --run-id baseline_trial_01
.\run_baseline_experiments.cmd --run-id baseline_trial_01 --epochs 30 --seed 42
```

To continue an existing run, add `--continue-run`. Completed experiments are
skipped; partial experiment artifacts cause the script to stop rather than
overwrite them. Preview a continuation with:

```powershell
.\run_baseline_experiments.cmd --continue-run --what-if --run-id disertatie_final_01
```

To run a single model without overwriting the preliminary results, use a new
run ID. If omitted, the training command creates and prints a timestamped ID;
pass that ID to comparison commands to read and write within the same run:

```powershell
python -m src.training.runner --representation rgb --split random --epochs 30 --run-id baseline_trial_01
python -m src.analysis.compare_experiments --run-id baseline_trial_01
python -m src.analysis.compare_class_performance --run-id baseline_trial_01
```
