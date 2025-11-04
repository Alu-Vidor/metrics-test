# Metrics Testbed

This repository bundles a lightweight pipeline for training time-series
forecasting models and evaluating them across multiple stress scenarios. The
evaluation produces accuracy, structure, robustness and cost components that
are combined into a composite score.

## Requirements

* Python 3.10+
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [PyTorch](https://pytorch.org/) (CPU build is sufficient)
* [PyYAML](https://pyyaml.org/)

Install the dependencies via `pip`:

```bash
pip install numpy pandas torch pyyaml
```

## Running an experiment

Experiments are configured via `configs/experiment.yaml`. The configuration
selects datasets, model architectures, scenario generators and weighting
schemes. A full end-to-end run can be triggered with:

```bash
python -m src.run_experiment --config configs/experiment.yaml
```

The script trains every listed model, evaluates them on validation and test
windows, searches for optimal component weights `(α, β, γ)` and finally writes a
set of CSV tables together with an HTML summary to `outputs/reports/`.

If the configured dataset file is unavailable the runner automatically falls
back to a synthetic sinusoidal series so the pipeline can be exercised without
additional resources.

## Configuration overview

* `seed`: Random seed used across the run.
* `horizon` / `window`: Forecast horizon and context window size.
* `datasets`: List of datasets with optional references into
  `configs/datasets.yaml` for metadata.
* `models`: Import path, name and hyper-parameters for each model. The runner
  automatically instantiates `<Name>Model` classes from the provided module.
* `scenarios`: Scenario definitions (clean, noise, shift, phase) used during
  robustness evaluation.
* `normalization`: Normalisation scheme for accuracy, structure and robustness.
* `weight_search`: Grid resolution for `(α, β, γ)` search.
* `lambda`: Weight applied to the latency cost penalty.

The default configuration demonstrates all components on a small electricity
dataset. Replace the dataset path or extend the configuration to plug in your
own data and models.
