# AutoDD: Learning to Select Drift Detectors for Evolving Data Streams

**Repository:** <https://github.com/TechyNilesh/AutoDD>

## Overview

AutoDD is a framework that jointly automates drift detector selection and hyperparameter configuration for evolving data streams. It uses a two-level hierarchical selection strategy: **LinUCB** (contextual bandit) selects the detector based on 15 online meta-features extracted from the stream, and **Thompson Sampling** chooses the hyperparameter configuration for the selected detector. The framework continually refines its choices as the stream evolves, without requiring ground-truth drift labels.

**Detector portfolio (K=7):** ADWIN, DDM, EDDM, HDDM-A, HDDM-W, Page-Hinkley, KSWIN

**Base classifiers:** GaussianNB, Hoeffding Adaptive Tree (HAT)

**Default settings:** window size `w=100`, exploration `α=1.0`, regularisation `λ=1.0`, `C=5` hyperparameter configurations per detector, averaged over 10 runs.

## Installation

### Prerequisites

Python 3.8+ and the following packages:

```bash
pip install numpy pandas scikit-learn river scipy matplotlib psutil tqdm
```

### Setup

```bash
git clone https://github.com/TechyNilesh/AutoDD.git
cd AutoDD/code
```

Generate synthetic datasets (if not already present):

```bash
python generate_synthetic_streams.py
```

## Datasets

All datasets are in the `datasets/` directory as CSV files (features + target column).

### Synthetic Datasets

| Dataset | Instances | Features | Drift Type |
|---------|-----------|----------|------------|
| SEA | 500,000 | 3 | Multiple (Sudden, Gradual, Incremental, Recurring) |
| Hyperplane (HYP) | 500,000 | 10 | Multiple |
| Agrawal (AGR) | 100,000 | 9 | Multiple |
| RBF | 200,000 | 10 | Multiple |
| Random Tree (RTG) | 100,000 | 30 | Multiple |

### Real-world Datasets

| Dataset | Instances | Features | Drift Type |
|---------|-----------|----------|------------|
| Electricity (Elec) | 45,312 | 8 | Unknown |
| Weather | 18,159 | 8 | Unknown |
| Spam | 4,601 | 57 | Unknown |
| Airlines | 539,383 | 7 | Unknown |
| Poker | 829,201 | 10 | Unknown |

## Usage

### Running AutoDD

```bash
# GaussianNB base model (default)
python scripts/autodd_run.py --dataset elec --seed 42

# Hoeffding Adaptive Tree base model
python scripts/autodd_run.py --dataset elec --seed 42 --base_model hat

# With custom AutoDD hyperparameters
python scripts/autodd_run.py --dataset sea --window_size 100 --alpha 1.0 --lambda_reg 1.0 --n_hp_configs 5
```

### Running All Experiments

```bash
# GaussianNB (default)
python run_experiments.py --mode all

# HAT
python run_experiments.py --mode all --base_model hat

# Both base models
python run_experiments.py --mode all --base_model all
```

### Baseline Detectors

```bash
# No detector (baseline)
python scripts/no_detector_run.py --dataset elec --seed 42

# ADWIN
python scripts/adwin_run.py --dataset elec --seed 42

# DDM
python scripts/ddm_run.py --dataset weather --seed 42

# EDDM
python scripts/eddm_run.py --dataset spam --seed 42

# HDDM-A
python scripts/hddm_a_run.py --dataset airlines --seed 42

# HDDM-W
python scripts/hddm_w_run.py --dataset poker --seed 42

# Page-Hinkley
python scripts/pagehinkley_run.py --dataset rbf --seed 42

# KSWIN
python scripts/kswin_run.py --dataset rtg --seed 42

# MDDM
python scripts/mddm_run.py --dataset sea --seed 42

# ADDM
python scripts/addm_run.py --dataset hyp --seed 42

# PUDD
python scripts/pudd_run.py --dataset elec --seed 42

# Meta-ADD
python scripts/meta_add_run.py --dataset hyp --seed 42
```

### Parameters

All scripts share these common parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | required | Dataset name: `sea`, `hyp`, `agr`, `rbf`, `rtg`, `elec`, `weather`, `spam`, `airlines`, `poker` |
| `--base_model` | `gaussiannb` | Base classifier: `gaussiannb` or `hat` |
| `--seed` | `42` | Random seed for reproducibility |
| `--run_count` | `None` | Run identifier for multiple runs |
| `--progress_interval` | `1000` | Instances between progress updates |
| `--output_dir` | `benchmark_results` | Directory to save results |
| `--verbose` | flag | Enable verbose output |

AutoDD-specific parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--window_size` | `100` | Sliding window size `w` for meta-feature extraction |
| `--alpha` | `1.0` | LinUCB exploration parameter `α_ucb` |
| `--lambda_reg` | `1.0` | LinUCB regularisation parameter `λ_reg` |
| `--n_hp_configs` | `5` | Hyperparameter configurations per detector `C` |

### Additional Experiments

```bash
# Ablation study (component contribution analysis)
python run_ablation_study.py

# Detector selection frequency analysis
python run_detector_selection_experiment.py

# Meta-feature type analysis (statistical / stationarity / temporal / all)
python run_meta_feature_experiment.py

# Hyperparameter sensitivity analysis
python run_sensitivity_experiment.py                          # window size
python run_sensitivity_experiment.py --experiment alpha       # α_ucb
python run_sensitivity_experiment.py --experiment lambda      # λ_reg
```

## Output Format

Each run produces a JSON file in `benchmark_results/` named `{DETECTOR}_{dataset}_{basemodel}_run{N}.json` (e.g., `AUTODD_elec_gaussiannb_run1.json`, `ADWIN_weather_hat_run3.json`). These files are **not tracked in git** — see [What is tracked](#what-is-tracked) below.

```json
{
    "dataset": "elec",
    "model": "AUTODD",
    "detector": "autodd",
    "base_model": "gaussiannb",
    "accuracy": 0.896,
    "f1": 0.688,
    "n_drifts": 22,
    "drift_points": [1015, 6974, ...],
    "total_runtime": 7.63,
    "total_memory_mb": 8.61,
    "n_instances": 45312,
    "window_metrics": {
        "interval": 1000,
        "instances": [0, 1029, ...],
        "accuracy_scores": [0.863, 0.884, ...],
        "f1_scores": [0.681, 0.684, ...]
    },
    "selection_distribution": {
        "ADWIN": 0.246,
        "DDM": 0.143,
        "EDDM": 0.118,
        "HDDM_A": 0.137,
        "HDDM_W": 0.120,
        "PageHinkley": 0.088,
        "KSWIN": 0.241
    }
}
```

## Ablation Study

The ablation study (Table 4 in the paper) evaluates each component's contribution averaged across the four real-world streams (Elec, Weather, Airlines, Poker) with GaussianNB, 10 runs each:

| Configuration | Accuracy | Description |
|---------------|----------|-------------|
| Full AutoDD | 74.5% | All components enabled |
| w/o Level 2 (Thompson Sampling) | 69.2% | Fixed default hyperparameters |
| w/o Level 1 (LinUCB → Random) | 56.6% | Random detector selection |
| w/o Statistical features | 64.9% | Without statistical meta-features |
| w/o Stationarity features | 67.5% | Without stationarity meta-features |
| w/o Temporal features | 67.9% | Without temporal meta-features |

## Project Structure

```
code/
├── AutoDD/                         # AutoDD implementation
│   ├── autodd.py                   # Main AutoDD class
│   ├── linucb_agent.py             # Level 1: LinUCB detector selection
│   ├── hyperparameter_selector.py  # Level 2: Thompson Sampling config selection
│   ├── meta_feature_extractor.py   # 15 online meta-features
│   ├── detector_portfolio.py       # Detector portfolio (K=7)
│   └── classifier_portfolio.py     # Base classifier management
├── MetaADD/                        # Meta-ADD implementation
├── MDDM/                           # MDDM implementation
├── PUDD/                           # PUDD implementation
├── ADDM/                           # ADDM implementation
├── scripts/                        # Run scripts for all detectors
│   ├── utils.py                    # Shared utilities
│   ├── autodd_run.py
│   ├── no_detector_run.py
│   ├── adwin_run.py
│   ├── ddm_run.py
│   ├── eddm_run.py
│   ├── hddm_a_run.py
│   ├── hddm_w_run.py
│   ├── pagehinkley_run.py
│   ├── kswin_run.py
│   ├── mddm_run.py
│   ├── addm_run.py
│   ├── pudd_run.py
│   └── meta_add_run.py
├── plotting/                       # Figure generation scripts
├── datasets/                       # CSV dataset files (tracked)
├── benchmark_results/              # Per-run JSON results          (not tracked)
├── ablation_results/               # Ablation study results        (not tracked)
├── experiment_results/             # Sensitivity/selection results (not tracked)
├── run_experiments.py              # Main experiment runner
├── run_ablation_study.py           # Ablation study
├── run_detector_selection_experiment.py
├── run_meta_feature_experiment.py
├── run_sensitivity_experiment.py
└── generate_synthetic_streams.py
```

## What is tracked

To keep the repository small, the raw experiment output is **not** committed. Everything
needed to regenerate it is.

| Path | Tracked | Notes |
|------|---------|-------|
| `AutoDD/`, `MetaADD/`, `MDDM/`, `PUDD/`, `ADDM/` | yes | All method implementations |
| `scripts/`, `run_*.py`, `generate_synthetic_streams.py` | yes | Everything needed to reproduce the runs |
| `datasets/*.csv` | yes | The exact CSVs used in the paper |
| `MetaADD/training_data.json`, `MetaADD/pretrained_model.pt` | yes | Meta-ADD baseline fixtures — inputs, not results |
| `benchmark_results/` | no | ~2.6k JSON files, regenerated by `run_experiments.py` |
| `ablation_results/` | no | 240 JSON files (4 real-world datasets x 6 configs x 10 runs), regenerated by `run_ablation_study.py --datasets elec weather airlines poker --n_runs 10` |
| `experiment_results/` | no | Regenerated by the `run_*_experiment.py` scripts |

The three results directories are listed in `.gitignore`. They are recreated automatically
the first time you run the corresponding script.

## Regenerating the results

```bash
# 1. Main benchmark: all detectors x all datasets x both base models x 10 runs
python run_experiments.py --mode all --base_model all   # -> benchmark_results/

# 2. Ablation study
python run_ablation_study.py                            # -> ablation_results/

# 3. Sensitivity, selection-frequency and meta-feature experiments
python run_sensitivity_experiment.py                    # -> experiment_results/
python run_sensitivity_experiment.py --experiment alpha
python run_sensitivity_experiment.py --experiment lambda
python run_detector_selection_experiment.py
python run_meta_feature_experiment.py
```

Step 1 is by far the longest: it evaluates 13 methods on 10 datasets (up to 829k instances)
under 2 base models for 10 seeds each. Steps 2 and 3 are smaller but still run full streams.
Seeds are fixed (uniformly sampled from [1, 100]), so repeated runs reproduce the paper's numbers.

Synthetic streams are regenerated with `python generate_synthetic_streams.py`; the CSVs in
`datasets/` are the exact files used in the paper and are tracked, so regeneration is only
needed if they are deleted.

## Citation

If you use AutoDD in your research, please cite:

```bibtex
@inproceedings{verma2026autodd,
  title     = {{AutoDD}: Learning to Select Drift Detectors for
               Evolving Data Streams},
  author    = {Verma, Nilesh and Bifet, Albert and Pfahringer, Bernhard and Bahri, Maroua},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Data Mining (ICDM)},
  publisher = {IEEE},
  address   = {Shenyang, China},
  year      = {2026},
  url       = {https://github.com/TechyNilesh/AutoDD}
}
```
