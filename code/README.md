# AutoDD - Automated Drift Detector Selection for Data Streams

## Overview

AutoDD is an adaptive framework for automated drift detector selection in data streams. It uses contextual multi-armed bandits (LinUCB) to dynamically select the most appropriate drift detector based on stream meta-features, enabling robust concept drift detection across varying data characteristics.

## Features

- **Adaptive Detector Selection**: Automatically selects the best drift detector using LinUCB contextual bandits
- **Meta-Feature Extraction**: Extracts temporal, statistical, and stationarity features from data streams
- **Multiple Base Classifiers**: Supports different base classifiers:
  - GaussianNB (Gaussian Naive Bayes) - default
  - HAT (Hoeffding Adaptive Tree)
- **Multiple Drift Detectors**: Supports various drift detection algorithms:
  - ADWIN (Adaptive Windowing)
  - DDM (Drift Detection Method)
  - EDDM (Early Drift Detection Method)
  - HDDM-A (Hoeffding's bound Drift Detection Method - A-test)
  - HDDM-W (Hoeffding's bound Drift Detection Method - W-test)
  - Page-Hinkley Test
  - KSWIN (Kolmogorov-Smirnov Windowing)
  - MDDM (McDiarmid Drift Detection Method)
  - PUDD (Prediction Uncertainty-based Drift Detector)
  - ADDM (Autoregressive-Based Drift Detection Method)

- **Comprehensive Evaluation**: Built-in benchmark framework with prequential evaluation
- **Ablation Studies**: Tools for analyzing component contributions

## Installation

### Prerequisites

Make sure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### Dependencies

- numpy
- pandas
- scikit-learn
- river
- scipy
- matplotlib
- psutil
- tqdm

### Setup

1. Clone the repository:
```bash
git clone https://github.com/anonymous/AutoDD.git
cd AutoDD/code
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate synthetic datasets (optional):
```bash
python generate_synthetic_streams.py
```

## Dataset Structure

The project includes both synthetic and real-world datasets in the `datasets/` directory.

### Datasets Used in Experiments

#### Synthetic Datasets

| **Dataset** | **# Instances** | **# Features** | **Drift Type** |
|-------------|-----------------|----------------|----------------|
| SEA | 500,000 | 3 | Abrupt |
| Hyperplane (HYP) | 500,000 | 10 | Gradual |
| Agrawal (AGR) | 100,000 | 9 | Abrupt |
| RBF | 200,000 | 10 | Gradual |
| RTG (Random Tree) | 100,000 | 30 | Abrupt |

#### Real-world Datasets

| **Dataset** | **# Instances** | **# Features** | **Drift Type** |
|-------------|-----------------|----------------|----------------|
| Electricity (ELEC) | 45,312 | 8 | Unknown |
| Weather | 18,159 | 8 | Unknown |
| Spam | 4,601 | 57 | Unknown |
| Airlines | 539,383 | 7 | Unknown |
| Poker | 829,201 | 10 | Unknown |

**Dataset Format**: CSV files with features and target column.

## Usage

### Quick Start

#### Running AutoDD
```bash
# With GaussianNB (default)
python scripts/autodd_run.py --dataset elec --seed 42

# With Hoeffding Adaptive Tree (HAT)
python scripts/autodd_run.py --dataset elec --seed 42 --base_model hat
```

#### Running All Experiments
```bash
# Run with GaussianNB (default)
python run_experiments.py --mode all

# Run with HAT
python run_experiments.py --mode all --base_model hat

# Run with both base models
python run_experiments.py --mode all --base_model all
```

### Individual Model Scripts

Each drift detector has its own script in the `scripts/` directory:

#### AutoDD (Adaptive Selection)
```bash
python scripts/autodd_run.py --dataset sea --run_count 1 --seed 42
```

#### Meta-ADD (Meta-learning based)
```bash
python scripts/meta_add_run.py --dataset hyp --run_count 1 --seed 42
```

#### Baseline Detectors
```bash
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
```

### Common Parameters

All scripts support the following parameters:

- `--dataset`: Name of the dataset (sea, hyp, agr, rbf, rtg, elec, weather, spam, airlines, poker)
- `--run_count`: Run identifier for multiple runs (optional)
- `--seed`: Random seed for reproducibility (default: 42)
- `--progress_interval`: Interval for progress updates (default: 1000)
- `--output_dir`: Directory to save results (default: "benchmark_results")
- `--verbose`: Enable verbose output
- `--base_model`: Base classifier to use (default: "gaussiannb")
  - `gaussiannb`: Gaussian Naive Bayes
  - `hat`: Hoeffding Adaptive Tree

### Base Model Selection

AutoDD supports two base classifiers that are reset when drift is detected:

| Base Model | Description | Best For |
|------------|-------------|----------|
| `gaussiannb` | Gaussian Naive Bayes | Fast, low memory, works well with continuous features |
| `hat` | Hoeffding Adaptive Tree | Better accuracy on complex patterns, handles concept drift internally |

```bash
# Run ADWIN with HAT base model
python scripts/adwin_run.py --dataset elec --base_model hat

# Run all detectors with both base models
python run_experiments.py --mode all --base_model all
```

### Running Multiple Experiments

```bash
# Run all models on all datasets with 10 runs each
python run_experiments.py --mode all

# Run ablation study
python run_ablation_study.py

# Run detector selection experiment
python run_detector_selection_experiment.py

# Run meta-feature type analysis (statistical, stationarity, temporal, all)
python run_meta_feature_experiment.py

# Run hyperparameter sensitivity analysis (window, alpha, lambda)
python run_sensitivity_experiment.py
python run_sensitivity_experiment.py --experiment alpha --n_runs 5
python run_sensitivity_experiment.py --experiment window --n_runs 3
```

## Output Format

Each experiment generates a JSON file with the following structure:

```json
{
    "dataset": "elec",
    "model": "AutoDD",
    "base_model": "GaussianNB",
    "accuracy": 0.789,
    "f1": 0.632,
    "n_drifts": 22,
    "drift_points": [1502, 3045, ...],
    "total_runtime": 1.88,
    "total_memory_mb": 6.76,
    "n_instances": 45312,
    "window_metrics": {
        "interval": 1000,
        "instances": [0, 1000, 2000, ...],
        "accuracy_scores": [0.82, 0.84, ...],
        "f1_scores": [0.78, 0.81, ...]
    },
    "selection_distribution": {
        "ADWIN": 0.225,
        "DDM": 0.123,
        "EDDM": 0.096,
        ...
    }
}
```

**Output Filename Format**: `{Detector}_{BaseModel}_{Dataset}.json` (e.g., `AutoDD_GaussianNB_elec.json`, `ADWIN_HAT_weather.json`)

## Ablation Study

The ablation study analyzes the contribution of different components:

| Configuration | Description |
|---------------|-------------|
| Full AutoDD | Complete model with all features |
| w/o Temporal | Without temporal features |
| w/o Statistical | Without statistical features |
| w/o Stationarity | Without stationarity features |
| Random | Random detector selection (no LinUCB) |

Run ablation study:
```bash
python run_ablation_study.py
```

## Generating Charts and Tables

The `plotting/` directory contains scripts for generating all figures and tables:

### Tables
```bash
# Generate all benchmark tables
python plotting/table_benchmark_accuracy.py
python plotting/table_benchmark_f1.py
python plotting/table_benchmark_time.py
python plotting/table_benchmark_memory.py
python plotting/table_drift_counts.py
python plotting/table_ablation.py
```

### Charts
```bash
# Prequential accuracy/F1 charts
python plotting/chart_prequential_accuracy.py
python plotting/chart_prequential_f1.py

# Ablation box plots
python plotting/chart_ablation.py

# CD diagram
python plotting/chart_cd_diagram.py

# Time vs Memory scatter plot
python plotting/chart_time_memory.py

# Detector selection timeline
python plotting/chart_detector_selection.py

# Error rate with drift regions
python plotting/chart_error_rate_drift.py
```

## Project Structure

```
AutoDD/
├── code/
│   ├── AutoDD/                    # AutoDD implementation
│   ├── MetaADD/                   # Meta-ADD implementation
│   ├── MDDM/                      # MDDM implementation
│   ├── PUDD/                      # PUDD implementation
│   ├── ADDM/                      # ADDM implementation
│   ├── scripts/                   # Individual run scripts
│   │   ├── utils.py              # Shared utilities (base model helpers)
│   │   ├── autodd_run.py
│   │   ├── meta_add_run.py
│   │   ├── no_detector_run.py    # Baseline without drift detection
│   │   ├── adwin_run.py
│   │   ├── ddm_run.py
│   │   ├── eddm_run.py
│   │   ├── hddm_a_run.py
│   │   ├── hddm_w_run.py
│   │   ├── pagehinkley_run.py
│   │   ├── kswin_run.py
│   │   ├── mddm_run.py
│   │   ├── pudd_run.py
│   │   └── addm_run.py
│   ├── plotting/                  # Visualization scripts
│   │   ├── charts/               # Generated charts
│   │   ├── tables/               # Generated tables
│   │   ├── chart_*.py            # Chart generation scripts
│   │   └── table_*.py            # Table generation scripts
│   ├── datasets/                  # Dataset files
│   ├── benchmark_results/         # Experiment results
│   ├── ablation_results/          # Ablation study results
│   ├── run_experiments.py         # Main experiment runner
│   ├── run_ablation_study.py      # Ablation study runner
│   └── generate_synthetic_streams.py
└── README.md
```

## Performance Tips

- Use `--progress_interval 500` for more frequent updates on large datasets
- For large-scale experiments, run models in parallel using multiple terminals
- Monitor memory usage during experiments on large datasets (airlines, poker)

## Troubleshooting

### Common Issues

1. **Memory Issues**: For large datasets (airlines, poker), ensure sufficient RAM (8GB+)
2. **Missing Dependencies**: Install with `pip install river numpy pandas scikit-learn scipy matplotlib psutil tqdm`
3. **Dataset Not Found**: Ensure datasets are in `datasets/` directory or run `generate_synthetic_streams.py`

## Citation

If you use AutoDD in your research, please cite:

```bibtex
@inproceedings{anonymous2026autodd,
  title={AutoDD: Automated Drift Detector Selection for Data Streams},
  author={Anonymous},
  booktitle={Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI-ECAI)},
  year={2026},
  note={Under review}
}
```

## License

MIT License
