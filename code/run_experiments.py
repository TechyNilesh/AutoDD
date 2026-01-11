"""
Main Experiment Runner for AutoDD

This script runs drift detection experiments on datasets using various detectors.

Usage:
    python run_experiments.py --mode dataset --name elec
    python run_experiments.py --mode detector --name adwin
    python run_experiments.py --mode all
    python run_experiments.py --mode random

Examples:
    # Run all detectors on a single dataset
    python run_experiments.py --mode dataset --name elec

    # Run a single detector on all datasets
    python run_experiments.py --mode detector --name adwin

    # Run all detectors on all datasets
    python run_experiments.py --mode all

    # Run multiple random runs
    python run_experiments.py --mode random --n_runs 10
"""

import concurrent.futures
import subprocess
import random
import argparse
import os

# Dataset lists
SYNTHETIC_DATASETS = ['sea', 'hyp', 'agr', 'rbf', 'rtg']
REALWORLD_DATASETS = ['elec', 'weather', 'spam', 'airlines', 'poker']
ALL_DATASETS = SYNTHETIC_DATASETS + REALWORLD_DATASETS

# Detector list (from paper's portfolio + baselines + AutoDD)
DETECTORS = [
    'gaussiannb',  # Baseline: No drift detection (*)
    'adwin',
    'ddm',
    'eddm',
    'hddm_a',
    'hddm_w',
    'pagehinkley',
    'kswin',
    'mddm',      # MDDM (McDiarmid Drift Detection Method)
    'pudd',      # PUDD (Prediction Uncertainty-based Drift Detector)
    'addm',      # ADDM (Autoregressive-Based Drift Detection Method)
    'meta_add',  # Meta-ADD baseline
    'autodd',    # AutoDD (our method)
]

# Detector script mapping
DETECTOR_SCRIPTS = {
    'gaussiannb': 'scripts/gaussiannb_run.py',
    'adwin': 'scripts/adwin_run.py',
    'meta_add': 'scripts/meta_add_run.py',
    'autodd': 'scripts/autodd_run.py',
    'ddm': 'scripts/ddm_run.py',
    'eddm': 'scripts/eddm_run.py',
    'hddm_a': 'scripts/hddm_a_run.py',
    'hddm_w': 'scripts/hddm_w_run.py',
    'pagehinkley': 'scripts/pagehinkley_run.py',
    'kswin': 'scripts/kswin_run.py',
    'mddm': 'scripts/mddm_run.py',
    'pudd': 'scripts/pudd_run.py',
    'addm': 'scripts/addm_run.py',
}


def run_script(dataset_name, detector_name, run_count=None, random_seed=False,
               progress_interval=1000, output_dir="benchmark_results", verbose=False):
    """
    Execute a detector script for a specified dataset.

    Args:
        dataset_name: Name of the dataset
        detector_name: Name of the detector
        run_count: Run identifier for multiple runs
        random_seed: Whether to use random seeds
        progress_interval: Interval for progress updates
        output_dir: Directory to save results
        verbose: Enable verbose output
    """
    if random_seed:
        seed = random.randint(1, 1000)
    else:
        seed = 42

    script_path = DETECTOR_SCRIPTS.get(detector_name)
    if script_path is None:
        print(f"Unknown detector: {detector_name}")
        return

    print(f"\n{'='*60}")
    print(f"Running {detector_name.upper()} on {dataset_name}")
    print(f"Seed: {seed}, Run: {run_count if run_count else 'single'}")
    print(f"{'='*60}")

    command = ['python', script_path, '--dataset', dataset_name]

    if run_count is not None:
        command += ['--run_count', str(run_count)]

    command += ['--seed', str(seed)]
    command += ['--progress_interval', str(progress_interval)]
    command += ['--output_dir', output_dir]

    if verbose:
        command += ['--verbose']

    # Execute the command
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Print output in real-time
    for line in process.stdout:
        print(line, end='')

    process.wait()

    # Check for errors
    if process.returncode != 0:
        print(f"Error running {detector_name} on {dataset_name}:")
        print(process.stderr.read())


def run_all_detectors_for_dataset(dataset_name, progress_interval=1000,
                                   output_dir="benchmark_results", verbose=False,
                                   parallel=True, max_workers=None):
    """Run all detectors on a single dataset."""
    print(f"\n{'#'*60}")
    print(f"Running all detectors on {dataset_name}")
    print(f"{'#'*60}")

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            futures = []
            for detector in DETECTORS:
                futures.append(executor.submit(
                    run_script, dataset_name, detector, None, False,
                    progress_interval, output_dir, verbose
                ))
            concurrent.futures.wait(futures)
    else:
        for detector in DETECTORS:
            run_script(dataset_name, detector, None, False,
                      progress_interval, output_dir, verbose)


def run_detector_on_all_datasets(detector_name, progress_interval=1000,
                                  output_dir="benchmark_results", verbose=False,
                                  parallel=True, max_workers=None):
    """Run a single detector on all datasets."""
    print(f"\n{'#'*60}")
    print(f"Running {detector_name.upper()} on all datasets")
    print(f"{'#'*60}")

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            futures = []
            for dataset in ALL_DATASETS:
                futures.append(executor.submit(
                    run_script, dataset, detector_name, None, False,
                    progress_interval, output_dir, verbose
                ))
            concurrent.futures.wait(futures)
    else:
        for dataset in ALL_DATASETS:
            run_script(dataset, detector_name, None, False,
                      progress_interval, output_dir, verbose)


def run_all_experiments(progress_interval=1000, output_dir="benchmark_results",
                        verbose=False, parallel=True, max_workers=None):
    """Run all detectors on all datasets."""
    print(f"\n{'#'*60}")
    print("Running ALL experiments")
    print(f"Detectors: {len(DETECTORS)}, Datasets: {len(ALL_DATASETS)}")
    print(f"Total runs: {len(DETECTORS) * len(ALL_DATASETS)}")
    print(f"{'#'*60}")

    tasks = [(dataset, detector) for dataset in ALL_DATASETS for detector in DETECTORS]

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            futures = []
            for dataset, detector in tasks:
                futures.append(executor.submit(
                    run_script, dataset, detector, None, False,
                    progress_interval, output_dir, verbose
                ))
            concurrent.futures.wait(futures)
    else:
        for dataset, detector in tasks:
            run_script(dataset, detector, None, False,
                      progress_interval, output_dir, verbose)


def run_random_experiments(n_runs=10, progress_interval=1000,
                           output_dir="benchmark_results", verbose=False,
                           parallel=True, max_workers=None):
    """Run multiple random experiments."""
    print(f"\n{'#'*60}")
    print(f"Running {n_runs} random experiments")
    print(f"Detectors: {len(DETECTORS)}, Datasets: {len(ALL_DATASETS)}")
    print(f"Total runs: {len(DETECTORS) * len(ALL_DATASETS) * n_runs}")
    print(f"{'#'*60}")

    tasks = [
        (dataset, detector, run)
        for run in range(1, n_runs + 1)
        for dataset in ALL_DATASETS
        for detector in DETECTORS
    ]

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            futures = []
            for dataset, detector, run in tasks:
                futures.append(executor.submit(
                    run_script, dataset, detector, run, True,
                    progress_interval, output_dir, verbose
                ))
            concurrent.futures.wait(futures)
    else:
        for dataset, detector, run in tasks:
            run_script(dataset, detector, run, True,
                      progress_interval, output_dir, verbose)


def run_synthetic_only(progress_interval=1000, output_dir="benchmark_results",
                       verbose=False, parallel=True, max_workers=None):
    """Run all detectors on synthetic datasets only."""
    print(f"\n{'#'*60}")
    print("Running experiments on SYNTHETIC datasets")
    print(f"{'#'*60}")

    tasks = [(dataset, detector) for dataset in SYNTHETIC_DATASETS for detector in DETECTORS]

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            futures = []
            for dataset, detector in tasks:
                futures.append(executor.submit(
                    run_script, dataset, detector, None, False,
                    progress_interval, output_dir, verbose
                ))
            concurrent.futures.wait(futures)
    else:
        for dataset, detector in tasks:
            run_script(dataset, detector, None, False,
                      progress_interval, output_dir, verbose)


def run_realworld_only(progress_interval=1000, output_dir="benchmark_results",
                       verbose=False, parallel=True, max_workers=None):
    """Run all detectors on real-world datasets only."""
    print(f"\n{'#'*60}")
    print("Running experiments on REAL-WORLD datasets")
    print(f"{'#'*60}")

    tasks = [(dataset, detector) for dataset in REALWORLD_DATASETS for detector in DETECTORS]

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            futures = []
            for dataset, detector in tasks:
                futures.append(executor.submit(
                    run_script, dataset, detector, None, False,
                    progress_interval, output_dir, verbose
                ))
            concurrent.futures.wait(futures)
    else:
        for dataset, detector in tasks:
            run_script(dataset, detector, None, False,
                      progress_interval, output_dir, verbose)


def run_detector_multiple_runs(detector_name, n_runs=10, progress_interval=1000,
                                output_dir="benchmark_results", verbose=False,
                                parallel=True, max_workers=None):
    """Run a single detector with multiple runs on all datasets."""
    print(f"\n{'#'*60}")
    print(f"Running {detector_name.upper()} with {n_runs} runs on all datasets")
    print(f"Datasets: {len(ALL_DATASETS)}")
    print(f"Total runs: {len(ALL_DATASETS) * n_runs}")
    print(f"{'#'*60}")

    tasks = [
        (dataset, run)
        for run in range(1, n_runs + 1)
        for dataset in ALL_DATASETS
    ]

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
            futures = []
            for dataset, run in tasks:
                futures.append(executor.submit(
                    run_script, dataset, detector_name, run, True,
                    progress_interval, output_dir, verbose
                ))
            concurrent.futures.wait(futures)
    else:
        for dataset, run in tasks:
            run_script(dataset, detector_name, run, True,
                      progress_interval, output_dir, verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run drift detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  dataset      - Run all detectors on a specific dataset
  detector     - Run a specific detector on all datasets (single run)
  detector_runs- Run a specific detector with multiple runs on all datasets
  all          - Run all detectors on all datasets
  random       - Run multiple random experiments (all detectors)
  synthetic    - Run all detectors on synthetic datasets only
  realworld    - Run all detectors on real-world datasets only

Examples:
  python run_experiments.py --mode dataset --name elec
  python run_experiments.py --mode detector --name adwin
  python run_experiments.py --mode detector_runs --name meta_add --n_runs 10
  python run_experiments.py --mode all
  python run_experiments.py --mode random --n_runs 10
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                        choices=['dataset', 'detector', 'detector_runs', 'all', 'random', 'synthetic', 'realworld'],
                        help='Experiment mode')
    parser.add_argument('--name', type=str, default=None,
                        help='Dataset or detector name (for dataset/detector modes)')
    parser.add_argument('--n_runs', type=int, default=10,
                        help='Number of runs for random mode')
    parser.add_argument('--progress_interval', type=int, default=1000,
                        help='Progress update interval')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--sequential', action='store_true',
                        help='Run experiments sequentially (not parallel)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Max parallel workers (default: CPU count)')

    args = parser.parse_args()
    parallel = not args.sequential
    max_workers = args.max_workers if args.max_workers else os.cpu_count()

    print(f"Running with max_workers={max_workers} (CPUs available: {os.cpu_count()})")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'dataset':
        if not args.name:
            print("Please provide dataset name with --name")
            print(f"Available datasets: {ALL_DATASETS}")
        else:
            run_all_detectors_for_dataset(
                args.name, args.progress_interval, args.output_dir,
                args.verbose, parallel, max_workers
            )

    elif args.mode == 'detector':
        if not args.name:
            print("Please provide detector name with --name")
            print(f"Available detectors: {DETECTORS}")
        else:
            run_detector_on_all_datasets(
                args.name, args.progress_interval, args.output_dir,
                args.verbose, parallel, max_workers
            )

    elif args.mode == 'detector_runs':
        if not args.name:
            print("Please provide detector name with --name")
            print(f"Available detectors: {DETECTORS}")
        else:
            run_detector_multiple_runs(
                args.name, args.n_runs, args.progress_interval, args.output_dir,
                args.verbose, parallel, max_workers
            )

    elif args.mode == 'all':
        run_all_experiments(
            args.progress_interval, args.output_dir, args.verbose, parallel, max_workers
        )

    elif args.mode == 'random':
        run_random_experiments(
            args.n_runs, args.progress_interval, args.output_dir,
            args.verbose, parallel, max_workers
        )

    elif args.mode == 'synthetic':
        run_synthetic_only(
            args.progress_interval, args.output_dir, args.verbose, parallel, max_workers
        )

    elif args.mode == 'realworld':
        run_realworld_only(
            args.progress_interval, args.output_dir, args.verbose, parallel, max_workers
        )

    print(f"\n{'='*60}")
    print("Experiments completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")
