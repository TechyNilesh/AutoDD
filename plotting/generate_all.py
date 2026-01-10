"""
Generate All Tables and Charts

Runs all plotting scripts to generate tables and charts for the paper.

Usage:
    python plotting/generate_all.py
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TABLE_SCRIPTS = [
    "table_benchmark_accuracy.py",
    "table_benchmark_f1.py",
    "table_benchmark_time.py",
    "table_benchmark_memory.py",
    "table_drift_counts.py",
    "table_ablation.py",
    "table_window_size.py",
    "table_detector_selection.py",
]

CHART_SCRIPTS = [
    "chart_prequential_accuracy.py",
    "chart_prequential_f1.py",
    "chart_ablation.py",
    "chart_cd_diagram.py",
    "chart_detector_selection.py",
    "chart_error_rate_drift.py",
    "chart_time_memory.py",
]


def run_script(script_name):
    """Run a script and return success status."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(SCRIPT_DIR)
    )
    return result.returncode == 0


def main():
    print("=" * 70)
    print("GENERATING ALL TABLES AND CHARTS")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    print("\n--- GENERATING TABLES ---")
    for script in TABLE_SCRIPTS:
        if run_script(script):
            success_count += 1
        else:
            fail_count += 1
            print(f"WARNING: {script} failed!")

    print("\n--- GENERATING CHARTS ---")
    for script in CHART_SCRIPTS:
        if run_script(script):
            success_count += 1
        else:
            fail_count += 1
            print(f"WARNING: {script} failed!")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"\nOutput locations:")
    print(f"  Tables: {os.path.join(SCRIPT_DIR, 'tables')}/")
    print(f"  Charts: {os.path.join(SCRIPT_DIR, 'charts')}/")


if __name__ == "__main__":
    main()
