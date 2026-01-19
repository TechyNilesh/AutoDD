"""
Generate 3D Sensitivity Analysis Surface Plot

Creates 3D surface plots showing the interaction between two parameters
and their effect on error rates (similar to OASF paper style).

This creates publication-quality 3D surface plots for hyperparameter
sensitivity analysis, visualizing average error rates across a 2D grid
of parameter combinations.

Usage:
    python plotting/chart_sensitivity_3d.py
    python plotting/chart_sensitivity_3d.py --data experiment_results/sensitivity_2d_alpha_lambda.json
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, rcParams

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(CODE_DIR, "experiment_results")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

# Plot settings for publication-quality figures
rcParams['font.weight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['font.size'] = 11


def plot_sensitivity_3d_single(param1_values, param2_values, error_matrix,
                               param1_name, param2_name, 
                               output_filename="sensitivity_3d.png",
                               colormap='viridis', view_angle=(25, 225)):
    """
    Create a single 3D surface plot for sensitivity analysis.
    
    Args:
        param1_values: Array of values for parameter 1 (X-axis)
        param2_values: Array of values for parameter 2 (Y-axis)
        error_matrix: 2D array of error rates, shape (len(param2), len(param1))
        param1_name: Label for parameter 1 (e.g., "Parameter λ")
        param2_name: Label for parameter 2 (e.g., "Parameter α")
        output_filename: Name of the output file
        colormap: Matplotlib colormap name ('viridis', 'coolwarm', 'plasma', etc.)
        view_angle: Tuple of (elevation, azimuth) for the 3D view
    
    Returns:
        Path to the saved figure
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D surface
    # For log-scale visualization, we use indices but label with actual values
    X_idx = np.arange(len(param1_values))
    Y_idx = np.arange(len(param2_values))
    X, Y = np.meshgrid(X_idx, Y_idx)
    Z = np.array(error_matrix)
    
    # Get the colormap
    cmap = cm.get_cmap(colormap)
    
    # Create the surface plot with edge lines (like reference image)
    surf = ax.plot_surface(X, Y, Z,
                           cmap=cmap,
                           edgecolor='black',
                           linewidth=0.3,
                           antialiased=True,
                           alpha=0.95)
    
    # Set axis labels
    ax.set_xlabel(param1_name, fontsize=12, fontweight='bold', labelpad=12)
    ax.set_ylabel(param2_name, fontsize=12, fontweight='bold', labelpad=12)
    ax.set_zlabel('Average Error Rates', fontsize=12, fontweight='bold', labelpad=10)
    
    # Set custom tick labels for log-scale parameters
    ax.set_xticks(X_idx)
    ax.set_xticklabels([f'{v}' for v in param1_values], fontsize=9, rotation=45, ha='right')
    ax.set_yticks(Y_idx)
    ax.set_yticklabels([f'{v}' for v in param2_values], fontsize=9, rotation=-15)
    
    # Format z-axis ticks
    ax.zaxis.set_tick_params(labelsize=9)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Error Rate', fontsize=11, fontweight='bold')
    
    # Set viewing angle (similar to reference image)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Improve grid visibility
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    
    plt.tight_layout()
    
    output_path = os.path.join(CHARTS_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def plot_sensitivity_3d_dual(data1, data2, output_filename="sensitivity_3d_dual.png",
                             colormap='viridis', view_angle=(25, 225)):
    """
    Create side-by-side 3D surface plots (like the OASF reference image).
    
    Args:
        data1: dict with keys:
            - 'param1_values': list of values for first parameter
            - 'param2_values': list of values for second parameter  
            - 'error_matrix': 2D list of error rates
            - 'param1_name': label for first parameter
            - 'param2_name': label for second parameter
        data2: same structure as data1 for the second plot
        output_filename: Name of the output file
        colormap: Matplotlib colormap name
        view_angle: Tuple of (elevation, azimuth) for the 3D view
    
    Returns:
        Path to the saved figure
    """
    
    fig = plt.figure(figsize=(16, 6))
    cmap = cm.get_cmap(colormap)
    
    # ==================== First Subplot ====================
    ax1 = fig.add_subplot(121, projection='3d')
    
    X1_idx = np.arange(len(data1['param1_values']))
    Y1_idx = np.arange(len(data1['param2_values']))
    X1, Y1 = np.meshgrid(X1_idx, Y1_idx)
    Z1 = np.array(data1['error_matrix'])
    
    surf1 = ax1.plot_surface(X1, Y1, Z1,
                             cmap=cmap,
                             edgecolor='black',
                             linewidth=0.3,
                             antialiased=True,
                             alpha=0.95)
    
    ax1.set_xlabel(data1['param1_name'], fontsize=11, fontweight='bold', labelpad=10)
    ax1.set_ylabel(data1['param2_name'], fontsize=11, fontweight='bold', labelpad=10)
    ax1.set_zlabel('Average Error Rates', fontsize=11, fontweight='bold', labelpad=8)
    
    ax1.set_xticks(X1_idx)
    ax1.set_xticklabels([f'{v}' for v in data1['param1_values']], fontsize=8, rotation=45)
    ax1.set_yticks(Y1_idx)
    ax1.set_yticklabels([f'{v}' for v in data1['param2_values']], fontsize=8, rotation=-15)
    
    ax1.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Colorbar for first plot
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.08)
    cbar1.ax.tick_params(labelsize=9)
    
    # Style panes
    for pane in [ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('gray')
    
    # ==================== Second Subplot ====================
    ax2 = fig.add_subplot(122, projection='3d')
    
    X2_idx = np.arange(len(data2['param1_values']))
    Y2_idx = np.arange(len(data2['param2_values']))
    X2, Y2 = np.meshgrid(X2_idx, Y2_idx)
    Z2 = np.array(data2['error_matrix'])
    
    surf2 = ax2.plot_surface(X2, Y2, Z2,
                             cmap=cmap,
                             edgecolor='black',
                             linewidth=0.3,
                             antialiased=True,
                             alpha=0.95)
    
    ax2.set_xlabel(data2['param1_name'], fontsize=11, fontweight='bold', labelpad=10)
    ax2.set_ylabel(data2['param2_name'], fontsize=11, fontweight='bold', labelpad=10)
    ax2.set_zlabel('Average Error Rates', fontsize=11, fontweight='bold', labelpad=8)
    
    ax2.set_xticks(X2_idx)
    ax2.set_xticklabels([f'{v}' for v in data2['param1_values']], fontsize=8, rotation=45)
    ax2.set_yticks(Y2_idx)
    ax2.set_yticklabels([f'{v}' for v in data2['param2_values']], fontsize=8, rotation=-15)
    
    ax2.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Colorbar for second plot
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.08)
    cbar2.ax.tick_params(labelsize=9)
    
    # Style panes
    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('gray')
    
    plt.tight_layout()
    
    output_path = os.path.join(CHARTS_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def load_2d_sensitivity_data(filepath):
    """
    Load 2D sensitivity experiment results from JSON file.
    
    Expected JSON format:
    {
        "param1_name": "λ",
        "param2_name": "α", 
        "param1_values": [0.01, 0.1, 1.0, ...],
        "param2_values": [0.1, 0.5, 1.0, ...],
        "results": {
            "dataset_name": {
                "error_matrix": [[...], [...], ...]
            }
        }
    }
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def create_demo_plots():
    """
    Create demonstration 3D sensitivity plots with synthetic data.
    This shows the expected output format.
    """
    print("Creating demonstration 3D sensitivity plots...")
    
    # Define parameter values (similar to OASF paper)
    lambda_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    alpha_values = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    window_values = [25, 50, 100, 150, 200]
    
    # Generate synthetic error matrix (replace with real data)
    # Shape: (len(param2), len(param1)) 
    np.random.seed(42)
    
    # Create bowl-shaped surface (optimal in middle)
    def create_sensitivity_surface(n_rows, n_cols, base_error=0.12, noise=0.02):
        """Create a realistic sensitivity surface with optimal region."""
        x = np.linspace(-1, 1, n_cols)
        y = np.linspace(-1, 1, n_rows)
        X, Y = np.meshgrid(x, y)
        # Bowl shape with optimal in center
        Z = base_error + 0.08 * (X**2 + Y**2) + noise * np.random.randn(n_rows, n_cols)
        return np.clip(Z, 0.10, 0.30)
    
    error_matrix_1 = create_sensitivity_surface(len(alpha_values), len(lambda_values))
    error_matrix_2 = create_sensitivity_surface(len(alpha_values), len(window_values))
    
    # Single 3D plot
    path1 = plot_sensitivity_3d_single(
        param1_values=lambda_values,
        param2_values=alpha_values,
        error_matrix=error_matrix_1,
        param1_name="Parameter λ",
        param2_name="Parameter α",
        output_filename="sensitivity_3d_lambda_alpha.png",
        colormap='viridis',
        view_angle=(25, 225)
    )
    print(f"  Single plot saved: {path1}")
    
    # Dual 3D plot (like reference image)
    data1 = {
        'param1_values': lambda_values,
        'param2_values': alpha_values,
        'error_matrix': error_matrix_1.tolist(),
        'param1_name': 'Parameter λ',
        'param2_name': 'Parameter α'
    }
    
    data2 = {
        'param1_values': window_values,
        'param2_values': alpha_values,
        'error_matrix': error_matrix_2.tolist(),
        'param1_name': 'Window Size',
        'param2_name': 'Parameter α'
    }
    
    path2 = plot_sensitivity_3d_dual(data1, data2, "sensitivity_3d_dual_demo.png")
    print(f"  Dual plot saved: {path2}")
    
    return path1, path2


def plot_from_existing_data():
    """
    Create 3D sensitivity plots by combining existing 1D sensitivity data.
    Uses lambda and alpha sensitivity results to create a pseudo-2D visualization.
    """
    print("Creating 3D plot from existing sensitivity data...")
    
    # Load existing 1D sensitivity results
    lambda_file = os.path.join(RESULTS_DIR, "lambda_sensitivity_summary.json")
    alpha_file = os.path.join(RESULTS_DIR, "alpha_sensitivity_summary.json")
    
    if not os.path.exists(lambda_file) or not os.path.exists(alpha_file):
        print("  Warning: Existing sensitivity data not found. Run demo instead.")
        return create_demo_plots()
    
    with open(lambda_file, 'r') as f:
        lambda_data = json.load(f)
    with open(alpha_file, 'r') as f:
        alpha_data = json.load(f)
    
    lambda_values = lambda_data['lambda_values']
    alpha_values = alpha_data['alpha_values']
    
    # For each dataset, create an interpolated 2D error matrix
    # This is an approximation since we don't have true 2D data
    for dataset in ['sea', 'hyp', 'rbf']:
        if dataset not in lambda_data['results'] or dataset not in alpha_data['results']:
            continue
        
        # Get 1D error rates (1 - accuracy)
        lambda_errors = [1 - lambda_data['results'][dataset][str(l)]['accuracy_mean'] 
                        for l in lambda_values]
        alpha_errors = [1 - alpha_data['results'][dataset][str(a)]['accuracy_mean'] 
                       for a in alpha_values]
        
        # Create approximate 2D surface by combining 1D effects
        # This assumes additive effect (real data would need actual 2D experiment)
        error_matrix = []
        for a_err in alpha_errors:
            row = []
            for l_err in lambda_errors:
                # Combine effects (simplistic model)
                combined = (a_err + l_err) / 2
                row.append(combined)
            error_matrix.append(row)
        
        path = plot_sensitivity_3d_single(
            param1_values=lambda_values,
            param2_values=alpha_values,
            error_matrix=error_matrix,
            param1_name="Parameter λ",
            param2_name="Parameter α",
            output_filename=f"sensitivity_3d_{dataset}.png",
            colormap='viridis'
        )
        print(f"  {dataset.upper()} plot saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='Generate 3D Sensitivity Analysis Charts')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to 2D sensitivity JSON file')
    parser.add_argument('--demo', action='store_true',
                        help='Create demonstration plots with synthetic data')
    parser.add_argument('--combine', action='store_true',
                        help='Combine existing 1D sensitivity data into 3D plot')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generating 3D Sensitivity Analysis Charts")
    print("=" * 60)
    
    if args.data:
        # Load and plot from provided data file
        data = load_2d_sensitivity_data(args.data)
        # Implementation depends on data format
        print(f"Loaded data from: {args.data}")
        # TODO: Plot based on data format
    elif args.combine:
        plot_from_existing_data()
    else:
        # Default: create demo plots
        create_demo_plots()
    
    print("\n" + "=" * 60)
    print("3D Sensitivity Charts Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
