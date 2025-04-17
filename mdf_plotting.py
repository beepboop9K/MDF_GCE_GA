#!/usr/bin/env python3.8
################################
# Plotting functions for MDF_GA
################################

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import pandas as pd

def ensure_dirs():
    """Ensure necessary directories exist"""
    os.makedirs('GA/loss', exist_ok=True)

def plot_mdf_curves(GalGA, feh, normalized_count, results_df=None):
    """Plot all model MDFs, highlight the best model, and overlay observational data"""
    plt.figure(figsize=(18, 12))
    
    # Get the best model from the results dataframe if provided
    if results_df is not None and not results_df.empty:
        best_model = results_df.iloc[0]
        best_params = np.array([best_model['sigma_2'], best_model['t_2'], best_model['infall_2']])
    else:
        # Otherwise just use the first model as the reference
        best_params = np.array([GalGA.results[0][5], GalGA.results[0][7], GalGA.results[0][9]])
    
    # Loop through each stored model result to plot its MDF curve
    for i in range(len(GalGA.mdf_data)):
        x_data, y_data = GalGA.mdf_data[i]  # each is an array for the MDF curve
        label = GalGA.labels[i]
        
        # Get the parameters of the current model from the results array
        params = np.array([GalGA.results[i][5], GalGA.results[i][7], GalGA.results[i][9]])
        
        # If these match (within tolerance) the best model's parameters, plot in red and thicker
        if np.allclose(params, best_params, rtol=1e-5):
            plt.plot(x_data, y_data, label=f'{label} (BEST)', color='red', linewidth=2, zorder=3)
        else:
            plt.plot(x_data, y_data, alpha=0.5, zorder=1)
    
    # Plot the raw observational data (black crosses)
    plt.plot(feh, normalized_count, label='Observational Data', color='black', 
             marker='x', linestyle='-', markersize=12, zorder=2)
    
    plt.xlabel('[Fe/H]')
    plt.ylabel('Normalized Number Density')
    plt.xlim(-2, 1)
    plt.legend()
    plt.title('Metallicity Distribution Functions (MDFs)')
    plt.savefig('GA/MDF_multiple_results.png', bbox_inches='tight')
    print("Generated MDF curves plot")
    
    return plt.gcf()

def plot_3d_scatter(x, y, z, color_metric, label, xlabel='sigma_2', ylabel='t_2', zlabel='infall_2'):
    """Plot 3D scatter plot with color indicating a specific metric"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f'3D Parameter Space Colored by {label}')
    plt.savefig(f'GA/loss/{label}_loss_3d.png', bbox_inches='tight')
    plt.close()
    
    return fig

def plot_2d_scatter(x, y, color_metric, label, xlabel='t_2', ylabel='infall_2'):
    """Plot 2D scatter plot with color indicating a specific metric"""
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(x, y, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{label} Loss')
    plt.savefig(f'GA/loss/{label}_loss_2d.png', bbox_inches='tight')
    plt.close()
    
    return plt.gcf()

def plot_walker_history(walker_history, param_names, param_indices):
    """Plot the evolution of parameters for each walker"""
    if not walker_history:
        print("Walker history data not available. Skipping walker evolution plots.")
        return None
        
    figs = []
    for idx, param_name in enumerate(param_names):
        fig = plt.figure(figsize=(12, 8))
        figs.append(fig)
        
        for walker_idx, history in walker_history.items():
            if not history:  # Skip if history is empty
                continue
                
            history = np.array(history)  # Convert to numpy array for easier slicing
            param_idx = param_indices[idx]
            
            if param_idx >= history.shape[1]:  # Skip if parameter index is out of bounds
                continue
                
            generations = np.arange(len(history))
            
            # Plot the parameter value for this walker
            plt.plot(
                generations, 
                history[:, param_idx], 
                label=f"Walker {walker_idx}",
                alpha=0.5  # Adjust transparency for better visualization
            )
        
        plt.xlabel("Generation")
        plt.ylabel(f"{param_name} Value")
        plt.title(f"Evolution of {param_name} Over Generations")
        plt.legend(loc="upper right", fontsize="small", ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'GA/loss/walker_evolution_{param_name}.png', bbox_inches='tight')
        plt.close()
    
    print("Generated walker evolution plots")
    return figs

def create_3d_animation(walker_history):
    """Create an animated 3D visualization of walker evolution"""
    if not walker_history:
        print("Walker history data not available. Skipping 3D animation.")
        return None
        
    # Get maximum number of generations
    num_generations = max(len(v) for v in walker_history.values()) if walker_history else 0
    if num_generations == 0:
        print("No generation data found. Skipping 3D animation.")
        return None
    
    # Initialize figure for 3D animation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for walkers
    colors = plt.cm.viridis(np.linspace(0, 1, len(walker_history)))
    
    # Animation function
    def update(num):
        ax.clear()
        ax.set_xlabel("Generation")
        ax.set_ylabel("tmax_2")
        ax.set_zlabel("infall_2")
        ax.set_title("Walker Evolution in 3D")
        ax.view_init(elev=20, azim=num)  # Rotate by 1 degree per frame
    
        for i, (walker_id, history) in enumerate(walker_history.items()):
            if not history:
                continue
            history = np.array(history)
            generations = np.arange(len(history))
            
            # Use correct indices for t_2 (7) and infall_2 (9)
            if num < num_generations:
                # During first rotation, show progressive evolution
                plot_up_to = min(num+1, len(history))
                ax.plot(generations[:plot_up_to], history[:plot_up_to, 7], history[:plot_up_to, 9], 
                        color=colors[i], alpha=0.7, label=f"Walker {i}")
            else:
                # Second rotation shows complete paths
                ax.plot(generations, history[:, 7], history[:, 9], 
                        color=colors[i], alpha=0.7, label=f"Walker {i}")
    
        ax.legend(loc="upper right", fontsize="small")
    
    # Create animation with two full rotations
    total_frames = 360 * 2  # Two full rotations at 1 degree per frame
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)
    
    # Save as GIF with lower frame rate
    gif_path = "GA/loss/walker_evolution_3D.gif"
    ani.save(gif_path, writer="pillow", fps=6)  # Lower fps for slower rotation
    plt.close()
    
    print(f"Generated 3D animation: {gif_path}")
    return ani

def extract_metrics(results_file):
    """Extract metrics from CSV file for plotting"""
    # Load the dataframe directly
    df = pd.read_csv(results_file)
    
    # Extract parameters using column names
    sigma_2_vals = df['sigma_2'].values
    t_2_vals = df['t_2'].values
    infall_2_vals = df['infall_2'].values
    
    # Extract metrics
    metrics_dict = {}
    for metric in ['wrmse', 'mae', 'mape', 'huber', 'cosine', 'log_cosh', 'ks', 'ensemble']:
        if metric in df.columns:
            metrics_dict[metric] = df[metric].values
    
    return sigma_2_vals, t_2_vals, infall_2_vals, metrics_dict, df

def generate_all_plots(GalGA, feh, normalized_count, results_file='GA/simulation_results.csv'):
    """Generate all plots from GalGA results"""
    # Ensure directories exist
    ensure_dirs()
    
    # Extract metrics for scatter plots
    sigma_2_vals, t_2_vals, infall_2_vals, metrics_dict, df = extract_metrics(results_file)
    
    # 1. Plot MDF curves
    plot_mdf_curves(GalGA, feh, normalized_count, df)
    
    # 2. Plot 3D scatter plots for various metrics
    print("Generating 3D scatter plots...")
    for metric_name, metric_vals in metrics_dict.items():
        plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, metric_vals, metric_name)
    
    # 3. Plot 2D scatter plots
    print("Generating 2D scatter plots...")
    for metric_name, metric_vals in metrics_dict.items():
        plot_2d_scatter(t_2_vals, infall_2_vals, metric_vals, metric_name)
    
    # 4. Plot walker history
    print("Generating walker evolution plots...")
    param_names = ["sigma_2", "t_2", "infall_2"]
    param_indices = [5, 7, 9]  # Indices in the individual arrays
    plot_walker_history(GalGA.walker_history, param_names, param_indices)
    
    # 5. Create 3D animation
    print("Generating 3D animation...")
    create_3d_animation(GalGA.walker_history)
    
    print("All plotting complete! Check the GA directory for results.")