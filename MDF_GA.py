#!/usr/bin/env python3.8
################################
# Author: N Miller, M Joyce, (Fixed version)
################################

# Importing required libraries
import matplotlib.pyplot as plt
import warnings
import numpy as np
import os

# Create GA directory if it doesn't exist
os.makedirs('GA/loss', exist_ok=True)

# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
from scipy.interpolate import CubicSpline
from deap import base, creator, tools
import csv
import random
import Gal_GA_PP as Gal_GA
from matplotlib import cm, colors
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import pandas as pd 

# Adding custom paths
sys.path.append('../')

from NuPyCEE import omega, read_yields, stellab, sygma
from JINAPyCEE import omega_plus

# Setting up color maps for plotting
cmap = cm.ocean
norm = colors.Normalize(vmin=1, vmax=375)
m = cm.ScalarMappable(norm=norm, cmap=cmap)

# Load and normalize observational data
obs_file = 'binned_dist_lat6_0.08dex.dat'
feh, count = np.loadtxt(obs_file, usecols=(0, 1), unpack=True)
normalized_count = count / count.max()  # Normalize count for comparison

# Parse parameters from the parameter card file
params = Gal_GA.parse_inlist('bulge_pcard.txt')

# Assign parsed parameters to variables
iniab_header = params['iniab_header']
sn1a_header = params['sn1a_header']
sigma_2_list = params['sigma_2_list']
tmax_1_list = params['tmax_1_list']
tmax_2_list = params['tmax_2_list']
infall_timescale_1_list = params['infall_timescale_1_list']
infall_timescale_2_list = params['infall_timescale_2_list']
comp_array = params['comp_array']
sfe_array = params['sfe_array']
imf_array = params['imf_array']
imf_upper_limits = params['imf_upper_limits']
sn1a_assumptions = params['sn1a_assumptions']
stellar_yield_assumptions = params['stellar_yield_assumptions']
mgal_values = params['mgal_values']
nb_array = params['nb_array']
sn1a_rates = params['sn1a_rates']
timesteps = params['timesteps']
A2 = params['A2']
A1 = params['A1']

popsize = params['popsize']
generations = params['generations']
crossover_probability = params['crossover_probability']
mutation_probability = params['mutation_probability']
tournament_size = params['tournament_size']
selection_threshold = params['selection_threshold']

loss_metric = params['loss_metric']
fancy_mutation = params['fancy_mutation']
shrink_range = params['shrink_range']

# Initialize the Galactic Evolution Genetic Algorithm class with parsed parameters
GalGA = Gal_GA.GalacticEvolutionGA(
    iniab_header=iniab_header,
    sn1a_header=sn1a_header,
    sigma_2_list=sigma_2_list,
    tmax_1_list=tmax_1_list,
    tmax_2_list=tmax_2_list,    
    infall_timescale_1_list=infall_timescale_1_list,
    infall_timescale_2_list=infall_timescale_2_list,
    comp_array=comp_array,
    imf_array=imf_array,
    sfe_array=sfe_array,
    imf_upper_limits=imf_upper_limits,
    sn1a_assumptions=sn1a_assumptions,
    stellar_yield_assumptions=stellar_yield_assumptions,
    mgal_values=mgal_values,
    nb_array=nb_array,
    sn1a_rates=sn1a_rates,
    timesteps=timesteps,
    A1=A1,
    A2=A2,
    feh=feh,
    normalized_count=normalized_count,
    loss_metric=loss_metric,
    fancy_mutation=fancy_mutation,
    shrink_range=shrink_range,
    tournament_size=tournament_size,
    threshold=selection_threshold,
    cxpb=crossover_probability, 
    mutpb=mutation_probability, 
    PP=True
)

# Initialize Genetic Algorithm population and toolbox
genal_population, genal_toolbox = GalGA.init_GenAl(population_size=popsize)

# Run the GA
#GalGA.GenAl(population_size=popsize, num_generations=generations, population=genal_population, toolbox=genal_toolbox)

# Define column names based on the actual structure of GalGA.results
# Note: These must match exactly what's being returned in the 'metrics' list in GalGA.evaluate
col_names = [
    'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
    'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
    'sfe', 'imf_upper', 'mgal', 'nb',
    'ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber', 'cosine', 'log_cosh'
]

results_df = pd.DataFrame(GalGA.results, columns=col_names)

# Use the chosen loss metric to define a loss column
results_df['loss'] = results_df[loss_metric]

# Sort the results DataFrame by loss (lowest first) and reset index
results_df.sort_values('loss', inplace=True)
results_df.reset_index(drop=True, inplace=True)

# Save the results to a CSV file
results_file = 'GA/simulation_results.csv'
results_df.to_csv(results_file, index=False)
print("Results saved to:", results_file)

# The best model is now the first row in the sorted DataFrame
best_model = results_df.iloc[0]
print("Best model from results dataframe:\n", best_model)

# --- Plotting: plot all model MDFs, highlight the best model, and overlay observational data
plt.figure(figsize=(18, 12))

# Extract the key parameters for comparison (using the right column indices)
# For our comparison, we'll use sigma_2, t_2, and infall_2 which are at indices 5, 7, and 9
best_params = np.array([best_model['sigma_2'], best_model['t_2'], best_model['infall_2']])

# Loop through each stored model result to plot its MDF curve
for i in range(len(GalGA.mdf_data)):
    x_data, y_data = GalGA.mdf_data[i]  # each is an array for the MDF curve
    label = GalGA.labels[i]
    
    # Get the parameters of the current model from the results array
    # We need to adjust this to match the actual indices in the results array
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
plt.savefig('GA/MDF_multiple_results.png', bbox_inches='tight')

def extract_metrics(results_file):
    """
    Extracts the metric arrays from the given CSV file.
    
    Returns:
        sigma_2_vals, t_2_vals, infall_2_vals, wrmse_vals, 
        mae_vals, mape_vals, huber_vals, cosine_vals, log_cosh_vals
    """
    # Load the dataframe directly instead of using genfromtxt
    df = pd.read_csv(results_file)
    
    # Extract parameters using the correct column names
    sigma_2_vals = df['sigma_2'].values
    t_2_vals = df['t_2'].values
    infall_2_vals = df['infall_2'].values
    
    # Extract metrics
    wrmse_vals = df['wrmse'].values
    mae_vals = df['mae'].values
    mape_vals = df['mape'].values
    huber_vals = df['huber'].values
    cosine_vals = df['cosine'].values
    log_cosh_vals = df['log_cosh'].values
    
    return sigma_2_vals, t_2_vals, infall_2_vals, wrmse_vals, mae_vals, mape_vals, huber_vals, cosine_vals, log_cosh_vals

# Extract metrics from the file
results_file = 'GA/simulation_results.csv'
sigma_2_vals, t_2_vals, infall_2_vals, wrmse_vals, mae_vals, mape_vals, huber_vals, cos_vals, log_cosh_vals = extract_metrics(results_file)

# 3D Scatter Plot function for loss metrics
def plot_3d_scatter(x, y, z, color_metric, label, xlabel='sigma_2', ylabel='t_2', zlabel='infall_2'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.savefig(f'GA/loss/{label}_loss_3d.png', bbox_inches='tight')
    plt.close()

# Plot 3D scatter plots for various loss metrics
plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, wrmse_vals, 'WRMSE')
plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, mae_vals, 'MAE')
plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, mape_vals, 'MAPE')
plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, huber_vals, 'Huber')
plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, cos_vals, 'Cosine')
plot_3d_scatter(sigma_2_vals, t_2_vals, infall_2_vals, log_cosh_vals, 'Log(cosh)')

# 2D Scatter Plot function for loss metrics
def plot_2d_scatter(x, y, color_metric, label, xlabel='t_2', ylabel='infall_2'):
    plt.figure()
    sc = plt.scatter(x, y, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{label} Loss')
    plt.savefig(f'GA/loss/{label}_loss_2d.png', bbox_inches='tight')
    plt.close()

# Plot 2D scatter plots for various loss metrics
plot_2d_scatter(t_2_vals, infall_2_vals, wrmse_vals, 'WRMSE')
plot_2d_scatter(t_2_vals, infall_2_vals, mae_vals, 'MAE')
plot_2d_scatter(t_2_vals, infall_2_vals, mape_vals, 'MAPE')
plot_2d_scatter(t_2_vals, infall_2_vals, huber_vals, 'Huber')
plot_2d_scatter(t_2_vals, infall_2_vals, cos_vals, 'Cosine')
plot_2d_scatter(t_2_vals, infall_2_vals, log_cosh_vals, 'Log(cosh)')

# Walker history plotting function
def plot_walker_history(walker_history, param_names):
    for param_idx, param_name in enumerate(param_names):
        plt.figure(figsize=(12, 8))

        for walker_idx, history in walker_history.items():
            if not history:  # Skip if history is empty
                continue
                
            history = np.array(history)  # Convert to numpy array for easier slicing
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

# Define the real parameter names we're tracking (sigma_2, t_2, infall_2)
param_names = ["sigma_2", "t_2", "infall_2"]
# Define indices of these parameters in the walker_history arrays
param_indices = [5, 7, 9]  # These are the indices in the individual arrays

# Plot the walker history
plot_walker_history(GalGA.walker_history, param_names)

# Extract total generations
num_generations = max(len(v) for v in GalGA.walker_history.values())

# Initialize figure for 3D animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Colors for walkers
colors = plt.cm.viridis(np.linspace(0, 1, len(GalGA.walker_history)))

# Labels
ax.set_xlabel("Generation")
ax.set_ylabel("tmax_2")
ax.set_zlabel("infall_2")
ax.set_title("Walker Evolution in 3D")

# Animation function
def update(num):
    ax.clear()
    ax.set_xlabel("Generation")
    ax.set_ylabel("tmax_2")
    ax.set_zlabel("infall_2")
    ax.set_title("Walker Evolution in 3D")
    ax.view_init(elev=20, azim=num)  # Rotate by 1 degree per frame

    for i, (walker_id, history) in enumerate(GalGA.walker_history.items()):
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

# Import animation module
import matplotlib.animation as animation

# Create animation with two full rotations:
# First rotation (360 degrees) shows progressive evolution
# Second rotation (360 degrees) shows the complete paths
total_frames = 360 * 2  # Two full rotations at 1 degree per frame
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=False)

# Save as GIF with lower frame rate
gif_path = "GA/loss/walker_evolution_3D.gif"
ani.save(gif_path, writer="pillow", fps=6)  # Lower fps for slower rotation

print("All plotting complete. Check the GA directory for results.")