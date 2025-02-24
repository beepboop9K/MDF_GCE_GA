#!/usr/bin/env python3.8
################################
# Author: M Joyce, N Miller
################################

# Importing required libraries
import matplotlib.pyplot as plt
import warnings
import numpy as np

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

# Adding custom paths
sys.path.append('../')
sys.path.append('/home/njm/bulge-main/iniabu/')

from NuPyCEE import omega, read_yields, stellab, sygma
from JINAPyCEE import omega_plus

# Setting up color maps for plotting
cmap = cm.ocean
norm = colors.Normalize(vmin=1, vmax=375)
m = cm.ScalarMappable(norm=norm, cmap=cmap)

# Load and normalize observational data
obs_file = 'binned_hist.dat'
obs_file = 'binned_dist_lat6_0.08dex.dat'
feh, count = np.loadtxt(obs_file, usecols=(0, 1), unpack=True)
normalized_count = count / count.max()  # Normalize count for comparison

# Parse parameters from the 'bulge_pcard.txt' file
#params = Gal_GA.parse_inlist('bulge_pcard.txt')
params = Gal_GA.parse_inlist('Meridith_bulge_pcard.txt')

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


loss_metric=params['loss_metric']
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
    fancy_mutation = fancy_mutation,
    shrink_range = shrink_range,
    tournament_size = tournament_size,
    threshold = selection_threshold,
    cxpb=crossover_probability, 
    mutpb=mutation_probability, 
    PP = True
)




# Initialize Genetic Algorithm population and toolbox
genal_population, genal_toolbox = GalGA.init_GenAl(population_size=popsize)

# Run the Genetic Algorithm with defined parameters
GalGA.GenAl(population_size=popsize, num_generations=generations, population=genal_population, toolbox=genal_toolbox)

# Select and print the best individual
best_individual = tools.selBest(genal_population, k=1)[0]
print("Best individual is: ", best_individual)

# Save Genetic Algorithm results to a CSV file
results = np.array(GalGA.results)
results_file = 'GA/simulation_results.csv'
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['sigma_2', 'tmax_2', 'infall_2', 'WRMSE', 'MAE', 'MAPE', 'Huber', 'Cos', 'log_cosh'])
    writer.writerows(results)

# Load saved GA results for further analysis and plotting
data = np.genfromtxt(results_file, delimiter=',', skip_header=1)
sigma_2_vals, tmax_2_vals, infall_2_vals, wrmse_vals, mae_vals, mape_vals, huber_vals, cos_vals, log_cosh_vals = data.T

# Extract relevant information from GA results
        
        
loss_vals = {'wrmse': wrmse_vals,'mae': mae_vals,'mape': mape_vals,'huber': huber_vals,'cosine_similarity': cos_vals,'log_cosh': log_cosh_vals,}
loss_val = loss_vals[loss_metric]
scores = np.array(huber_vals)
MDFs = np.array(GalGA.MDFs)
model_numbers = np.array(GalGA.model_numbers)
sorted_scores = np.sort(scores)

# Compare best individual with results
best_individual = np.array(best_individual[:3])
best_model_idx = None
for idx, result in enumerate(GalGA.results):
    result_values = np.array(result[:3])
    if np.allclose(result_values, best_individual, rtol=1e-5):
        best_model_idx = idx
        break

if best_model_idx is not None:
    print(f"Best individual corresponds to model index: {best_model_idx}")
else:
    raise ValueError("Best individual not found in the results.")

# Plotting multiple models and observational data
plt.figure(figsize=(18, 12))
j = 0
num_models = len(sigma_2_vals)
colors = ['#e487ea', '#80d2af', '#a8a68c', '#4e73d6'] * num_models
linestyles = ['-', '--', ':'] * num_models
markers = ['o', '^', 'v', 'D'] * num_models
plot_once = True
with open('GA/MDF_data_results.txt', 'w') as text_file:
    for i in range(len(sorted_scores)):
        this_one = np.where(scores == sorted_scores[i])[0][0]
        best_model = model_numbers[this_one]
        label = GalGA.labels[this_one]
        y_data = GalGA.mdf_data[this_one][2]
        x_data = GalGA.mdf_data[this_one][1]
        text_file.write(f'Model: {best_model}\n')
        for x_i, x in enumerate(x_data):
            text_file.write(f'{x: <8} {y_data[x_i]: <8}\n')

        if this_one == best_model_idx and plot_once:
            plot_once = False
            plt.plot(x_data, y_data, label=f'{label} (Best Individual)', color='red', linestyle='-', marker='s', markersize=12, alpha=0.8, zorder=3)
        else:
            plt.plot(x_data, y_data, label=label, color=colors[j % len(colors)], linestyle=linestyles[j % len(linestyles)], marker=markers[j % len(markers)], markersize=10, alpha=0.5, zorder=1)
        j += 1

# Plot real observational data
plt.plot(feh, normalized_count, label='Real Data [Fe/H]', color='k', marker='x', linestyle='-', markersize=15, zorder=2)

# Set axis labels, limits, and legend
plt.xlabel('[Fe/H]')
plt.ylabel('Number Density (Normalized)')
plt.xlim(-2, 1)
#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=max(1, int(len(sorted_scores) / 40)))
plt.savefig('GA/MDF_multiple_results.png', bbox_inches='tight')
plt.close()





# 3D Scatter Plot function for loss metrics
def plot_3d_scatter(x, y, z, color_metric, label, xlabel='sigma_2', ylabel='tmax_2', zlabel='infall_2'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.savefig(f'GA/loss/{label}_loss_3d.png', bbox_inches='tight')

# Plot 3D scatter plots for various loss metrics
plot_3d_scatter(sigma_2_vals, tmax_2_vals, infall_2_vals, wrmse_vals, 'WRMSE')
plot_3d_scatter(sigma_2_vals, tmax_2_vals, infall_2_vals, mae_vals, 'MAE')
plot_3d_scatter(sigma_2_vals, tmax_2_vals, infall_2_vals, mape_vals, 'MAPE')
plot_3d_scatter(sigma_2_vals, tmax_2_vals, infall_2_vals, huber_vals, 'Huber')
plot_3d_scatter(sigma_2_vals, tmax_2_vals, infall_2_vals, cos_vals, 'Cosine')
plot_3d_scatter(sigma_2_vals, tmax_2_vals, infall_2_vals, log_cosh_vals, 'Log(cosh)')


# 2D Scatter Plot function for loss metrics
def plot_2d_scatter(x, y, color_metric, label, xlabel='tmax_2', ylabel='infall_2'):
    plt.figure()
    sc = plt.scatter(x, y, c=color_metric, cmap='brg')
    plt.colorbar(sc, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{label} Loss')
    plt.savefig(f'GA/loss/{label}_loss_2d.png', bbox_inches='tight')
    plt.close()

# Plot 2D scatter plots for various loss metrics
plot_2d_scatter(tmax_2_vals, infall_2_vals, wrmse_vals, 'WRMSE')
plot_2d_scatter(tmax_2_vals, infall_2_vals, mae_vals, 'MAE')
plot_2d_scatter(tmax_2_vals, infall_2_vals, mape_vals, 'MAPE')
plot_2d_scatter(tmax_2_vals, infall_2_vals, huber_vals, 'Huber')
plot_2d_scatter(tmax_2_vals, infall_2_vals, cos_vals, 'Cosine')
plot_2d_scatter(tmax_2_vals, infall_2_vals, log_cosh_vals, 'Log(cosh)')






def plot_walker_history(walker_history, param_names):
    for param_idx, param_name in enumerate(param_names):
        plt.figure(figsize=(12, 8))

        for walker_idx, history in walker_history.items():
            history = np.array(history)  # Convert to numpy array for easier slicing
            generations = np.arange(len(history))

            # Plot the parameter value for this walker
            plt.plot(
                generations, 
                history[:, param_idx], 
                label=f"Walker {walker_idx}",
                alpha=0.5  # Adjust transparency for better visualization
            )
        print(history[:, param_idx])
        plt.xlabel("Generation")
        plt.ylabel(f"{param_name} Value")
        plt.title(f"Evolution of {param_name} Over Generations")
        plt.legend(loc="upper right", fontsize="small", ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'GA/loss/walker_evolution_{param_name}.png', bbox_inches='tight')
        #plt.show()


# Assuming walker_history tracks ['sigma_2', 'tmax_2', 'infall_2']
param_names = ["sigma_2", "tmax_2", "infall_2"]
plot_walker_history(GalGA.walker_history, param_names)






