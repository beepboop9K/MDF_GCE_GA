#!/usr/bin/env python3.8
################################
# Author: N Miller, M Joyce
################################

# Importing required libraries
import matplotlib.pyplot as plt
import warnings
import numpy as np
import sys
import argparse
from scipy.interpolate import CubicSpline
from deap import base, creator, tools
import random
import Gal_GA_PP as Gal_GA
import pandas as pd
import os
# Import plotting module
import mdf_plotting

# Suppress specific RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Adding custom paths
sys.path.append('../')

# Create argument parser
parser = argparse.ArgumentParser(description='Run MDF Genetic Algorithm with optional plotting only')
parser.add_argument('--plot-only', action='store_true', help='Skip computation and only generate plots')
parser.add_argument('--results-file', type=str, default='GA/simulation_results.csv', 
                   help='CSV file containing results (for plot-only mode)')
args = parser.parse_args()

# Load and normalize observational data
obs_file = 'binned_dist_lat6_0.08dex.dat'
feh, count = np.loadtxt(obs_file, usecols=(0, 1), unpack=True)
normalized_count = count / count.max()  # Normalize count for comparison

# Parse parameters from the 'bulge_pcard.txt' file
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

# Global GalGA object to be used for both computation and plotting
GalGA = None


# Save/load walker history
def save_walker_history():
    if not hasattr(GalGA, 'walker_history'):
        return
        
    np.savez_compressed(
        'GA/walker_history.npz',
        walker_ids=np.array(list(GalGA.walker_history.keys()), dtype=np.int32),
        histories=[np.array(h) for h in GalGA.walker_history.values()]
    )
    print("Walker history saved")

def load_walker_history():
    if not os.path.exists('GA/walker_history.npz'):
        return {}
        
    data = np.load('GA/walker_history.npz', allow_pickle=True)
    walker_ids = data['walker_ids']
    histories = data['histories']
    
    walker_history = {}
    for i, walker_id in enumerate(walker_ids):
        walker_history[int(walker_id)] = histories[i]
    
    print("Walker history loaded")
    return walker_history




def run_ga():
    """Run the genetic algorithm"""
    global GalGA
    
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
    GalGA.GenAl(population_size=popsize, num_generations=generations, 
                population=genal_population, toolbox=genal_toolbox)

    # Define column names based on the structure of GalGA.results
    col_names = [
        'comp_idx', 'imf_idx', 'sn1a_idx', 'sy_idx', 'sn1ar_idx',
        'sigma_2', 't_1', 't_2', 'infall_1', 'infall_2', 
        'sfe', 'imf_upper', 'mgal', 'nb',
        'ks', 'ensemble', 'wrmse', 'mae', 'mape', 'huber', 'cosine', 'log_cosh'
    ]

    # Create DataFrame from results
    results_df = pd.DataFrame(GalGA.results, columns=col_names)

    # Use the chosen loss metric to define a loss column
    results_df['loss'] = results_df[loss_metric]

    # Sort the results DataFrame by loss (lowest first) and reset index
    results_df.sort_values('loss', inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Save the results to a CSV file
    results_file = 'GA/simulation_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}")

    # The best model is now the first row in the sorted DataFrame
    best_model = results_df.iloc[0]
    print("Best model from results dataframe:")
    print(best_model)
    
    return results_file

def load_ga_for_plotting():
    """Load GA object for plotting only"""
    global GalGA
    
    # For plot-only mode, we create a minimal GalGA object that has the properties 
    # needed for plotting, but doesn't run any computations
    
    print(f"Loading existing results from {args.results_file}")
    
    # Make sure results file exists
    import os
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} not found")
        sys.exit(1)
    
    # Initialize a basic GalGA object
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
        PP=False  # Don't use parallel processing for plot-only
    )
    
    # Load results from CSV
    try:
        df = pd.read_csv(args.results_file)
        
        # Extract results from the dataframe
        GalGA.results = df.values.tolist()
        
        # We need to generate some placeholder data for plotting functions
        # that require mdf_data and labels
        x_vals = np.linspace(-2, 1, 100)
        y_vals = np.zeros_like(x_vals)
        GalGA.mdf_data = [(x_vals, y_vals)]
        GalGA.labels = ["Placeholder"]
        
        # Create an empty walker_history
        GalGA.walker_history = {}
        
        # Check if log files or other data sources might have the actual MDFs
        # and walker history data, but this is beyond the scope of this example
        
        print(f"Loaded {len(df)} model results")
    
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)
    
    return args.results_file

if __name__ == "__main__":

    make_history = True
    # Load walker history if requested
    if make_history == True:
        results_file = run_ga()
        save_walker_history()
    else:
        GalGA.walker_history = load_walker_history()
    
    # Generate all plots using the plotting module
    mdf_plotting.generate_all_plots(GalGA, feh, normalized_count, results_file)