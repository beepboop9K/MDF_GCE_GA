#!/usr/bin/env python3.8
################################
#
# Author: M Joyce, N Miller, chatgpt
#
################################
import matplotlib
#import imp
import time
import matplotlib.pyplot as plt
import numpy as np
import re
import itertools
import sys
#from sklearn import preprocessing
sys.path.append('../')
from NuPyCEE import omega

import multiprocessing
import gc
from string import printable
from scipy.interpolate import CubicSpline
from matplotlib import cm
from matplotlib.lines import *
from matplotlib.patches import *
from NuPyCEE import read_yields
from JINAPyCEE import omega_plus
from NuPyCEE import stellab
from NuPyCEE import sygma
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from multiprocessing.pool import ThreadPool
from deap import base, creator, tools, algorithms
import random
import matplotlib as mpl  # Importing the matplotlib module
import csv

# Function to find the index of the nearest value in an array
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

# Function to compute WRMSE
def compute_wrmse(predicted, observed, sigma):
    return np.sqrt(np.mean(((predicted - observed) / sigma) ** 2))

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()


# Function to model inflow rates
def two_inflow_fn(t, exp_inflow):
    if t < exp_inflow[1][1]:
        return exp_inflow[0][0] * np.exp(-t / exp_inflow[0][2])
    else:
        return (exp_inflow[0][0] * np.exp(-t / exp_inflow[0][2]) +
                exp_inflow[1][0] * np.exp(-(t - exp_inflow[1][1]) / exp_inflow[1][2]))


# Function to parse the inlist file into a dictionary
def parse_inlist(file_path):
    params = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed_value = eval(value)  # Try to evaluate if it's a list or expression
            except:
                parsed_value = value  # Keep original string if eval fails
            params[key] = parsed_value
    return params


class GalacticEvolutionGA:
    def __init__(self, sn1a_header, iniab_header, sigma_2_list, tmax_1_list, tmax_2_list, infall_timescale_1_list, infall_timescale_2_list, comp_array, imf_array, sfe_array, imf_upper_limits, 
                 sn1a_assumptions, stellar_yield_assumptions, mgal_values, nb_array, sn1a_rates, timesteps,A1, A2, feh, normalized_count, loss_metric='huber', fancy_mutation = 'gaussian', shrink_range = False, tournament_size = 3, threshold = -1, cxpb=0.5, mutpb=0.5,  PP = False):
        # Initialize parameters as instance variables
        self.sn1a_header = sn1a_header
        self.iniab_header = iniab_header
        self.sigma_2_list = sigma_2_list
        self.tmax_1_list = tmax_1_list
        self.tmax_2_list = tmax_2_list
        self.infall_timescale_1_list = infall_timescale_1_list
        self.infall_timescale_2_list = infall_timescale_2_list        
        self.comp_array = comp_array
        self.imf_array = imf_array
        self.sfe_array = sfe_array
        self.imf_upper_limits = imf_upper_limits
        self.sn1a_assumptions = sn1a_assumptions
        self.stellar_yield_assumptions = stellar_yield_assumptions
        self.mgal_values = mgal_values
        self.nb_array = nb_array
        self.sn1a_rates = sn1a_rates
        self.timesteps = timesteps
        self.A1 = A1
        self.A2 = A2        
        self.feh = feh
        self.normalized_count = normalized_count
        self.placeholder_sigma_array = np.zeros(len(normalized_count)) + 1  # Assume all sigmas are 1
        self.fancy_mutation = fancy_mutation
        self.PP = PP
        self.quant_individuals = False
        self.model_count = 0
        self.mdf_data = []
        self.results = []
        self.labels = []
        self.MDFs = []
        self.model_numbers = []
        self.shrink_range = shrink_range
        # Min and max values for sigma_2, t_2, and infall_2
        self.sigma_2_min, self.sigma_2_max = min(sigma_2_list), max(sigma_2_list)
        self.t_2_min, self.t_2_max = min(tmax_2_list), max(tmax_2_list)
        self.infall_2_min, self.infall_2_max = min(infall_timescale_2_list), max(infall_timescale_2_list)

        self.cxpb=cxpb
        self.mutpb=mutpb
        
        print('############################')
        print(f'Doing {self.fancy_mutation} mutations with {loss_metric} loss and parallel processing is {self.PP}')
        print('############################')
        
        # Define available loss metrics
        self.loss_functions = {
            'wrmse': self.compute_wrmse,
            'mae': self.compute_mae,
            'mape': self.compute_mape,
            'huber': self.compute_huber,
            'cosine_similarity': self.compute_cosine_similarity,
            'log_cosh': self.compute_log_cosh
        }

        # Select the loss function based on user input
        if loss_metric not in self.loss_functions:
            raise ValueError(f"Invalid loss metric. Available options are: {list(self.loss_functions.keys())}")
        
        self.selected_loss_function = self.loss_functions[loss_metric]

        self.all_gene_values_successful = []
        self.all_gene_values_unsuccessful = []
        self.all_losses_successful = []
        self.all_losses_unsuccessful = []
        self.gene_bounds = []
        self.global_min_loss = None
        self.global_max_loss = None
        
        self.threshold = threshold
        self.tournament_size = tournament_size


    def init_GenAl(self, population_size):
        # DEAP framework setup for Genetic Algorithm (GA)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Toolbox to define how individuals (solutions) are created and evolve
        toolbox = base.Toolbox()

        # Register attribute generators with independent selection for each parameter
        toolbox.register("sigma_2_attr", lambda: self.sigma_2_list[random.randint(0, len(self.sigma_2_list) - 1)])
        toolbox.register("t_2_attr", lambda: self.tmax_2_list[random.randint(0, len(self.tmax_2_list) - 1)])
        toolbox.register("infall_2_attr", lambda: self.infall_timescale_2_list[random.randint(0, len(self.infall_timescale_2_list) - 1)])

        # Create an individual by combining the three attributes (sigma_2, t_2, infall_2)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.sigma_2_attr, toolbox.t_2_attr, toolbox.infall_2_attr), n=1)

        # Create a population by repeating individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Register the evaluation function
        toolbox.register("evaluate", self.evaluate)

        # Register genetic operations: crossover (blending) and mutation (custom)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        
        
        
        mutators  ={'gaussian': self.custom_gaussian_mutate, 'uniform': self.custom_uniform_mutate, '': self.custom_mutate}
        mutator = mutators[self.fancy_mutation]
        toolbox.register("mutate", self.custom_mutate)

        # Register selection mechanism (tournament selection)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        # Create the initial population
        population = toolbox.population(n=population_size)
        return population, toolbox

    def print_population(self, population, generation):
        """Helper function to print population details."""
        print(f"\nGeneration {generation+1}:")
        for i, individual in enumerate(population):
            print(f"Individual {i}: {individual}, Fitness: {individual.fitness.values if individual.fitness.valid else 'Not evaluated'}")




    def plot_mutation_info_3D(self, population, fitnesses, base_sigma=1.0, mutation_type='gaussian'):
        #print('Starting plot...')

        # Calculate losses
        losses = [fit[0] for fit in fitnesses]
        max_loss = max(losses)
        min_loss = min(losses)

        # Update global min and max loss
        if self.global_min_loss is None or min_loss < self.global_min_loss:
            self.global_min_loss = min_loss
        if self.global_max_loss is None or max_loss > self.global_max_loss:
            self.global_max_loss = max_loss

        threshold = np.median(losses)

        # Identify successful and unsuccessful individuals
        successful_inds = []
        unsuccessful_inds = []
        for ind, fit in zip(population, fitnesses):
            if fit[0] <= threshold:
                successful_inds.append((ind, fit[0]))
            else:
                unsuccessful_inds.append((ind, fit[0]))

        # Number of genes
        gene_names = ['sigma_2', 't_2', 'infall_2']
        num_genes = len(gene_names)

        # Collect data for accumulation
        # Successful individuals
        gene_values_successful = []
        losses_successful = []
        for ind, loss in successful_inds:
            genes = ind[:num_genes]
            gene_values_successful.append(genes)
            losses_successful.append(loss)
        self.all_gene_values_successful.extend(gene_values_successful)
        self.all_losses_successful.extend(losses_successful)

        # Unsuccessful individuals
        gene_values_unsuccessful = []
        losses_unsuccessful = []
        for ind, loss in unsuccessful_inds:
            genes = ind[:num_genes]
            gene_values_unsuccessful.append(genes)
            losses_unsuccessful.append(loss)
        self.all_gene_values_unsuccessful.extend(gene_values_unsuccessful)
        self.all_losses_unsuccessful.extend(losses_unsuccessful)

        # Store gene bounds
        current_gene_bounds = {
            'xmin': self.sigma_2_min,
            'xmax': self.sigma_2_max,
            'ymin': self.t_2_min,
            'ymax': self.t_2_max,
            'zmin': self.infall_2_min,
            'zmax': self.infall_2_max
        }
        self.gene_bounds.append(current_gene_bounds)

        # At the end of all generations, plot the accumulated data
        if self.gen + 1 == self.num_generations:
            # Prepare the colormap for losses
            all_losses = self.all_losses_successful + self.all_losses_unsuccessful
            min_loss = self.global_min_loss
            max_loss = self.global_max_loss
            loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

            # Normalize losses
            losses_successful_norm = [(loss - min_loss) / loss_range for loss in self.all_losses_successful]
            losses_unsuccessful_norm = [(loss - min_loss) / loss_range for loss in self.all_losses_unsuccessful]

            # Create colormap (darker color for lower loss)
            succmap = cm.get_cmap('YlGn')  # Reverse Greys for darker color at lower values
            unsuccmap = cm.get_cmap('Reds_r')  # Reverse Greys for darker color at lower values
            
            colors_successful = [succmap(loss_norm) for loss_norm in losses_successful_norm]
            colors_unsuccessful = [unsuccmap(loss_norm) for loss_norm in losses_unsuccessful_norm]


            # Prepare the colormap for bounding boxes
            num_generations = self.num_generations
            bbox_cmap = cm.get_cmap('Greys')
            colors_bounding_boxes = [bbox_cmap(i / (num_generations - 1)) for i in range(num_generations)]

            # Create a 3D scatter plot
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')

            # Plot successful individuals
            if len(self.all_gene_values_successful) > 0:
                gene_values_successful = np.array(self.all_gene_values_successful)
                ax.scatter(
                    gene_values_successful[:, 0],
                    gene_values_successful[:, 1],
                    gene_values_successful[:, 2],
                    color=colors_successful,
                    label='Successful',
                    alpha=0.6,
                    marker='o'
                )

            # Plot unsuccessful individuals
            if len(self.all_gene_values_unsuccessful) > 0:
                gene_values_unsuccessful = np.array(self.all_gene_values_unsuccessful)
                ax.scatter(
                    gene_values_unsuccessful[:, 0],
                    gene_values_unsuccessful[:, 1],
                    gene_values_unsuccessful[:, 2],
                    color=colors_unsuccessful,
                    label='Unsuccessful',
                    alpha=0.6,
                    marker='^'
                )

            # Define the edges of the bounding box
            edges = [
                [0, 1], [0, 2], [0, 4],
                [1, 3], [1, 5],
                [2, 3], [2, 6],
                [3, 7],
                [4, 5], [4, 6],
                [5, 7],
                [6, 7]
            ]

            # Plot the bounding boxes
            for i, gene_bound in enumerate(self.gene_bounds):
                color = colors_bounding_boxes[i]
                # Extract bounds
                xmin = gene_bound['xmin']
                xmax = gene_bound['xmax']
                ymin = gene_bound['ymin']
                ymax = gene_bound['ymax']
                zmin = gene_bound['zmin']
                zmax = gene_bound['zmax']

                # Define the corners of the bounding box
                corners = np.array([
                    [xmin, ymin, zmin],
                    [xmin, ymin, zmax],
                    [xmin, ymax, zmin],
                    [xmin, ymax, zmax],
                    [xmax, ymin, zmin],
                    [xmax, ymin, zmax],
                    [xmax, ymax, zmin],
                    [xmax, ymax, zmax]
                ])

                # Plot the edges of the bounding box
                for edge in edges:
                    x = [corners[edge[0], 0], corners[edge[1], 0]]
                    y = [corners[edge[0], 1], corners[edge[1], 1]]
                    z = [corners[edge[0], 2], corners[edge[1], 2]]
                    ax.plot(x, y, z, color=color, linestyle='--', alpha=0.5)

            # Customize plot
            ax.set_title("3D Scatter Plot of Individuals with Gene Bounds")
            ax.set_xlabel(gene_names[0])
            ax.set_ylabel(gene_names[1])
            ax.set_zlabel(gene_names[2])
            ax.legend()

            # Adjust the viewing angle for better visualization
            ax.view_init(elev=20., azim=-35)

            plt.tight_layout()
            plt.savefig('GA/MDF_individuals_3D.png', bbox_inches='tight')
            #plt.show()
            print('...plot made!')

    def plot_mutation_info_2d(self, population, fitnesses, base_sigma=1.0, mutation_type='gaussian'):
        # Calculate losses
        losses = [fit[0] for fit in fitnesses]
        max_loss = max(losses)
        min_loss = min(losses)

        # Update global min and max loss
        if self.global_min_loss is None or min_loss < self.global_min_loss:
            self.global_min_loss = min_loss
        if self.global_max_loss is None or max_loss > self.global_max_loss:
            self.global_max_loss = max_loss

        threshold = np.median(losses)

        # Identify successful and unsuccessful individuals
        successful_inds = []
        unsuccessful_inds = []
        for ind, fit in zip(population, fitnesses):
            if fit[0] <= threshold:
                successful_inds.append((ind, fit[0]))
            else:
                unsuccessful_inds.append((ind, fit[0]))

        # Number of genes (excluding sigma)
        gene_names = ['t_2', 'infall_2']
        num_genes = len(gene_names)

        # Collect data for accumulation
        # Successful individuals
        gene_values_successful = []
        losses_successful = []
        for ind, loss in successful_inds:
            genes = ind[1:num_genes+1]  # Only take `t_2` and `infall_2`
            gene_values_successful.append(genes)
            losses_successful.append(loss)
        self.all_gene_values_successful.extend(gene_values_successful)
        self.all_losses_successful.extend(losses_successful)

        # Unsuccessful individuals
        gene_values_unsuccessful = []
        losses_unsuccessful = []
        for ind, loss in unsuccessful_inds:
            genes = ind[1:num_genes+1]  # Only take `t_2` and `infall_2`
            gene_values_unsuccessful.append(genes)
            losses_unsuccessful.append(loss)
        self.all_gene_values_unsuccessful.extend(gene_values_unsuccessful)
        self.all_losses_unsuccessful.extend(losses_unsuccessful)

        # Store gene bounds
        current_gene_bounds = {
            'xmin': self.t_2_min,
            'xmax': self.t_2_max,
            'ymin': self.infall_2_min,
            'ymax': self.infall_2_max
        }
        self.gene_bounds.append(current_gene_bounds)

        # At the end of all generations, plot the accumulated data
        if self.gen + 1 == self.num_generations:
            # Prepare the colormap for losses
            all_losses = self.all_losses_successful + self.all_losses_unsuccessful
            min_loss = self.global_min_loss
            max_loss = self.global_max_loss
            loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

            # Normalize losses
            losses_successful_norm = [(loss - min_loss) / loss_range for loss in self.all_losses_successful]
            losses_unsuccessful_norm = [(loss - min_loss) / loss_range for loss in self.all_losses_unsuccessful]

            # Create colormaps
            succmap = cm.get_cmap('YlGn')
            unsuccmap = cm.get_cmap('Reds_r')

            colors_successful = [succmap(loss_norm) for loss_norm in losses_successful_norm]
            colors_unsuccessful = [unsuccmap(loss_norm) for loss_norm in losses_unsuccessful_norm]

            # Prepare the colormap for bounding boxes
            num_generations = self.num_generations
            bbox_cmap = cm.get_cmap('Greys')
            colors_bounding_boxes = [bbox_cmap(i / (num_generations - 1)) for i in range(num_generations)]

            # Create a 2D scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot successful individuals
            if len(self.all_gene_values_successful) > 0:
                gene_values_successful = np.array(self.all_gene_values_successful)
                ax.scatter(
                    gene_values_successful[:, 0],  # t_2
                    gene_values_successful[:, 1],  # infall_2
                    color=colors_successful,
                    label='Successful',
                    alpha=0.6,
                    marker='o'
                )

            # Plot unsuccessful individuals
            if len(self.all_gene_values_unsuccessful) > 0:
                gene_values_unsuccessful = np.array(self.all_gene_values_unsuccessful)
                ax.scatter(
                    gene_values_unsuccessful[:, 0],  # t_2
                    gene_values_unsuccessful[:, 1],  # infall_2
                    color=colors_unsuccessful,
                    label='Unsuccessful',
                    alpha=0.6,
                    marker='^'
                )

            # Plot the bounding boxes
            for i, gene_bound in enumerate(self.gene_bounds):
                color = colors_bounding_boxes[i]
                xmin, xmax = gene_bound['xmin'], gene_bound['xmax']
                ymin, ymax = gene_bound['ymin'], gene_bound['ymax']

                # Plot the bounding box as a rectangle
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           edgecolor=color, fill=False, linestyle='--', alpha=0.5))

            # Customize plot
            ax.set_title("2D Scatter Plot of Individuals with Gene Bounds")
            ax.set_xlabel(gene_names[0])
            ax.set_ylabel(gene_names[1])
            ax.legend()
            plt.tight_layout()
            plt.savefig('GA/MDF_individuals_2D.png', bbox_inches='tight')
            print('...2D plot made!')


    def compute_wrmse(self, theory_count_array):
        return compute_wrmse(theory_count_array, self.normalized_count, self.placeholder_sigma_array)

    def compute_mae(self, theory_count_array):
        return np.mean(np.abs(np.array(theory_count_array) - np.array(self.normalized_count)))

    def compute_mape(self, theory_count_array):
        return np.mean(np.abs((np.array(theory_count_array) - np.array(self.normalized_count)) / np.array(self.normalized_count))) * 100

    def compute_huber(self, theory_count_array):
        return np.mean(huber_loss(self.normalized_count, theory_count_array))

    def compute_cosine_similarity(self, theory_count_array):
        return np.dot(self.normalized_count, theory_count_array) / (np.linalg.norm(self.normalized_count) * np.linalg.norm(theory_count_array))

    def compute_log_cosh(self, theory_count_array):
        return np.mean(np.log(np.cosh(theory_count_array - self.normalized_count)))

    def calculate_all_metrics(self, theory_count_array):
        # Calculate all metrics
        wrmse = self.compute_wrmse(theory_count_array)
        mae = self.compute_mae(theory_count_array)
        mape = self.compute_mape(theory_count_array)
        huber = self.compute_huber(theory_count_array)
        cos_similarity = self.compute_cosine_similarity(theory_count_array)
        log_cosh = self.compute_log_cosh(theory_count_array)
        return wrmse, mae, mape, huber, cos_similarity, log_cosh


    def custom_gaussian_mutate(self, individual, loss, population, fitnesses, base_range=1.0, shrink_range=False):
     
        base_sigma = base_range
        def calculate_successful_diversity(population, fitnesses, threshold):
            successful_inds = [ind for ind, fit in zip(population, fitnesses) if fit <= threshold]
            if not successful_inds:
                return [1.0 for _ in range(len(population[0]))]
            genes = list(zip(*successful_inds))
            return [np.std(gene_values) if len(gene_values) > 1 else 0.0 for gene_values in genes]
     
     
        def calculate_sigma(loss, max_loss, min_loss, gene_diversity, max_diversity, min_diversity, base_sigma, shrink_range=False):
            if not shrink_range:
                return base_sigma  # Keep sigma constant if range shrinking is disabled

            # Normalize loss and diversity to [0, 1]
            loss_norm = (loss - min_loss) / (max_loss - min_loss) if max_loss != min_loss else 1.0
            diversity_norm = (gene_diversity - min_diversity) / (max_diversity - min_diversity) if max_diversity != min_diversity else 1.0

            # Invert loss_norm so that lower loss leads to smaller sigma
            loss_factor = 1 - loss_norm

            # Combine factors (you can adjust weights)
            sigma = base_sigma * (loss_factor + diversity_norm) / 2

            # Ensure sigma is within reasonable bounds
            sigma = max(sigma, base_sigma * 0.1)  # Minimum sigma
            sigma = min(sigma, base_sigma * 10)   # Maximum sigma

            return sigma

        # Calculate loss statistics
        losses = [fit[0] for fit in fitnesses]
        max_loss = max(losses)
        min_loss = min(losses)

        # Calculate diversity of successful population
        if self.threshold == -1:
            self.threshold = np.median(losses)
        threshold = self.threshold
        gene_diversity = calculate_successful_diversity(population, losses, threshold)
        max_diversity = max(gene_diversity)
        min_diversity = min(gene_diversity)

        # For each gene in the individual
        for i in range(len(individual)):

            # Calculate sigma for this gene using 'loss'
            sigma = calculate_sigma(loss, max_loss, min_loss, gene_diversity[i], max_diversity, min_diversity, base_sigma, self.shrink_range)

            # Apply Gaussian mutation
            individual[i] += random.gauss(0, sigma)

            # Ensure the mutated gene is within bounds
            if i == 0:  # sigma_2
                min_bound = self.sigma_2_min
                max_bound = self.sigma_2_max
            elif i == 1:  # t_2
                min_bound = self.t_2_min
                max_bound = self.t_2_max
            elif i == 2:  # infall_2
                min_bound = self.infall_2_min
                max_bound = self.infall_2_max
            individual[i] = min(max(individual[i], min_bound), max_bound)
        return individual



    def custom_uniform_mutate(self, individual, loss, population, fitnesses, base_range=1.0, shrink_range=False):


        def calculate_successful_diversity(population, fitnesses, threshold):
            successful_inds = [ind for ind, fit in zip(population, fitnesses) if fit <= threshold]
            if not successful_inds:
                return [1.0 for _ in range(len(population[0]))]
            genes = list(zip(*successful_inds))
            return [np.std(gene_values) if len(gene_values) > 1 else 0.0 for gene_values in genes]


        def calculate_mutation_range(loss, max_loss, min_loss, gene_diversity, max_diversity, min_diversity, base_range, shrink_range=False):
            if not shrink_range:
                return base_range  # Keep mutation range constant if range shrinking is disabled

            # Normalize loss and diversity to [0, 1]
            loss_norm = (loss - min_loss) / (max_loss - min_loss) if max_loss != min_loss else 1.0
            diversity_norm = (gene_diversity - min_diversity) / (max_diversity - min_diversity) if max_diversity != min_diversity else 1.0

            # Invert loss_norm so that lower loss leads to smaller range
            loss_factor = 1 - loss_norm

            # Combine factors (you can adjust weights)
            mutation_range = base_range * (loss_factor + diversity_norm) / 2

            # Ensure mutation_range is within reasonable bounds
            mutation_range = max(mutation_range, base_range * 0.1)  # Minimum range
            mutation_range = min(mutation_range, base_range * 10)   # Maximum range

            return mutation_range

        # Calculate loss statistics
        losses = [fit[0] for fit in fitnesses]
        max_loss = max(losses)
        min_loss = min(losses)

        # Calculate diversity of successful population
        if self.threshold == -1:
            self.threshold = np.median(losses)
        threshold = self.threshold
        gene_diversity = calculate_successful_diversity(population, losses, threshold)
        max_diversity = max(gene_diversity)
        min_diversity = min(gene_diversity)

        # For each gene in the individual
        for i in range(len(individual)):

            # Calculate mutation range for this gene
            mutation_range = calculate_mutation_range(loss, max_loss, min_loss,gene_diversity[i], max_diversity, min_diversity,base_range, self.shrink_range)

            # Apply uniform mutation within the calculated range
            mutation_value = random.uniform(-mutation_range, mutation_range)
            individual[i] += mutation_value

            # Ensure the mutated gene is within bounds
            if i == 0:  # sigma_2
                min_bound = self.sigma_2_min
                max_bound = self.sigma_2_max
            elif i == 1:  # t_2
                min_bound = self.t_2_min
                max_bound = self.t_2_max
            elif i == 2:  # infall_2
                min_bound = self.infall_2_min
                max_bound = self.infall_2_max
            individual[i] = min(max(individual[i], min_bound), max_bound)
        return individual

    
    #def custom_mutate(self, individual, loss, population, fitnesses, base_range=1.0, shrink_range=False):
    def custom_mutate(self, individual, shrink_range=False):
        # Custom mutation function
        for i in range(len(individual)):

            if i == 0:  # sigma_2
                individual[i] = random.uniform(self.sigma_2_min, self.sigma_2_max)
            elif i == 1:  # t_2
                individual[i] = random.uniform(self.t_2_min, self.t_2_max)
            elif i == 2:  # infall_2
                individual[i] = random.uniform(self.infall_2_min, self.infall_2_max)
        return individual,
        
        
    def GenAl(self, population_size, num_generations, population, toolbox):
        total_eval_time = 0
        total_eval_steps = 0
        total_start_time = time.time()

        # Define helper function for re-quantization
        def requantize(ind):
            ind[0] = min(self.sigma_2_list, key=lambda x: abs(x - ind[0]))  # Snap sigma_2 to nearest
            ind[1] = min(self.tmax_2_list, key=lambda x: abs(x - ind[1]))   # Snap tmax_2 to nearest
            ind[2] = min(self.infall_timescale_2_list, key=lambda x: abs(x - ind[2]))  # Snap infall_2 to nearest
            return ind

        # Use a context manager for the multiprocessing pool
        if self.PP:        
            with ThreadPool(processes=16) as pool:
            #with multiprocessing.Pool() as pool:
                toolbox.register("map", pool.map)
                self._run_genetic_algorithm(population, toolbox, num_generations, requantize)
        else:
            self._run_genetic_algorithm(population, toolbox, num_generations, requantize)

        total_time = time.time() - total_start_time

        # Calculate and print the average evaluation time per individual
        if total_eval_steps > 0:
            eff_avg_eval_time = total_time / total_eval_steps
            overall_avg_eval_time = total_eval_time / total_eval_steps
            print(f"Overall average evaluation time per individual: {overall_avg_eval_time:.4f} seconds.")
            print(f"Effective overall average evaluation time per individual: {eff_avg_eval_time:.4f} seconds.")
        else:
            print("No evaluations were performed.")
        
        gc.collect()  # Final garbage collection


    def _run_genetic_algorithm(self, population, toolbox, num_generations, requantize):
        for gen in range(num_generations):
            print(f"-- Generation {gen + 1}/{num_generations} --")
            self.gen = gen
            self.walker_history = {i: [] for i in range(len(population))}  # Track each walker's history
            # Step 1: Evaluate individuals with invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            if invalid_ind:
                if self.PP:
                    fitnesses_and_results = toolbox.map(toolbox.evaluate, invalid_ind)
                else:
                    fitnesses_and_results = [toolbox.evaluate(ind) for ind in invalid_ind]

                for (ind, (fit, result)) in zip(invalid_ind, fitnesses_and_results):
                    ind.fitness.values = fit
                    self.labels.append(result['label'])
                    self.mdf_data.append([result['resampled_x'], result['x_data'], result['y_data']])
                    self.results.append(result['metrics'])
                    self.MDFs.append(result['cs_MDF'])
                    self.model_numbers.append(result['model_number'])
                    self.model_count += 1


            gc.collect()

            # Step 2: Select the next generation
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            # Step 3: Apply mutation and crossover
            for mutant in offspring:
                if random.random() < self.mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            if self.quant_individuals:
                offspring = [requantize(ind) for ind in offspring]

            if round(gen % (num_generations / 4)) == 0:
                self.print_population(population, generation=gen)


            # Step 4: Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                if self.PP:
                    fitnesses_and_results = toolbox.map(toolbox.evaluate, invalid_ind)
                else:
                    fitnesses_and_results = [toolbox.evaluate(ind) for ind in invalid_ind]

                for (ind, (fit, result)) in zip(invalid_ind, fitnesses_and_results):
                    ind.fitness.values = fit
                    self.labels.append(result['label'])
                    self.mdf_data.append([result['resampled_x'], result['x_data'], result['y_data']])
                    self.results.append(result['metrics'])
                    self.MDFs.append(result['cs_MDF'])
                    self.model_numbers.append(result['model_number'])
                    self.model_count += 1

            gc.collect()

            # After evaluating the population in each generation
            for idx, ind in enumerate(population):
                self.walker_history[idx].append(list(ind))
            # Replace the old population with the offspring
            population[:] = offspring
            #print(f"  Offspring evaluation time for generation {gen+1}: {step_eval_time:.2f} seconds for {num_offspring_evals} evaluations. There were {baby_counter} cross overs & {mutant_counter} mutations")


    def evaluate(self, individual):
        sigma_2, t_2, infall_2 = individual

        # Fixed parameters from the pcard
        t_1 = self.tmax_2_list[0]
        infall_1 = self.infall_timescale_1_list[0]
        comp = self.comp_array[0]
        imf_val = self.imf_array[0]
        sfe_val = self.sfe_array[0]
        imf_upper = self.imf_upper_limits[0]
        sn1a = self.sn1a_assumptions[0]
        sy = self.stellar_yield_assumptions[0]
        mgal = self.mgal_values[0]
        nb = self.nb_array[0]
        sn1ar = self.sn1a_rates[0]
        A1 = self.A1
        A2 = self.A2
        sn1a_header = self.sn1a_header
        iniab_header = self.iniab_header

        # GCE Model kwargs
        kwargs = {
            'special_timesteps': self.timesteps,
            'twoinfall_sigmas': [1300, sigma_2],
            'galradius': 1800,
            'exp_infall':[[A1, t_1*1e9, infall_1*1e9], [A2, t_2*1e9, infall_2*1e9]],            
            'tauup': [0.02e9, 0.02e9],
            'mgal': mgal,
            'iniZ': 0.0,
            'mass_loading': 0.0,
            'table': sn1a_header + sy,
            'sfe': sfe_val,
            'imf_type': imf_val,
            'sn1a_table': sn1a_header + sn1a,
            'imf_yields_range': [1, imf_upper],
            'iniabu_table': iniab_header + comp,
            'nb_1a_per_m': nb,
            'sn1a_rate': sn1ar
        }
        
        
        # Run GCE model and compute MDF
        GCE_model = omega_plus.omega_plus(**kwargs)
        x_data, y_data = GCE_model.inner.plot_mdf(axis_mdf='[Fe/H]', sigma_gauss=0.1, norm=True, return_x_y=True)
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        sample_at = 0.0001
        resampled_x = np.arange(x_data.min(), x_data.max(), sample_at)
        cs_MDF = CubicSpline(x_data, y_data)
        theory_count_array = [cs_MDF(resampled_x)[find_nearest(resampled_x, self.feh[i])[0]] for i in range(len(self.feh))]
        
        # Calculate errors and similarities
        wrmse, mae, mape, huber, cos_similarity, log_cosh = self.calculate_all_metrics(theory_count_array)
        
        # Use the selected loss function for decision-making
        primary_loss_value = self.selected_loss_function(theory_count_array)

        # Return the necessary data instead of appending to shared lists
        label = f'sigma2: {sigma_2:.3f}, t2: {t_2:.3f}, infall2: {infall_2:.3f}'
        result = {
            'label': label,
            'resampled_x': resampled_x,
            'x_data': x_data,
            'y_data': y_data,
            'metrics': [sigma_2, t_2, infall_2, wrmse, mae, mape, huber, cos_similarity, log_cosh],
            'cs_MDF': cs_MDF,
            'model_number': self.model_count
        }
        
        # Increment model count (assuming this happens in the main thread after evaluation)
        return (primary_loss_value,), result

