# parameter_optimization.py: Implements metaheuristic optimization for kernel parameters

import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time
from kfcm_clustering import KernelFuzzyCMeansClustering
from mkfcm_clustering import ModifiedKernelFuzzyCMeansClustering

class KernelParameterOptimizer:
    """
    Implements a Genetic Algorithm to find optimal kernel parameters 
    for KFCM and MKFCM algorithms
    """
    
    def __init__(self, algorithm='kfcm', n_clusters=4, pop_size=20, n_generations=50, 
                crossover_rate=0.8, mutation_rate=0.2, elitism=2, random_state=None,
                param_ranges=None):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        algorithm : str, default='kfcm'
            Algorithm to optimize ('kfcm' or 'mkfcm')
        n_clusters : int, default=4
            Number of clusters for clustering algorithm
        pop_size : int, default=20
            Population size for GA
        n_generations : int, default=50
            Number of generations to run
        crossover_rate : float, default=0.8
            Rate of crossover
        mutation_rate : float, default=0.2
            Rate of mutation
        elitism : int, default=2
            Number of best individuals to keep unchanged
        random_state : int, default=None
            Random state for reproducibility
        param_ranges : dict, default=None
            Ranges for parameters in format {param_name: (min_val, max_val)}
            If None, default ranges are used
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        
        # Set default parameter ranges if not provided
        if param_ranges is None:
            self.param_ranges = {
                'sigma_squared': (0.1, 100.0),  # Range for kernel width
                'm': (1.1, 3.0)                 # Range for fuzzifier
            }
        else:
            self.param_ranges = param_ranges
            
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
            
        # Fitness history for each generation
        self.fitness_history = []
        self.best_fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.best_labels = None
        self.best_centroids = None
        
    def initialize_population(self):
        """
        Initialize population using a combination of seeded values from research
        and random individuals within the specified ranges.
        """
        population = []

        # --- Define candidate seeds based on research (e.g., Graves & Pedrycz) ---
        # Adjust these based on your review of the paper's tables for Gaussian Kernels
        seeds = [
            {'sigma_squared': 10.0, 'm': 2.0},    # Default-ish baseline
            {'sigma_squared': 0.5, 'm': 1.4},     # Example from paper (MKFCM Ring c=2)
            {'sigma_squared': 2.0, 'm': 1.4},     # Example from paper (KFCM XOR c=2)
            {'sigma_squared': 8.0, 'm': 2.5},     # Example from paper (MKFCM XOR c=2)
            {'sigma_squared': 1.0, 'm': 1.2},     # Example from paper (KFCM Ring c=2)
            # Add maybe one more distinct, successful combination if found
        ]

        print(f"Initializing population. Will attempt to add {len(seeds)} seeds.")

        # Add seeded individuals first (ensure they are within defined bounds)
        added_seed_count = 0
        for seed in seeds:
            if len(population) >= self.pop_size:
                break # Stop if population is already full

            valid_seed = {}
            in_bounds = True
            required_params = set(self.param_ranges.keys())
            seed_params = set(seed.keys())

            # Check if seed has all required parameters and if they are in bounds
            if not required_params.issubset(seed_params):
                 print(f"  Skipping seed {seed}: Missing required parameters.")
                 continue # Seed doesn't have all parameters we are optimizing

            for param, val in seed.items():
                if param in self.param_ranges:
                    min_val, max_val = self.param_ranges[param]
                    if min_val <= val <= max_val:
                        valid_seed[param] = val
                    else:
                        in_bounds = False
                        print(f"  Skipping seed {seed}: Parameter '{param}' value {val} out of bounds [{min_val}, {max_val}].")
                        break
                # else: # Parameter in seed but not being optimized - ignore it
                #    pass

            if in_bounds:
                # Check if we already added an identical seed (optional, prevents duplicates)
                is_duplicate = any(valid_seed == p for p in population)
                if not is_duplicate:
                     population.append(valid_seed)
                     added_seed_count += 1
                # else:
                #    print(f"  Skipping seed {seed}: Duplicate already added.")


        print(f"Successfully added {added_seed_count} unique seeds within bounds.")

        # Fill the rest of the population with random individuals (uniform random)
        print(f"Filling remaining {self.pop_size - len(population)} slots randomly.")
        while len(population) < self.pop_size:
            individual = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                # Generate random value within range
                individual[param] = min_val + np.random.random() * (max_val - min_val)
            population.append(individual)

        # Optional: Shuffle the initial population if desired
        # np.random.shuffle(population)

        print(f"Initialization complete. Population size: {len(population)}.")
        return population
    
    def fitness_function(self, individual, data):
        """
        Evaluate fitness of an individual
        
        Parameters:
        -----------
        individual : dict
            Dictionary of parameter values
        data : array-like
            Data to cluster
            
        Returns:
        --------
        fitness : float
            Fitness value (silhouette score)
        labels : array
            Cluster labels
        centroids : array
            Cluster centroids
        """
        # Extract parameters
        sigma_squared = individual['sigma_squared']
        m = individual['m']
        
        # Initialize clustering algorithm with these parameters
        if self.algorithm == 'kfcm':
            clustering = KernelFuzzyCMeansClustering(
                n_clusters=self.n_clusters,
                m=m,
                sigma_squared=sigma_squared
            )
        else:  # 'mkfcm'
            clustering = ModifiedKernelFuzzyCMeansClustering(
                n_clusters=self.n_clusters,
                m=m,
                sigma_squared=sigma_squared
            )
            
        # Run clustering
        try:
            labels = clustering.fit(data)
            
            # Calculate silhouette score (higher is better)
            if len(np.unique(labels)) < 2:
                # If all points are assigned to the same cluster, return worst score
                fitness = -1
            else:
                fitness = silhouette_score(data, labels)
                
            return fitness, labels, clustering.get_centroids()
        except Exception as e:
            # If clustering fails, return worst score
            print(f"Clustering failed with parameters {individual}: {e}")
            return -1, None, None
    
    def select_parents(self, population, fitnesses):
        """
        Select parents for reproduction using tournament selection
        
        Parameters:
        -----------
        population : list
            List of individuals
        fitnesses : list
            List of fitness values
            
        Returns:
        --------
        parents : list
            List of selected parents
        """
        tournament_size = 3
        num_parents = len(population)
        parents = []
        
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # Select the best individual from tournament
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            parents.append(population[winner_idx])
            
        return parents
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Parameters:
        -----------
        parent1, parent2 : dict
            Parent individuals
            
        Returns:
        --------
        child1, child2 : dict
            Child individuals
        """
        if np.random.random() < self.crossover_rate:
            # Create children by blending parameters
            child1, child2 = {}, {}
            
            for param in parent1.keys():
                # Blend ratio for this parameter
                alpha = np.random.random()
                
                # Create children using blend crossover
                child1[param] = alpha * parent1[param] + (1 - alpha) * parent2[param]
                child2[param] = alpha * parent2[param] + (1 - alpha) * parent1[param]
                
                # Ensure values are within range
                min_val, max_val = self.param_ranges[param]
                child1[param] = np.clip(child1[param], min_val, max_val)
                child2[param] = np.clip(child2[param], min_val, max_val)
                
            return child1, child2
        else:
            # No crossover
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """
        Perform mutation on an individual
        
        Parameters:
        -----------
        individual : dict
            Individual to mutate
            
        Returns:
        --------
        mutated : dict
            Mutated individual
        """
        mutated = individual.copy()
        
        for param, (min_val, max_val) in self.param_ranges.items():
            # Mutate each parameter with probability mutation_rate
            if np.random.random() < self.mutation_rate:
                # Add Gaussian noise
                sigma = (max_val - min_val) * 0.1  # 10% of range as standard deviation
                noise = np.random.normal(0, sigma)
                
                # Apply mutation
                mutated[param] += noise
                
                # Ensure values are within range
                mutated[param] = np.clip(mutated[param], min_val, max_val)
                
        return mutated
    
    def optimize(self, data):
        """
        Run the genetic algorithm to find optimal parameters
        
        Parameters:
        -----------
        data : array-like
            Data to cluster
            
        Returns:
        --------
        best_params : dict
            Best parameters found
        best_fitness : float
            Best fitness value
        best_labels : array
            Best cluster labels
        """
        print(f"Starting optimization for {self.algorithm.upper()} with population size {self.pop_size} for {self.n_generations} generations")
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population()
        
        # Main GA loop
        for generation in range(self.n_generations):
            gen_start_time = time.time()
            
            # Evaluate fitness for all individuals
            fitnesses = []
            labels_list = []
            centroids_list = []
            
            for individual in population:
                fitness, labels, centroids = self.fitness_function(individual, data)
                fitnesses.append(fitness)
                labels_list.append(labels)
                centroids_list.append(centroids)
                
                # Update best individual
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
                    self.best_labels = labels
                    self.best_centroids = centroids
            
            # Store fitness history
            self.fitness_history.append(np.mean(fitnesses))
            self.best_fitness_history.append(np.max(fitnesses))
            
            # Print progress
            gen_time = time.time() - gen_start_time
            print(f"Generation {generation+1}/{self.n_generations} - Avg Fitness: {np.mean(fitnesses):.4f}, "
                  f"Best Fitness: {np.max(fitnesses):.4f}, Time: {gen_time:.2f}s")
            
            # Apply elitism - keep best individuals
            elite_indices = np.argsort(fitnesses)[-self.elitism:]
            elite = [population[i].copy() for i in elite_indices]
            
            # Select parents
            parents = self.select_parents(population, fitnesses)
            
            # Create next generation through crossover and mutation
            next_gen = []
            
            # Add elite individuals
            next_gen.extend(elite)
            
            # Fill the rest of population with offspring
            while len(next_gen) < self.pop_size:
                # Select two parents
                idx1, idx2 = np.random.choice(len(parents), 2, replace=False)
                parent1, parent2 = parents[idx1], parents[idx2]
                
                # Create offspring
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate offspring
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Add offspring to next generation
                next_gen.append(child1)
                if len(next_gen) < self.pop_size:
                    next_gen.append(child2)
            
            # Update population
            population = next_gen
        
        total_time = time.time() - start_time
        print(f"\nOptimization complete in {total_time:.2f}s")
        print(f"Best parameters: {self.best_individual}")
        print(f"Best fitness (Silhouette Score): {self.best_fitness:.4f}")
        
        return self.best_individual, self.best_fitness, self.best_labels
    
    def plot_convergence(self, filename="ga_convergence.png"):
        """
        Plot the convergence history of the genetic algorithm
        
        Parameters:
        -----------
        filename : str, default="ga_convergence.png"
            Filename to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        generations = range(1, len(self.fitness_history) + 1)
        
        plt.plot(generations, self.fitness_history, 'b-', label='Average Fitness')
        plt.plot(generations, self.best_fitness_history, 'r-', label='Best Fitness')
        
        plt.title(f'Genetic Algorithm Convergence for {self.algorithm.upper()}')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Silhouette Score)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def get_best_parameters(self):
        """Return the best parameters found"""
        return self.best_individual
    
    def get_best_fitness(self):
        """Return the best fitness found"""
        return self.best_fitness
    
    def get_best_labels(self):
        """Return the best cluster labels found"""
        return self.best_labels
    
    def get_best_centroids(self):
        """Return the best centroids found"""
        return self.best_centroids
    
    def get_convergence_history(self):
        """Return the convergence history"""
        return {
            'average_fitness': self.fitness_history,
            'best_fitness': self.best_fitness_history
        }


class DifferentialEvolutionOptimizer:
    """
    Implements Differential Evolution to find optimal kernel parameters 
    for KFCM and MKFCM algorithms
    """
    
    def __init__(self, algorithm='kfcm', n_clusters=4, pop_size=20, n_generations=50, 
                F=0.8, CR=0.7, random_state=None, param_ranges=None):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        algorithm : str, default='kfcm'
            Algorithm to optimize ('kfcm' or 'mkfcm')
        n_clusters : int, default=4
            Number of clusters for clustering algorithm
        pop_size : int, default=20
            Population size for DE
        n_generations : int, default=50
            Number of generations to run
        F : float, default=0.8
            Differential weight (mutation factor)
        CR : float, default=0.7
            Crossover probability
        random_state : int, default=None
            Random state for reproducibility
        param_ranges : dict, default=None
            Ranges for parameters in format {param_name: (min_val, max_val)}
            If None, default ranges are used
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        
        # Set default parameter ranges if not provided
        if param_ranges is None:
            self.param_ranges = {
                'sigma_squared': (0.1, 100.0),  # Range for kernel width
                'm': (1.1, 3.0)                 # Range for fuzzifier
            }
        else:
            self.param_ranges = param_ranges
            
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
            
        # Fitness history for each generation
        self.fitness_history = []
        self.best_fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        self.best_labels = None
        self.best_centroids = None
        
    def initialize_population(self):
        """
        Initialize population using same approach as GA, with seeds from research
        and random individuals within the specified ranges.
        """
        population = []

        # --- Define candidate seeds based on research (e.g., Graves & Pedrycz) ---
        seeds = [
            {'sigma_squared': 10.0, 'm': 2.0},    # Default-ish baseline
            {'sigma_squared': 0.5, 'm': 1.4},     # Example from paper (MKFCM Ring c=2)
            {'sigma_squared': 2.0, 'm': 1.4},     # Example from paper (KFCM XOR c=2)
            {'sigma_squared': 8.0, 'm': 2.5},     # Example from paper (MKFCM XOR c=2)
            {'sigma_squared': 1.0, 'm': 1.2},     # Example from paper (KFCM Ring c=2)
        ]

        print(f"Initializing population. Will attempt to add {len(seeds)} seeds.")

        # Add seeded individuals first (ensure they are within defined bounds)
        added_seed_count = 0
        for seed in seeds:
            if len(population) >= self.pop_size:
                break # Stop if population is already full

            valid_seed = {}
            in_bounds = True
            required_params = set(self.param_ranges.keys())
            seed_params = set(seed.keys())

            # Check if seed has all required parameters and if they are in bounds
            if not required_params.issubset(seed_params):
                 print(f"  Skipping seed {seed}: Missing required parameters.")
                 continue # Seed doesn't have all parameters we are optimizing

            for param, val in seed.items():
                if param in self.param_ranges:
                    min_val, max_val = self.param_ranges[param]
                    if min_val <= val <= max_val:
                        valid_seed[param] = val
                    else:
                        in_bounds = False
                        print(f"  Skipping seed {seed}: Parameter '{param}' value {val} out of bounds [{min_val}, {max_val}].")
                        break

            if in_bounds:
                # Check if we already added an identical seed (optional, prevents duplicates)
                is_duplicate = any(valid_seed == p for p in population)
                if not is_duplicate:
                     population.append(valid_seed)
                     added_seed_count += 1

        print(f"Successfully added {added_seed_count} unique seeds within bounds.")

        # Fill the rest of the population with random individuals (uniform random)
        print(f"Filling remaining {self.pop_size - len(population)} slots randomly.")
        while len(population) < self.pop_size:
            individual = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                # Generate random value within range
                individual[param] = min_val + np.random.random() * (max_val - min_val)
            population.append(individual)

        print(f"Initialization complete. Population size: {len(population)}.")
        return population
    
    def fitness_function(self, individual, data):
        """
        Evaluate fitness of an individual
        
        Parameters:
        -----------
        individual : dict
            Dictionary of parameter values
        data : array-like
            Data to cluster
            
        Returns:
        --------
        fitness : float
            Fitness value (silhouette score)
        labels : array
            Cluster labels
        centroids : array
            Cluster centroids
        """
        # Extract parameters
        sigma_squared = individual['sigma_squared']
        m = individual['m']
        
        # Initialize clustering algorithm with these parameters
        if self.algorithm == 'kfcm':
            clustering = KernelFuzzyCMeansClustering(
                n_clusters=self.n_clusters,
                m=m,
                sigma_squared=sigma_squared
            )
        else:  # 'mkfcm'
            clustering = ModifiedKernelFuzzyCMeansClustering(
                n_clusters=self.n_clusters,
                m=m,
                sigma_squared=sigma_squared
            )
            
        # Run clustering
        try:
            labels = clustering.fit(data)
            
            # Calculate silhouette score (higher is better)
            if len(np.unique(labels)) < 2:
                # If all points are assigned to the same cluster, return worst score
                fitness = -1
            else:
                fitness = silhouette_score(data, labels)
                
            return fitness, labels, clustering.get_centroids()
        except Exception as e:
            # If clustering fails, return worst score
            print(f"Clustering failed with parameters {individual}: {e}")
            return -1, None, None
    
    def optimize(self, data):
        """
        Run the differential evolution algorithm to find optimal parameters
        
        Parameters:
        -----------
        data : array-like
            Data to cluster
            
        Returns:
        --------
        best_params : dict
            Best parameters found
        best_fitness : float
            Best fitness value
        best_labels : array
            Best cluster labels
        """
        print(f"Starting DE optimization for {self.algorithm.upper()} with population size {self.pop_size} for {self.n_generations} generations")
        start_time = time.time()
        
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial population
        fitnesses = []
        labels_list = []
        centroids_list = []
        
        for individual in population:
            fitness, labels, centroids = self.fitness_function(individual, data)
            fitnesses.append(fitness)
            labels_list.append(labels)
            centroids_list.append(centroids)
            
            # Update best individual
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual.copy()
                self.best_labels = labels
                self.best_centroids = centroids
        
        # Store initial fitness
        self.fitness_history.append(np.mean(fitnesses))
        self.best_fitness_history.append(np.max(fitnesses))
        
        # Main DE loop
        for generation in range(self.n_generations):
            gen_start_time = time.time()
            
            # For each individual in the population
            for i in range(self.pop_size):
                # Select three random distinct individuals different from i
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Create trial vector through mutation and crossover
                trial = {}
                
                for param in population[i].keys():
                    # Randomly decide if we should use mutated value or original value
                    if np.random.random() < self.CR:
                        # DE/rand/1 mutation
                        # F controls the amplification of the differential variation
                        min_val, max_val = self.param_ranges[param]
                        mutated_value = population[a][param] + self.F * (population[b][param] - population[c][param])
                        # Ensure value is within bounds
                        trial[param] = np.clip(mutated_value, min_val, max_val)
                    else:
                        # Keep original value
                        trial[param] = population[i][param]
                
                # Evaluate trial vector
                trial_fitness, trial_labels, trial_centroids = self.fitness_function(trial, data)
                
                # Selection: if trial is better, replace current individual
                if trial_fitness > fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fitness
                    labels_list[i] = trial_labels
                    centroids_list[i] = trial_centroids
                    
                    # Update best individual
                    if trial_fitness > self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_individual = trial.copy()
                        self.best_labels = trial_labels
                        self.best_centroids = trial_centroids
            
            # Store fitness history
            self.fitness_history.append(np.mean(fitnesses))
            self.best_fitness_history.append(np.max(fitnesses))
            
            # Print progress
            gen_time = time.time() - gen_start_time
            print(f"Generation {generation+1}/{self.n_generations} - Avg Fitness: {np.mean(fitnesses):.4f}, "
                  f"Best Fitness: {np.max(fitnesses):.4f}, Time: {gen_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"\nOptimization complete in {total_time:.2f}s")
        print(f"Best parameters: {self.best_individual}")
        print(f"Best fitness (Silhouette Score): {self.best_fitness:.4f}")
        
        return self.best_individual, self.best_fitness, self.best_labels
    
    def plot_convergence(self, filename="de_convergence.png"):
        """
        Plot the convergence history of the differential evolution algorithm
        
        Parameters:
        -----------
        filename : str, default="de_convergence.png"
            Filename to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        generations = range(1, len(self.fitness_history) + 1)
        
        plt.plot(generations, self.fitness_history, 'b-', label='Average Fitness')
        plt.plot(generations, self.best_fitness_history, 'r-', label='Best Fitness')
        
        plt.title(f'Differential Evolution Convergence for {self.algorithm.upper()}')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Silhouette Score)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    def get_best_parameters(self):
        """Return the best parameters found"""
        return self.best_individual
    
    def get_best_fitness(self):
        """Return the best fitness found"""
        return self.best_fitness
    
    def get_best_labels(self):
        """Return the best cluster labels found"""
        return self.best_labels
    
    def get_best_centroids(self):
        """Return the best centroids found"""
        return self.best_centroids
    
    def get_convergence_history(self):
        """Return the convergence history"""
        return {
            'average_fitness': self.fitness_history,
            'best_fitness': self.best_fitness_history
        }


def run_parameter_optimization(data, algorithm='kfcm', n_clusters=4, pop_size=20, 
                              n_generations=30, random_state=None, method='ga'):
    """
    Run parameter optimization and return results
    
    Parameters:
    -----------
    data : array-like
        Data to cluster
    algorithm : str, default='kfcm'
        Algorithm to optimize ('kfcm' or 'mkfcm')
    n_clusters : int, default=4
        Number of clusters
    pop_size : int, default=20
        Population size for optimization algorithm
    n_generations : int, default=30
        Number of generations to run
    random_state : int, default=None
        Random state for reproducibility
    method : str, default='ga'
        Optimization method ('ga' for Genetic Algorithm or 'de' for Differential Evolution)
        
    Returns:
    --------
    results : dict
        Dictionary with optimization results
    """
    # Initialize optimizer based on method
    if method.lower() == 'ga':
        optimizer = KernelParameterOptimizer(
            algorithm=algorithm,
            n_clusters=n_clusters,
            pop_size=pop_size,
            n_generations=n_generations,
            random_state=random_state
        )
        method_name = "GA"
    elif method.lower() == 'de':
        optimizer = DifferentialEvolutionOptimizer(
            algorithm=algorithm,
            n_clusters=n_clusters,
            pop_size=pop_size,
            n_generations=n_generations,
            random_state=random_state
        )
        method_name = "DE"
    else:
        raise ValueError(f"Unknown optimization method: {method}. Use 'ga' or 'de'.")
    
    # Run optimization
    best_params, best_fitness, best_labels = optimizer.optimize(data)
    
    # Plot convergence
    optimizer.plot_convergence(f"{algorithm}_{method.lower()}_convergence.png")
    
    # Create results dictionary
    results = {
        'algorithm': algorithm,
        'optimization_method': method,
        'best_params': best_params,
        'best_fitness': best_fitness,
        'best_labels': best_labels,
        'best_centroids': optimizer.get_best_centroids(),
        'convergence_history': optimizer.get_convergence_history()
    }
    
    return results


def compare_optimization_methods(data, algorithm='kfcm', n_clusters=4, pop_size=20, 
                               n_generations=30, random_state=None):
    """
    Compare GA and DE optimization methods and return results
    
    Parameters:
    -----------
    data : array-like
        Data to cluster
    algorithm : str, default='kfcm'
        Algorithm to optimize ('kfcm' or 'mkfcm')
    n_clusters : int, default=4
        Number of clusters
    pop_size : int, default=20
        Population size for optimization algorithms
    n_generations : int, default=30
        Number of generations to run
    random_state : int, default=None
        Random state for reproducibility
        
    Returns:
    --------
    comparison_results : dict
        Dictionary with comparison results
    """
    print(f"\nComparing GA and DE optimization methods for {algorithm.upper()}...")
    start_time = time.time()
    
    # Run GA optimization
    print("\n--- Genetic Algorithm Optimization ---")
    ga_results = run_parameter_optimization(
        data, algorithm, n_clusters, pop_size, n_generations, random_state, method='ga'
    )
    
    # Run DE optimization
    print("\n--- Differential Evolution Optimization ---")
    de_results = run_parameter_optimization(
        data, algorithm, n_clusters, pop_size, n_generations, random_state, method='de'
    )
    
    # Compute time taken
    total_time = time.time() - start_time
    
    # Compare results
    print("\n--- Optimization Methods Comparison ---")
    print(f"GA Best Parameters: {ga_results['best_params']}")
    print(f"GA Best Fitness: {ga_results['best_fitness']:.4f}")
    print(f"DE Best Parameters: {de_results['best_params']}")
    print(f"DE Best Fitness: {de_results['best_fitness']:.4f}")
    
    # Determine winner
    if ga_results['best_fitness'] > de_results['best_fitness']:
        winner = "Genetic Algorithm"
        improvement = ((ga_results['best_fitness'] - de_results['best_fitness']) / 
                      de_results['best_fitness'] * 100)
    else:
        winner = "Differential Evolution"
        improvement = ((de_results['best_fitness'] - ga_results['best_fitness']) / 
                      ga_results['best_fitness'] * 100)
    
    print(f"Winner: {winner} with {improvement:.2f}% improvement")
    
    # Plot comparison of convergence
    plot_optimization_comparison(ga_results, de_results, algorithm)
    
    # Create comparison results dictionary
    comparison_results = {
        'algorithm': algorithm,
        'ga_results': ga_results,
        'de_results': de_results,
        'winner': winner,
        'improvement': improvement,
        'total_time': total_time
    }
    
    return comparison_results


def plot_optimization_comparison(ga_results, de_results, algorithm, 
                              filename=None):
    """
    Plot comparison of GA and DE convergence histories
    
    Parameters:
    -----------
    ga_results : dict
        Results from GA optimization
    de_results : dict
        Results from DE optimization
    algorithm : str
        Algorithm being optimized ('kfcm' or 'mkfcm')
    filename : str, default=None
        Filename to save the plot. If None, uses algorithm name.
    """
    if filename is None:
        filename = f"{algorithm}_optimization_comparison.png"
    
    plt.figure(figsize=(12, 8))
    
    # Get convergence histories
    ga_history = ga_results['convergence_history']
    de_history = de_results['convergence_history']
    
    # Handle case where histories might have different lengths
    ga_avg_fitness = ga_history['average_fitness']
    ga_best_fitness = ga_history['best_fitness']
    de_avg_fitness = de_history['average_fitness']
    de_best_fitness = de_history['best_fitness']
    
    # Find min length to ensure compatible shapes
    min_length = min(len(ga_avg_fitness), len(de_avg_fitness))
    
    # Truncate to same length
    ga_avg_fitness = ga_avg_fitness[:min_length]
    ga_best_fitness = ga_best_fitness[:min_length]
    de_avg_fitness = de_avg_fitness[:min_length]
    de_best_fitness = de_best_fitness[:min_length]
    
    generations = range(1, min_length + 1)
    
    # Plot average fitness
    plt.subplot(2, 1, 1)
    plt.plot(generations, ga_avg_fitness, 'b-', label='GA - Average Fitness')
    plt.plot(generations, de_avg_fitness, 'g-', label='DE - Average Fitness')
    
    plt.title(f'GA vs DE Average Fitness Comparison for {algorithm.upper()}')
    plt.ylabel('Average Fitness (Silhouette Score)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot best fitness
    plt.subplot(2, 1, 2)
    plt.plot(generations, ga_best_fitness, 'r-', label='GA - Best Fitness')
    plt.plot(generations, de_best_fitness, 'm-', label='DE - Best Fitness')
    
    plt.title(f'GA vs DE Best Fitness Comparison for {algorithm.upper()}')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness (Silhouette Score)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Comparison plot saved to {filename}")


def run_multi_trial_comparison(data, algorithm='kfcm', n_clusters=4, pop_size=20, 
                               n_generations=15, n_trials=3, base_seed=42):
    """
    Run multiple trials of GA vs DE comparison with different random seeds
    and perform statistical analysis on the results.
    
    Parameters:
    -----------
    data : array-like
        Data to cluster
    algorithm : str, default='kfcm'
        Algorithm to optimize ('kfcm' or 'mkfcm')
    n_clusters : int, default=4
        Number of clusters
    pop_size : int, default=20
        Population size for optimization algorithms
    n_generations : int, default=15
        Number of generations to run
    n_trials : int, default=3
        Number of trials to run with different random seeds
    base_seed : int, default=42
        Base random seed to use (will be incremented for each trial)
        
    Returns:
    --------
    results : dict
        Dictionary with aggregated comparison results
    """
    print(f"\nRunning {n_trials} trials of GA vs DE comparison for {algorithm.upper()}...")
    
    # Lists to store results from each trial
    ga_fitnesses = []
    de_fitnesses = []
    ga_times = []
    de_times = []
    ga_params = []
    de_params = []
    winners = []
    improvements = []
    
    # Run n_trials comparisons with different random seeds
    for trial in range(n_trials):
        random_seed = base_seed + trial
        print(f"\n--- Trial {trial+1}/{n_trials} (seed: {random_seed}) ---")
        
        # Run comparison with current random seed
        comparison_results = compare_optimization_methods(
            data, algorithm, n_clusters, pop_size, n_generations, random_seed
        )
        
        # Extract and store results
        ga_results = comparison_results['ga_results']
        de_results = comparison_results['de_results']
        
        ga_fitnesses.append(ga_results['best_fitness'])
        de_fitnesses.append(de_results['best_fitness'])
        
        ga_times.append(ga_results['total_time'] if 'total_time' in ga_results else None)
        de_times.append(de_results['total_time'] if 'total_time' in de_results else None)
        
        ga_params.append(ga_results['best_params'])
        de_params.append(de_results['best_params'])
        
        winners.append(comparison_results['winner'])
        improvements.append(comparison_results['improvement'])
    
    # Calculate aggregate metrics
    ga_wins = winners.count("Genetic Algorithm")
    de_wins = winners.count("Differential Evolution")
    
    avg_ga_fitness = sum(ga_fitnesses) / n_trials
    avg_de_fitness = sum(de_fitnesses) / n_trials
    
    std_ga_fitness = np.std(ga_fitnesses)
    std_de_fitness = np.std(de_fitnesses)
    
    # Determine overall winner based on average fitness
    if avg_ga_fitness > avg_de_fitness:
        overall_winner = "Genetic Algorithm"
        avg_improvement = (avg_ga_fitness - avg_de_fitness) / avg_de_fitness * 100
    else:
        overall_winner = "Differential Evolution"
        avg_improvement = (avg_de_fitness - avg_ga_fitness) / avg_ga_fitness * 100
    
    # Print aggregate results
    print("\n" + "="*50)
    print(f"AGGREGATE RESULTS OVER {n_trials} TRIALS")
    print("="*50)
    print(f"GA Wins: {ga_wins}/{n_trials}, DE Wins: {de_wins}/{n_trials}")
    print(f"Average GA Fitness: {avg_ga_fitness:.4f} ± {std_ga_fitness:.4f}")
    print(f"Average DE Fitness: {avg_de_fitness:.4f} ± {std_de_fitness:.4f}")
    print(f"Overall Winner: {overall_winner} with {avg_improvement:.2f}% average improvement")
    
    # Average parameter values
    avg_ga_sigma = sum(p['sigma_squared'] for p in ga_params) / n_trials
    avg_ga_m = sum(p['m'] for p in ga_params) / n_trials
    avg_de_sigma = sum(p['sigma_squared'] for p in de_params) / n_trials
    avg_de_m = sum(p['m'] for p in de_params) / n_trials
    
    print(f"Average GA Parameters: σ²={avg_ga_sigma:.4f}, m={avg_ga_m:.4f}")
    print(f"Average DE Parameters: σ²={avg_de_sigma:.4f}, m={avg_de_m:.4f}")
    
    # Prepare return value
    results = {
        'algorithm': algorithm,
        'n_trials': n_trials,
        'ga_fitnesses': ga_fitnesses,
        'de_fitnesses': de_fitnesses,
        'ga_params': ga_params,
        'de_params': de_params,
        'winners': winners,
        'improvements': improvements,
        'ga_wins': ga_wins,
        'de_wins': de_wins,
        'avg_ga_fitness': avg_ga_fitness,
        'avg_de_fitness': avg_de_fitness,
        'std_ga_fitness': std_ga_fitness,
        'std_de_fitness': std_de_fitness,
        'overall_winner': overall_winner,
        'avg_improvement': avg_improvement,
        'avg_ga_params': {'sigma_squared': avg_ga_sigma, 'm': avg_ga_m},
        'avg_de_params': {'sigma_squared': avg_de_sigma, 'm': avg_de_m}
    }
    
    # Plot comparison of average fitnesses
    plot_multi_trial_comparison(results, algorithm)
    
    return results


def plot_multi_trial_comparison(results, algorithm, filename=None):
    """
    Plot comparison of GA and DE results across multiple trials
    
    Parameters:
    -----------
    results : dict
        Results from multiple trials
    algorithm : str
        Algorithm being optimized ('kfcm' or 'mkfcm')
    filename : str, default=None
        Filename to save the plot. If None, uses algorithm name.
    """
    if filename is None:
        filename = f"{algorithm}_multi_trial_comparison.png"
    
    plt.figure(figsize=(12, 10))
    
    # ---- Plot 1: Best Fitness per Trial ----
    plt.subplot(2, 1, 1)
    trials = range(1, results['n_trials'] + 1)
    
    plt.bar([x - 0.2 for x in trials], results['ga_fitnesses'], width=0.4, label='GA', color='blue', alpha=0.7)
    plt.bar([x + 0.2 for x in trials], results['de_fitnesses'], width=0.4, label='DE', color='green', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(results['ga_fitnesses']):
        plt.text(i + 1 - 0.2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(results['de_fitnesses']):
        plt.text(i + 1 + 0.2, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.title(f'Best Fitness per Trial for {algorithm.upper()}')
    plt.xlabel('Trial')
    plt.ylabel('Fitness (Silhouette Score)')
    plt.xticks(trials)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ---- Plot 2: Average Fitness with Standard Deviation ----
    plt.subplot(2, 1, 2)
    labels = ['Genetic Algorithm', 'Differential Evolution']
    avg_values = [results['avg_ga_fitness'], results['avg_de_fitness']]
    std_values = [results['std_ga_fitness'], results['std_de_fitness']]
    
    plt.bar(labels, avg_values, yerr=std_values, alpha=0.7, capsize=10, color=['blue', 'green'])
    
    # Add value labels
    for i, v in enumerate(avg_values):
        plt.text(i, v + std_values[i] + 0.01, f'{v:.4f} ± {std_values[i]:.4f}', ha='center', va='bottom')
    
    plt.title(f'Average Fitness Across {results["n_trials"]} Trials')
    plt.ylabel('Average Fitness (Silhouette Score)')
    plt.grid(True, alpha=0.3)
    
    # Add a note about the overall winner
    plt.figtext(0.5, 0.01, 
                f"Overall Winner: {results['overall_winner']} "
                f"with {results['avg_improvement']:.2f}% average improvement\n"
                f"Wins: GA {results['ga_wins']}/{results['n_trials']}, "
                f"DE {results['de_wins']}/{results['n_trials']}",
                ha='center', 
                fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    
    print(f"Multi-trial comparison plot saved to {filename}") 