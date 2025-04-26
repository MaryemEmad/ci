experiment_runner.py:
# experiment_runner.py: Runs experiments and collects metrics
import numpy as np
import time
from sklearn.metrics import silhouette_score
from visualization import plot_clusters_2d, plot_clusters_3d

def run_experiments(data_2d, data_3d, algorithms, n_runs=30, algorithm_runs=None):
    """
    Run multiple experiments with all algorithms and collect performance metrics.
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    data_3d : array-like
        3D dataset (Age, Annual Income, Spending Score)
    algorithms : dict
        Dictionary of algorithms to run {'algorithm_name': algorithm_instance}
    n_runs : int
        Default number of runs per algorithm (default: 30)
    algorithm_runs : dict, optional
        Dictionary specifying number of runs for specific algorithms
        Example: {'K-means': 10, 'GA': 50}
        
    Returns:
    --------
    results : dict
        Dictionary with results for all algorithms
    seeds : dict
        Dictionary of random seeds used for reproducibility for each algorithm
    """
    results = {}
    seeds = {}  # Store seeds for reproducibility by algorithm
    
    # Initialize algorithm_runs if not provided
    if algorithm_runs is None:
        algorithm_runs = {}

    for algo_name, algo in algorithms.items():
        # Get number of runs for this algorithm (use default if not specified)
        runs_for_algorithm = algorithm_runs.get(algo_name, n_runs)
        print(f"Running experiments for {algo_name} ({runs_for_algorithm} runs)...")
        
        # Generate seeds for this algorithm
        algorithm_seeds = list(range(runs_for_algorithm))
        seeds[algo_name] = algorithm_seeds
        
        # Initialize results dictionary for this algorithm
        algo_results = {
            "silhouette_scores": [],  # Clustering quality
            "iterations": [],         # Convergence speed
            "times": [],              # Computational time
            "labels_2d": None,        # Final cluster assignments (2D)
            "labels_3d": None,        # Final cluster assignments (3D)
            "centroids_2d": None,     # Final centroids (2D)
            "centroids_3d": None,     # Final centroids (3D)
            "n_runs": runs_for_algorithm,  # Store number of runs for reference
        }
        
        # Run experiments with 2D data
        for run in range(runs_for_algorithm):
            # Set random seed for reproducibility
            np.random.seed(algorithm_seeds[run])

            # Measure execution time
            start_time = time.time()
            labels = algo.fit(data_2d)
            execution_time = time.time() - start_time

            # Calculate silhouette score (clustering quality)
            score = silhouette_score(data_2d, labels)

            # Get number of iterations (if available, for convergence speed)
            iterations = getattr(algo.model, 'n_iter_', 0) if hasattr(algo, 'model') else 0

            # Store results
            algo_results["silhouette_scores"].append(score)
            algo_results["iterations"].append(iterations)
            algo_results["times"].append(execution_time)

            # Show progress for algorithms with many runs
            if runs_for_algorithm > 10 and (run + 1) % 10 == 0:
                print(f"  {algo_name} - Completed {run + 1}/{runs_for_algorithm} runs")
            
            # Save results from last run for visualization
            if run == runs_for_algorithm - 1:
                algo_results["labels_2d"] = labels
                if hasattr(algo, 'get_centroids'):
                    algo_results["centroids_2d"] = algo.get_centroids()

        # Run experiments with 3D data (only for the last run)
        np.random.seed(algorithm_seeds[0])
        labels_3d = algo.fit(data_3d)
        algo_results["labels_3d"] = labels_3d
        if hasattr(algo, 'get_centroids'):
            algo_results["centroids_3d"] = algo.get_centroids()
        
        # Store results for this algorithm
        results[algo_name] = algo_results
        
        # Print summary
        print(f"  {algo_name} - Avg Silhouette Score: {np.mean(algo_results['silhouette_scores']):.4f}")
        print(f"  {algo_name} - Avg Time: {np.mean(algo_results['times']):.4f}s")

    return results, seeds
