# experiment_runner.py: Runs experiments and collects metrics
import numpy as np
import time
from sklearn.metrics import silhouette_score
from visualization import plot_clusters_2d, plot_clusters_3d

def run_experiments(data_2d, data_3d, algorithms, n_runs=30):
    results = {}
    seeds = list(range(n_runs))  # Store seeds for reproducibility

    for algo_name, algo in algorithms.items():
        algo_results = {"best_score": [], "convergence_speed": [], "time": [], "labels_2d": None, "labels_3d": None, "centroids_2d": None, "centroids_3d": None}
        
        # Experiments with 2D data (Annual Income, Spending Score)
        for run in range(n_runs):
            np.random.seed(seeds[run])

            # Measure time
            start_time = time.time()
            labels = algo.fit(data_2d)
            elapsed_time = time.time() - start_time

            # Compute silhouette score as a proxy for clustering quality
            score = silhouette_score(data_2d, labels)

            # Placeholder for convergence speed (max iterations for K-means)
            convergence_speed = algo.model.n_iter_ if hasattr(algo.model, 'n_iter_') else 0

            algo_results["best_score"].append(score)
            algo_results["convergence_speed"].append(convergence_speed)
            algo_results["time"].append(elapsed_time)

            # Store labels and centroids from the last run for visualization
            if run == n_runs - 1:
                algo_results["labels_2d"] = labels
                algo_results["centroids_2d"] = algo.get_centroids() if hasattr(algo, 'get_centroids') else None

        # Experiments with 3D data (Age, Annual Income, Spending Score)
        for run in range(n_runs):
            np.random.seed(seeds[run])
            labels = algo.fit(data_3d)
            if run == n_runs - 1:
                algo_results["labels_3d"] = labels
                algo_results["centroids_3d"] = algo.get_centroids() if hasattr(algo, 'get_centroids') else None

        results[algo_name] = algo_results
        print(f"{algo_name} - Avg Silhouette Score (2D): {np.mean(algo_results['best_score'])}, Avg Time: {np.mean(algo_results['time'])}s")

    return results, seeds
