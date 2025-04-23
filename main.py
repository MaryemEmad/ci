
# main.py: Entry point for running experiments and launching the UI
import numpy as np
from data_loader import load_mall_customers_data
from kmeans_clustering import KMeansClustering
from ga_clustering import GAClustering
from aco_clustering import ACOClustering
from abc_clustering import ABCClustering
from fa_clustering import FAClustering
from de_clustering import DEClustering
from experiment_runner import run_experiments
from visualization import plot_clusters_2d, plot_clusters_3d, plot_elbow
from ui import start_ui

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)

    # Load and preprocess data
    df, features_2d, features_3d, scaler_2d, scaler_3d = load_mall_customers_data()

    # Plot elbow curve to determine optimal k (optional, for report)
    plot_elbow(features_2d, max_k=10, init_method='k-means++')
    plot_elbow(features_3d, max_k=10, init_method='k-means++')

    # Initialize clustering algorithms
    algorithms = {
        "K-means": KMeansClustering(n_clusters=4, init_method='k-means++'),
        "GA": GAClustering(n_clusters=4),
        "ACO": ACOClustering(n_clusters=4),
        "ABC": ABCClustering(n_clusters=4),
        "FA": FAClustering(n_clusters=4),
        "DE": DEClustering(n_clusters=4)
    }

    # Run experiments (30 runs per algorithm setting)
    results, seeds = run_experiments(features_2d, features_3d, algorithms)

    # Visualize results for the last run
    for algo_name, algo in algorithms.items():
        # 2D visualization
        if results[algo_name]["labels_2d"] is not None and results[algo_name]["centroids_2d"] is not None:
            plot_clusters_2d(features_2d, results[algo_name]["labels_2d"], results[algo_name]["centroids_2d"], algo_name, scaler_2d)
        # 3D visualization
        if results[algo_name]["labels_3d"] is not None and results[algo_name]["centroids_3d"] is not None:
            plot_clusters_3d(features_3d, results[algo_name]["labels_3d"], results[algo_name]["centroids_3d"], algo_name, scaler_3d)

    # Start the UI
    start_ui(features_2d, algorithms, results)

if __name__ == "__main__":
    main()
