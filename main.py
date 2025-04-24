# main.py: Entry point for running experiments and launching the UI
import numpy as np
from data_loader import load_mall_customers_data
from kmeans_clustering import KMeansClustering
from ga_clustering import GAClustering
from aco_clustering import ACOClustering
from abc_clustering import ABCClustering
from fa_clustering import FAClustering
from de_clustering import DEClustering
from visualization import plot_clusters_2d, plot_clusters_3d, plot_elbow
from ui import start_ui

def main():
    # Set random seeds for reproducibility
    main_seed = 42
    np.random.seed(main_seed)

    # Load and preprocess data
    df, features_2d, features_3d, scaler_2d, scaler_3d = load_mall_customers_data()

    # Plot elbow curve to determine optimal k (optional, for report)
    plot_elbow(features_2d, max_k=10, init_method='k-means++')
    plot_elbow(features_3d, max_k=10, init_method='k-means++')

    # Initialize clustering algorithms with the same random seed for fair comparison
    algorithms = {
        "K-means": KMeansClustering(n_clusters=4, init_method='k-means++'),
        "GA": GAClustering(n_clusters=4, random_state=main_seed),
        "ACO": ACOClustering(n_clusters=4, random_state=main_seed),
        "ABC": ABCClustering(n_clusters=4, random_state=main_seed),
        "FA": FAClustering(n_clusters=4, random_state=main_seed),
        "DE": DEClustering(n_clusters=4, random_state=main_seed)
    }

    # Start the UI directly without pre-running experiments
    # Pass both 2D and 3D data to enable both visualizations
    start_ui(data_2d=features_2d, algorithms=algorithms, data_3d=features_3d)

if __name__ == "__main__":
    main()
