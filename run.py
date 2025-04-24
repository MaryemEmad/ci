#!/usr/bin/env python
# run.py: Simpler entry point for running just the K-means experiment

import numpy as np
from data_loader import load_mall_customers_data
from kmeans_clustering import KMeansClustering
from visualization import plot_clusters_2d, plot_clusters_3d, plot_elbow
from ui import start_ui

def main():
    print("Running Customer Segmentation with K-means...")
    
    # Set random seeds for reproducibility
    main_seed = 42
    np.random.seed(main_seed)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    df, features_2d, features_3d, scaler_2d, scaler_3d = load_mall_customers_data()

    # Plot elbow curve to determine optimal k
    print("Generating elbow plot...")
    plot_elbow(features_2d, max_k=10, init_method='k-means++')

    # Initialize just the K-means algorithm
    algorithms = {
        "K-means": KMeansClustering(n_clusters=4, init_method='k-means++'),
    }

    # Start the UI with both 2D and 3D data
    print("Starting UI...")
    start_ui(data_2d=features_2d, algorithms=algorithms, data_3d=features_3d)

if __name__ == "__main__":
    main() 
