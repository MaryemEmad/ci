#!/usr/bin/env python
# run.py: Main script to run clustering experiments and generate visualizations

import numpy as np
import json
import os
import time
from sklearn.metrics import silhouette_score
from data_loader import load_mall_customers_data
from kmeans_clustering import KMeansClustering
from fcm_clustering import FuzzyCMeansClustering
from visualization import (
    plot_clusters_2d, 
    plot_clusters_3d,
    plot_elbow, 
    compare_kmeans_fcm,
    plot_wcss_comparison,
    plot_fcm_m_comparison,
    plot_kmeans_init_comparison,
    plot_convergence_curves
)

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

def save_experiment_seeds(seeds, filename="experiment_seeds.json"):
    """Save experiment seeds to a file for reproducibility"""
    # Ensure all data is JSON serializable
    for key in seeds:
        if hasattr(seeds[key], 'tolist'):
            seeds[key] = seeds[key].tolist()
    
    with open(filename, 'w') as f:
        json.dump(seeds, f, indent=4)

def load_experiment_seeds(filename="experiment_seeds.json"):
    """Load experiment seeds from file"""
    try:
        with open(filename, 'r') as f:
            seeds = json.load(f)
            return seeds
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load experiment seeds: {e}")
        return {}

def run_kmeans_fcm_comparison(data_2d, data_3d, n_clusters=4, n_runs=30, random_seeds=None):
    """
    Run comparison between K-means and FCM algorithms
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    data_3d : array-like
        3D dataset (Age, Annual Income, Spending Score)
    n_clusters : int, default=4
        Number of clusters
    n_runs : int, default=30
        Number of runs (for statistical significance)
    random_seeds : list or None, default=None
        List of random seeds for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with experiment results
    seeds : list
        Random seeds used for experiments
    """
    print(f"Running K-means vs FCM comparison ({n_runs} runs each)...")
    
    # Generate or use provided random seeds
    if random_seeds is None:
        random_seeds = np.random.randint(0, 10000, size=n_runs)
        # Convert to list if it's a numpy array
        if hasattr(random_seeds, 'tolist'):
            random_seeds = random_seeds.tolist()
    # Ensure we have the right number of seeds
    if len(random_seeds) < n_runs:
        additional_seeds = np.random.randint(0, 10000, size=(n_runs - len(random_seeds))).tolist()
        random_seeds.extend(additional_seeds)
    
    # Results containers
    kmeans_silhouette_2d = []
    kmeans_wcss_2d = []
    kmeans_time_2d = []
    
    fcm_silhouette_2d = []
    fcm_wcss_2d = []
    fcm_time_2d = []
    
    # Store final results from last run for visualization
    final_results = {}
    
    # Run experiments
    for i, seed in enumerate(random_seeds):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        # Run K-means
        start_time = time.time()
        kmeans = KMeansClustering(n_clusters=n_clusters, init_method='k-means++', random_state=seed)
        kmeans_labels_2d = kmeans.fit(data_2d)
        kmeans_time_2d.append(time.time() - start_time)
        
        # Calculate K-means metrics
        kmeans_wcss_2d.append(kmeans.compute_fitness(data_2d))
        kmeans_silhouette_2d.append(silhouette_score(data_2d, kmeans_labels_2d))
        
        # Run FCM
        start_time = time.time()
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
        fcm_labels_2d = fcm.fit(data_2d)
        fcm_time_2d.append(time.time() - start_time)
        
        # Calculate FCM metrics
        fcm_wcss_2d.append(fcm.get_crisp_inertia())  # Use crisp WCSS for fair comparison
        fcm_silhouette_2d.append(silhouette_score(data_2d, fcm_labels_2d))
        
        # Store results from final run for visualization
        if i == n_runs - 1:
            final_results["kmeans_labels_2d"] = kmeans_labels_2d
            final_results["kmeans_centroids_2d"] = kmeans.get_centroids()
            final_results["fcm_labels_2d"] = fcm_labels_2d
            final_results["fcm_centroids_2d"] = fcm.get_centroids()
            
            # Also run on 3D data for the final run
            kmeans_3d = KMeansClustering(n_clusters=n_clusters, init_method='k-means++', random_state=seed)
            fcm_3d = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
            
            kmeans_labels_3d = kmeans_3d.fit(data_3d)
            fcm_labels_3d = fcm_3d.fit(data_3d)
            
            final_results["kmeans_labels_3d"] = kmeans_labels_3d
            final_results["kmeans_centroids_3d"] = kmeans_3d.get_centroids()
            final_results["fcm_labels_3d"] = fcm_labels_3d
            final_results["fcm_centroids_3d"] = fcm_3d.get_centroids()
            
            # Save convergence history
            final_results["kmeans_history"] = kmeans.get_fitness_history()
            final_results["fcm_history"] = fcm.get_fitness_history()
    
    # Calculate average metrics
    results = {
        "kmeans_avg_silhouette": np.mean(kmeans_silhouette_2d),
        "kmeans_avg_wcss": np.mean(kmeans_wcss_2d),
        "kmeans_avg_time": np.mean(kmeans_time_2d),
        "fcm_avg_silhouette": np.mean(fcm_silhouette_2d),
        "fcm_avg_wcss": np.mean(fcm_wcss_2d),
        "fcm_avg_time": np.mean(fcm_time_2d),
        "final_run": final_results
    }
    
    # Print summary
    print("\nK-means vs FCM Results Summary:")
    print(f"  K-means - Avg Silhouette: {results['kmeans_avg_silhouette']:.4f}")
    print(f"  K-means - Avg WCSS: {results['kmeans_avg_wcss']:.4f}")
    print(f"  K-means - Avg Time: {results['kmeans_avg_time']:.4f}s")
    print(f"  FCM - Avg Silhouette: {results['fcm_avg_silhouette']:.4f}")
    print(f"  FCM - Avg WCSS: {results['fcm_avg_wcss']:.4f}")
    print(f"  FCM - Avg Time: {results['fcm_avg_time']:.4f}s")
    
    return results, random_seeds

def run_fcm_m_comparison(data_2d, n_clusters=4, m_values=None, random_seed=42):
    """
    Compare FCM with different m values
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    n_clusters : int, default=4
        Number of clusters
    m_values : list or None, default=None
        List of m values to compare (default: [1.1, 1.5, 2.0, 2.5, 3.0])
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with experiment results
    """
    if m_values is None:
        m_values = [1.1, 1.5, 2.0, 2.5, 3.0]
    
    print(f"Running FCM m-value comparison for m in {m_values}...")
    
    results = {}
    fcm_results = []
    
    # Run FCM for each m value
    for m in m_values:
        print(f"  FCM with m={m}")
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=m, random_state=random_seed)
        labels = fcm.fit(data_2d)
        
        # Calculate metrics
        silhouette = silhouette_score(data_2d, labels)
        wcss = fcm.get_crisp_inertia()
        
        fcm_results.append({
            "m": m,
            "labels": labels,
            "centroids": fcm.get_centroids(),
            "silhouette": silhouette,
            "wcss": wcss,
            "history": fcm.get_fitness_history()
        })
        
        print(f"    Silhouette: {silhouette:.4f}, WCSS: {wcss:.4f}")
    
    results["fcm_results"] = fcm_results
    return results

def run_kmeans_init_comparison(data_2d, n_clusters=4, n_runs=10, random_seeds=None):
    """
    Compare K-means with different initialization methods
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    n_clusters : int, default=4
        Number of clusters
    n_runs : int, default=10
        Number of runs (for statistical significance)
    random_seeds : list or None, default=None
        List of random seeds for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with experiment results
    seeds : list
        Random seeds used for experiments
    """
    print(f"Running K-means initialization comparison ({n_runs} runs each)...")
    
    # Generate or use provided random seeds
    if random_seeds is None:
        random_seeds = np.random.randint(0, 10000, size=n_runs)
        # Convert to list if it's a numpy array
        if hasattr(random_seeds, 'tolist'):
            random_seeds = random_seeds.tolist()
    # Ensure we have the right number of seeds
    if len(random_seeds) < n_runs:
        additional_seeds = np.random.randint(0, 10000, size=(n_runs - len(random_seeds))).tolist()
        random_seeds.extend(additional_seeds)
    
    # Results containers
    kmeans_pp_silhouette = []
    kmeans_pp_wcss = []
    kmeans_pp_time = []
    
    kmeans_random_silhouette = []
    kmeans_random_wcss = []
    kmeans_random_time = []
    
    # Store final results from last run for visualization
    final_results = {}
    
    # Run experiments
    for i, seed in enumerate(random_seeds):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        # Run K-means with k-means++ initialization
        start_time = time.time()
        kmeans_pp = KMeansClustering(n_clusters=n_clusters, init_method='k-means++', random_state=seed)
        labels_pp = kmeans_pp.fit(data_2d)
        kmeans_pp_time.append(time.time() - start_time)
        
        # Calculate metrics
        kmeans_pp_wcss.append(kmeans_pp.compute_fitness(data_2d))
        kmeans_pp_silhouette.append(silhouette_score(data_2d, labels_pp))
        
        # Run K-means with random initialization
        start_time = time.time()
        kmeans_random = KMeansClustering(n_clusters=n_clusters, init_method='random', random_state=seed)
        labels_random = kmeans_random.fit(data_2d)
        kmeans_random_time.append(time.time() - start_time)
        
        # Calculate metrics
        kmeans_random_wcss.append(kmeans_random.compute_fitness(data_2d))
        kmeans_random_silhouette.append(silhouette_score(data_2d, labels_random))
        
        # Store results from final run for visualization
        if i == n_runs - 1:
            final_results["kmeans_pp_labels"] = labels_pp
            final_results["kmeans_pp_centroids"] = kmeans_pp.get_centroids()
            final_results["kmeans_random_labels"] = labels_random
            final_results["kmeans_random_centroids"] = kmeans_random.get_centroids()
            
            # Save convergence history
            final_results["kmeans_pp_history"] = kmeans_pp.get_fitness_history()
            final_results["kmeans_random_history"] = kmeans_random.get_fitness_history()
    
    # Calculate average metrics
    results = {
        "kmeans_pp_avg_silhouette": np.mean(kmeans_pp_silhouette),
        "kmeans_pp_avg_wcss": np.mean(kmeans_pp_wcss),
        "kmeans_pp_avg_time": np.mean(kmeans_pp_time),
        "kmeans_random_avg_silhouette": np.mean(kmeans_random_silhouette),
        "kmeans_random_avg_wcss": np.mean(kmeans_random_wcss),
        "kmeans_random_avg_time": np.mean(kmeans_random_time),
        "final_run": final_results
    }
    
    # Print summary
    print("\nK-means Initialization Results Summary:")
    print(f"  k-means++ - Avg Silhouette: {results['kmeans_pp_avg_silhouette']:.4f}")
    print(f"  k-means++ - Avg WCSS: {results['kmeans_pp_avg_wcss']:.4f}")
    print(f"  k-means++ - Avg Time: {results['kmeans_pp_avg_time']:.4f}s")
    print(f"  Random - Avg Silhouette: {results['kmeans_random_avg_silhouette']:.4f}")
    print(f"  Random - Avg WCSS: {results['kmeans_random_avg_wcss']:.4f}")
    print(f"  Random - Avg Time: {results['kmeans_random_avg_time']:.4f}s")
    
    return results, random_seeds

def main():
    # Set random seed for reproducibility
    main_seed = 42
    np.random.seed(main_seed)
    
    # Load experiment seeds if available
    experiment_seeds = load_experiment_seeds()
    
    # Load and preprocess data
    print("Loading Mall Customers dataset...")
    df, features_2d, features_3d, scaler_2d, scaler_3d = load_mall_customers_data()

    # Plot elbow curve to determine optimal k (for report)
    print("Plotting elbow curve for optimal k...")
    plot_elbow(features_2d, max_k=10, init_method='k-means++')
    
    # Run K-means vs FCM comparison
    kmeans_fcm_seeds = experiment_seeds.get("kmeans_fcm_comparison", [])
    kmeans_fcm_results, kmeans_fcm_seeds = run_kmeans_fcm_comparison(
        features_2d, features_3d, n_clusters=4, n_runs=30, random_seeds=kmeans_fcm_seeds
    )
    
    # Run FCM m-value comparison
    fcm_m_results = run_fcm_m_comparison(
        features_2d, n_clusters=4, m_values=[1.1, 1.5, 2.0, 2.5, 3.0], random_seed=main_seed
    )
    
    # Run K-means initialization comparison
    kmeans_init_seeds = experiment_seeds.get("kmeans_init_comparison", [])
    kmeans_init_results, kmeans_init_seeds = run_kmeans_init_comparison(
        features_2d, n_clusters=4, n_runs=10, random_seeds=kmeans_init_seeds
    )
    
    # Save experiment seeds for reproducibility
    experiment_seeds["kmeans_fcm_comparison"] = kmeans_fcm_seeds
    experiment_seeds["kmeans_init_comparison"] = kmeans_init_seeds
    save_experiment_seeds(experiment_seeds)
    
    # Generate visualizations
    
    # 1. 2D/3D Clustering Visualizations for K-means and FCM
    print("Generating 2D/3D clustering visualizations...")
    # K-means
    plot_clusters_2d(
        features_2d, 
        kmeans_fcm_results["final_run"]["kmeans_labels_2d"],
        kmeans_fcm_results["final_run"]["kmeans_centroids_2d"],
        "K-means",
        scaler_2d
    )
    plot_clusters_3d(
        features_3d, 
        kmeans_fcm_results["final_run"]["kmeans_labels_3d"],
        kmeans_fcm_results["final_run"]["kmeans_centroids_3d"],
        "K-means",
        scaler_3d
    )
    
    # FCM
    plot_clusters_2d(
        features_2d, 
        kmeans_fcm_results["final_run"]["fcm_labels_2d"],
        kmeans_fcm_results["final_run"]["fcm_centroids_2d"],
        "Fuzzy_C-Means",
        scaler_2d
    )
    plot_clusters_3d(
        features_3d, 
        kmeans_fcm_results["final_run"]["fcm_labels_3d"],
        kmeans_fcm_results["final_run"]["fcm_centroids_3d"],
        "Fuzzy_C-Means",
        scaler_3d
    )
    
    # 2. Side-by-side comparison of K-means and FCM
    print("Generating K-means vs FCM comparison plots...")
    compare_kmeans_fcm(
        features_2d,
        kmeans_fcm_results["final_run"]["kmeans_labels_2d"],
        kmeans_fcm_results["final_run"]["fcm_labels_2d"],
        kmeans_fcm_results["final_run"]["kmeans_centroids_2d"],
        kmeans_fcm_results["final_run"]["fcm_centroids_2d"],
        scaler_2d
    )
    
    # 3. WCSS comparison
    plot_wcss_comparison(
        kmeans_fcm_results["kmeans_avg_wcss"],
        kmeans_fcm_results["fcm_avg_wcss"]
    )
    
    # 4. FCM m-value comparison
    print("Generating FCM m-value comparison plots...")
    plot_fcm_m_comparison(
        features_2d,
        [result["m"] for result in fcm_m_results["fcm_results"]],
        n_clusters=4,
        scaler=scaler_2d,
        random_state=main_seed
    )
    
    # 5. K-means initialization comparison
    print("Generating K-means initialization comparison plots...")
    plot_kmeans_init_comparison(
        features_2d,
        n_clusters=4,
        scaler=scaler_2d,
        random_state=main_seed
    )
    
    # 6. Convergence curves
    print("Generating convergence curve plots...")
    # K-means vs FCM convergence
    plot_convergence_curves(
        {
            "K-means": kmeans_fcm_results["final_run"]["kmeans_history"],
            "FCM": kmeans_fcm_results["final_run"]["fcm_history"]
        }
    )
    
    # K-means initialization convergence
    plot_convergence_curves(
        {
            "K-means++ Initialization": kmeans_init_results["final_run"]["kmeans_pp_history"],
            "Random Initialization": kmeans_init_results["final_run"]["kmeans_random_history"]
        }
    )
    
    # FCM m-value convergence
    fcm_histories = {}
    for result in fcm_m_results["fcm_results"]:
        fcm_histories[f"FCM m={result['m']}"] = result["history"]
    plot_convergence_curves(fcm_histories)
    
    print("\nAll experiments and visualizations complete!")
    print("Seed data saved for reproducibility.")

if __name__ == "__main__":
    main() 
