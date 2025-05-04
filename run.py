#!/usr/bin/env python
# run.py: Main script to run clustering experiments and generate visualizations

import numpy as np
import json
import os
import time
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_loader import load_mall_customers_data
from kmeans_clustering import KMeansClustering
from fcm_clustering import FuzzyCMeansClustering
from gkfcm_clustering import GKFuzzyCMeansClustering
from kfcm_clustering import KernelFuzzyCMeansClustering
from mkfcm_clustering import ModifiedKernelFuzzyCMeansClustering
from rsekfcm_clustering import RseKFCMClustering
from spkfcm_clustering import SpKFCMClustering
from okfcm_clustering import OKFCMClustering

from visualization import (
    plot_clusters_2d, 
    plot_clusters_3d,
    plot_elbow, 
    compare_kmeans_fcm,
    plot_wcss_comparison,
    plot_fcm_m_comparison,
    plot_kmeans_init_comparison,
    plot_convergence_curves,
    plot_algorithm_metrics_comparison,
    compare_all_fuzzy,
    compare_fuzzy_metrics,
    plot_kernel_sigma_comparison,
    compare_fuzzy_metrics_with_error_bars,
    plot_sigma_parameter_study,
    plot_incremental_params_comparison
)

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

def save_experiment_seeds(seeds, filename="experiment_seeds.json"):
    """Save experiment seeds to a file for reproducibility."""
    # Ensure all data is JSON serializable
    serializable_seeds = {}
    for key, value in seeds.items():
        if isinstance(value, np.ndarray):
            serializable_seeds[key] = value.tolist()
        else:
            serializable_seeds[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_seeds, f, indent=4)

def load_experiment_seeds(filename="experiment_seeds.json"):
    """Load experiment seeds from file."""
    try:
        with open(filename, 'r') as f:
            seeds = json.load(f)
            return seeds
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load experiment seeds: {e}")
        return {}

def plot_elbow_curve(data, max_k=10):
    """
    Plot the elbow curve to determine the optimal number of clusters and return optimal_k.
    
    Parameters:
    -----------
    data : array-like
        Dataset to cluster
    max_k : int, default=10
        Maximum number of clusters to test
        
    Returns:
    --------
    optimal_k : int
        Optimal number of clusters based on the elbow method
    """
    print("Plotting elbow curve for optimal k...")
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve for Optimal k')
    plt.grid(True)
    plt.savefig('results/elbow_curve.png')
    plt.close()
    
    # Simple heuristic to find the elbow: largest difference in inertia
    diffs = np.diff(inertias)
    diffs = -diffs  # We want the largest decrease
    optimal_k = np.argmax(diffs) + 2  # +2 because diffs is offset by 1, and k starts at 1
    return optimal_k

def plot_data_distribution(data_2d):
    """
    Plot a scatter plot of the 2D data to visualize its distribution.
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    """
    print("Plotting scatter plot of the data distribution...")
    plt.figure(figsize=(8, 6))
    plt.scatter(data_2d[:, 0], data_2d[:, 1], s=50, alpha=0.6)
    plt.xlabel('Annual Income (scaled)')
    plt.ylabel('Spending Score (scaled)')
    plt.title('Scatter Plot of Mall Customers Data (2D)')
    plt.grid(True)
    plt.savefig('results/data_distribution_2d.png')
    plt.close()
    

def run_kmeans_fcm_comparison(data_2d, data_3d, n_clusters=4, n_runs=30, random_seeds=None):
    """
    Run comparison between K-means and FCM algorithms.
    
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
        random_seeds = np.random.randint(0, 10000, size=n_runs).tolist()
    if len(random_seeds) < n_runs:
        additional_seeds = np.random.randint(0, 10000, size=(n_runs - len(random_seeds))).tolist()
        random_seeds.extend(additional_seeds)
    
    # Results containers
    kmeans_silhouette_2d = []
    kmeans_inertia_2d = []
    kmeans_time_2d = []
    
    fcm_silhouette_2d = []
    fcm_fuzzy_inertia = []
    fcm_time_2d = []
    
    all_run_results = []
    final_results = {}
    
    for i, seed in enumerate(random_seeds[:n_runs]):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        # Run K-means
        start_time = time.time()
        kmeans = KMeansClustering(n_clusters=n_clusters, init_method='k-means++', random_state=seed)
        kmeans_labels_2d = kmeans.fit(data_2d)
        kmeans_time = time.time() - start_time
        kmeans_time_2d.append(kmeans_time)
        
        kmeans_inertia = kmeans.compute_fitness(data_2d)
        kmeans_inertia_2d.append(kmeans_inertia)
        kmeans_silhouette = silhouette_score(data_2d, kmeans_labels_2d)
        kmeans_silhouette_2d.append(kmeans_silhouette)
        
        # Run FCM
        start_time = time.time()
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed)
        fcm_labels_2d = fcm.fit(data_2d)
        fcm_time = time.time() - start_time
        fcm_time_2d.append(fcm_time)
        
        fcm_inertia = fcm.inertia_
        fcm_fuzzy_inertia.append(fcm_inertia)
        fcm_silhouette = silhouette_score(data_2d, fcm_labels_2d)
        fcm_silhouette_2d.append(fcm_silhouette)
        
        run_result = {
            "run": i+1,
            "seed": seed,
            "kmeans_silhouette": kmeans_silhouette,
            "kmeans_inertia": kmeans_inertia,
            "kmeans_time": kmeans_time,
            "fcm_silhouette": fcm_silhouette,
            "fcm_fuzzy_inertia": fcm_inertia,
            "fcm_time": fcm_time
        }
        all_run_results.append(run_result)
        
        if i == n_runs - 1:
            final_results["kmeans_labels_2d"] = kmeans_labels_2d
            final_results["kmeans_centroids_2d"] = kmeans.get_centroids()
            final_results["fcm_labels_2d"] = fcm_labels_2d
            final_results["fcm_centroids_2d"] = fcm.get_centroids()
            
            kmeans_3d = KMeansClustering(n_clusters=n_clusters, init_method='k-means++', random_state=seed)
            fcm_3d = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
            
            kmeans_labels_3d = kmeans_3d.fit(data_3d)
            fcm_labels_3d = fcm_3d.fit(data_3d)
            
            final_results["kmeans_labels_3d"] = kmeans_labels_3d
            final_results["kmeans_centroids_3d"] = kmeans_3d.get_centroids()
            final_results["fcm_labels_3d"] = fcm_labels_3d
            final_results["fcm_centroids_3d"] = fcm_3d.get_centroids()
            final_results["fcm_history"] = fcm.get_fitness_history()
    
    results = {
        "kmeans_avg_silhouette": np.mean(kmeans_silhouette_2d),
        "kmeans_avg_inertia": np.mean(kmeans_inertia_2d),
        "kmeans_avg_time": np.mean(kmeans_time_2d),
        "fcm_avg_silhouette": np.mean(fcm_silhouette_2d),
        "fcm_avg_fuzzy_inertia": np.mean(fcm_fuzzy_inertia),
        "fcm_avg_time": np.mean(fcm_time_2d),
        "kmeans_std_silhouette": np.std(kmeans_silhouette_2d),
        "kmeans_std_inertia": np.std(kmeans_inertia_2d),
        "kmeans_std_time": np.std(kmeans_time_2d),
        "fcm_std_silhouette": np.std(fcm_silhouette_2d),
        "fcm_std_fuzzy_inertia": np.std(fcm_fuzzy_inertia),
        "fcm_std_time": np.std(fcm_time_2d),
        "all_run_results": all_run_results,
        "final_run": final_results
    }
    
    print("\nK-means vs FCM Results Summary:")
    print(f"  Common Comparison Metric (Silhouette Score):")
    print(f"    K-means - Avg Silhouette: {results['kmeans_avg_silhouette']:.4f} ± {results['kmeans_std_silhouette']:.4f}")
    print(f"    FCM - Avg Silhouette: {results['fcm_avg_silhouette']:.4f} ± {results['fcm_std_silhouette']:.4f}")
    print(f"  Algorithm-Specific Metrics:")
    print(f"    K-means - Avg Inertia: {results['kmeans_avg_inertia']:.4f} ± {results['kmeans_std_inertia']:.4f}")
    print(f"    FCM - Avg Fuzzy Inertia: {results['fcm_avg_fuzzy_inertia']:.4f} ± {results['fcm_std_fuzzy_inertia']:.4f}")
    print(f"  Performance:")
    print(f"    K-means - Avg Time: {results['kmeans_avg_time']:.4f}s ± {results['kmeans_std_time']:.4f}s")
    print(f"    FCM - Avg Time: {results['fcm_avg_time']:.4f}s ± {results['fcm_std_time']:.4f}s")
    
    return results, random_seeds[:n_runs]

def run_fcm_m_comparison(data_2d, n_clusters=4, m_values=None, random_seed=42):
    """
    Compare FCM with different m values.
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    n_clusters : int, default=4
        Number of clusters
    m_values : list or None, default=None
        List of m values to compare
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
    
    fcm_results = []
    
    for m in m_values:
        print(f"  FCM with m={m}")
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=m, random_state=random_seed)
        labels = fcm.fit(data_2d)
        
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
    
    return {"fcm_results": fcm_results}

def run_kmeans_init_comparison(data_2d, n_clusters=4, n_runs=10, random_seeds=None):
    """
    Compare K-means with different initialization methods.
    
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
    
    if random_seeds is None:
        random_seeds = np.random.randint(0, 10000, size=n_runs).tolist()
    if len(random_seeds) < n_runs:
        additional_seeds = np.random.randint(0, 10000, size=(n_runs - len(random_seeds))).tolist()
        random_seeds.extend(additional_seeds)
    
    kmeans_pp_silhouette = []
    kmeans_pp_wcss = []
    kmeans_pp_time = []
    
    kmeans_random_silhouette = []
    kmeans_random_wcss = []
    kmeans_random_time = []
    
    final_results = {}
    
    for i, seed in enumerate(random_seeds[:n_runs]):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        start_time = time.time()
        kmeans_pp = KMeansClustering(n_clusters=n_clusters, init_method='k-means++', random_state=seed)
        labels_pp = kmeans_pp.fit(data_2d)
        kmeans_pp_time.append(time.time() - start_time)
        
        kmeans_pp_wcss.append(kmeans_pp.compute_fitness(data_2d))
        kmeans_pp_silhouette.append(silhouette_score(data_2d, labels_pp))
        
        start_time = time.time()
        kmeans_random = KMeansClustering(n_clusters=n_clusters, init_method='random', random_state=seed)
        labels_random = kmeans_random.fit(data_2d)
        kmeans_random_time.append(time.time() - start_time)
        
        kmeans_random_wcss.append(kmeans_random.compute_fitness(data_2d))
        kmeans_random_silhouette.append(silhouette_score(data_2d, labels_random))
        
        if i == n_runs - 1:
            final_results["kmeans_pp_labels"] = labels_pp
            final_results["kmeans_pp_centroids"] = kmeans_pp.get_centroids()
            final_results["kmeans_random_labels"] = labels_random
            final_results["kmeans_random_centroids"] = kmeans_random.get_centroids()
    
    results = {
        "kmeans_pp_avg_silhouette": np.mean(kmeans_pp_silhouette),
        "kmeans_pp_avg_wcss": np.mean(kmeans_pp_wcss),
        "kmeans_pp_avg_time": np.mean(kmeans_pp_time),
        "kmeans_random_avg_silhouette": np.mean(kmeans_random_silhouette),
        "kmeans_random_avg_wcss": np.mean(kmeans_random_wcss),
        "kmeans_random_avg_time": np.mean(kmeans_random_time),
        "final_run": final_results
    }
    
    print("\nK-means Initialization Results Summary:")
    print(f"  k-means++ - Avg Silhouette: {results['kmeans_pp_avg_silhouette']:.4f}")
    print(f"  k-means++ - Avg WCSS: {results['kmeans_pp_avg_wcss']:.4f}")
    print(f"  k-means++ - Avg Time: {results['kmeans_pp_avg_time']:.4f}s")
    print(f"  Random - Avg Silhouette: {results['kmeans_random_avg_silhouette']:.4f}")
    print(f"  Random - Avg WCSS: {results['kmeans_random_avg_wcss']:.4f}")
    print(f"  Random - Avg Time: {results['kmeans_random_avg_time']:.4f}s")
    
    return results, random_seeds[:n_runs]

def run_fcm_gkfcm_comparison(data_2d, data_3d, n_clusters=4, n_runs=30, random_seeds=None):
    """
    Run comparison between FCM and GK-FCM algorithms.
    
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
    print(f"Running FCM vs GK-FCM comparison ({n_runs} runs each)...")
    
    if random_seeds is None:
        random_seeds = np.random.randint(0, 10000, size=n_runs).tolist()
    if len(random_seeds) < n_runs:
        additional_seeds = np.random.randint(0, 10000, size=(n_runs - len(random_seeds))).tolist()
        random_seeds.extend(additional_seeds)
    
    fcm_silhouette_2d = []
    fcm_fuzzy_inertia = []
    fcm_time_2d = []
    
    gkfcm_silhouette_2d = []
    gkfcm_fuzzy_inertia = []
    gkfcm_time_2d = []
    
    final_results = {}
    
    for i, seed in enumerate(random_seeds[:n_runs]):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        start_time = time.time()
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed)
        fcm_labels_2d = fcm.fit(data_2d)
        fcm_time_2d.append(time.time() - start_time)
        
        fcm_fuzzy_inertia.append(fcm.inertia_)
        fcm_silhouette_2d.append(silhouette_score(data_2d, fcm_labels_2d))
        
        start_time = time.time()
        gkfcm = GKFuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed)
        gkfcm_labels_2d = gkfcm.fit(data_2d)
        gkfcm_time_2d.append(time.time() - start_time)
        
        gkfcm_fuzzy_inertia.append(gkfcm.inertia_)
        gkfcm_silhouette_2d.append(silhouette_score(data_2d, gkfcm_labels_2d))
        
        if i == n_runs - 1:
            final_results["fcm_labels_2d"] = fcm_labels_2d
            final_results["fcm_centroids_2d"] = fcm.get_centroids()
            final_results["gkfcm_labels_2d"] = gkfcm_labels_2d
            final_results["gkfcm_centroids_2d"] = gkfcm.get_centroids()
            
            fcm_3d = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed)
            gkfcm_3d = GKFuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed)
            
            fcm_labels_3d = fcm_3d.fit(data_3d)
            gkfcm_labels_3d = gkfcm_3d.fit(data_3d)
            
            final_results["fcm_labels_3d"] = fcm_labels_3d
            final_results["fcm_centroids_3d"] = fcm_3d.get_centroids()
            final_results["gkfcm_labels_3d"] = gkfcm_labels_3d
            final_results["gkfcm_centroids_3d"] = gkfcm_3d.get_centroids()
            
            final_results["fcm_history"] = fcm.get_fitness_history()
            final_results["gkfcm_history"] = gkfcm.get_fitness_history()
            
            if gkfcm.centroids.shape[1] != data_2d.shape[1]:
                gkfcm.centroids = gkfcm.centroids[:, :data_2d.shape[1]]
            
            gkfcm.covariance_matrices = gkfcm.update_covariance_matrices(data_2d, gkfcm.centroids, gkfcm.membership)
            
            n_features = data_2d.shape[1]
            gkfcm.inv_covariance_matrices = np.zeros((gkfcm.n_clusters, n_features, n_features))
            for j in range(gkfcm.n_clusters):
                try:
                    gkfcm.inv_covariance_matrices[j] = np.linalg.inv(gkfcm.covariance_matrices[j])
                except np.linalg.LinAlgError:
                    gkfcm.inv_covariance_matrices[j] = np.linalg.pinv(gkfcm.covariance_matrices[j])
            
            gkfcm.norm_matrices = gkfcm.calculate_norm_matrices(gkfcm.covariance_matrices, data_2d.shape[1])
    
    results = {
        "fcm_avg_silhouette": np.mean(fcm_silhouette_2d),
        "fcm_avg_fuzzy_inertia": np.mean(fcm_fuzzy_inertia),
        "fcm_avg_time": np.mean(fcm_time_2d),
        "gkfcm_avg_silhouette": np.mean(gkfcm_silhouette_2d),
        "gkfcm_avg_fuzzy_inertia": np.mean(gkfcm_fuzzy_inertia),
        "gkfcm_avg_time": np.mean(gkfcm_time_2d),
        "final_run": final_results
    }
    
    print("\nFCM vs GK-FCM Results Summary:")
    print(f"  Common Comparison Metric (Silhouette Score):")
    print(f"    FCM - Avg Silhouette: {results['fcm_avg_silhouette']:.4f}")
    print(f"    GK-FCM - Avg Silhouette: {results['gkfcm_avg_silhouette']:.4f}")
    print(f"  Algorithm-Specific Metrics:")
    print(f"    FCM - Avg Fuzzy Inertia: {results['fcm_avg_fuzzy_inertia']:.4f}")
    print(f"    GK-FCM - Avg Fuzzy Inertia: {results['gkfcm_avg_fuzzy_inertia']:.4f}")
    print(f"  Performance:")
    print(f"    FCM - Avg Time: {results['fcm_avg_time']:.4f}s")
    print(f"    GK-FCM - Avg Time: {results['gkfcm_avg_time']:.4f}s")
    
    return results, random_seeds[:n_runs]

def compare_fcm_gkfcm(data_2d, fcm_labels, gkfcm_labels, fcm_centroids, gkfcm_centroids, scaler=None):
    """
    Create visualizations to compare FCM and GK-FCM clustering results.
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    fcm_labels : array-like
        Cluster labels from FCM
    gkfcm_labels : array-like
        Cluster labels from GK-FCM
    fcm_centroids : array-like
        Centroids from FCM
    gkfcm_centroids : array-like
        Centroids from GK-FCM
    scaler : object, default=None
        Scaler used to normalize data
    """
    compare_kmeans_fcm(
        data_2d, fcm_labels, gkfcm_labels, fcm_centroids, gkfcm_centroids,
        scaler, title_left="FCM", title_right="GK-FCM", filename="comparison_fcm_gkfcm_2d.png"
    )
    
    fcm_silhouette = silhouette_score(data_2d, fcm_labels)
    gkfcm_silhouette = silhouette_score(data_2d, gkfcm_labels)
    
    fcm = FuzzyCMeansClustering(n_clusters=len(np.unique(fcm_labels)))
    fcm.centroids = fcm_centroids
    fcm.membership = np.zeros((len(data_2d), fcm.n_clusters))
    for i, label in enumerate(fcm_labels):
        fcm.membership[i, label] = 1.0
    fcm_fuzzy_inertia = fcm._calculate_inertia(data_2d)
    
    gkfcm = GKFuzzyCMeansClustering(n_clusters=len(np.unique(gkfcm_labels)))
    gkfcm.centroids = gkfcm_centroids
    gkfcm.membership = np.zeros((len(data_2d), gkfcm.n_clusters))
    for i, label in enumerate(gkfcm_labels):
        gkfcm.membership[i, label] = 1.0
    
    if gkfcm.centroids.shape[1] != data_2d.shape[1]:
        gkfcm.centroids = gkfcm.centroids[:, :data_2d.shape[1]]
        
    gkfcm.covariance_matrices = gkfcm.update_covariance_matrices(data_2d, gkfcm.centroids, gkfcm.membership)
    
    n_features = data_2d.shape[1]
    gkfcm.inv_covariance_matrices = np.zeros((gkfcm.n_clusters, n_features, n_features))
    for j in range(gkfcm.n_clusters):
        try:
            gkfcm.inv_covariance_matrices[j] = np.linalg.inv(gkfcm.covariance_matrices[j])
        except np.linalg.LinAlgError:
            gkfcm.inv_covariance_matrices[j] = np.linalg.pinv(gkfcm.covariance_matrices[j])
    
    gkfcm.norm_matrices = gkfcm.calculate_norm_matrices(gkfcm.covariance_matrices, data_2d.shape[1])
    
    gkfcm_fuzzy_inertia = gkfcm._calculate_inertia(data_2d)
    
    plot_algorithm_metrics_comparison(
        fcm_fuzzy_inertia, 
        gkfcm_fuzzy_inertia,
        algorithms=['FCM', 'GK-FCM'],
        inertia1=fcm_silhouette,
        inertia2=gkfcm_silhouette,
        filename="fcm_gkfcm_metrics_comparison.png",
        title="FCM vs GK-FCM: Algorithm-Specific Objective Functions"
    )

def run_all_fuzzy_comparison(data_2d, data_3d, n_clusters=4, n_runs=10, random_seeds=None):
    """
    Run comparison between all fuzzy clustering algorithms: FCM, GK-FCM, KFCM, MKFCM, rseKFCM, spKFCM, oKFCM.
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    data_3d : array-like
        3D dataset (Age, Annual Income, Spending Score)
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
    print(f"Running comparison of all fuzzy clustering algorithms ({n_runs} runs each)...")
    
    if random_seeds is None:
        random_seeds = np.random.randint(0, 10000, size=n_runs).tolist()
    if len(random_seeds) < n_runs:
        additional_seeds = np.random.randint(0, 10000, size=(n_runs - len(random_seeds))).tolist()
        random_seeds.extend(additional_seeds)
    
    fcm_silhouette = []
    gkfcm_silhouette = []
    kfcm_silhouette = []
    mkfcm_silhouette = []
    rsekfcm_silhouette = []
    spkfcm_silhouette = []
    okfcm_silhouette = []
    
    fcm_fuzzy_inertia = []
    gkfcm_fuzzy_inertia = []
    kfcm_fuzzy_inertia = []
    mkfcm_fuzzy_inertia = []
    rsekfcm_fuzzy_inertia = []
    spkfcm_fuzzy_inertia = []
    okfcm_fuzzy_inertia = []
    
    fcm_time = []
    gkfcm_time = []
    kfcm_time = []
    mkfcm_time = []
    rsekfcm_time = []
    spkfcm_time = []
    okfcm_time = []
    
    all_run_results = []
    final_results = {}
    
    def get_centroids_safe(clustering_obj):
        """Helper function to safely retrieve centroids from a clustering object."""
        if hasattr(clustering_obj, 'get_centroids') and callable(getattr(clustering_obj, 'get_centroids')):
            return clustering_obj.get_centroids()
        elif hasattr(clustering_obj, 'centroids'):
            return clustering_obj.centroids
        else:
            raise AttributeError(f"{clustering_obj.__class__.__name__} has neither 'get_centroids()' method nor 'centroids' attribute")

    def get_fitness_history_safe(clustering_obj):
        """Helper function to safely retrieve fitness history from a clustering object."""
        if hasattr(clustering_obj, 'get_fitness_history') and callable(getattr(clustering_obj, 'get_fitness_history')):
            return clustering_obj.get_fitness_history()
        elif hasattr(clustering_obj, 'fitness_history'):
            return clustering_obj.fitness_history
        else:
            print(f"    Warning: {clustering_obj.__class__.__name__} has neither 'get_fitness_history()' method nor 'fitness_history' attribute. Skipping fitness history.")
            return []

    for i, seed in enumerate(random_seeds[:n_runs]):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        # Run FCM
        start_time = time.time()
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed)
        fcm_labels = fcm.fit(data_2d)
        fcm_exec_time = time.time() - start_time
        fcm_time.append(fcm_exec_time)
        
        if len(np.unique(fcm_labels)) > 1:
            fcm_silh = silhouette_score(data_2d, fcm_labels)
        else:
            print(f"    Warning: FCM produced only {len(np.unique(fcm_labels))} unique label(s). Skipping silhouette score.")
            fcm_silh = np.nan
        fcm_silhouette.append(fcm_silh)
        fcm_inertia = fcm.inertia_
        fcm_fuzzy_inertia.append(fcm_inertia)
        
        # Run GK-FCM
        start_time = time.time()
        gkfcm = GKFuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed)
        gkfcm_labels = gkfcm.fit(data_2d)
        gkfcm_exec_time = time.time() - start_time
        gkfcm_time.append(gkfcm_exec_time)
        
        if len(np.unique(gkfcm_labels)) > 1:
            gkfcm_silh = silhouette_score(data_2d, gkfcm_labels)
        else:
            print(f"    Warning: GK-FCM produced only {len(np.unique(gkfcm_labels))} unique label(s). Skipping silhouette score.")
            gkfcm_silh = np.nan
        gkfcm_silhouette.append(gkfcm_silh)
        gkfcm_inertia = gkfcm.inertia_
        gkfcm_fuzzy_inertia.append(gkfcm_inertia)
        
        # Run KFCM
        start_time = time.time()
        kfcm = KernelFuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed, sigma_squared=1.0)
        kfcm_labels = kfcm.fit(data_2d)
        kfcm_exec_time = time.time() - start_time
        kfcm_time.append(kfcm_exec_time)
        
        if len(np.unique(kfcm_labels)) > 1:
            kfcm_silh = silhouette_score(data_2d, kfcm_labels)
        else:
            print(f"    Warning: KFCM produced only {len(np.unique(kfcm_labels))} unique label(s). Skipping silhouette score.")
            kfcm_silh = np.nan
        kfcm_silhouette.append(kfcm_silh)
        kfcm_inertia = kfcm.inertia_
        kfcm_fuzzy_inertia.append(kfcm_inertia)
        
        # Run MKFCM
        start_time = time.time()
        mkfcm = ModifiedKernelFuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, random_state=seed, sigma_squared=1.0)
        mkfcm_labels = mkfcm.fit(data_2d)
        mkfcm_exec_time = time.time() - start_time
        mkfcm_time.append(mkfcm_exec_time)
        
        if len(np.unique(mkfcm_labels)) > 1:
            mkfcm_silh = silhouette_score(data_2d, mkfcm_labels)
        else:
            print(f"    Warning: MKFCM produced only {len(np.unique(mkfcm_labels))} unique label(s). Skipping silhouette score.")
            mkfcm_silh = np.nan
        mkfcm_silhouette.append(mkfcm_silh)
        mkfcm_inertia = mkfcm.inertia_
        mkfcm_fuzzy_inertia.append(mkfcm_inertia)
        
        # Run rseKFCM
        start_time = time.time()
        rsekfcm = RseKFCMClustering(n_clusters=n_clusters, m=2.5, random_state=seed, sample_size=50)
        rsekfcm_labels = rsekfcm.fit(data_2d)
        rsekfcm_exec_time = time.time() - start_time
        rsekfcm_time.append(rsekfcm_exec_time)
        
        if len(np.unique(rsekfcm_labels)) > 1:
            rsekfcm_silh = silhouette_score(data_2d, rsekfcm_labels)
        else:
            print(f"    Warning: rseKFCM produced only {len(np.unique(rsekfcm_labels))} unique label(s). Skipping silhouette score.")
            rsekfcm_silh = np.nan
        rsekfcm_silhouette.append(rsekfcm_silh)
        rsekfcm_inertia = rsekfcm.inertia_
        rsekfcm_fuzzy_inertia.append(rsekfcm_inertia)
        
        # Run spKFCM with adjusted parameters
        start_time = time.time()
        spkfcm = SpKFCMClustering(n_clusters=n_clusters,m=2.5 , random_state=seed, n_chunks=10)
        spkfcm_labels = spkfcm.fit(data_2d)
        spkfcm_exec_time = time.time() - start_time
        spkfcm_time.append(spkfcm_exec_time)
        
        if len(np.unique(spkfcm_labels)) > 1:
            spkfcm_silh = silhouette_score(data_2d, spkfcm_labels)
        else:
            print(f"    Warning: spKFCM produced only {len(np.unique(spkfcm_labels))} unique label(s). Skipping silhouette score.")
            spkfcm_silh = np.nan
        spkfcm_silhouette.append(spkfcm_silh)
        spkfcm_inertia = spkfcm.inertia_
        spkfcm_fuzzy_inertia.append(spkfcm_inertia)
        
        # Run oKFCM with adjusted parameters
        start_time = time.time()
        okfcm = OKFCMClustering(n_clusters=n_clusters, m=2.5, random_state=seed, n_chunks=10)
        okfcm_labels = okfcm.fit(data_2d)
        okfcm_exec_time = time.time() - start_time
        okfcm_time.append(okfcm_exec_time)
        
        if len(np.unique(okfcm_labels)) > 1:
            okfcm_silh = silhouette_score(data_2d, okfcm_labels)
        else:
            print(f"    Warning: oKFCM produced only {len(np.unique(okfcm_labels))} unique label(s). Skipping silhouette score.")
            okfcm_silh = np.nan
        okfcm_silhouette.append(okfcm_silh)
        okfcm_inertia = okfcm.inertia_
        okfcm_fuzzy_inertia.append(okfcm_inertia)
        
        run_result = {
            "run": i+1,
            "seed": seed,
            "fcm_silhouette": fcm_silh,
            "fcm_fuzzy_inertia": fcm_inertia,
            "fcm_time": fcm_exec_time,
            "gkfcm_silhouette": gkfcm_silh,
            "gkfcm_fuzzy_inertia": gkfcm_inertia,
            "gkfcm_time": gkfcm_exec_time,
            "kfcm_silhouette": kfcm_silh,
            "kfcm_fuzzy_inertia": kfcm_inertia,
            "kfcm_time": kfcm_exec_time,
            "mkfcm_silhouette": mkfcm_silh, 
            "mkfcm_fuzzy_inertia": mkfcm_inertia,
            "mkfcm_time": mkfcm_exec_time,
            "rsekfcm_silhouette": rsekfcm_silh,
            "rsekfcm_fuzzy_inertia": rsekfcm_inertia,
            "rsekfcm_time": rsekfcm_exec_time,
            "spkfcm_silhouette": spkfcm_silh,
            "spkfcm_fuzzy_inertia": spkfcm_inertia,
            "spkfcm_time": spkfcm_exec_time,
            "okfcm_silhouette": okfcm_silh,
            "okfcm_fuzzy_inertia": okfcm_inertia,
            "okfcm_time": okfcm_exec_time
        }
        all_run_results.append(run_result)
        
        if i == n_runs - 1:
            final_results["fcm_labels"] = fcm_labels
            final_results["fcm_centroids"] = get_centroids_safe(fcm)
            final_results["gkfcm_labels"] = gkfcm_labels
            final_results["gkfcm_centroids"] = get_centroids_safe(gkfcm)
            final_results["kfcm_labels"] = kfcm_labels
            final_results["kfcm_centroids"] = get_centroids_safe(kfcm)
            final_results["mkfcm_labels"] = mkfcm_labels
            final_results["mkfcm_centroids"] = get_centroids_safe(mkfcm)
            final_results["rsekfcm_labels"] = rsekfcm_labels
            final_results["rsekfcm_centroids"] = get_centroids_safe(rsekfcm)
            final_results["spkfcm_labels"] = spkfcm_labels
            final_results["spkfcm_centroids"] = get_centroids_safe(spkfcm)
            final_results["okfcm_labels"] = okfcm_labels
            final_results["okfcm_centroids"] = get_centroids_safe(okfcm)
            
            final_results["fcm_history"] = get_fitness_history_safe(fcm)
            final_results["gkfcm_history"] = get_fitness_history_safe(gkfcm)
            final_results["kfcm_history"] = get_fitness_history_safe(kfcm)
            final_results["mkfcm_history"] = get_fitness_history_safe(mkfcm)
            final_results["rsekfcm_history"] = get_fitness_history_safe(rsekfcm)
            final_results["spkfcm_history"] = get_fitness_history_safe(spkfcm)
            final_results["okfcm_history"] = get_fitness_history_safe(okfcm)
            
            if hasattr(gkfcm, 'get_covariance_matrices'):
                final_results["gkfcm_covariance_matrices"] = gkfcm.get_covariance_matrices()
            else:
                print("    Warning: GKFCM does not have 'get_covariance_matrices' method. Skipping covariance matrices.")
                final_results["gkfcm_covariance_matrices"] = None
    
    # Remove NaN values for averaging (in case some algorithms failed to produce multiple clusters)
    fcm_silhouette = [x for x in fcm_silhouette if not np.isnan(x)]
    gkfcm_silhouette = [x for x in gkfcm_silhouette if not np.isnan(x)]
    kfcm_silhouette = [x for x in kfcm_silhouette if not np.isnan(x)]
    mkfcm_silhouette = [x for x in mkfcm_silhouette if not np.isnan(x)]
    rsekfcm_silhouette = [x for x in rsekfcm_silhouette if not np.isnan(x)]
    spkfcm_silhouette = [x for x in spkfcm_silhouette if not np.isnan(x)]
    okfcm_silhouette = [x for x in okfcm_silhouette if not np.isnan(x)]

    # Compute averages and standard deviations, handling empty lists
    results = {
        "fcm_avg_silhouette": np.mean(fcm_silhouette) if fcm_silhouette else np.nan,
        "fcm_avg_fuzzy_inertia": np.mean(fcm_fuzzy_inertia),
        "fcm_avg_time": np.mean(fcm_time),
        "fcm_std_silhouette": np.std(fcm_silhouette) if fcm_silhouette else np.nan,
        "fcm_std_fuzzy_inertia": np.std(fcm_fuzzy_inertia),
        "fcm_std_time": np.std(fcm_time),
        
        "gkfcm_avg_silhouette": np.mean(gkfcm_silhouette) if gkfcm_silhouette else np.nan,
        "gkfcm_avg_fuzzy_inertia": np.mean(gkfcm_fuzzy_inertia),
        "gkfcm_avg_time": np.mean(gkfcm_time),
        "gkfcm_std_silhouette": np.std(gkfcm_silhouette) if gkfcm_silhouette else np.nan,
        "gkfcm_std_fuzzy_inertia": np.std(gkfcm_fuzzy_inertia),
        "gkfcm_std_time": np.std(gkfcm_time),
        
        "kfcm_avg_silhouette": np.mean(kfcm_silhouette) if kfcm_silhouette else np.nan,
        "kfcm_avg_fuzzy_inertia": np.mean(kfcm_fuzzy_inertia),
        "kfcm_avg_time": np.mean(kfcm_time),
        "kfcm_std_silhouette": np.std(kfcm_silhouette) if kfcm_silhouette else np.nan,
        "kfcm_std_fuzzy_inertia": np.std(kfcm_fuzzy_inertia),
        "kfcm_std_time": np.std(kfcm_time),
        
        "mkfcm_avg_silhouette": np.mean(mkfcm_silhouette) if mkfcm_silhouette else np.nan,
        "mkfcm_avg_fuzzy_inertia": np.mean(mkfcm_fuzzy_inertia),
        "mkfcm_avg_time": np.mean(mkfcm_time),
        "mkfcm_std_silhouette": np.std(mkfcm_silhouette) if mkfcm_silhouette else np.nan,
        "mkfcm_std_fuzzy_inertia": np.std(mkfcm_fuzzy_inertia),
        "mkfcm_std_time": np.std(mkfcm_time),
        
        "rsekfcm_avg_silhouette": np.mean(rsekfcm_silhouette) if rsekfcm_silhouette else np.nan,
        "rsekfcm_avg_fuzzy_inertia": np.mean(rsekfcm_fuzzy_inertia),
        "rsekfcm_avg_time": np.mean(rsekfcm_time),
        "rsekfcm_std_silhouette": np.std(rsekfcm_silhouette) if rsekfcm_silhouette else np.nan,
        "rsekfcm_std_fuzzy_inertia": np.std(rsekfcm_fuzzy_inertia),
        "rsekfcm_std_time": np.std(rsekfcm_time),
        
        "spkfcm_avg_silhouette": np.mean(spkfcm_silhouette) if spkfcm_silhouette else np.nan,
        "spkfcm_avg_fuzzy_inertia": np.mean(spkfcm_fuzzy_inertia),
        "spkfcm_avg_time": np.mean(spkfcm_time),
        "spkfcm_std_silhouette": np.std(spkfcm_silhouette) if spkfcm_silhouette else np.nan,
        "spkfcm_std_fuzzy_inertia": np.std(spkfcm_fuzzy_inertia),
        "spkfcm_std_time": np.std(spkfcm_time),
        
        "okfcm_avg_silhouette": np.mean(okfcm_silhouette) if okfcm_silhouette else np.nan,
        "okfcm_avg_fuzzy_inertia": np.mean(okfcm_fuzzy_inertia),
        "okfcm_avg_time": np.mean(okfcm_time),
        "okfcm_std_silhouette": np.std(okfcm_silhouette) if okfcm_silhouette else np.nan,
        "okfcm_std_fuzzy_inertia": np.std(okfcm_fuzzy_inertia),
        "okfcm_std_time": np.std(okfcm_time),
        
        "all_run_results": all_run_results,
        "final_run": final_results
    }
    
    print("\nFuzzy Clustering Algorithms Comparison Results Summary:")
    print(f"  Common Comparison Metric (Silhouette Score):")
    print(f"    FCM     - Avg Silhouette: {results['fcm_avg_silhouette']:.4f} ± {results['fcm_std_silhouette']:.4f}")
    print(f"    GK-FCM  - Avg Silhouette: {results['gkfcm_avg_silhouette']:.4f} ± {results['gkfcm_std_silhouette']:.4f}")
    print(f"    KFCM    - Avg Silhouette: {results['kfcm_avg_silhouette']:.4f} ± {results['kfcm_std_silhouette']:.4f}")
    print(f"    MKFCM   - Avg Silhouette: {results['mkfcm_avg_silhouette']:.4f} ± {results['mkfcm_std_silhouette']:.4f}")
    print(f"    rseKFCM - Avg Silhouette: {results['rsekfcm_avg_silhouette']:.4f} ± {results['rsekfcm_std_silhouette']:.4f}")
    print(f"    spKFCM  - Avg Silhouette: {results['spkfcm_avg_silhouette']:.4f} ± {results['spkfcm_std_silhouette']:.4f}")
    print(f"    oKFCM   - Avg Silhouette: {results['okfcm_avg_silhouette']:.4f} ± {results['okfcm_std_silhouette']:.4f}")
    
    print(f"  Algorithm-Specific Metrics (not directly comparable):")
    print(f"    FCM     - Avg Fuzzy Inertia: {results['fcm_avg_fuzzy_inertia']:.4f} ± {results['fcm_std_fuzzy_inertia']:.4f}")
    print(f"    GK-FCM  - Avg Fuzzy Inertia: {results['gkfcm_avg_fuzzy_inertia']:.4f} ± {results['gkfcm_std_fuzzy_inertia']:.4f}")
    print(f"    KFCM    - Avg Fuzzy Inertia: {results['kfcm_avg_fuzzy_inertia']:.4f} ± {results['kfcm_std_fuzzy_inertia']:.4f}")
    print(f"    MKFCM   - Avg Fuzzy Inertia: {results['mkfcm_avg_fuzzy_inertia']:.4f} ± {results['mkfcm_std_fuzzy_inertia']:.4f}")
    print(f"    rseKFCM - Avg Fuzzy Inertia: {results['rsekfcm_avg_fuzzy_inertia']:.4f} ± {results['rsekfcm_std_fuzzy_inertia']:.4f}")
    print(f"    spKFCM  - Avg Fuzzy Inertia: {results['spkfcm_avg_fuzzy_inertia']:.4f} ± {results['spkfcm_std_fuzzy_inertia']:.4f}")
    print(f"    oKFCM   - Avg Fuzzy Inertia: {results['okfcm_avg_fuzzy_inertia']:.4f} ± {results['okfcm_std_fuzzy_inertia']:.4f}")
    
    print(f"  Performance:")
    print(f"    FCM     - Avg Time: {results['fcm_avg_time']:.4f}s ± {results['fcm_std_time']:.4f}s")
    print(f"    GK-FCM  - Avg Time: {results['gkfcm_avg_time']:.4f}s ± {results['gkfcm_std_time']:.4f}s")
    print(f"    KFCM    - Avg Time: {results['kfcm_avg_time']:.4f}s ± {results['kfcm_std_time']:.4f}s")
    print(f"    MKFCM   - Avg Time: {results['mkfcm_avg_time']:.4f}s ± {results['mkfcm_std_time']:.4f}s")
    print(f"    rseKFCM - Avg Time: {results['rsekfcm_avg_time']:.4f}s ± {results['rsekfcm_std_time']:.4f}s")
    print(f"    spKFCM  - Avg Time: {results['spkfcm_avg_time']:.4f}s ± {results['spkfcm_std_time']:.4f}s")
    print(f"    oKFCM   - Avg Time: {results['okfcm_avg_time']:.4f}s ± {results['okfcm_std_time']:.4f}s")
    
    return results, random_seeds[:n_runs]

def run_kernel_sigma_comparison(data_2d, n_clusters=4, sigma_values=None, n_runs=5, random_seeds=None):
    """
    Compare KFCM and MKFCM with different sigma_squared values.
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    n_clusters : int, default=4
        Number of clusters
    sigma_values : list or None, default=None
        List of sigma_squared values to compare
    n_runs : int, default=5
        Number of runs for each sigma value
    random_seeds : list or None, default=None
        List of random seeds for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with experiment results
    """
    if sigma_values is None:
        sigma_values = [0.1, 0.5, 2.0, 5.0]
    
    print(f"Running kernel sigma_squared comparison for values: {sigma_values}...")
    
    if random_seeds is None:
        random_seeds = np.random.randint(0, 10000, size=n_runs).tolist()
    if len(random_seeds) < n_runs:
        additional_seeds = np.random.randint(0, 10000, size=(n_runs - len(random_seeds))).tolist()
        random_seeds.extend(additional_seeds)
    
    results = {}
    kfcm_results = []
    mkfcm_results = []
    kfcm_all_runs = []
    mkfcm_all_runs = []
    
    for sigma in sigma_values:
        print(f"  Testing KFCM with sigma_squared={sigma} ({n_runs} runs)")
        
        sigma_silhouette_scores = []
        sigma_inertia_values = []
        sigma_times = []
        
        for i, seed in enumerate(random_seeds[:n_runs]):
            print(f"    Run {i+1}/{n_runs} (seed: {seed})")
            
            start_time = time.time()
            kfcm = KernelFuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, 
                                              random_state=seed, sigma_squared=sigma)
            labels = kfcm.fit(data_2d)
            execution_time = time.time() - start_time
            
            silhouette = silhouette_score(data_2d, labels)
            inertia = kfcm.inertia_
            
            sigma_silhouette_scores.append(silhouette)
            sigma_inertia_values.append(inertia)
            sigma_times.append(execution_time)
            
            run_result = {
                "sigma_squared": sigma,
                "run": i+1,
                "seed": seed,
                "silhouette": silhouette,
                "inertia": inertia,
                "time": execution_time
            }
            kfcm_all_runs.append(run_result)
            
            if i == n_runs - 1:
                last_run = {
                    "sigma_squared": sigma,
                    "labels": labels,
                    "centroids": kfcm.get_centroids(),
                    "silhouette": silhouette,
                    "inertia": inertia,
                    "history": kfcm.get_fitness_history(),
                    "time": execution_time
                }
                kfcm_results.append(last_run)
        
        avg_silhouette = np.mean(sigma_silhouette_scores)
        std_silhouette = np.std(sigma_silhouette_scores)
        avg_inertia = np.mean(sigma_inertia_values)
        std_inertia = np.std(sigma_inertia_values)
        avg_time = np.mean(sigma_times)
        std_time = np.std(sigma_times)
        
        print(f"    Avg Silhouette: {avg_silhouette:.4f} ± {std_silhouette:.4f}")
        print(f"    Avg Kernel Inertia: {avg_inertia:.4f} ± {std_inertia:.4f}")
        print(f"    Avg Time: {avg_time:.4f}s ± {std_time:.4f}s")
        
        print(f"  Testing MKFCM with sigma_squared={sigma} ({n_runs} runs)")
        
        sigma_silhouette_scores = []
        sigma_inertia_values = []
        sigma_times = []
        
        for i, seed in enumerate(random_seeds[:n_runs]):
            print(f"    Run {i+1}/{n_runs} (seed: {seed})")
            
            start_time = time.time()
            mkfcm = ModifiedKernelFuzzyCMeansClustering(n_clusters=n_clusters, m=2.5, 
                                                      random_state=seed, sigma_squared=sigma)
            labels = mkfcm.fit(data_2d)
            execution_time = time.time() - start_time
            
            silhouette = silhouette_score(data_2d, labels)
            inertia = mkfcm.inertia_
            
            sigma_silhouette_scores.append(silhouette)
            sigma_inertia_values.append(inertia)
            sigma_times.append(execution_time)
            
            run_result = {
                "sigma_squared": sigma,
                "run": i+1,
                "seed": seed,
                "silhouette": silhouette,
                "inertia": inertia,
                "time": execution_time
            }
            mkfcm_all_runs.append(run_result)
            
            if i == n_runs - 1:
                last_run = {
                    "sigma_squared": sigma,
                    "labels": labels,
                    "centroids": mkfcm.get_centroids(),
                    "silhouette": silhouette,
                    "inertia": inertia,
                    "history": mkfcm.get_fitness_history(),
                    "time": execution_time
                }
                mkfcm_results.append(last_run)
        
        avg_silhouette = np.mean(sigma_silhouette_scores)
        std_silhouette = np.std(sigma_silhouette_scores)
        avg_inertia = np.mean(sigma_inertia_values)
        std_inertia = np.std(sigma_inertia_values)
        avg_time = np.mean(sigma_times)
        std_time = np.std(sigma_times)
        
        print(f"    Avg Silhouette: {avg_silhouette:.4f} ± {std_silhouette:.4f}")
        print(f"    Avg Kernel Inertia: {avg_inertia:.4f} ± {std_inertia:.4f}")
        print(f"    Avg Time: {avg_time:.4f}s ± {std_time:.4f}s")
    
    results["kfcm_results"] = kfcm_results
    results["mkfcm_results"] = mkfcm_results
    results["kfcm_all_runs"] = kfcm_all_runs
    results["mkfcm_all_runs"] = mkfcm_all_runs
    return results

def run_incremental_params_comparison(data_2d, n_clusters=4, n_runs=5, random_seeds=None):
    """
    Run comparison of incremental parameters for rseKFCM, spKFCM, and oKFCM.
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    n_clusters : int, default=4
        Number of clusters
    n_runs : int, default=5
        Number of runs (for statistical significance)
    random_seeds : list or None, default=None
        List of random seeds for reproducibility
        
    Returns:
    --------
    results : dict
        Dictionary with experiment results
    """
    print("Running incremental parameter comparison (sample_size: [50, 100, 200], n_chunks: [5, 10, 20])...")
    
    if random_seeds is None:
        random_seeds = np.random.randint(0, 10000, size=n_runs).tolist()
    if len(random_seeds) < n_runs:
        additional_seeds = np.random.randint(0, 10000, size=(n_runs - len(random_seeds))).tolist()
        random_seeds.extend(additional_seeds)
    
    sample_sizes = [20, 30, 50]
    n_chunks_list = [2, 3]
    
    rsekfcm_results = []
    spkfcm_results = []
    okfcm_results = []
    
    def get_centroids_safe(clustering_obj):
        """Helper function to safely retrieve centroids from a clustering object."""
        if hasattr(clustering_obj, 'get_centroids') and callable(getattr(clustering_obj, 'get_centroids')):
            return clustering_obj.get_centroids()
        elif hasattr(clustering_obj, 'centroids'):
            return clustering_obj.centroids
        else:
            raise AttributeError(f"{clustering_obj.__class__.__name__} has neither 'get_centroids()' method nor 'centroids' attribute")

    def get_fitness_history_safe(clustering_obj):
        """Helper function to safely retrieve fitness history from a clustering object."""
        if hasattr(clustering_obj, 'get_fitness_history') and callable(getattr(clustering_obj, 'get_fitness_history')):
            return clustering_obj.get_fitness_history()
        elif hasattr(clustering_obj, 'fitness_history'):
            return clustering_obj.fitness_history
        else:
            print(f"    Warning: {clustering_obj.__class__.__name__} has neither 'get_fitness_history()' method nor 'fitness_history' attribute. Skipping fitness history.")
            return []
    
    # Test rseKFCM with varying sample sizes
    for sample_size in sample_sizes:
        print(f"  Testing rseKFCM with sample_size={sample_size} ({n_runs} runs)")
        silhouettes = []
        inertias = []
        times = []
        for i, seed in enumerate(random_seeds[:n_runs]):
            print(f"    Run {i+1}/{n_runs} (seed: {seed})")
            start_time = time.time()
            rsekfcm = RseKFCMClustering(n_clusters=n_clusters, m=2.5, random_state=seed, sample_size=sample_size)
            labels = rsekfcm.fit(data_2d)
            exec_time = time.time() - start_time
            times.append(exec_time)
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data_2d, labels)
            else:
                print(f"    Warning: rseKFCM produced only {len(np.unique(labels))} unique label(s). Skipping silhouette score.")
                silhouette = np.nan
            silhouettes.append(silhouette)
            inertia = rsekfcm.inertia_
            inertias.append(inertia)
            
            result = {
                "run": i + 1,
                "seed": seed,
                "param_value": sample_size,
                "labels": labels,
                "centroids": get_centroids_safe(rsekfcm),
                "fitness_history": get_fitness_history_safe(rsekfcm),
                "silhouette": silhouette,
                "inertia": inertia,
                "time": exec_time
            }
            if i == n_runs - 1:
                rsekfcm_results.append(result)
    
    # Test spKFCM with varying number of chunks
    for n_chunks in n_chunks_list:
        print(f"  Testing spKFCM with n_chunks={n_chunks} ({n_runs} runs)")
        silhouettes = []
        inertias = []
        times = []
        for i, seed in enumerate(random_seeds[:n_runs]):
            print(f"    Run {i+1}/{n_runs} (seed: {seed})")
            start_time = time.time()
            spkfcm = SpKFCMClustering(n_clusters=n_clusters, m=2.5, random_state=seed, n_chunks=n_chunks)
            labels = spkfcm.fit(data_2d)
            exec_time = time.time() - start_time
            times.append(exec_time)
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data_2d, labels)
            else:
                print(f"    Warning: spKFCM produced only {len(np.unique(labels))} unique label(s). Skipping silhouette score.")
                silhouette = np.nan
            silhouettes.append(silhouette)
            inertia = spkfcm.inertia_
            inertias.append(inertia)
            
            result = {
                "run": i + 1,
                "seed": seed,
                "param_value": n_chunks,
                "labels": labels,
                "centroids": get_centroids_safe(spkfcm),
                "fitness_history": get_fitness_history_safe(spkfcm),
                "silhouette": silhouette,
                "inertia": inertia,
                "time": exec_time
            }
            if i == n_runs - 1:
                spkfcm_results.append(result)
    
    # Test oKFCM with varying number of chunks
    for n_chunks in n_chunks_list:
        print(f"  Testing oKFCM with n_chunks={n_chunks} ({n_runs} runs)")
        silhouettes = []
        inertias = []
        times = []
        for i, seed in enumerate(random_seeds[:n_runs]):
            print(f"    Run {i+1}/{n_runs} (seed: {seed})")
            start_time = time.time()
            okfcm = OKFCMClustering(n_clusters=n_clusters, m=2.5, random_state=seed, n_chunks=n_chunks)
            labels = okfcm.fit(data_2d)
            exec_time = time.time() - start_time
            times.append(exec_time)
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(data_2d, labels)
            else:
                print(f"    Warning: oKFCM produced only {len(np.unique(labels))} unique label(s). Skipping silhouette score.")
                silhouette = np.nan
            silhouettes.append(silhouette)
            inertia = okfcm.inertia_
            inertias.append(inertia)
            
            result = {
                "run": i + 1,
                "seed": seed,
                "param_value": n_chunks,
                "labels": labels,
                "centroids": get_centroids_safe(okfcm),
                "fitness_history": get_fitness_history_safe(okfcm),
                "silhouette": silhouette,
                "inertia": inertia,
                "time": exec_time
            }
            if i == n_runs - 1:
                okfcm_results.append(result)
    
    results = {
        "rsekfcm_results": rsekfcm_results,
        "spkfcm_results": spkfcm_results,
        "okfcm_results": okfcm_results
    }
    
    return results

def main():
    main_seed = 42
    np.random.seed(main_seed)
    
    experiment_seeds = load_experiment_seeds()

    print("Loading Mall Customers dataset...")
    df, features_2d, features_3d, scaler_2d, scaler_3d = load_mall_customers_data()
    
    # Visualize the data distribution
    plot_data_distribution(features_2d)

    print("Plotting elbow curve for optimal k...")
    optimal_k = plot_elbow_curve(features_2d, max_k=10)
    print(f"Optimal k determined by Elbow method: {optimal_k}")
    
    # [MODIFIED] Override optimal_k to 5 as suggested
    optimal_k = 5
    print(f"Manually set number of clusters: {optimal_k}")
    
    # Load seeds for all experiments
    kmeans_fcm_seeds = experiment_seeds.get("kmeans_fcm_comparison", [])
    kmeans_init_seeds = experiment_seeds.get("kmeans_init_comparison", [])
    fcm_gkfcm_seeds = experiment_seeds.get("fcm_gkfcm_comparison", [])
    all_fuzzy_seeds = experiment_seeds.get("all_fuzzy_comparison", [])
    kernel_sigma_seeds = experiment_seeds.get("kernel_sigma_comparison", [])
    incremental_params_seeds = experiment_seeds.get("incremental_params_comparison", [])

    kmeans_fcm_results, kmeans_fcm_seeds = run_kmeans_fcm_comparison(
        features_2d, features_3d, n_clusters=optimal_k, n_runs=30, random_seeds=kmeans_fcm_seeds
    )
    
    fcm_m_results = run_fcm_m_comparison(
        features_2d, n_clusters=optimal_k, m_values=[1.1, 1.5, 2.0, 2.5, 3.0], random_seed=main_seed
    )
    
    kmeans_init_results, kmeans_init_seeds = run_kmeans_init_comparison(
        features_2d, n_clusters=optimal_k, n_runs=10, random_seeds=kmeans_init_seeds
    )
    
    fcm_gkfcm_results, fcm_gkfcm_seeds = run_fcm_gkfcm_comparison(
        features_2d, features_3d, n_clusters=optimal_k, n_runs=30, random_seeds=fcm_gkfcm_seeds
    )
    
    all_fuzzy_results, all_fuzzy_seeds = run_all_fuzzy_comparison(
        features_2d, features_3d, n_clusters=optimal_k, n_runs=30, random_seeds=all_fuzzy_seeds
    )
    
    kernel_sigma_results = run_kernel_sigma_comparison(
        features_2d, n_clusters=optimal_k, sigma_values=[0.1, 1.0, 10.0, 50.0, 100.0], n_runs=30, random_seeds=kernel_sigma_seeds
    )
    
    incremental_params_results = run_incremental_params_comparison(
        features_2d,
        n_clusters=optimal_k,
        n_runs=30,
        random_seeds=incremental_params_seeds
    )
    
    experiment_seeds["kmeans_fcm_comparison"] = kmeans_fcm_seeds
    experiment_seeds["kmeans_init_comparison"] = kmeans_init_seeds
    experiment_seeds["fcm_gkfcm_comparison"] = fcm_gkfcm_seeds
    experiment_seeds["all_fuzzy_comparison"] = all_fuzzy_seeds
    experiment_seeds["kernel_sigma_comparison"] = kernel_sigma_seeds
    experiment_seeds["incremental_params_comparison"] = incremental_params_seeds
    save_experiment_seeds(experiment_seeds)
    
    print("Generating 2D/3D clustering visualizations...")
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
    
    print("Generating K-means vs FCM comparison plots...")
    compare_kmeans_fcm(
        features_2d,
        kmeans_fcm_results["final_run"]["kmeans_labels_2d"],
        kmeans_fcm_results["final_run"]["fcm_labels_2d"],
        kmeans_fcm_results["final_run"]["kmeans_centroids_2d"],
        kmeans_fcm_results["final_run"]["fcm_centroids_2d"],
        scaler_2d
    )
    
    plot_algorithm_metrics_comparison(
        kmeans_fcm_results["kmeans_avg_inertia"],
        kmeans_fcm_results["fcm_avg_fuzzy_inertia"],
        algorithms=['K-means', 'FCM'],
        filename="algorithm_specific_metrics_comparison.png",
        title="K-means vs FCM: Algorithm-Specific Objective Functions"
    )
    
    print("Generating FCM m-value comparison plots...")
    plot_fcm_m_comparison(
        features_2d,
        [result["m"] for result in fcm_m_results["fcm_results"]],
        [result["labels"] for result in fcm_m_results["fcm_results"]],
        [result["centroids"] for result in fcm_m_results["fcm_results"]],
        [result["silhouette"] for result in fcm_m_results["fcm_results"]],
        [result["wcss"] for result in fcm_m_results["fcm_results"]],
        scaler_2d,
        filename="fcm_m_comparison.png",
        title="FCM: Effect of Fuzziness Parameter (m)"
    )

    print("Generating K-means initialization comparison plots...")
    plot_kmeans_init_comparison(
        features_2d,
        kmeans_init_results["final_run"]["kmeans_pp_labels"],
        kmeans_init_results["final_run"]["kmeans_random_labels"],
        kmeans_init_results["final_run"]["kmeans_pp_centroids"],
        kmeans_init_results["final_run"]["kmeans_random_centroids"],
        scaler_2d,
        filename="kmeans_init_comparison.png",
        title="K-means: k-means++ vs Random Initialization"
    )

    print("Generating FCM vs GK-FCM comparison plots...")
    compare_fcm_gkfcm(
        features_2d,
        fcm_gkfcm_results["final_run"]["fcm_labels_2d"],
        fcm_gkfcm_results["final_run"]["gkfcm_labels_2d"],
        fcm_gkfcm_results["final_run"]["fcm_centroids_2d"],
        fcm_gkfcm_results["final_run"]["gkfcm_centroids_2d"],
        scaler_2d
    )

    print("Generating convergence curves for FCM and GK-FCM...")
    plot_convergence_curves(
        {"FCM": fcm_gkfcm_results["final_run"]["fcm_history"],
         "GK-FCM": fcm_gkfcm_results["final_run"]["gkfcm_history"]},
        filename="convergence_fcm_gkfcm.png",
        title="Convergence Curves: FCM vs GK-FCM"
    )

    print("Generating comparison plots for all fuzzy clustering algorithms...")
    compare_all_fuzzy(
        features_2d,
        {
            "FCM": all_fuzzy_results["final_run"]["fcm_labels"],
            "GK-FCM": all_fuzzy_results["final_run"]["gkfcm_labels"],
            "KFCM": all_fuzzy_results["final_run"]["kfcm_labels"],
            "MKFCM": all_fuzzy_results["final_run"]["mkfcm_labels"],
            "rseKFCM": all_fuzzy_results["final_run"]["rsekfcm_labels"],
            "spKFCM": all_fuzzy_results["final_run"]["spkfcm_labels"],
            "oKFCM": all_fuzzy_results["final_run"]["okfcm_labels"]
        },
        {
            "FCM": all_fuzzy_results["final_run"]["fcm_centroids"],
            "GK-FCM": all_fuzzy_results["final_run"]["gkfcm_centroids"],
            "KFCM": all_fuzzy_results["final_run"]["kfcm_centroids"],
            "MKFCM": all_fuzzy_results["final_run"]["mkfcm_centroids"],
            "rseKFCM": all_fuzzy_results["final_run"]["rsekfcm_centroids"],
            "spKFCM": all_fuzzy_results["final_run"]["spkfcm_centroids"],
            "oKFCM": all_fuzzy_results["final_run"]["okfcm_centroids"]
        },
        scaler_2d,
        filename="all_fuzzy_comparison.png",
        title="Comparison of All Fuzzy Clustering Algorithms"
    )

    print("Generating metrics comparison for all fuzzy clustering algorithms...")
    compare_fuzzy_metrics_with_error_bars(
        [all_fuzzy_results["fcm_avg_silhouette"], all_fuzzy_results["gkfcm_avg_silhouette"],
         all_fuzzy_results["kfcm_avg_silhouette"], all_fuzzy_results["mkfcm_avg_silhouette"],
         all_fuzzy_results["rsekfcm_avg_silhouette"], all_fuzzy_results["spkfcm_avg_silhouette"],
         all_fuzzy_results["okfcm_avg_silhouette"]],
        [all_fuzzy_results["fcm_std_silhouette"], all_fuzzy_results["gkfcm_std_silhouette"],
         all_fuzzy_results["kfcm_std_silhouette"], all_fuzzy_results["mkfcm_std_silhouette"],
         all_fuzzy_results["rsekfcm_std_silhouette"], all_fuzzy_results["spkfcm_std_silhouette"],
         all_fuzzy_results["okfcm_std_silhouette"]],
        [all_fuzzy_results["fcm_avg_time"], all_fuzzy_results["gkfcm_avg_time"],
         all_fuzzy_results["kfcm_avg_time"], all_fuzzy_results["mkfcm_avg_time"],
         all_fuzzy_results["rsekfcm_avg_time"], all_fuzzy_results["spkfcm_avg_time"],
         all_fuzzy_results["okfcm_avg_time"]],
        [all_fuzzy_results["fcm_std_time"], all_fuzzy_results["gkfcm_std_time"],
         all_fuzzy_results["kfcm_std_time"], all_fuzzy_results["mkfcm_std_time"],
         all_fuzzy_results["rsekfcm_std_time"], all_fuzzy_results["spkfcm_std_time"],
         all_fuzzy_results["okfcm_std_time"]],
        algorithms=["FCM", "GK-FCM", "KFCM", "MKFCM", "rseKFCM", "spKFCM", "oKFCM"],
        filename="fuzzy_metrics_comparison.png",
        title="Fuzzy Clustering: Metrics Comparison"
    )

    print("Generating convergence curves for all fuzzy clustering algorithms...")
    plot_convergence_curves(
    {"FCM": fcm_gkfcm_results["final_run"]["fcm_history"],
    "GK-FCM": fcm_gkfcm_results["final_run"]["gkfcm_history"]},
    filename="convergence_fcm_gkfcm.png",  # This is the problematic argument
    title="Convergence Curves: FCM vs GK-FCM"
)

    print("Generating kernel sigma comparison plots...")
    plot_sigma_parameter_study(
        kernel_sigma_results["kfcm_results"],
        kernel_sigma_results["mkfcm_results"],
        sigma_values=[0.1, 1.0, 10.0, 50.0, 100.0],
        filename="kernel_sigma_comparison.png",
        title="KFCM vs MKFCM: Effect of Sigma Parameter"
    )
    
    print("Generating incremental parameters comparison plots...")
    plot_incremental_params_comparison(
        features_2d,
        incremental_params_results["rsekfcm_results"],
        incremental_params_results["spkfcm_results"],
        incremental_params_results["okfcm_results"],
        scaler_2d,
        filename="incremental_params_comparison.png",
        title="Incremental Parameters Comparison"
    )

    print("All experiments and visualizations completed successfully!")

if __name__ == "__main__":
    main()