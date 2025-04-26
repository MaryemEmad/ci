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
from gkfcm_clustering import GKFuzzyCMeansClustering
from kfcm_clustering import KernelFuzzyCMeansClustering
from mkfcm_clustering import ModifiedKernelFuzzyCMeansClustering
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
    plot_sigma_parameter_study
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
    kmeans_inertia_2d = []  # K-means specific metric
    kmeans_time_2d = []
    
    fcm_silhouette_2d = []
    fcm_fuzzy_inertia = []  # FCM specific metric
    fcm_time_2d = []
    
    # Store all individual run results for statistical analysis
    all_run_results = []
    
    # Store final results from last run for visualization
    final_results = {}
    
    # Run experiments
    for i, seed in enumerate(random_seeds):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        # Run K-means
        start_time = time.time()
        kmeans = KMeansClustering(n_clusters=n_clusters, init_method='k-means++', random_state=seed)
        kmeans_labels_2d = kmeans.fit(data_2d)
        kmeans_time = time.time() - start_time
        kmeans_time_2d.append(kmeans_time)
        
        # Calculate K-means metrics
        kmeans_inertia = kmeans.compute_fitness(data_2d)  # K-means inertia
        kmeans_inertia_2d.append(kmeans_inertia)
        kmeans_silhouette = silhouette_score(data_2d, kmeans_labels_2d)
        kmeans_silhouette_2d.append(kmeans_silhouette)
        
        # Run FCM
        start_time = time.time()
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
        fcm_labels_2d = fcm.fit(data_2d)
        fcm_time = time.time() - start_time
        fcm_time_2d.append(fcm_time)
        
        # Calculate FCM metrics
        fcm_inertia = fcm.inertia_  # Fuzzy inertia
        fcm_fuzzy_inertia.append(fcm_inertia)
        fcm_silhouette = silhouette_score(data_2d, fcm_labels_2d)
        fcm_silhouette_2d.append(fcm_silhouette)
        
        # Store individual run results
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
            
            # Save convergence history - only for FCM (KMeans doesn't provide real convergence data)
            final_results["fcm_history"] = fcm.get_fitness_history()
    
    # Calculate average metrics
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
    
    # Print summary with Silhouette Score as primary comparison metric
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
            
            # Note: KMeans doesn't provide real convergence history data
    
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

def run_fcm_gkfcm_comparison(data_2d, data_3d, n_clusters=4, n_runs=30, random_seeds=None):
    """
    Run comparison between FCM and GK-FCM algorithms
    
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
    fcm_silhouette_2d = []
    fcm_fuzzy_inertia = []  # FCM-specific metric (fuzzy inertia)
    fcm_time_2d = []
    
    gkfcm_silhouette_2d = []
    gkfcm_fuzzy_inertia = []  # GK-FCM-specific metric (fuzzy inertia with Mahalanobis distance)
    gkfcm_time_2d = []
    
    # Store final results from last run for visualization
    final_results = {}
    
    # Run experiments
    for i, seed in enumerate(random_seeds):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        # Run FCM
        start_time = time.time()
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
        fcm_labels_2d = fcm.fit(data_2d)
        fcm_time_2d.append(time.time() - start_time)
        
        # Calculate FCM metrics
        fcm_fuzzy_inertia.append(fcm.inertia_)  # Fuzzy inertia (FCM-specific)
        fcm_silhouette_2d.append(silhouette_score(data_2d, fcm_labels_2d))
        
        # Run GK-FCM
        start_time = time.time()
        gkfcm = GKFuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
        gkfcm_labels_2d = gkfcm.fit(data_2d)
        gkfcm_time_2d.append(time.time() - start_time)
        
        # Calculate GK-FCM metrics
        gkfcm_fuzzy_inertia.append(gkfcm.inertia_)  # Fuzzy inertia with Mahalanobis distance (GK-FCM-specific)
        gkfcm_silhouette_2d.append(silhouette_score(data_2d, gkfcm_labels_2d))
        
        # Store results from final run for visualization
        if i == n_runs - 1:
            final_results["fcm_labels_2d"] = fcm_labels_2d
            final_results["fcm_centroids_2d"] = fcm.get_centroids()
            final_results["gkfcm_labels_2d"] = gkfcm_labels_2d
            final_results["gkfcm_centroids_2d"] = gkfcm.get_centroids()
            
            # Also run on 3D data for the final run
            fcm_3d = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
            gkfcm_3d = GKFuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
            
            fcm_labels_3d = fcm_3d.fit(data_3d)
            gkfcm_labels_3d = gkfcm_3d.fit(data_3d)
            
            final_results["fcm_labels_3d"] = fcm_labels_3d
            final_results["fcm_centroids_3d"] = fcm_3d.get_centroids()
            final_results["gkfcm_labels_3d"] = gkfcm_labels_3d
            final_results["gkfcm_centroids_3d"] = gkfcm_3d.get_centroids()
            
            # Save convergence history
            final_results["fcm_history"] = fcm.get_fitness_history()
            final_results["gkfcm_history"] = gkfcm.get_fitness_history()
            
            # Calculate covariance matrices for GK-FCM
            # Ensure centroids have the right shape
            print(f"Data shape: {data_2d.shape}, Centroids shape: {gkfcm.centroids.shape}")
            # Make sure centroids have the same number of features as data
            if gkfcm.centroids.shape[1] != data_2d.shape[1]:
                print(f"Warning: Centroid dimensions don't match data. Reshaping centroids.")
                # Take only the first data_2d.shape[1] features from each centroid
                gkfcm.centroids = gkfcm.centroids[:, :data_2d.shape[1]]
            
            gkfcm.covariance_matrices = gkfcm.update_covariance_matrices(data_2d, gkfcm.centroids, gkfcm.membership)
            
            # Calculate inverse covariance matrices
            n_features = data_2d.shape[1]
            gkfcm.inv_covariance_matrices = np.zeros((gkfcm.n_clusters, n_features, n_features))
            for j in range(gkfcm.n_clusters):
                try:
                    gkfcm.inv_covariance_matrices[j] = np.linalg.inv(gkfcm.covariance_matrices[j])
                except np.linalg.LinAlgError:
                    # If matrix is singular, use pseudoinverse
                    gkfcm.inv_covariance_matrices[j] = np.linalg.pinv(gkfcm.covariance_matrices[j])
            
            gkfcm.norm_matrices = gkfcm.calculate_norm_matrices(gkfcm.covariance_matrices, data_2d.shape[1])
    
    # Calculate average metrics
    results = {
        "fcm_avg_silhouette": np.mean(fcm_silhouette_2d),
        "fcm_avg_fuzzy_inertia": np.mean(fcm_fuzzy_inertia),
        "fcm_avg_time": np.mean(fcm_time_2d),
        "gkfcm_avg_silhouette": np.mean(gkfcm_silhouette_2d),
        "gkfcm_avg_fuzzy_inertia": np.mean(gkfcm_fuzzy_inertia),
        "gkfcm_avg_time": np.mean(gkfcm_time_2d),
        "final_run": final_results
    }
    
    # Print summary with Silhouette Score as primary comparison metric
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
    
    return results, random_seeds

def compare_fcm_gkfcm(data_2d, fcm_labels, gkfcm_labels, fcm_centroids, gkfcm_centroids, scaler=None):
    """
    Create visualizations to compare FCM and GK-FCM clustering results
    
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
    # This is a placeholder for a custom comparison function
    # We'll reuse the existing comparison function for now
    compare_kmeans_fcm(data_2d, fcm_labels, gkfcm_labels, fcm_centroids, gkfcm_centroids, 
                      scaler, title_left="FCM", title_right="GK-FCM", filename="comparison_fcm_gkfcm_2d.png")
    
    # Calculate Silhouette Score for both algorithms (common comparison metric)
    fcm_silhouette = silhouette_score(data_2d, fcm_labels)
    gkfcm_silhouette = silhouette_score(data_2d, gkfcm_labels)
    
    # Calculate algorithm-specific metrics
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
    
    # Calculate covariance matrices for GK-FCM
    # Ensure centroids have the right shape
    print(f"Data shape: {data_2d.shape}, Centroids shape: {gkfcm.centroids.shape}")
    # Make sure centroids have the same number of features as data
    if gkfcm.centroids.shape[1] != data_2d.shape[1]:
        print(f"Warning: Centroid dimensions don't match data. Reshaping centroids.")
        # Take only the first data_2d.shape[1] features from each centroid
        gkfcm.centroids = gkfcm.centroids[:, :data_2d.shape[1]]
        
    gkfcm.covariance_matrices = gkfcm.update_covariance_matrices(data_2d, gkfcm.centroids, gkfcm.membership)
    
    # Calculate inverse covariance matrices
    n_features = data_2d.shape[1]
    gkfcm.inv_covariance_matrices = np.zeros((gkfcm.n_clusters, n_features, n_features))
    for j in range(gkfcm.n_clusters):
        try:
            gkfcm.inv_covariance_matrices[j] = np.linalg.inv(gkfcm.covariance_matrices[j])
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudoinverse
            gkfcm.inv_covariance_matrices[j] = np.linalg.pinv(gkfcm.covariance_matrices[j])
    
    gkfcm.norm_matrices = gkfcm.calculate_norm_matrices(gkfcm.covariance_matrices, data_2d.shape[1])
    
    gkfcm_fuzzy_inertia = gkfcm._calculate_inertia(data_2d)
    
    # Use the algorithm-specific metrics comparison function
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
    Run comparison between all fuzzy clustering algorithms: FCM, GK-FCM, KFCM, and MKFCM
    
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
    
    # Results containers for silhouette score (common metric)
    fcm_silhouette = []
    gkfcm_silhouette = []
    kfcm_silhouette = []
    mkfcm_silhouette = []
    
    # Results containers for algorithm-specific metrics
    fcm_fuzzy_inertia = []
    gkfcm_fuzzy_inertia = []
    kfcm_fuzzy_inertia = []
    mkfcm_fuzzy_inertia = []
    
    # Results containers for execution time
    fcm_time = []
    gkfcm_time = []
    kfcm_time = []
    mkfcm_time = []
    
    # Store all individual run results for statistical analysis
    all_run_results = []
    
    # Store final results from last run for visualization
    final_results = {}
    
    # Run experiments
    for i, seed in enumerate(random_seeds):
        print(f"  Run {i+1}/{n_runs} (seed: {seed})")
        
        # Run FCM
        start_time = time.time()
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
        fcm_labels = fcm.fit(data_2d)
        fcm_exec_time = time.time() - start_time
        fcm_time.append(fcm_exec_time)
        
        # Calculate FCM metrics
        fcm_silh = silhouette_score(data_2d, fcm_labels)
        fcm_silhouette.append(fcm_silh)
        fcm_inertia = fcm.inertia_
        fcm_fuzzy_inertia.append(fcm_inertia)
        
        # Run GK-FCM
        start_time = time.time()
        gkfcm = GKFuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
        gkfcm_labels = gkfcm.fit(data_2d)
        gkfcm_exec_time = time.time() - start_time
        gkfcm_time.append(gkfcm_exec_time)
        
        # Calculate GK-FCM metrics
        gkfcm_silh = silhouette_score(data_2d, gkfcm_labels)
        gkfcm_silhouette.append(gkfcm_silh)
        gkfcm_inertia = gkfcm.inertia_
        gkfcm_fuzzy_inertia.append(gkfcm_inertia)
        
        # Run KFCM
        start_time = time.time()
        kfcm = KernelFuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
        kfcm_labels = kfcm.fit(data_2d)
        kfcm_exec_time = time.time() - start_time
        kfcm_time.append(kfcm_exec_time)
        
        # Calculate KFCM metrics
        kfcm_silh = silhouette_score(data_2d, kfcm_labels)
        kfcm_silhouette.append(kfcm_silh)
        kfcm_inertia = kfcm.inertia_
        kfcm_fuzzy_inertia.append(kfcm_inertia)
        
        # Run MKFCM
        start_time = time.time()
        mkfcm = ModifiedKernelFuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, random_state=seed)
        mkfcm_labels = mkfcm.fit(data_2d)
        mkfcm_exec_time = time.time() - start_time
        mkfcm_time.append(mkfcm_exec_time)
        
        # Calculate MKFCM metrics
        mkfcm_silh = silhouette_score(data_2d, mkfcm_labels)
        mkfcm_silhouette.append(mkfcm_silh)
        mkfcm_inertia = mkfcm.inertia_
        mkfcm_fuzzy_inertia.append(mkfcm_inertia)
        
        # Store individual run results
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
            "mkfcm_time": mkfcm_exec_time
        }
        all_run_results.append(run_result)
        
        # Store results from final run for visualization
        if i == n_runs - 1:
            final_results["fcm_labels"] = fcm_labels
            final_results["fcm_centroids"] = fcm.get_centroids()
            final_results["gkfcm_labels"] = gkfcm_labels
            final_results["gkfcm_centroids"] = gkfcm.get_centroids()
            final_results["kfcm_labels"] = kfcm_labels
            final_results["kfcm_centroids"] = kfcm.get_centroids()
            final_results["mkfcm_labels"] = mkfcm_labels
            final_results["mkfcm_centroids"] = mkfcm.get_centroids()
            
            # Save convergence history
            final_results["fcm_history"] = fcm.get_fitness_history()
            final_results["gkfcm_history"] = gkfcm.get_fitness_history()
            final_results["kfcm_history"] = kfcm.get_fitness_history()
            final_results["mkfcm_history"] = mkfcm.get_fitness_history()
            
            # For GK-FCM, also save covariance matrices
            final_results["gkfcm_covariance_matrices"] = gkfcm.get_covariance_matrices()
    
    # Calculate average metrics
    results = {
        "fcm_avg_silhouette": np.mean(fcm_silhouette),
        "fcm_avg_fuzzy_inertia": np.mean(fcm_fuzzy_inertia),
        "fcm_avg_time": np.mean(fcm_time),
        "fcm_std_silhouette": np.std(fcm_silhouette),
        "fcm_std_fuzzy_inertia": np.std(fcm_fuzzy_inertia),
        "fcm_std_time": np.std(fcm_time),
        
        "gkfcm_avg_silhouette": np.mean(gkfcm_silhouette),
        "gkfcm_avg_fuzzy_inertia": np.mean(gkfcm_fuzzy_inertia),
        "gkfcm_avg_time": np.mean(gkfcm_time),
        "gkfcm_std_silhouette": np.std(gkfcm_silhouette),
        "gkfcm_std_fuzzy_inertia": np.std(gkfcm_fuzzy_inertia),
        "gkfcm_std_time": np.std(gkfcm_time),
        
        "kfcm_avg_silhouette": np.mean(kfcm_silhouette),
        "kfcm_avg_fuzzy_inertia": np.mean(kfcm_fuzzy_inertia),
        "kfcm_avg_time": np.mean(kfcm_time),
        "kfcm_std_silhouette": np.std(kfcm_silhouette),
        "kfcm_std_fuzzy_inertia": np.std(kfcm_fuzzy_inertia),
        "kfcm_std_time": np.std(kfcm_time),
        
        "mkfcm_avg_silhouette": np.mean(mkfcm_silhouette),
        "mkfcm_avg_fuzzy_inertia": np.mean(mkfcm_fuzzy_inertia),
        "mkfcm_avg_time": np.mean(mkfcm_time),
        "mkfcm_std_silhouette": np.std(mkfcm_silhouette),
        "mkfcm_std_fuzzy_inertia": np.std(mkfcm_fuzzy_inertia),
        "mkfcm_std_time": np.std(mkfcm_time),
        
        "all_run_results": all_run_results,
        "final_run": final_results
    }
    
    # Print summary with Silhouette Score as primary comparison metric
    print("\nFuzzy Clustering Algorithms Comparison Results Summary:")
    print(f"  Common Comparison Metric (Silhouette Score):")
    print(f"    FCM   - Avg Silhouette: {results['fcm_avg_silhouette']:.4f} ± {results['fcm_std_silhouette']:.4f}")
    print(f"    GK-FCM - Avg Silhouette: {results['gkfcm_avg_silhouette']:.4f} ± {results['gkfcm_std_silhouette']:.4f}")
    print(f"    KFCM  - Avg Silhouette: {results['kfcm_avg_silhouette']:.4f} ± {results['kfcm_std_silhouette']:.4f}")
    print(f"    MKFCM - Avg Silhouette: {results['mkfcm_avg_silhouette']:.4f} ± {results['mkfcm_std_silhouette']:.4f}")
    
    print(f"  Algorithm-Specific Metrics (not directly comparable):")
    print(f"    FCM   - Avg Fuzzy Inertia: {results['fcm_avg_fuzzy_inertia']:.4f} ± {results['fcm_std_fuzzy_inertia']:.4f}")
    print(f"    GK-FCM - Avg Fuzzy Inertia: {results['gkfcm_avg_fuzzy_inertia']:.4f} ± {results['gkfcm_std_fuzzy_inertia']:.4f}")
    print(f"    KFCM  - Avg Fuzzy Inertia: {results['kfcm_avg_fuzzy_inertia']:.4f} ± {results['kfcm_std_fuzzy_inertia']:.4f}")
    print(f"    MKFCM - Avg Fuzzy Inertia: {results['mkfcm_avg_fuzzy_inertia']:.4f} ± {results['mkfcm_std_fuzzy_inertia']:.4f}")
    
    print(f"  Performance:")
    print(f"    FCM   - Avg Time: {results['fcm_avg_time']:.4f}s ± {results['fcm_std_time']:.4f}s")
    print(f"    GK-FCM - Avg Time: {results['gkfcm_avg_time']:.4f}s ± {results['gkfcm_std_time']:.4f}s")
    print(f"    KFCM  - Avg Time: {results['kfcm_avg_time']:.4f}s ± {results['kfcm_std_time']:.4f}s")
    print(f"    MKFCM - Avg Time: {results['mkfcm_avg_time']:.4f}s ± {results['mkfcm_std_time']:.4f}s")
    
    return results, random_seeds

def run_kernel_sigma_comparison(data_2d, n_clusters=4, sigma_values=None, n_runs=5, random_seeds=None):
    """
    Compare KFCM and MKFCM with different sigma_squared values
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset (Annual Income, Spending Score)
    n_clusters : int, default=4
        Number of clusters
    sigma_values : list or None, default=None
        List of sigma_squared values to compare (default: [0.1, 1.0, 10.0, 50.0, 100.0])
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
        sigma_values = [0.1, 1.0, 10.0, 50.0, 100.0]
    
    print(f"Running kernel sigma_squared comparison for values: {sigma_values}...")
    
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
    
    results = {}
    kfcm_results = []
    mkfcm_results = []
    
    # Store all individual run results for statistical analysis
    kfcm_all_runs = []
    mkfcm_all_runs = []
    
    # Run KFCM for each sigma value
    for sigma in sigma_values:
        print(f"  Testing KFCM with sigma_squared={sigma} ({n_runs} runs)")
        
        # Results for this sigma value
        sigma_silhouette_scores = []
        sigma_inertia_values = []
        sigma_times = []
        
        # Run multiple times with different seeds
        for i, seed in enumerate(random_seeds):
            print(f"    Run {i+1}/{n_runs} (seed: {seed})")
            
            # Run KFCM
            start_time = time.time()
            kfcm = KernelFuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, 
                                              random_state=seed, sigma_squared=sigma)
            labels = kfcm.fit(data_2d)
            execution_time = time.time() - start_time
            
            # Calculate metrics
            silhouette = silhouette_score(data_2d, labels)
            inertia = kfcm.inertia_
            
            # Store metrics for this run
            sigma_silhouette_scores.append(silhouette)
            sigma_inertia_values.append(inertia)
            sigma_times.append(execution_time)
            
            # Store full run result
            run_result = {
                "sigma_squared": sigma,
                "run": i+1,
                "seed": seed,
                "silhouette": silhouette,
                "inertia": inertia,
                "time": execution_time
            }
            kfcm_all_runs.append(run_result)
            
            # Store last run for visualization
            if i == n_runs - 1:
                last_run = {
                    "sigma_squared": sigma,
                    "labels": labels,
                    "centroids": kfcm.get_centroids(),
                    "silhouette": silhouette,
                    "inertia": inertia,
                    "history": kfcm.get_fitness_history()
                }
                kfcm_results.append(last_run)
        
        # Calculate average metrics for this sigma value
        avg_silhouette = np.mean(sigma_silhouette_scores)
        std_silhouette = np.std(sigma_silhouette_scores)
        avg_inertia = np.mean(sigma_inertia_values)
        std_inertia = np.std(sigma_inertia_values)
        avg_time = np.mean(sigma_times)
        std_time = np.std(sigma_times)
        
        print(f"    Avg Silhouette: {avg_silhouette:.4f} ± {std_silhouette:.4f}")
        print(f"    Avg Kernel Inertia: {avg_inertia:.4f} ± {std_inertia:.4f}")
        print(f"    Avg Time: {avg_time:.4f}s ± {std_time:.4f}s")
        
        # Run MKFCM for the same sigma value
        print(f"  Testing MKFCM with sigma_squared={sigma} ({n_runs} runs)")
        
        # Results for this sigma value
        sigma_silhouette_scores = []
        sigma_inertia_values = []
        sigma_times = []
        
        # Run multiple times with different seeds
        for i, seed in enumerate(random_seeds):
            print(f"    Run {i+1}/{n_runs} (seed: {seed})")
            
            # Run MKFCM
            start_time = time.time()
            mkfcm = ModifiedKernelFuzzyCMeansClustering(n_clusters=n_clusters, m=2.0, 
                                                      random_state=seed, sigma_squared=sigma)
            labels = mkfcm.fit(data_2d)
            execution_time = time.time() - start_time
            
            # Calculate metrics
            silhouette = silhouette_score(data_2d, labels)
            inertia = mkfcm.inertia_
            
            # Store metrics for this run
            sigma_silhouette_scores.append(silhouette)
            sigma_inertia_values.append(inertia)
            sigma_times.append(execution_time)
            
            # Store full run result
            run_result = {
                "sigma_squared": sigma,
                "run": i+1,
                "seed": seed,
                "silhouette": silhouette,
                "inertia": inertia,
                "time": execution_time
            }
            mkfcm_all_runs.append(run_result)
            
            # Store last run for visualization
            if i == n_runs - 1:
                last_run = {
                    "sigma_squared": sigma,
                    "labels": labels,
                    "centroids": mkfcm.get_centroids(),
                    "silhouette": silhouette,
                    "inertia": inertia,
                    "history": mkfcm.get_fitness_history()
                }
                mkfcm_results.append(last_run)
        
        # Calculate average metrics for this sigma value
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
    
    # Run FCM vs GK-FCM comparison
    fcm_gkfcm_seeds = experiment_seeds.get("fcm_gkfcm_comparison", [])
    fcm_gkfcm_results, fcm_gkfcm_seeds = run_fcm_gkfcm_comparison(
        features_2d, features_3d, n_clusters=4, n_runs=30, random_seeds=fcm_gkfcm_seeds
    )
    
    # Run all fuzzy clustering comparison
    all_fuzzy_seeds = experiment_seeds.get("all_fuzzy_comparison", [])
    all_fuzzy_results, all_fuzzy_seeds = run_all_fuzzy_comparison(
        features_2d, features_3d, n_clusters=4, n_runs=10, random_seeds=all_fuzzy_seeds
    )
    
    # Run kernel sigma parameter study
    kernel_sigma_results = run_kernel_sigma_comparison(
        features_2d, n_clusters=4, sigma_values=[0.1, 1.0, 10.0, 50.0, 100.0], n_runs=5, random_seeds=None
    )
    
    # Save experiment seeds for reproducibility
    experiment_seeds["kmeans_fcm_comparison"] = kmeans_fcm_seeds
    experiment_seeds["kmeans_init_comparison"] = kmeans_init_seeds
    experiment_seeds["fcm_gkfcm_comparison"] = fcm_gkfcm_seeds
    experiment_seeds["all_fuzzy_comparison"] = all_fuzzy_seeds
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
    
    # 3. Algorithm-specific metrics comparison
    plot_algorithm_metrics_comparison(
        kmeans_fcm_results["kmeans_avg_inertia"],
        kmeans_fcm_results["fcm_avg_fuzzy_inertia"],
        algorithms=['K-means', 'FCM'],
        filename="algorithm_specific_metrics_comparison.png",
        title="K-means vs FCM: Algorithm-Specific Objective Functions"
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
    # Scikit-learn's KMeans implementation doesn't expose the actual iteration-by-iteration convergence data.
    # Only FCM-based algorithms with true convergence histories are plotted.
    
    # Only plot FCM convergence (no comparison with KMeans)
    plot_convergence_curves(
        {
            "FCM": kmeans_fcm_results["final_run"]["fcm_history"]
        }
    )
    
    # FCM m-value convergence
    fcm_histories = {}
    for result in fcm_m_results["fcm_results"]:
        fcm_histories[f"FCM m={result['m']}"] = result["history"]
    plot_convergence_curves(fcm_histories)
    
    # FCM vs GK-FCM comparison
    compare_fcm_gkfcm(
        features_2d,
        fcm_gkfcm_results["final_run"]["fcm_labels_2d"],
        fcm_gkfcm_results["final_run"]["gkfcm_labels_2d"],
        fcm_gkfcm_results["final_run"]["fcm_centroids_2d"],
        fcm_gkfcm_results["final_run"]["gkfcm_centroids_2d"],
        scaler_2d
    )
    
    # All fuzzy clustering comparison
    compare_fcm_gkfcm(
        features_2d,
        all_fuzzy_results["final_run"]["fcm_labels"],
        all_fuzzy_results["final_run"]["gkfcm_labels"],
        all_fuzzy_results["final_run"]["fcm_centroids"],
        all_fuzzy_results["final_run"]["gkfcm_centroids"],
        scaler_2d
    )
    
    # All fuzzy clustering comparison visualization
    # Create dictionaries for labels and centroids
    labels_dict = {
        'fcm': all_fuzzy_results["final_run"]["fcm_labels"],
        'gkfcm': all_fuzzy_results["final_run"]["gkfcm_labels"],
        'kfcm': all_fuzzy_results["final_run"]["kfcm_labels"],
        'mkfcm': all_fuzzy_results["final_run"]["mkfcm_labels"]
    }
    
    centroids_dict = {
        'fcm': all_fuzzy_results["final_run"]["fcm_centroids"],
        'gkfcm': all_fuzzy_results["final_run"]["gkfcm_centroids"],
        'kfcm': all_fuzzy_results["final_run"]["kfcm_centroids"],
        'mkfcm': all_fuzzy_results["final_run"]["mkfcm_centroids"]
    }
    
    # Create visualization of all fuzzy clustering algorithms
    compare_all_fuzzy(
        features_2d,
        labels_dict,
        centroids_dict,
        title="Comparison of All Fuzzy Clustering Algorithms",
        filename="all_fuzzy_comparison.png",
        scaler=scaler_2d
    )
    
    # Compare Silhouette scores
    silhouette_dict = {
        'FCM': all_fuzzy_results["fcm_avg_silhouette"],
        'GK-FCM': all_fuzzy_results["gkfcm_avg_silhouette"],
        'KFCM': all_fuzzy_results["kfcm_avg_silhouette"],
        'MKFCM': all_fuzzy_results["mkfcm_avg_silhouette"]
    }
    
    compare_fuzzy_metrics(
        silhouette_dict,
        metric_name='Silhouette Score',
        higher_better=True,
        filename='fuzzy_silhouette_comparison.png'
    )
    
    # Create dictionaries with standard deviations for error bars
    silhouette_std_dict = {
        'FCM': all_fuzzy_results["fcm_std_silhouette"],
        'GK-FCM': all_fuzzy_results["gkfcm_std_silhouette"],
        'KFCM': all_fuzzy_results["kfcm_std_silhouette"],
        'MKFCM': all_fuzzy_results["mkfcm_std_silhouette"]
    }
    
    fuzzy_inertia_dict = {
        'FCM': all_fuzzy_results["fcm_avg_fuzzy_inertia"],
        'GK-FCM': all_fuzzy_results["gkfcm_avg_fuzzy_inertia"],
        'KFCM': all_fuzzy_results["kfcm_avg_fuzzy_inertia"],
        'MKFCM': all_fuzzy_results["mkfcm_avg_fuzzy_inertia"]
    }
    
    fuzzy_inertia_std_dict = {
        'FCM': all_fuzzy_results["fcm_std_fuzzy_inertia"],
        'GK-FCM': all_fuzzy_results["gkfcm_std_fuzzy_inertia"],
        'KFCM': all_fuzzy_results["kfcm_std_fuzzy_inertia"],
        'MKFCM': all_fuzzy_results["mkfcm_std_fuzzy_inertia"]
    }
    
    time_dict = {
        'FCM': all_fuzzy_results["fcm_avg_time"],
        'GK-FCM': all_fuzzy_results["gkfcm_avg_time"],
        'KFCM': all_fuzzy_results["kfcm_avg_time"],
        'MKFCM': all_fuzzy_results["mkfcm_avg_time"]
    }
    
    time_std_dict = {
        'FCM': all_fuzzy_results["fcm_std_time"],
        'GK-FCM': all_fuzzy_results["gkfcm_std_time"],
        'KFCM': all_fuzzy_results["kfcm_std_time"],
        'MKFCM': all_fuzzy_results["mkfcm_std_time"]
    }
    
    # Create bar charts with error bars for all fuzzy algorithms
    compare_fuzzy_metrics_with_error_bars(
        silhouette_dict,
        silhouette_std_dict,
        metric_name='Silhouette Score',
        higher_better=True,
        filename='fuzzy_silhouette_with_error.png'
    )
    
    compare_fuzzy_metrics_with_error_bars(
        fuzzy_inertia_dict,
        fuzzy_inertia_std_dict,
        metric_name='Fuzzy Inertia',
        higher_better=False,
        filename='fuzzy_inertia_with_error.png'
    )
    
    compare_fuzzy_metrics_with_error_bars(
        time_dict,
        time_std_dict,
        metric_name='Execution Time (s)',
        higher_better=False,
        filename='fuzzy_time_with_error.png'
    )
    
    compare_fuzzy_metrics(
        time_dict,
        metric_name='Execution Time (s)',
        higher_better=False,
        filename='fuzzy_time_comparison.png'
    )
    
    # Plot convergence curves for all fuzzy algorithms
    plot_convergence_curves(
        {
            'FCM': all_fuzzy_results["final_run"]["fcm_history"],
            'GK-FCM': all_fuzzy_results["final_run"]["gkfcm_history"],
            'KFCM': all_fuzzy_results["final_run"]["kfcm_history"],
            'MKFCM': all_fuzzy_results["final_run"]["mkfcm_history"]
        }
    )
    
    # After other visualization code, add kernel sigma parameter study visualization
    print("Generating kernel sigma parameter study visualizations...")
    
    # Visualize the effect of different sigma values on clustering
    plot_kernel_sigma_comparison(
        features_2d,
        kernel_sigma_results['kfcm_results'],
        'KFCM',
        scaler_2d,
        "Effect of σ² on KFCM Clustering",
        "kfcm_sigma_clustering_comparison.png"
    )
    
    plot_kernel_sigma_comparison(
        features_2d,
        kernel_sigma_results['mkfcm_results'],
        'MKFCM',
        scaler_2d,
        "Effect of σ² on MKFCM Clustering",
        "mkfcm_sigma_clustering_comparison.png"
    )
    
    # Create dictionaries for metric comparison
    kfcm_silhouette_dict = {f"σ²={result['sigma_squared']}": result['silhouette'] 
                           for result in kernel_sigma_results['kfcm_results']}
    mkfcm_silhouette_dict = {f"σ²={result['sigma_squared']}": result['silhouette'] 
                            for result in kernel_sigma_results['mkfcm_results']}
    
    kfcm_inertia_dict = {f"σ²={result['sigma_squared']}": result['inertia'] 
                        for result in kernel_sigma_results['kfcm_results']}
    mkfcm_inertia_dict = {f"σ²={result['sigma_squared']}": result['inertia'] 
                         for result in kernel_sigma_results['mkfcm_results']}
    
    # Visualize silhouette scores for different sigma values
    compare_fuzzy_metrics(
        kfcm_silhouette_dict,
        algorithms=list(kfcm_silhouette_dict.keys()),
        metric_name='KFCM Silhouette Score by σ²',
        higher_better=True,
        filename='kfcm_sigma_silhouette_comparison.png'
    )
    
    compare_fuzzy_metrics(
        mkfcm_silhouette_dict,
        algorithms=list(mkfcm_silhouette_dict.keys()),
        metric_name='MKFCM Silhouette Score by σ²',
        higher_better=True,
        filename='mkfcm_sigma_silhouette_comparison.png'
    )
    
    # Visualize kernel inertia for different sigma values
    compare_fuzzy_metrics(
        kfcm_inertia_dict,
        algorithms=list(kfcm_inertia_dict.keys()),
        metric_name='KFCM Kernel Inertia by σ²',
        higher_better=False,
        filename='kfcm_sigma_inertia_comparison.png'
    )
    
    compare_fuzzy_metrics(
        mkfcm_inertia_dict,
        algorithms=list(mkfcm_inertia_dict.keys()),
        metric_name='MKFCM Kernel Inertia by σ²',
        higher_better=False,
        filename='mkfcm_sigma_inertia_comparison.png'
    )
    
    # Create advanced statistical visualizations for sigma parameter study
    from visualization import compare_fuzzy_metrics_with_error_bars, plot_sigma_parameter_study
    
    # Extract all runs data for statistical analysis
    sigma_values = sorted(list(set([run["sigma_squared"] for run in kernel_sigma_results['kfcm_all_runs']])))
    
    # Prepare data for KFCM sigma study
    kfcm_silhouette_by_sigma = {sigma: [] for sigma in sigma_values}
    kfcm_inertia_by_sigma = {sigma: [] for sigma in sigma_values}
    kfcm_time_by_sigma = {sigma: [] for sigma in sigma_values}
    
    for run in kernel_sigma_results['kfcm_all_runs']:
        sigma = run['sigma_squared']
        kfcm_silhouette_by_sigma[sigma].append(run['silhouette'])
        kfcm_inertia_by_sigma[sigma].append(run['inertia'])
        kfcm_time_by_sigma[sigma].append(run['time'])
    
    # Calculate means and std for each sigma value
    kfcm_sil_means = {sigma: np.mean(values) for sigma, values in kfcm_silhouette_by_sigma.items()}
    kfcm_sil_stds = {sigma: np.std(values) for sigma, values in kfcm_silhouette_by_sigma.items()}
    
    kfcm_inertia_means = {sigma: np.mean(values) for sigma, values in kfcm_inertia_by_sigma.items()}
    kfcm_inertia_stds = {sigma: np.std(values) for sigma, values in kfcm_inertia_by_sigma.items()}
    
    kfcm_time_means = {sigma: np.mean(values) for sigma, values in kfcm_time_by_sigma.items()}
    kfcm_time_stds = {sigma: np.std(values) for sigma, values in kfcm_time_by_sigma.items()}
    
    # Create bar charts with error bars for KFCM
    compare_fuzzy_metrics_with_error_bars(
        kfcm_sil_means,
        kfcm_sil_stds,
        algorithms=[f"σ²={sigma}" for sigma in sigma_values],
        metric_name='Silhouette Score',
        higher_better=True,
        filename='kfcm_sigma_silhouette_with_error.png'
    )
    
    compare_fuzzy_metrics_with_error_bars(
        kfcm_inertia_means,
        kfcm_inertia_stds,
        algorithms=[f"σ²={sigma}" for sigma in sigma_values],
        metric_name='Kernel Inertia',
        higher_better=False,
        filename='kfcm_sigma_inertia_with_error.png'
    )
    
    compare_fuzzy_metrics_with_error_bars(
        kfcm_time_means,
        kfcm_time_stds,
        algorithms=[f"σ²={sigma}" for sigma in sigma_values],
        metric_name='Execution Time (s)',
        higher_better=False,
        filename='kfcm_sigma_time_with_error.png'
    )
    
    # Create line plots for KFCM sigma parameter study
    plot_sigma_parameter_study(
        sigma_values,
        [kfcm_sil_means[sigma] for sigma in sigma_values],
        [kfcm_sil_stds[sigma] for sigma in sigma_values],
        metric_name='Silhouette Score',
        algorithm_name='KFCM',
        filename='kfcm_sigma_silhouette_study.png'
    )
    
    plot_sigma_parameter_study(
        sigma_values,
        [kfcm_inertia_means[sigma] for sigma in sigma_values],
        [kfcm_inertia_stds[sigma] for sigma in sigma_values],
        metric_name='Kernel Inertia',
        algorithm_name='KFCM',
        filename='kfcm_sigma_inertia_study.png'
    )
    
    # Do the same for MKFCM
    mkfcm_silhouette_by_sigma = {sigma: [] for sigma in sigma_values}
    mkfcm_inertia_by_sigma = {sigma: [] for sigma in sigma_values}
    mkfcm_time_by_sigma = {sigma: [] for sigma in sigma_values}
    
    for run in kernel_sigma_results['mkfcm_all_runs']:
        sigma = run['sigma_squared']
        mkfcm_silhouette_by_sigma[sigma].append(run['silhouette'])
        mkfcm_inertia_by_sigma[sigma].append(run['inertia'])
        mkfcm_time_by_sigma[sigma].append(run['time'])
    
    # Calculate means and std for each sigma value
    mkfcm_sil_means = {sigma: np.mean(values) for sigma, values in mkfcm_silhouette_by_sigma.items()}
    mkfcm_sil_stds = {sigma: np.std(values) for sigma, values in mkfcm_silhouette_by_sigma.items()}
    
    mkfcm_inertia_means = {sigma: np.mean(values) for sigma, values in mkfcm_inertia_by_sigma.items()}
    mkfcm_inertia_stds = {sigma: np.std(values) for sigma, values in mkfcm_inertia_by_sigma.items()}
    
    mkfcm_time_means = {sigma: np.mean(values) for sigma, values in mkfcm_time_by_sigma.items()}
    mkfcm_time_stds = {sigma: np.std(values) for sigma, values in mkfcm_time_by_sigma.items()}
    
    # Create bar charts with error bars for MKFCM
    compare_fuzzy_metrics_with_error_bars(
        mkfcm_sil_means,
        mkfcm_sil_stds,
        algorithms=[f"σ²={sigma}" for sigma in sigma_values],
        metric_name='Silhouette Score',
        higher_better=True,
        filename='mkfcm_sigma_silhouette_with_error.png'
    )
    
    compare_fuzzy_metrics_with_error_bars(
        mkfcm_inertia_means,
        mkfcm_inertia_stds,
        algorithms=[f"σ²={sigma}" for sigma in sigma_values],
        metric_name='Kernel Inertia',
        higher_better=False,
        filename='mkfcm_sigma_inertia_with_error.png'
    )
    
    compare_fuzzy_metrics_with_error_bars(
        mkfcm_time_means,
        mkfcm_time_stds,
        algorithms=[f"σ²={sigma}" for sigma in sigma_values],
        metric_name='Execution Time (s)',
        higher_better=False,
        filename='mkfcm_sigma_time_with_error.png'
    )
    
    # Create line plots for MKFCM sigma parameter study
    plot_sigma_parameter_study(
        sigma_values,
        [mkfcm_sil_means[sigma] for sigma in sigma_values],
        [mkfcm_sil_stds[sigma] for sigma in sigma_values],
        metric_name='Silhouette Score',
        algorithm_name='MKFCM',
        filename='mkfcm_sigma_silhouette_study.png'
    )
    
    plot_sigma_parameter_study(
        sigma_values,
        [mkfcm_inertia_means[sigma] for sigma in sigma_values],
        [mkfcm_inertia_stds[sigma] for sigma in sigma_values],
        metric_name='Kernel Inertia',
        algorithm_name='MKFCM',
        filename='mkfcm_sigma_inertia_study.png'
    )
    
    # Plot convergence curves for different sigma values
    kfcm_sigma_histories = {}
    for result in kernel_sigma_results['kfcm_results']:
        kfcm_sigma_histories[f"KFCM σ²={result['sigma_squared']}"] = result["history"]
    
    plot_convergence_curves(kfcm_sigma_histories)
    
    mkfcm_sigma_histories = {}
    for result in kernel_sigma_results['mkfcm_results']:
        mkfcm_sigma_histories[f"MKFCM σ²={result['sigma_squared']}"] = result["history"]
    
    plot_convergence_curves(mkfcm_sigma_histories)

    print("\nAll experiments and visualizations complete!")
    print("Seed data saved for reproducibility.")

if __name__ == "__main__":
    main() 
