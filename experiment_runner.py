import numpy as np
import pandas as pd
import time
import os
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import euclidean_distances
from rsekfcm_clustering import RseKFCMClustering
from spkfcm_clustering import SpKFCMClustering
from okfcm_clustering import OKFCMClustering
from fcm_clustering import FuzzyCMeansClustering
from kfcm_clustering import KernelFuzzyCMeansClustering
from mkfcm_clustering import ModifiedKernelFuzzyCMeansClustering
from gkfcm_clustering import GKFuzzyCMeansClustering
from kmeans_clustering import KMeansClustering

def calculate_partition_coefficient(memberships):
    """Calculate Partition Coefficient (PC) for fuzzy clustering."""
    return np.mean(np.sum(memberships ** 2, axis=1))

def calculate_xie_beni_index(data, memberships, centroids, m=2):
    """Calculate Xie-Beni Index for fuzzy clustering."""
    n_clusters = centroids.shape[0]
    memberships_pow = memberships ** m
    intra_cluster = np.sum(memberships_pow.T * np.sum((data - centroids[:, np.newaxis])**2, axis=2))
    
    dist_matrix = euclidean_distances(centroids, centroids)
    np.fill_diagonal(dist_matrix, np.inf)
    min_inter_cluster = np.min(dist_matrix)
    min_inter_cluster = max(min_inter_cluster, 1e-6)
    
    return intra_cluster / (len(data) * min_inter_cluster)

def compute_sigma(data):
    """Compute sigma dynamically based on data distribution."""
    n = min(100, data.shape[0])
    distances = []
    indices = np.random.choice(data.shape[0], size=n, replace=False)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sum((data[indices[i]] - data[indices[j]]) ** 2)
            distances.append(dist)
    return np.sqrt(np.mean(distances))

def compute_kernel_matrix(data, kernel='rbf', sigma=None):
    """Compute kernel matrix using specified kernel function."""
    if sigma is None:
        sigma = compute_sigma(data)
    if kernel == 'rbf':
        n = len(data)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = np.exp(-np.sum((data[i] - data[j])**2) / (2 * (sigma ** 2)))
        return K
    else:
        raise ValueError("Unsupported kernel function. Use 'rbf'.")

def compute_wcss(data, memberships, centroids, m=2, is_kmeans=False):
    """Compute Within-Cluster Sum of Squares (WCSS)."""
    if is_kmeans:
        # Ensure memberships are hard assignments (0 or 1)
        labels = np.argmax(memberships, axis=1)
        hard_memberships = np.zeros_like(memberships)
        hard_memberships[np.arange(len(labels)), labels] = 1
        
        # Compute WCSS using vectorized operations
        wcss = 0
        for i in range(centroids.shape[0]):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:  # Avoid empty clusters
                wcss += np.sum((cluster_points - centroids[i]) ** 2)
        return wcss
    
    # For fuzzy clustering
    memberships_pow = memberships ** m
    return np.sum(memberships_pow.T * np.sum((data - centroids[:, np.newaxis])**2, axis=2))

def run_experiments(data_2d, data_3d, algorithms, n_runs=30, algorithm_runs=None, m=2.0, kernel='rbf', sigma=None):
    results = {}
    seeds = {}
    
    # Create a 'results' folder if it doesn't exist
    results_dir = "run_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if algorithm_runs is None:
        algorithm_runs = {}

    for algo_name, algo in algorithms.items():
        runs_for_algorithm = algorithm_runs.get(algo_name, n_runs)
        print(f"Running experiments for {algo_name} ({runs_for_algorithm} runs)...")

        algorithm_seeds = np.random.randint(0, 100000, size=runs_for_algorithm).tolist()
        seeds[algo_name] = algorithm_seeds

        algo_results = {
            "silhouette_scores_2d": [],
            "silhouette_scores_3d": [],
            "wcss_2d": [],
            "wcss_3d": [],
            "davies_bouldin_2d": [],
            "davies_bouldin_3d": [],
            "partition_coeff_2d": [],
            "partition_coeff_3d": [],
            "xie_beni_2d": [],
            "xie_beni_3d": [],
            "iterations": [],
            "times": [],
            "labels_2d": None,
            "labels_3d": None,
            "centroids_2d": None,
            "centroids_3d": None,
            "memberships_2d": None,
            "memberships_3d": None,
            "best_silhouette_2d": -float('inf'),
            "best_silhouette_3d": -float('inf'),
            "n_runs": runs_for_algorithm,
        }

        # Compute kernel matrices for 2D and 3D data if needed
        K_2d = compute_kernel_matrix(data_2d, kernel, sigma) if algo_name not in ['FCM', 'GK-FCM', 'K-Means'] else None
        K_3d = compute_kernel_matrix(data_3d, kernel, sigma) if algo_name not in ['FCM', 'GK-FCM', 'K-Means'] else None

        for run in range(runs_for_algorithm):
            np.random.seed(algorithm_seeds[run])
            algo.random_state = algorithm_seeds[run]

            try:
                # 2D Data
                start_time = time.time()
                memberships_2d, centroids_2d, labels_2d, iterations = algo.fit(data_2d, K_2d)
                execution_time = time.time() - start_time

                score_2d = silhouette_score(data_2d, labels_2d) if len(np.unique(labels_2d)) > 1 else -1
                db_score_2d = davies_bouldin_score(data_2d, labels_2d) if len(np.unique(labels_2d)) > 1 else float('inf')
                wcss_2d = compute_wcss(data_2d, memberships_2d, centroids_2d, m, is_kmeans=(algo_name == 'K-Means'))
                pc_2d = calculate_partition_coefficient(memberships_2d)
                xb_2d = calculate_xie_beni_index(data_2d, memberships_2d, centroids_2d, m)

                # 3D Data
                start_time = time.time()
                memberships_3d, centroids_3d, labels_3d, iterations = algo.fit(data_3d, K_3d)
                execution_time = max(execution_time, time.time() - start_time)

                score_3d = silhouette_score(data_3d, labels_3d) if len(np.unique(labels_3d)) > 1 else -1
                db_score_3d = davies_bouldin_score(data_3d, labels_3d) if len(np.unique(labels_3d)) > 1 else float('inf')
                wcss_3d = compute_wcss(data_3d, memberships_3d, centroids_3d, m, is_kmeans=(algo_name == 'K-Means'))
                pc_3d = calculate_partition_coefficient(memberships_3d)
                xb_3d = calculate_xie_beni_index(data_3d, memberships_3d, centroids_3d, m)

                algo_results["silhouette_scores_2d"].append(score_2d)
                algo_results["silhouette_scores_3d"].append(score_3d)
                algo_results["wcss_2d"].append(wcss_2d)
                algo_results["wcss_3d"].append(wcss_3d)
                algo_results["davies_bouldin_2d"].append(db_score_2d)
                algo_results["davies_bouldin_3d"].append(db_score_3d)
                algo_results["partition_coeff_2d"].append(pc_2d)
                algo_results["partition_coeff_3d"].append(pc_3d)
                algo_results["xie_beni_2d"].append(xb_2d)
                algo_results["xie_beni_3d"].append(xb_3d)
                algo_results["iterations"].append(iterations)
                algo_results["times"].append(execution_time)

                # Store best run based on Silhouette Score
                if score_2d > algo_results["best_silhouette_2d"]:
                    algo_results["best_silhouette_2d"] = score_2d
                    algo_results["labels_2d"] = labels_2d
                    algo_results["centroids_2d"] = centroids_2d
                    algo_results["memberships_2d"] = memberships_2d
                if score_3d > algo_results["best_silhouette_3d"]:
                    algo_results["best_silhouette_3d"] = score_3d
                    algo_results["labels_3d"] = labels_3d
                    algo_results["centroids_3d"] = centroids_3d
                    algo_results["memberships_3d"] = memberships_3d

                if runs_for_algorithm > 10 and (run + 1) % 10 == 0:
                    print(f"  {algo_name} - Completed {run + 1}/{runs_for_algorithm} runs")

            except Exception as e:
                print(f"Error in {algo_name} run {run + 1}: {str(e)}")
                continue

        results_df = pd.DataFrame({
            "Silhouette_2D": algo_results["silhouette_scores_2d"],
            "Silhouette_3D": algo_results["silhouette_scores_3d"],
            "WCSS_2D": algo_results["wcss_2d"],
            "WCSS_3D": algo_results["wcss_3d"],
            "Davies_Bouldin_2D": algo_results["davies_bouldin_2d"],
            "Davies_Bouldin_3D": algo_results["davies_bouldin_3d"],
            "Partition_Coefficient_2D": algo_results["partition_coeff_2d"],
            "Partition_Coefficient_3D": algo_results["partition_coeff_3d"],
            "Xie_Beni_2D": algo_results["xie_beni_2d"],
            "Xie_Beni_3D": algo_results["xie_beni_3d"],
            "Iterations": algo_results["iterations"],
            "Time": algo_results["times"]
        })
        results_df.to_csv(os.path.join(results_dir, f"{algo_name}_run_results.csv"), index=False)

        np.save(os.path.join(results_dir, f"{algo_name}_labels_2d.npy"), algo_results["labels_2d"])
        np.save(os.path.join(results_dir, f"{algo_name}_labels_3d.npy"), algo_results["labels_3d"])
        np.save(os.path.join(results_dir, f"{algo_name}_centroids_2d.npy"), algo_results["centroids_2d"])
        np.save(os.path.join(results_dir, f"{algo_name}_centroids_3d.npy"), algo_results["centroids_3d"])
        np.save(os.path.join(results_dir, f"{algo_name}_memberships_2d.npy"), algo_results["memberships_2d"])
        np.save(os.path.join(results_dir, f"{algo_name}_memberships_3d.npy"), algo_results["memberships_3d"])

        print(f"  {algo_name} - Avg Silhouette Score (2D): {np.mean(algo_results['silhouette_scores_2d']):.4f}")
        print(f"  {algo_name} - Avg Silhouette Score (3D): {np.mean(algo_results['silhouette_scores_3d']):.4f}")
        print(f"  {algo_name} - Avg WCSS (2D): {np.mean(algo_results['wcss_2d']):.4f}")
        print(f"  {algo_name} - Avg Davies-Bouldin (2D): {np.mean(algo_results['davies_bouldin_2d']):.4f}")
        print(f"  {algo_name} - Avg Partition Coefficient (2D): {np.mean(algo_results['partition_coeff_2d']):.4f}")
        print(f"  {algo_name} - Avg Xie-Beni Index (2D): {np.mean(algo_results['xie_beni_2d']):.4f}")
        print(f"  {algo_name} - Avg Time: {np.mean(algo_results['times']):.4f}s")

        results[algo_name] = algo_results

    return results, seeds

# Load Mall Customers data
data = pd.read_csv('Mall_Customers.csv')
data_2d = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
data_3d = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_2d = scaler.fit_transform(data_2d)
data_3d = scaler.fit_transform(data_3d)

# Initialize algorithms
algorithms = {
    'rseKFCM': RseKFCMClustering(n_clusters=5, m=2.0, sample_size=50, max_iter=100, epsilon=1e-3, random_state=42),
    'spKFCM': SpKFCMClustering(n_clusters=5, m=2.0, n_chunks=2, max_iter=100, epsilon=1e-3, random_state=42),
    'oKFCM': OKFCMClustering(n_clusters=5, m=2.0, n_chunks=2, max_iter=100, epsilon=1e-3, random_state=42),
    'FCM': FuzzyCMeansClustering(n_clusters=5, m=2.0, max_iter=100, tol=1e-4, random_state=42),
    'KFCM': KernelFuzzyCMeansClustering(n_clusters=5, m=2.0, max_iter=100, tol=1e-4, random_state=42),
    'MKFCM': ModifiedKernelFuzzyCMeansClustering(n_clusters=5, m=2.0, max_iter=100, tol=1e-4, random_state=42),
    'GK-FCM': GKFuzzyCMeansClustering(n_clusters=5, m=2.0, max_iter=100, tol=1e-4, random_state=42, min_det_value=1e-3),
    'K-Means': KMeansClustering(n_clusters=5, init_method='k-means++', random_state=42, max_iter=300)
}

# Run experiments
results, seeds = run_experiments(data_2d, data_3d, algorithms, n_runs=30, m=2.0, kernel='rbf')
