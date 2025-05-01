import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score
import os
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Ensure the 'results' directory exists
def ensure_results_directory():
    if not os.path.exists("results"):
        os.makedirs("results")

# 1. Scatter Plots for 2D Clustering
def plot_clusters_2d(data, labels, centroids, algo_name, scaler=None):
    ensure_results_directory()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data.shape[1]} dimensions")
    if centroids.shape[1] != 2:
        raise ValueError(f"Expected 2D centroids, but got centroids with {centroids.shape[1]} dimensions")
    
    # Inverse transform data and centroids for visualization
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", alpha=0.8, s=100)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centroids')
    plt.title(f"Clustering Results (2D) - {algo_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Annual Income (k$)", fontsize=12)
    plt.ylabel("Spending Score (1-100)", fontsize=12)
    plt.colorbar(label="Cluster")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{algo_name}_clusters_2d.png", dpi=300)
    plt.close()

# 2. Scatter Plots for 3D Clustering
def plot_clusters_3d(data, labels, centroids, algo_name, scaler=None):
    ensure_results_directory()
    if data.shape[1] != 3:
        raise ValueError(f"Expected 3D data, but got data with {data.shape[1]} dimensions")
    if centroids.shape[1] != 3:
        raise ValueError(f"Expected 3D centroids, but got centroids with {centroids.shape[1]} dimensions")
    
    # Inverse transform data and centroids for visualization
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="viridis", s=100, alpha=0.8)
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, linewidths=3, color='red', label='Centroids')
    
    ax.legend()
    plt.colorbar(scatter, label="Cluster")
    
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Annual Income (k$)", fontsize=12)
    ax.set_zlabel("Spending Score (1-100)", fontsize=12)
    ax.set_title(f"Clustering Results (3D) - {algo_name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"results/{algo_name}_clusters_3d.png", dpi=300)
    plt.close()

# 3. Bar Plots for Metrics Comparison
def plot_metrics_bar_comparison(summary_df, metrics, title="Metrics Comparison (2D)", filename="metrics_bar_comparison_2d.png"):
    ensure_results_directory()
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=summary_df.index, y=summary_df[metric], hue=summary_df.index, palette='viridis', legend=False)
        plt.title(f"{metric} Comparison (2D)", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Algorithm", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"results/{metric}_bar_comparison_2d.png", dpi=300)
        plt.close()

# 4. Heatmap for All Metrics
def plot_metrics_heatmap(summary_df, metrics, title="Metrics Heatmap (2D)", filename="metrics_heatmap_2d.png"):
    ensure_results_directory()
    plt.figure(figsize=(10, 8))
    sns.heatmap(summary_df[metrics], annot=True, cmap='viridis', fmt='.4f', annot_kws={"size": 10})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

# 5. Compare K-Means vs FCM
def compare_kmeans_fcm(data, kmeans_labels, fcm_labels, kmeans_centroids, fcm_centroids, scaler=None, 
                       title_left="K-Means", title_right="Fuzzy C-Means", 
                       filename="comparison_kmeans_fcm_2d.png"):
    ensure_results_directory()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data.shape[1]} dimensions")
    
    # Inverse transform the data and centroids for visualization
    if scaler:
        data = scaler.inverse_transform(data)
        kmeans_centroids = scaler.inverse_transform(kmeans_centroids)
        fcm_centroids = scaler.inverse_transform(fcm_centroids)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # K-Means plot (left)
    scatter1 = ax1.scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap="viridis", alpha=0.8, s=100)
    ax1.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax1.set_title(f"{title_left} Clustering Results", fontsize=14)
    ax1.set_xlabel("Annual Income (k$)", fontsize=12)
    ax1.set_ylabel("Spending Score (1-100)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # FCM plot (right)
    scatter2 = ax2.scatter(data[:, 0], data[:, 1], c=fcm_labels, cmap="viridis", alpha=0.8, s=100)
    ax2.scatter(fcm_centroids[:, 0], fcm_centroids[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centroids')
    ax2.set_title(f"{title_right} Clustering Results", fontsize=14)
    ax2.set_xlabel("Annual Income (k$)", fontsize=12)
    ax2.set_ylabel("Spending Score (1-100)", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label="Cluster")
    plt.colorbar(scatter2, ax=ax2, label="Cluster")
    
    plt.suptitle(f"Comparison of {title_left} and {title_right} Clustering", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

    # Calculate silhouette scores
    kmeans_silhouette = np.nan
    if len(np.unique(kmeans_labels)) > 1:
        kmeans_silhouette = silhouette_score(data, kmeans_labels)
    
    fcm_silhouette = np.nan
    if len(np.unique(fcm_labels)) > 1:
        fcm_silhouette = silhouette_score(data, fcm_labels)
    
    return {
        "kmeans_silhouette": kmeans_silhouette,
        "fcm_silhouette": fcm_silhouette
    }

# 6. Compare FCM vs GK-FCM
def compare_fcm_gkfcm(data_2d, fcm_labels, gkfcm_labels, fcm_centroids, gkfcm_centroids, 
                      title_left="Fuzzy C-Means", title_right="Gustafson-Kessel FCM", 
                      filename="comparison_fcm_gkfcm_2d.png", scaler=None):
    ensure_results_directory()
    if data_2d.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data_2d.shape[1]} dimensions")
    
    # Inverse transform the data and centroids
    if scaler:
        data_2d = scaler.inverse_transform(data_2d)
        fcm_centroids = scaler.inverse_transform(fcm_centroids)
        gkfcm_centroids = scaler.inverse_transform(gkfcm_centroids)
    
    n_clusters = len(fcm_centroids)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Plot FCM results (left)
    for i in range(n_clusters):
        cluster_points = data_2d[fcm_labels == i]
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=colors[i], alpha=0.7, s=80, label=f'Cluster {i+1}')
    
    ax1.scatter(fcm_centroids[:, 0], fcm_centroids[:, 1], 
               marker='x', s=200, linewidths=3, color='red', label='Centroids')
    
    ax1.set_title(title_left, fontsize=14)
    ax1.set_xlabel("Annual Income (k$)", fontsize=12)
    ax1.set_ylabel("Spending Score (1-100)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot GK-FCM results (right)
    for i in range(n_clusters):
        cluster_points = data_2d[gkfcm_labels == i]
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=colors[i], alpha=0.7, s=80, label=f'Cluster {i+1}')
    
    ax2.scatter(gkfcm_centroids[:, 0], gkfcm_centroids[:, 1], 
               marker='x', s=200, linewidths=3, color='red', label='Centroids')
    
    ax2.set_title(title_right, fontsize=14)
    ax2.set_xlabel("Annual Income (k$)", fontsize=12)
    ax2.set_ylabel("Spending Score (1-100)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
              bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=n_clusters+1)
    
    fig.suptitle(f"Comparison: {title_left} vs {title_right}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

    # Calculate silhouette scores
    fcm_silhouette = np.nan
    if len(np.unique(fcm_labels)) > 1:
        fcm_silhouette = silhouette_score(data_2d, fcm_labels)
    
    gkfcm_silhouette = np.nan
    if len(np.unique(gkfcm_labels)) > 1:
        gkfcm_silhouette = silhouette_score(data_2d, gkfcm_labels)
    
    return {
        "fcm_silhouette": fcm_silhouette,
        "gkfcm_silhouette": gkfcm_silhouette
    }

# 7. Effect of Fuzziness Parameter (m) on FCM
def plot_fcm_m_comparison(data, m_values, labels_list, centroids_list, silhouette_scores, wcss_values, scaler=None, 
                          filename="fcm_m_comparison.png", title="FCM: Effect of Fuzziness Parameter (m)"):
    ensure_results_directory()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data.shape[1]} dimensions")
    
    # Inverse transform the data for visualization
    display_data = data.copy()
    if scaler:
        display_data = scaler.inverse_transform(data)
    
    # Create figure with subplots in a grid
    n_plots = len(m_values)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each m value's clustering result
    for i, (m, labels, centroids) in enumerate(zip(m_values, labels_list, centroids_list)):
        # Transform centroids for visualization if needed
        display_centroids = centroids.copy()
        if scaler:
            display_centroids = scaler.inverse_transform(centroids)
        
        scatter = axes[i].scatter(display_data[:, 0], display_data[:, 1], 
                                 c=labels, cmap="viridis", alpha=0.8, s=80)
        axes[i].scatter(display_centroids[:, 0], display_centroids[:, 1], 
                       marker='x', s=200, linewidths=3, color='red', label='Centroids')
        axes[i].set_title(f"FCM with m={m}", fontsize=12)
        axes[i].set_xlabel("Annual Income (k$)", fontsize=10)
        axes[i].set_ylabel("Spending Score (1-100)", fontsize=10)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle("FCM Clustering with Different m Values", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"results/fcm_m_clustering_results.png", dpi=300)
    plt.close()
    
    # Plot metrics comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # WCSS plot
    ax1.plot(m_values, wcss_values, 'bo-', label='WCSS')
    ax1.set_title("WCSS vs m Value (Lower is Better)", fontsize=12)
    ax1.set_xlabel("Fuzziness Parameter (m)", fontsize=10)
    ax1.set_ylabel("WCSS", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Silhouette score plot
    ax2.plot(m_values, silhouette_scores, 'ro-', label='Silhouette Score')
    ax2.set_title("Silhouette Score vs m Value (Higher is Better)", fontsize=12)
    ax2.set_xlabel("Fuzziness Parameter (m)", fontsize=10)
    ax2.set_ylabel("Silhouette Score", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()
    
    return {
        "m_values": m_values,
        "wcss_values": wcss_values,
        "silhouette_scores": silhouette_scores
    }

# 8. Effect of Sigma on KFCM and MKFCM
def plot_sigma_parameter_study(kfcm_results, mkfcm_results, sigma_values, title="KFCM vs MKFCM: Effect of Sigma Parameter", filename="kernel_sigma_comparison.png"):
    ensure_results_directory()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Silhouette scores
    ax1.plot(sigma_values, [r["silhouette"] for r in kfcm_results], label="KFCM", marker='o', color='blue')
    ax1.plot(sigma_values, [r["silhouette"] for r in mkfcm_results], label="MKFCM", marker='s', color='orange')
    ax1.set_xlabel("Sigma Squared", fontsize=12)
    ax1.set_ylabel("Silhouette Score", fontsize=12)
    ax1.set_title("Silhouette Score vs Sigma", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Kernel inertia
    ax2.plot(sigma_values, [r["inertia"] for r in kfcm_results], label="KFCM", marker='o', color='blue')
    ax2.plot(sigma_values, [r["inertia"] for r in mkfcm_results], label="MKFCM", marker='s', color='orange')
    ax2.set_xlabel("Sigma Squared", fontsize=12)
    ax2.set_ylabel("Kernel Inertia", fontsize=12)
    ax2.set_title("Kernel Inertia vs Sigma", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Execution time
    ax3.plot(sigma_values, [r["time"] for r in kfcm_results], label="KFCM", marker='o', color='blue')
    ax3.plot(sigma_values, [r["time"] for r in mkfcm_results], label="MKFCM", marker='s', color='orange')
    ax3.set_xlabel("Sigma Squared", fontsize=12)
    ax3.set_ylabel("Time (seconds)", fontsize=12)
    ax3.set_title("Execution Time vs Sigma", fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

# 9. GK-FCM Covariance Matrices
def plot_gkfcm_covariance_matrices(data, labels, centroids, covariance_matrices, scaler=None):
    ensure_results_directory()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data.shape[1]} dimensions")
    
    # Inverse transform the data and centroids
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)
    
    plt.figure(figsize=(12, 9))
    
    n_clusters = len(centroids)
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Plot the data points
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=colors[i], alpha=0.7, s=80, label=f'Cluster {i+1}')
    
    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               marker='x', s=200, linewidths=3, color='red', label='Centroids')
    
    # Function to draw confidence ellipses
    def confidence_ellipse(cov, mean, ax, n_std=2.0, facecolor='none', **kwargs):
        try:
            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            if np.isnan(pearson) or not (-1 <= pearson <= 1):
                return None
        except (ValueError, ZeroDivisionError):
            return None
        
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                         facecolor=facecolor, **kwargs)
        
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        
        if scale_x <= 0 or scale_y <= 0:
            return None
        
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean[0], mean[1])
        
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    
    # Plot the covariance matrices as confidence ellipses
    for i in range(n_clusters):
        ellipse = confidence_ellipse(
            covariance_matrices[i][:2, :2],
            centroids[i][:2],
            plt.gca(),
            n_std=2.0,
            edgecolor=colors[i],
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            label=f'Covariance {i+1}'
        )
        if ellipse is None:
            print(f"Warning: Could not plot covariance ellipse for cluster {i+1}")
    
    plt.xlabel("Annual Income (k$)", fontsize=12)
    plt.ylabel("Spending Score (1-100)", fontsize=12)
    plt.title("GK-FCM Clustering with Covariance Matrices", fontsize=14, fontweight='bold')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    seen_labels = set()
    for h, l in zip(handles, labels):
        if l not in seen_labels:
            seen_labels.add(l)
            unique_labels.append(l)
            unique_handles.append(h)
    
    plt.legend(unique_handles, unique_labels, loc='upper center', 
              bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    
    plt.tight_layout()
    plt.savefig("results/gkfcm_covariance_matrices.png", dpi=300)
    plt.close()

# 10. Compare All Fuzzy Algorithms
def compare_all_fuzzy(data, labels_dict, centroids_dict, scaler=None, title="Comparison of All Fuzzy Clustering Algorithms", filename="all_fuzzy_comparison.png"):
    ensure_results_directory()
    algorithms = list(labels_dict.keys())
    n_algos = len(algorithms)
    ncols = 3
    nrows = (n_algos + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.ravel()
    
    # Inverse transform data and centroids if scaler is provided
    if scaler:
        data = scaler.inverse_transform(data)
        centroids_dict = {algo: scaler.inverse_transform(centroids) for algo, centroids in centroids_dict.items()}
    
    for idx, algo in enumerate(algorithms):
        labels = labels_dict[algo]
        centroids = centroids_dict[algo]
        
        axes[idx].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
        axes[idx].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
        axes[idx].set_title(algo, fontsize=12)
        axes[idx].set_xlabel("Annual Income (k$)", fontsize=10)
        axes[idx].set_ylabel("Spending Score (1-100)", fontsize=10)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Turn off unused subplots
    for idx in range(len(algorithms), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

# 11. Incremental Parameters Comparison for rseKFCM, spKFCM, oKFCM
def plot_incremental_params_comparison(data, rsekfcm_results, spkfcm_results, okfcm_results, scaler=None, 
                                       title="Incremental Parameters Comparison", filename="incremental_params_comparison.png"):
    ensure_results_directory()
    n_subplots = len(rsekfcm_results) + len(spkfcm_results) + len(okfcm_results)
    ncols = 3
    nrows = (n_subplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.ravel()

    # Inverse transform data if scaler is provided
    if scaler:
        data_transformed = scaler.inverse_transform(data)
        for results in [rsekfcm_results, spkfcm_results, okfcm_results]:
            for result in results:
                result["centroids"] = scaler.inverse_transform(result["centroids"])
    else:
        data_transformed = data

    idx = 0

    # Plot rseKFCM results
    for result in rsekfcm_results:
        sample_size = result.get("sample_size")
        labels = result.get("labels")
        centroids = result.get("centroids")
        
        axes[idx].scatter(data_transformed[:, 0], data_transformed[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
        axes[idx].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
        axes[idx].set_title(f"rseKFCM (sample_size={sample_size})", fontsize=12)
        axes[idx].set_xlabel("Annual Income (k$)", fontsize=10)
        axes[idx].set_ylabel("Spending Score (1-100)", fontsize=10)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Plot spKFCM results
    for result in spkfcm_results:
        n_chunks = result.get("n_chunks")
        labels = result.get("labels")
        centroids = result.get("centroids")
        
        axes[idx].scatter(data_transformed[:, 0], data_transformed[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
        axes[idx].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
        axes[idx].set_title(f"spKFCM (n_chunks={n_chunks})", fontsize=12)
        axes[idx].set_xlabel("Annual Income (k$)", fontsize=10)
        axes[idx].set_ylabel("Spending Score (1-100)", fontsize=10)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Plot oKFCM results
    for result in okfcm_results:
        n_chunks = result.get("n_chunks")
        labels = result.get("labels")
        centroids = result.get("centroids")
        
        axes[idx].scatter(data_transformed[:, 0], data_transformed[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
        axes[idx].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
        axes[idx].set_title(f"oKFCM (n_chunks={n_chunks})", fontsize=12)
        axes[idx].set_xlabel("Annual Income (k$)", fontsize=10)
        axes[idx].set_ylabel("Spending Score (1-100)", fontsize=10)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        idx += 1

    # Turn off unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

# 12. Main Function to Run All Visualizations
def run_all_visualizations(data_2d, data_3d, scaler_2d, scaler_3d):
    algorithms = ['rseKFCM', 'spKFCM', 'oKFCM', 'FCM', 'KFCM', 'MKFCM', 'GK-FCM', 'K-Means']

    # Load results from the 'run_results' folder
    summary = {}
    labels_dict = {}
    centroids_dict = {}
    for algo in algorithms:
        df = pd.read_csv(f"run_results/{algo}_run_results.csv")
        summary[algo] = df.mean()
        labels_dict[algo] = np.load(f"run_results/{algo}_labels_2d.npy")
        centroids_dict[algo] = np.load(f"run_results/{algo}_centroids_2d.npy")

    summary_df = pd.DataFrame(summary).T
    metrics = ['Silhouette_2D', 'WCSS_2D', 'Davies_Bouldin_2D', 'Partition_Coefficient_2D', 'Xie_Beni_2D', 'Time']

    # 1. Bar Plots for Metrics Comparison
    plot_metrics_bar_comparison(summary_df, metrics)

    # 2. Heatmap for Metrics Comparison
    plot_metrics_heatmap(summary_df, metrics)

    # 3. Scatter Plots for Each Algorithm (2D)
    for algo in algorithms:
        plot_clusters_2d(data_2d, labels_dict[algo], centroids_dict[algo], algo, scaler_2d)

    # 4. Scatter Plots for Each Algorithm (3D)
    for algo in algorithms:
        labels_3d = np.load(f"run_results/{algo}_labels_3d.npy")
        centroids_3d = np.load(f"run_results/{algo}_centroids_3d.npy")
        plot_clusters_3d(data_3d, labels_3d, centroids_3d, algo, scaler_3d)

    # 5. Compare K-Means vs FCM
    compare_kmeans_fcm(data_2d, labels_dict['K-Means'], labels_dict['FCM'], 
                       centroids_dict['K-Means'], centroids_dict['FCM'], scaler_2d)

    # 6. Compare FCM vs GK-FCM
    compare_fcm_gkfcm(data_2d, labels_dict['FCM'], labels_dict['GK-FCM'], 
                      centroids_dict['FCM'], centroids_dict['GK-FCM'], scaler_2d)

    # 7. Compare All Fuzzy Algorithms
    fuzzy_algorithms = ['rseKFCM', 'spKFCM', 'oKFCM', 'FCM', 'KFCM', 'MKFCM', 'GK-FCM']
    fuzzy_labels_dict = {algo: labels_dict[algo] for algo in fuzzy_algorithms}
    fuzzy_centroids_dict = {algo: centroids_dict[algo] for algo in fuzzy_algorithms}
    compare_all_fuzzy(data_2d, fuzzy_labels_dict, fuzzy_centroids_dict, scaler_2d)

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('Mall_Customers.csv')
    data_2d = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
    data_3d = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

    from sklearn.preprocessing import StandardScaler
    # Create separate scalers for 2D and 3D data
    scaler_2d = StandardScaler()
    scaler_3d = StandardScaler()
    
    # Fit and transform the data
    data_2d = scaler_2d.fit_transform(data_2d)
    data_3d = scaler_3d.fit_transform(data_3d)

    # Run all visualizations
    run_all_visualizations(data_2d, data_3d, scaler_2d, scaler_3d)