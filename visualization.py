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
def plot_clusters_2d(data, labels, centroids, algo_name, scaler=None, m=None, n_clusters=None, filename_prefix=""):
    ensure_results_directory()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data.shape[1]} dimensions")
    if centroids.shape[1] != 2:
        raise ValueError(f"Expected 2D centroids, but got centroids with {centroids.shape[1]} dimensions")
    
    # Check for valid labels and centroids
    if np.any(np.isnan(labels)) or np.any(np.isnan(centroids)):
        print(f"Warning: Invalid labels or centroids for {algo_name} (m={m}, n_clusters={n_clusters}). Skipping plot.")
        return
    
    # Inverse transform data and centroids for visualization
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", alpha=0.8, s=100)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centroids')
    plt.title(f"Clustering Results (2D) - {algo_name} (m={m}, n_clusters={n_clusters})", fontsize=16, fontweight='bold')
    plt.xlabel("Annual Income (k$)", fontsize=14)
    plt.ylabel("Spending Score (1-100)", fontsize=14)
    plt.colorbar(label="Cluster")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"results/{filename_prefix}{algo_name}_clusters_2d_m_{m}_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# 2. Scatter Plots for 3D Clustering
def plot_clusters_3d(data, labels, centroids, algo_name, scaler=None, m=None, n_clusters=None, filename_prefix=""):
    ensure_results_directory()
    if data.shape[1] != 3:
        raise ValueError(f"Expected 3D data, but got data with {data.shape[1]} dimensions")
    if centroids.shape[1] != 3:
        raise ValueError(f"Expected 3D centroids, but got centroids with {centroids.shape[1]} dimensions")
    
    # Check for valid labels and centroids
    if np.any(np.isnan(labels)) or np.any(np.isnan(centroids)):
        print(f"Warning: Invalid labels or centroids for {algo_name} (m={m}, n_clusters={n_clusters}). Skipping plot.")
        return
    
    # Inverse transform data and centroids for visualization
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="viridis", s=100, alpha=0.8)
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, linewidths=3, color='red', label='Centroids')
    
    ax.legend(fontsize=12)
    plt.colorbar(scatter, label="Cluster")
    
    ax.set_xlabel("Age", fontsize=14)
    ax.set_ylabel("Annual Income (k$)", fontsize=14)
    ax.set_zlabel("Spending Score (1-100)", fontsize=14)
    ax.set_title(f"Clustering Results (3D) - {algo_name} (m={m}, n_clusters={n_clusters})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    filename = f"results/{filename_prefix}{algo_name}_clusters_3d_m_{m}_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# 3. Bar Plots for Metrics Comparison
def plot_metrics_bar_comparison(summary_df, metrics, m, n_clusters, title="Metrics Comparison (2D)", filename_prefix=""):
    ensure_results_directory()
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=summary_df.index, y=summary_df[metric], hue=summary_df.index, palette='viridis', legend=False)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f', fontsize=10, padding=3)  # Add values on bars
        plt.title(f"{metric} Comparison (2D, m={m}, n_clusters={n_clusters})", fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.ylabel(metric, fontsize=14)
        plt.xlabel("Algorithm", fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        filename = f"results/{filename_prefix}{metric}_bar_comparison_2d_m_{m}_n_clusters_{n_clusters}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

# 4. Heatmap for All Metrics
def plot_metrics_heatmap(summary_df, metrics, m, n_clusters, title="Metrics Heatmap (2D)", filename_prefix=""):
    ensure_results_directory()
    plt.figure(figsize=(12, 8))
    sns.heatmap(summary_df[metrics], annot=True, cmap='viridis', fmt='.4f', annot_kws={"size": 12})
    plt.title(f"{title} (m={m}, n_clusters={n_clusters})", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    filename = f"results/{filename_prefix}metrics_heatmap_2d_m_{m}_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# 5. Compare K-Means vs FCM
def compare_kmeans_fcm(data, kmeans_labels, fcm_labels, kmeans_centroids, fcm_centroids, scaler=None, 
                       m=None, n_clusters=None, title_left="K-Means", title_right="Fuzzy C-Means", filename_prefix=""):
    ensure_results_directory()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data.shape[1]} dimensions")
    
    # Check for valid labels and centroids
    if np.any(np.isnan(kmeans_labels)) or np.any(np.isnan(fcm_labels)) or np.any(np.isnan(kmeans_centroids)) or np.any(np.isnan(fcm_centroids)):
        print(f"Warning: Invalid labels or centroids for K-Means or FCM (m={m}, n_clusters={n_clusters}). Skipping plot.")
        return
    
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
    
    plt.suptitle(f"Comparison of {title_left} and {title_right} (m={m}, n_clusters={n_clusters})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"results/{filename_prefix}comparison_kmeans_fcm_2d_m_{m}_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
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
                      scaler=None, m=None, n_clusters=None, 
                      title_left="Fuzzy C-Means", title_right="Gustafson-Kessel FCM", filename_prefix=""):
    ensure_results_directory()
    if data_2d.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data_2d.shape[1]} dimensions")
    
    # Check for valid labels and centroids
    if np.any(np.isnan(fcm_labels)) or np.any(np.isnan(gkfcm_labels)) or np.any(np.isnan(fcm_centroids)) or np.any(np.isnan(gkfcm_centroids)):
        print(f"Warning: Invalid labels or centroids for FCM or GK-FCM (m={m}, n_clusters={n_clusters}). Skipping plot.")
        return
    
    # Inverse transform the data and centroids
    if scaler:
        data_2d = scaler.inverse_transform(data_2d)
        fcm_centroids = scaler.inverse_transform(fcm_centroids)
        gkfcm_centroids = scaler.inverse_transform(gkfcm_centroids)
    
    n_clusters_val = len(fcm_centroids)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters_val))
    
    # Plot FCM results (left)
    for i in range(n_clusters_val):
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
    for i in range(n_clusters_val):
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
              bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=n_clusters_val+1)
    
    fig.suptitle(f"Comparison: {title_left} vs {title_right} (m={m}, n_clusters={n_clusters})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    filename = f"results/{filename_prefix}comparison_fcm_gkfcm_2d_m_{m}_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
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

# 7. Effect of m on Performance for a Single Algorithm
def plot_m_comparison(data, algo_name, m_values, labels_dict, centroids_dict, metrics_dict, scaler=None, n_clusters=None, filename_prefix=""):
    ensure_results_directory()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data.shape[1]} dimensions")
    
    # Inverse transform the data for visualization
    display_data = data.copy()
    if scaler:
        display_data = scaler.inverse_transform(data)
    
    # Create figure with subplots for clustering results
    n_plots = len(m_values)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each m value's clustering result
    for i, m in enumerate(m_values):
        labels = labels_dict.get(m)
        centroids = centroids_dict.get(m)
        
        # Check for valid labels and centroids
        if labels is None or centroids is None or np.any(np.isnan(labels)) or np.any(np.isnan(centroids)):
            print(f"Warning: Invalid labels or centroids for {algo_name} with m={m}, n_clusters={n_clusters}. Skipping subplot.")
            axes[i].axis('off')
            continue
        
        # Transform centroids for visualization if needed
        display_centroids = centroids.copy()
        if scaler:
            display_centroids = scaler.inverse_transform(centroids)
        
        scatter = axes[i].scatter(display_data[:, 0], display_data[:, 1], 
                                 c=labels, cmap="viridis", alpha=0.8, s=80)
        axes[i].scatter(display_centroids[:, 0], display_centroids[:, 1], 
                       marker='x', s=200, linewidths=3, color='red', label='Centroids')
        axes[i].set_title(f"{algo_name} with m={m}", fontsize=12)
        axes[i].set_xlabel("Annual Income (k$)", fontsize=10)
        axes[i].set_ylabel("Spending Score (1-100)", fontsize=10)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f"{algo_name} Clustering with Different m Values (n_clusters={n_clusters})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    filename = f"results/{filename_prefix}{algo_name}_m_comparison_clusters_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    # Plot metrics comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Silhouette Score plot
    silhouette_scores = [metrics_dict[m]['Avg_Silhouette_2D'] for m in m_values]
    ax1.plot(m_values, silhouette_scores, 'ro-', label='Silhouette Score')
    ax1.set_title("Silhouette Score vs m Value (Higher is Better)", fontsize=12)
    ax1.set_xlabel("Fuzziness Parameter (m)", fontsize=10)
    ax1.set_ylabel("Silhouette Score", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # WCSS plot
    wcss_values = [metrics_dict[m]['Avg_WCSS_2D'] for m in m_values]
    ax2.plot(m_values, wcss_values, 'bo-', label='WCSS')
    ax2.set_title("WCSS vs m Value (Lower is Better)", fontsize=12)
    ax2.set_xlabel("Fuzziness Parameter (m)", fontsize=10)
    ax2.set_ylabel("WCSS", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(f"{algo_name}: Effect of m (n_clusters={n_clusters})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    filename = f"results/{filename_prefix}{algo_name}_m_comparison_metrics_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# 8. Effect of n_clusters on Performance
def plot_n_clusters_comparison(summary_df, n_clusters_values, metric="Avg_Silhouette_2D", m=None, filename_prefix=""):
    ensure_results_directory()
    plt.figure(figsize=(12, 6))
    for algo in summary_df['Algorithm'].unique():
        algo_data = summary_df[(summary_df['Algorithm'] == algo) & (summary_df['m'] == m)]
        if not algo_data.empty:
            plt.plot(algo_data['n_clusters'], algo_data[metric], marker='o', label=algo)
    
    plt.title(f"Effect of n_clusters on {metric} (m={m})", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Clusters (n_clusters)", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"results/{filename_prefix}n_clusters_comparison_{metric}_m_{m}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# 9. GK-FCM Covariance Matrices
def plot_gkfcm_covariance_matrices(data, labels, centroids, covariance_matrices, scaler=None, m=None, n_clusters=None, filename_prefix=""):
    ensure_results_directory()
    if data.shape[1] != 2:
        raise ValueError(f"Expected 2D data, but got data with {data.shape[1]} dimensions")
    
    # Check for valid inputs
    if np.any(np.isnan(labels)) or np.any(np.isnan(centroids)) or np.any(np.isnan(covariance_matrices)):
        print(f"Warning: Invalid labels, centroids, or covariance matrices for GK-FCM (m={m}, n_clusters={n_clusters}). Skipping plot.")
        return
    
    # Inverse transform the data and centroids
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)
    
    plt.figure(figsize=(12, 9))
    
    n_clusters_val = len(centroids)
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters_val))
    
    # Plot the data points
    for i in range(n_clusters_val):
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
    for i in range(n_clusters_val):
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
            print(f"Warning: Could not plot covariance ellipse for cluster {i+1} (m={m}, n_clusters={n_clusters})")
    
    plt.xlabel("Annual Income (k$)", fontsize=12)
    plt.ylabel("Spending Score (1-100)", fontsize=12)
    plt.title(f"GK-FCM Clustering with Covariance Matrices (m={m}, n_clusters={n_clusters})", fontsize=14, fontweight='bold')
    
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
    filename = f"results/{filename_prefix}gkfcm_covariance_matrices_m_{m}_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# 10. Compare All Algorithms
def compare_all_algorithms(data, labels_dict, centroids_dict, scaler=None, m=None, n_clusters=None, filename_prefix=""):
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
        
        # Check for valid labels and centroids
        if np.any(np.isnan(labels)) or np.any(np.isnan(centroids)):
            print(f"Warning: Invalid labels or centroids for {algo} (m={m}, n_clusters={n_clusters}). Skipping subplot.")
            axes[idx].axis('off')
            continue
        
        axes[idx].scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
        axes[idx].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
        axes[idx].set_title(f"{algo} (m={m}, n_clusters={n_clusters})", fontsize=12)
        axes[idx].set_xlabel("Annual Income (k$)", fontsize=10)
        axes[idx].set_ylabel("Spending Score (1-100)", fontsize=10)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    # Turn off unused subplots
    for idx in range(len(algorithms), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Comparison of All Clustering Algorithms (m={m}, n_clusters={n_clusters})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    filename = f"results/{filename_prefix}all_algorithms_comparison_m_{m}_n_clusters_{n_clusters}.png"
    plt.savefig(filename, dpi=300)
    plt.close()

# 11. Main Function to Run All Visualizations
def run_all_visualizations(data_2d, data_3d, scaler_2d, scaler_3d,m_values=[1.5, 2.0, 2.5], n_clusters_values=[3, 5]):
    algorithms = ['rseKFCM', 'spKFCM', 'oKFCM', 'FCM', 'KFCM', 'MKFCM', 'GK-FCM', 'K-Means', 'ImprovedGathGeva', 'IFCM']
    metrics = ['Avg_Silhouette_2D', 'Avg_WCSS_2D', 'Avg_Davies_Bouldin_2D', 
               'Avg_Partition_Coefficient_2D', 'Avg_Xie_Beni_2D', 'Avg_Time']

    # Load summary results
    try:
        summary_df = pd.read_csv('summary_results.csv')
    except FileNotFoundError:
        print("Error: summary_results.csv not found. Please run experiment_runner.py first.")
        return

    for m in m_values:
        for n_clusters in n_clusters_values:
            results_dir = f"run_results_m_{m}_n_clusters_{n_clusters}"
            print(f"Generating visualizations for m={m}, n_clusters={n_clusters}...")

            # Load labels and centroids for all algorithms
            labels_dict_2d = {}
            centroids_dict_2d = {}
            labels_dict_3d = {}
            centroids_dict_3d = {}
            for algo in algorithms:
                try:
                    labels_dict_2d[algo] = np.load(f"{results_dir}/{algo}_labels_2d.npy")
                    centroids_dict_2d[algo] = np.load(f"{results_dir}/{algo}_centroids_2d.npy")
                    labels_dict_3d[algo] = np.load(f"{results_dir}/{algo}_labels_3d.npy")
                    centroids_dict_3d[algo] = np.load(f"{results_dir}/{algo}_centroids_3d.npy")
                except FileNotFoundError:
                    print(f"Warning: Results for {algo} (m={m}, n_clusters={n_clusters}) not found. Skipping.")
                    continue

            # Filter summary data for current m and n_clusters
            subset_df = summary_df[(summary_df['m'] == m) & (summary_df['n_clusters'] == n_clusters)]
            subset_df = subset_df.set_index('Algorithm')

            # 1. Bar Plots for Metrics Comparison
            plot_metrics_bar_comparison(subset_df, metrics, m, n_clusters)

            # 2. Heatmap for Metrics Comparison
            plot_metrics_heatmap(subset_df, metrics, m, n_clusters)

            # 3. Scatter Plots for Each Algorithm (2D and 3D)
            for algo in algorithms:
                if algo in labels_dict_2d and algo in centroids_dict_2d:
                    plot_clusters_2d(data_2d, labels_dict_2d[algo], centroids_dict_2d[algo], algo, scaler_2d, m, n_clusters)
                if algo in labels_dict_3d and algo in centroids_dict_3d:
                    plot_clusters_3d(data_3d, labels_dict_3d[algo], centroids_dict_3d[algo], algo, scaler_3d, m, n_clusters)

            # 4. Compare K-Means vs FCM
            if 'K-Means' in labels_dict_2d and 'FCM' in labels_dict_2d:
                compare_kmeans_fcm(data_2d, labels_dict_2d['K-Means'], labels_dict_2d['FCM'], 
                                   centroids_dict_2d['K-Means'], centroids_dict_2d['FCM'], scaler_2d, m, n_clusters)

            # 5. Compare FCM vs GK-FCM
            if 'FCM' in labels_dict_2d and 'GK-FCM' in labels_dict_2d:
                compare_fcm_gkfcm(data_2d, labels_dict_2d['FCM'], labels_dict_2d['GK-FCM'], 
                                  centroids_dict_2d['FCM'], centroids_dict_2d['GK-FCM'], scaler_2d, m, n_clusters)

            # 6. Compare All Algorithms
            compare_all_algorithms(data_2d, labels_dict_2d, centroids_dict_2d, scaler_2d, m, n_clusters)

            # 7. Plot GK-FCM Covariance Matrices
            if 'GK-FCM' in labels_dict_2d:
                try:
                    covariance_matrices = np.load(f"{results_dir}/GK-FCM_covariances_2d.npy")
                    plot_gkfcm_covariance_matrices(data_2d, labels_dict_2d['GK-FCM'], centroids_dict_2d['GK-FCM'], 
                                                  covariance_matrices, scaler_2d, m, n_clusters)
                except FileNotFoundError:
                    print(f"Warning: Covariance matrices for GK-FCM (m={m}, n_clusters={n_clusters}) not found. Skipping.")

    # 8. Effect of m for Each Algorithm
    for n_clusters in n_clusters_values:
        for algo in algorithms:
            algo_summary = summary_df[(summary_df['Algorithm'] == algo) & (summary_df['n_clusters'] == n_clusters)]
            if algo_summary.empty:
                print(f"Warning: No summary data for {algo} (n_clusters={n_clusters}). Skipping m comparison.")
                continue
            
            labels_dict = {}
            centroids_dict = {}
            metrics_dict = {}
            for m in m_values:
                results_dir = f"run_results_m_{m}_n_clusters_{n_clusters}"
                try:
                    labels_dict[m] = np.load(f"{results_dir}/{algo}_labels_2d.npy")
                    centroids_dict[m] = np.load(f"{results_dir}/{algo}_centroids_2d.npy")
                    metrics_dict[m] = algo_summary[algo_summary['m'] == m].iloc[0].to_dict()
                except FileNotFoundError:
                    print(f"Warning: Results for {algo} (m={m}, n_clusters={n_clusters}) not found. Skipping m={m}.")
                    continue
            
            if labels_dict and centroids_dict:
                plot_m_comparison(data_2d, algo, m_values, labels_dict, centroids_dict, metrics_dict, scaler_2d, n_clusters)

    # 9. Effect of n_clusters for Each m
    for m in m_values:
        plot_n_clusters_comparison(summary_df, n_clusters_values, metric="Avg_Silhouette_2D", m=m)
        plot_n_clusters_comparison(summary_df, n_clusters_values, metric="Avg_WCSS_2D", m=m)

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('Mall_Customers.csv')
    data_2d = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
    data_3d = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

    from sklearn.preprocessing import StandardScaler
    scaler_2d = StandardScaler()
    scaler_3d = StandardScaler()
    data_2d = scaler_2d.fit_transform(data_2d)
    data_3d = scaler_3d.fit_transform(data_3d)

    # Run all visualizations
    run_all_visualizations(data_2d, data_3d, scaler_2d, scaler_3d,m_values=[1.5, 2.0, 2.5], n_clusters_values=[3, 5]) 