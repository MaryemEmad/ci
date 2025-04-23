# visualization.py: Generates scatter plots and cluster visualizations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def plot_clusters_2d(data, labels, centroids, algo_name, scaler=None):
    # If scaler is provided, inverse transform the data and centroids for visualization
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis", alpha=0.8, s=100)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='black', label='Centroids')
    plt.title(f"Clustering Results (2D) - {algo_name}")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.colorbar(label="Cluster")
    plt.legend()
    plt.savefig(f"{algo_name}_clusters_2d.png")
    plt.close()

def plot_clusters_3d(data, labels, centroids, algo_name, scaler=None):
    # If scaler is provided, inverse transform the data and centroids for visualization
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="viridis", s=100, alpha=0.8)
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, linewidths=3, color='black', label='Centroids')
    
    # Add legend and colorbar
    ax.legend()
    plt.colorbar(scatter, label="Cluster")
    
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    ax.set_title(f"Clustering Results (3D) - {algo_name}")
    plt.savefig(f"{algo_name}_clusters_3d.png")
    plt.close()

def plot_elbow(data, max_k=10, init_method='k-means++'):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init=init_method, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(7, 3))
    plt.plot(range(1, max_k + 1), wcss, 'bx-')
    plt.title(f"Elbow Plot using KMeans ({init_method})")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.savefig("elbow_plot.png")
    plt.close()
