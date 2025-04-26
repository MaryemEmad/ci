# visualization.py: Generates scatter plots and cluster visualizations
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.gridspec import GridSpec

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
    
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, max_k + 1), wcss, 'bx-')
    plt.title(f"Elbow Plot using KMeans ({init_method})")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.savefig("elbow_plot.png")
    plt.close()

def compare_kmeans_fcm(data, kmeans_labels, fcm_labels, kmeans_centroids, fcm_centroids, scaler=None):
    """
    Compare K-means and FCM clustering results side by side
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        Data used for clustering (2D)
    kmeans_labels : array-like of shape (n_samples,)
        Cluster labels from K-means
    fcm_labels : array-like of shape (n_samples,)
        Cluster labels from FCM
    kmeans_centroids : array-like
        Centroids from K-means
    fcm_centroids : array-like
        Centroids from FCM
    scaler : object, default=None
        Scaler used to normalize data (for inverse transform)
    """
    # If scaler is provided, inverse transform the data and centroids for visualization
    if scaler:
        data = scaler.inverse_transform(data)
        kmeans_centroids = scaler.inverse_transform(kmeans_centroids)
        fcm_centroids = scaler.inverse_transform(fcm_centroids)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # K-means plot (left)
    scatter1 = ax1.scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap="viridis", alpha=0.8, s=100)
    ax1.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], marker='x', s=200, linewidths=3, color='black', label='Centroids')
    ax1.set_title("K-means Clustering Results")
    ax1.set_xlabel("Annual Income (k$)")
    ax1.set_ylabel("Spending Score (1-100)")
    ax1.legend()
    
    # FCM plot (right)
    scatter2 = ax2.scatter(data[:, 0], data[:, 1], c=fcm_labels, cmap="viridis", alpha=0.8, s=100)
    ax2.scatter(fcm_centroids[:, 0], fcm_centroids[:, 1], marker='x', s=200, linewidths=3, color='black', label='Centroids')
    ax2.set_title("Fuzzy C-Means Clustering Results")
    ax2.set_xlabel("Annual Income (k$)")
    ax2.set_ylabel("Spending Score (1-100)")
    ax2.legend()
    
    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label="Cluster")
    plt.colorbar(scatter2, ax=ax2, label="Cluster")
    
    # Set common title
    plt.suptitle("Comparison of K-means and Fuzzy C-Means Clustering", fontsize=16)
    plt.tight_layout()
    plt.savefig("comparison_kmeans_fcm_2d.png")
    plt.close()

def plot_wcss_comparison(kmeans_wcss, fcm_wcss, kmeans_inertia=None, fcm_inertia=None):
    """
    Create a bar chart to compare WCSS values for K-means and FCM
    
    Parameters:
    -----------
    kmeans_wcss : float
        WCSS value for K-means
    fcm_wcss : float
        WCSS value for FCM
    kmeans_inertia : float, optional
        Inertia value from K-means algorithm (if different from calculated WCSS)
    fcm_inertia : float, optional
        Inertia value from FCM algorithm (if different from calculated WCSS)
    """
    algorithms = ['K-means', 'FCM']
    wcss_values = [kmeans_wcss, fcm_wcss]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, wcss_values, color=['#1f77b4', '#ff7f0e'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # Add inertia values as text annotations if provided
    if kmeans_inertia is not None:
        plt.annotate(f'Inertia: {kmeans_inertia:.4f}', 
                    xy=(0, kmeans_wcss/2),
                    ha='center',
                    va='center')
    
    if fcm_inertia is not None:
        plt.annotate(f'Inertia: {fcm_inertia:.4f}', 
                    xy=(1, fcm_wcss/2),
                    ha='center',
                    va='center')
    
    # Add improvement percentage
    wcss_improvement = (kmeans_wcss - fcm_wcss) / kmeans_wcss * 100
    
    # Check if we have inertia values to compare
    if kmeans_inertia is not None and fcm_inertia is not None:
        inertia_improvement = (kmeans_inertia - fcm_inertia) / kmeans_inertia * 100
        
        # Add text for both metrics
        plt.text(0.5, max(wcss_values) * 0.92, 
                 f'FCM crisp WCSS: {wcss_improvement:.2f}% change vs K-means',
                 ha='center',
                 fontsize=11,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.text(0.5, max(wcss_values) * 0.84, 
                 f'FCM fuzzy inertia: {inertia_improvement:.2f}% improvement vs K-means',
                 ha='center',
                 fontsize=11,
                 color='green',
                 bbox=dict(facecolor='white', alpha=0.8))
    else:
        # Just add the WCSS comparison
        plt.text(0.5, max(wcss_values) * 0.9, 
                 f'WCSS change: {wcss_improvement:.2f}%',
                 ha='center',
                 fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Within-Cluster Sum of Squares (WCSS) Comparison')
    plt.ylabel('WCSS (lower is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("wcss_comparison.png")
    plt.close()

def plot_fcm_m_comparison(data, m_values, n_clusters=4, scaler=None, random_state=42):
    """
    Compare FCM clustering with different m values
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        Data used for clustering (2D)
    m_values : list
        List of m values to compare
    n_clusters : int, default=4
        Number of clusters
    scaler : object, default=None
        Scaler used to normalize data (for inverse transform)
    random_state : int, default=42
        Random state for reproducibility
    """
    from fcm_clustering import FuzzyCMeansClustering
    
    # If scaler is provided, inverse transform the data for visualization
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
    
    # Track metrics for comparison
    wcss_values = []
    silhouette_scores = []
    
    # Run FCM for each m value
    for i, m in enumerate(m_values):
        # Initialize and fit FCM
        fcm = FuzzyCMeansClustering(n_clusters=n_clusters, m=m, random_state=random_state)
        labels = fcm.fit(data)
        centroids = fcm.get_centroids()
        
        # Transform centroids for visualization if needed
        display_centroids = centroids.copy()
        if scaler:
            display_centroids = scaler.inverse_transform(centroids)
        
        # Plot results
        scatter = axes[i].scatter(display_data[:, 0], display_data[:, 1], 
                                 c=labels, cmap="viridis", alpha=0.8, s=80)
        axes[i].scatter(display_centroids[:, 0], display_centroids[:, 1], 
                       marker='x', s=200, linewidths=3, color='black', label='Centroids')
        axes[i].set_title(f"FCM with m={m}")
        axes[i].set_xlabel("Annual Income (k$)")
        axes[i].set_ylabel("Spending Score (1-100)")
        axes[i].legend()
        
        # Store metrics
        wcss_values.append(fcm.get_crisp_inertia())
        
        # Calculate silhouette score if sklearn is available
        try:
            from sklearn.metrics import silhouette_score
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
        except:
            silhouette_scores.append(0)
    
    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig("fcm_m_clustering_results.png")
    plt.close()
    
    # Plot metrics comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # WCSS plot
    ax1.plot(m_values, wcss_values, 'bo-')
    ax1.set_title("WCSS vs m value (lower is better)")
    ax1.set_xlabel("Fuzziness Parameter (m)")
    ax1.set_ylabel("WCSS")
    ax1.grid(True, alpha=0.3)
    
    # Silhouette score plot
    ax2.plot(m_values, silhouette_scores, 'ro-')
    ax2.set_title("Silhouette Score vs m value (higher is better)")
    ax2.set_xlabel("Fuzziness Parameter (m)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Impact of Fuzziness Parameter (m) on FCM Clustering Quality", fontsize=16)
    plt.tight_layout()
    plt.savefig("fcm_m_comparison.png")
    plt.close()
    
    return {
        "m_values": m_values,
        "wcss_values": wcss_values,
        "silhouette_scores": silhouette_scores
    }

def plot_kmeans_init_comparison(data, n_clusters=4, scaler=None, random_state=42):
    """
    Compare K-means clustering with different initialization methods
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        Data used for clustering (2D)
    n_clusters : int, default=4
        Number of clusters
    scaler : object, default=None
        Scaler used to normalize data (for inverse transform)
    random_state : int, default=42
        Random state for reproducibility
    """
    from kmeans_clustering import KMeansClustering
    
    # If scaler is provided, inverse transform the data for visualization
    display_data = data.copy()
    if scaler:
        display_data = scaler.inverse_transform(data)
    
    # Initialize for both initialization methods
    kmeans_pp = KMeansClustering(n_clusters=n_clusters, 
                                init_method='k-means++', 
                                random_state=random_state)
    kmeans_random = KMeansClustering(n_clusters=n_clusters, 
                                   init_method='random', 
                                   random_state=random_state)
    
    # Fit both models
    labels_pp = kmeans_pp.fit(data)
    labels_random = kmeans_random.fit(data)
    
    # Get centroids
    centroids_pp = kmeans_pp.get_centroids()
    centroids_random = kmeans_random.get_centroids()
    
    # Transform centroids for visualization if needed
    display_centroids_pp = centroids_pp.copy()
    display_centroids_random = centroids_random.copy()
    if scaler:
        display_centroids_pp = scaler.inverse_transform(centroids_pp)
        display_centroids_random = scaler.inverse_transform(centroids_random)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # K-means++ plot
    scatter1 = ax1.scatter(display_data[:, 0], display_data[:, 1], 
                          c=labels_pp, cmap="viridis", alpha=0.8, s=100)
    ax1.scatter(display_centroids_pp[:, 0], display_centroids_pp[:, 1], 
               marker='x', s=200, linewidths=3, color='black', label='Centroids')
    ax1.set_title("K-means++ Initialization")
    ax1.set_xlabel("Annual Income (k$)")
    ax1.set_ylabel("Spending Score (1-100)")
    ax1.legend()
    
    # Random initialization plot
    scatter2 = ax2.scatter(display_data[:, 0], display_data[:, 1], 
                          c=labels_random, cmap="viridis", alpha=0.8, s=100)
    ax2.scatter(display_centroids_random[:, 0], display_centroids_random[:, 1], 
               marker='x', s=200, linewidths=3, color='black', label='Centroids')
    ax2.set_title("Random Initialization")
    ax2.set_xlabel("Annual Income (k$)")
    ax2.set_ylabel("Spending Score (1-100)")
    ax2.legend()
    
    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label="Cluster")
    plt.colorbar(scatter2, ax=ax2, label="Cluster")
    
    # Set common title
    plt.suptitle("Comparison of K-means Initialization Methods", fontsize=16)
    plt.tight_layout()
    plt.savefig("kmeans_init_comparison.png")
    plt.close()
    
    # Plot metrics comparison
    inertia_pp = kmeans_pp.compute_fitness(data)
    inertia_random = kmeans_random.compute_fitness(data)
    
    # Silhouette scores
    silhouette_pp = 0
    silhouette_random = 0
    try:
        from sklearn.metrics import silhouette_score
        silhouette_pp = silhouette_score(data, labels_pp)
        silhouette_random = silhouette_score(data, labels_random)
    except:
        pass
    
    # Create comparison figure
    plt.figure(figsize=(10, 6))
    
    # Plot WCSS comparison
    init_methods = ['k-means++', 'random']
    inertia_values = [inertia_pp, inertia_random]
    silhouette_values = [silhouette_pp, silhouette_random]
    
    x = np.arange(len(init_methods))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # WCSS plot
    bars1 = ax1.bar(init_methods, inertia_values, width, color=['#1f77b4', '#ff7f0e'])
    ax1.set_ylabel('WCSS (lower is better)')
    ax1.set_title('WCSS by Initialization Method')
    ax1.set_ylim(0, max(inertia_values) * 1.1)
    
    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(inertia_values),
                f'{height:.2f}', ha='center', va='bottom')
    
    # Silhouette plot
    bars2 = ax2.bar(init_methods, silhouette_values, width, color=['#1f77b4', '#ff7f0e'])
    ax2.set_ylabel('Silhouette Score (higher is better)')
    ax2.set_title('Silhouette Score by Initialization Method')
    ax2.set_ylim(0, max(silhouette_values) * 1.1)
    
    # Add values on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(silhouette_values),
                f'{height:.4f}', ha='center', va='bottom')
    
    # Calculate and display improvement percentage
    inertia_improvement = (inertia_random - inertia_pp) / inertia_random * 100
    plt.text(0.5, 0.01, 
             f'k-means++ improves WCSS by {inertia_improvement:.2f}% over random initialization',
             ha='center',
             fontsize=12,
             transform=fig.transFigure,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.suptitle("Impact of K-means Initialization Method", fontsize=16)
    plt.tight_layout()
    plt.savefig("kmeans_init_improvement.png")
    plt.close()
    
    return {
        "inertia_values": inertia_values,
        "silhouette_values": silhouette_values,
        "inertia_improvement": inertia_improvement
    }

def plot_convergence_curves(algorithms_history, algorithm_names=None):
    """
    Plot convergence curves for multiple algorithms
    
    Parameters:
    -----------
    algorithms_history : dict or list
        Dict of algorithm name -> fitness history list, or 
        list of fitness history lists
    algorithm_names : list, optional
        List of algorithm names (required if algorithms_history is a list)
    """
    plt.figure(figsize=(12, 8))
    
    # Handle both dict and list inputs
    if isinstance(algorithms_history, dict):
        for algo_name, history in algorithms_history.items():
            plt.plot(range(1, len(history) + 1), history, marker='o', markersize=4, 
                     label=algo_name)
    else:
        if algorithm_names is None:
            algorithm_names = [f"Algorithm {i+1}" for i in range(len(algorithms_history))]
        
        for i, history in enumerate(algorithms_history):
            plt.plot(range(1, len(history) + 1), history, marker='o', markersize=4, 
                     label=algorithm_names[i])
    
    plt.title("Convergence Curves (Iterations vs. Objective Function)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value (WCSS)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergence_curves.png")
    plt.close()
