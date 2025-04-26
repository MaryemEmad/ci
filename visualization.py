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

def compare_kmeans_fcm(data, kmeans_labels, fcm_labels, kmeans_centroids, fcm_centroids, scaler=None, 
                      title_left="K-means", title_right="Fuzzy C-Means", 
                      filename="comparison_kmeans_fcm_2d.png"):
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
    title_left : str, default="K-means"
        Title for the left subplot
    title_right : str, default="Fuzzy C-Means"
        Title for the right subplot
    filename : str, default="comparison_kmeans_fcm_2d.png"
        Filename for the saved plot
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
    ax1.set_title(f"{title_left} Clustering Results")
    ax1.set_xlabel("Annual Income (k$)")
    ax1.set_ylabel("Spending Score (1-100)")
    ax1.legend()
    
    # FCM plot (right)
    scatter2 = ax2.scatter(data[:, 0], data[:, 1], c=fcm_labels, cmap="viridis", alpha=0.8, s=100)
    ax2.scatter(fcm_centroids[:, 0], fcm_centroids[:, 1], marker='x', s=200, linewidths=3, color='black', label='Centroids')
    ax2.set_title(f"{title_right} Clustering Results")
    ax2.set_xlabel("Annual Income (k$)")
    ax2.set_ylabel("Spending Score (1-100)")
    ax2.legend()
    
    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label="Cluster")
    plt.colorbar(scatter2, ax=ax2, label="Cluster")
    
    # Set common title
    plt.suptitle(f"Comparison of {title_left} and {title_right} Clustering", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_wcss_comparison(wcss1, wcss2, algorithms=['K-means', 'FCM'], 
                       inertia1=None, inertia2=None, 
                       filename="wcss_comparison.png"):
    """
    Create a bar chart to compare WCSS values for two algorithms
    
    Parameters:
    -----------
    wcss1 : float
        WCSS value for first algorithm
    wcss2 : float
        WCSS value for second algorithm
    algorithms : list, default=['K-means', 'FCM']
        Names of the algorithms to compare
    inertia1 : float, optional
        Inertia value from first algorithm (if different from calculated WCSS)
    inertia2 : float, optional
        Inertia value from second algorithm (if different from calculated WCSS)
    filename : str, default="wcss_comparison.png"
        Filename for the saved plot
    """
    wcss_values = [wcss1, wcss2]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, wcss_values, color=['#1f77b4', '#ff7f0e'])
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')
    
    # Add inertia values as text annotations if provided
    if inertia1 is not None:
        plt.annotate(f'Inertia: {inertia1:.4f}', 
                    xy=(0, wcss1/2),
                    ha='center',
                    va='center')
    
    if inertia2 is not None:
        plt.annotate(f'Inertia: {inertia2:.4f}', 
                    xy=(1, wcss2/2),
                    ha='center',
                    va='center')
    
    # Add improvement percentage
    wcss_improvement = (wcss1 - wcss2) / wcss1 * 100
    
    # Check if we have inertia values to compare
    if inertia1 is not None and inertia2 is not None:
        inertia_improvement = (inertia1 - inertia2) / inertia1 * 100
        
        # Add text for both metrics
        plt.text(0.5, max(wcss_values) * 0.92, 
                 f'{algorithms[1]} WCSS: {wcss_improvement:.2f}% change vs {algorithms[0]}',
                 ha='center',
                 fontsize=11,
                 bbox=dict(facecolor='white', alpha=0.8))
        
        plt.text(0.5, max(wcss_values) * 0.84, 
                 f'{algorithms[1]} fuzzy inertia: {inertia_improvement:.2f}% improvement vs {algorithms[0]}',
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
    plt.savefig(filename)
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
        
    Notes:
    ------
    This function should only be used with algorithms that provide real convergence history.
    Scikit-learn's KMeans does not expose iteration-by-iteration convergence data, so it
    should not be included in these plots.
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

def plot_gkfcm_covariance_matrices(data, labels, centroids, covariance_matrices, scaler=None):
    """
    Visualize the clusters and their covariance matrices for GK-FCM
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, n_features)
        2D dataset
    labels : array-like of shape (n_samples,)
        Cluster labels
    centroids : array-like
        Cluster centroids
    covariance_matrices : array-like
        Covariance matrices for each cluster
    scaler : object, default=None
        Scaler used to normalize data
    """
    import matplotlib.patches as patches
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    # If scaler is provided, inverse transform the data and centroids
    if scaler:
        data = scaler.inverse_transform(data)
        centroids = scaler.inverse_transform(centroids)
    
    # Create a figure for visualization
    plt.figure(figsize=(12, 9))
    
    # Get the number of clusters
    n_clusters = len(centroids)
    
    # Create a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Plot the data points
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=colors[i], alpha=0.7, s=80, label=f'Cluster {i+1}')
    
    # Plot the centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               marker='x', s=200, linewidths=3, color='black', label='Centroids')
    
    # Function to draw confidence ellipses
    def confidence_ellipse(cov, mean, ax, n_std=2.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse.
        
        Parameters:
        -----------
        cov : array-like of shape (2, 2)
            Covariance matrix
        mean : array-like of shape (2,)
            Center of ellipse
        ax : matplotlib.axes.Axes
            The axes on which to draw
        n_std : float
            Number of standard deviations for ellipse size
        facecolor : str
            Color for the ellipse
        **kwargs : dict
            Additional parameters to pass to Ellipse
        """
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        
        # Using a special case to obtain the eigenvalues of this
        # two-dimensional dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                         facecolor=facecolor, **kwargs)
        
        # Calculating the standard deviation of x from the square root of
        # the variance and multiplying with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        
        # Calculating the standard deviation of y
        scale_y = np.sqrt(cov[1, 1]) * n_std
        
        # Transform ellipse to the given mean and covariance
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean[0], mean[1])
        
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
    
    # Plot the covariance matrices as confidence ellipses
    for i in range(n_clusters):
        confidence_ellipse(
            covariance_matrices[i][:2, :2],  # Use only 2D part for visualization
            centroids[i][:2],                # Use only 2D part of centroid
            plt.gca(),
            n_std=2.0,                       # 2 standard deviations ~ 95% confidence
            edgecolor=colors[i],
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            label=f'Covariance {i+1}'
        )
    
    # Set labels and title
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("GK-FCM Clustering with Covariance Matrices", fontsize=14)
    
    # Add legend
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter to show one entry per cluster and one for centroids
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
    plt.savefig("gkfcm_covariance_matrices.png")
    plt.close()

def compare_fcm_gkfcm(data_2d, fcm_labels, fcm_centroids, gkfcm_labels, gkfcm_centroids, 
                   title_left="Fuzzy C-Means", title_right="Gustafson-Kessel FCM", 
                   filename="comparison_fcm_gkfcm_2d.png", scaler=None):
    """
    Create a side-by-side comparison of clustering results between FCM and GK-FCM algorithms
    
    Parameters:
    -----------
    data_2d : array-like of shape (n_samples, 2)
        2D dataset
    fcm_labels : array-like of shape (n_samples,)
        FCM cluster labels
    fcm_centroids : array-like
        FCM cluster centroids
    gkfcm_labels : array-like of shape (n_samples,)
        GK-FCM cluster labels
    gkfcm_centroids : array-like
        GK-FCM cluster centroids
    title_left : str, default="Fuzzy C-Means"
        Title for the left plot (FCM)
    title_right : str, default="Gustafson-Kessel FCM"
        Title for the right plot (GK-FCM)
    filename : str, default="comparison_fcm_gkfcm_2d.png"
        Filename for the saved plot
    scaler : object, default=None
        Scaler used to normalize data
    """
    # If scaler is provided, inverse transform the data and centroids
    if scaler:
        data_2d = scaler.inverse_transform(data_2d)
        fcm_centroids = scaler.inverse_transform(fcm_centroids)
        gkfcm_centroids = scaler.inverse_transform(gkfcm_centroids)
    
    n_clusters = len(fcm_centroids)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create a colormap
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    
    # Plot FCM results (left)
    for i in range(n_clusters):
        cluster_points = data_2d[fcm_labels == i]
        ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=colors[i], alpha=0.7, s=80, label=f'Cluster {i+1}')
    
    ax1.scatter(fcm_centroids[:, 0], fcm_centroids[:, 1], 
               marker='x', s=200, linewidths=3, color='black', label='Centroids')
    
    ax1.set_title(title_left, fontsize=14)
    ax1.set_xlabel("Annual Income (k$)")
    ax1.set_ylabel("Spending Score (1-100)")
    
    # Plot GK-FCM results (right)
    for i in range(n_clusters):
        cluster_points = data_2d[gkfcm_labels == i]
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   color=colors[i], alpha=0.7, s=80, label=f'Cluster {i+1}')
    
    ax2.scatter(gkfcm_centroids[:, 0], gkfcm_centroids[:, 1], 
               marker='x', s=200, linewidths=3, color='black', label='Centroids')
    
    ax2.set_title(title_right, fontsize=14)
    ax2.set_xlabel("Annual Income (k$)")
    ax2.set_ylabel("Spending Score (1-100)")
    
    # Add overall title
    fig.suptitle(f"Comparison: {title_left} vs {title_right}", fontsize=16, y=0.98)
    
    # Add legend (only to the second plot to avoid duplication)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
              bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=n_clusters+1)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for legend
    plt.savefig(filename)
    plt.close()

def plot_algorithm_comparison(metrics, filename="algorithm_comparison.png"):
    """
    Plot a comparison of performance metrics between FCM and GK-FCM algorithms.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metric names as keys and tuples of (fcm_value, gkfcm_value) as values
        Example: {'Silhouette': (0.65, 0.72), 'DB Index': (0.45, 0.38)}
    filename : str, default="algorithm_comparison.png"
        Filename for the saved plot
    """
    metric_names = list(metrics.keys())
    fcm_values = [metrics[metric][0] for metric in metric_names]
    gkfcm_values = [metrics[metric][1] for metric in metric_names]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    barWidth = 0.35
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(metric_names))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    ax.bar(r1, fcm_values, width=barWidth, color='skyblue', edgecolor='black', 
           label='Fuzzy C-Means')
    ax.bar(r2, gkfcm_values, width=barWidth, color='lightgreen', edgecolor='black', 
           label='Gustafson-Kessel FCM')
    
    # Add value labels on top of bars
    for i, value in enumerate(fcm_values):
        ax.text(r1[i], value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    
    for i, value in enumerate(gkfcm_values):
        ax.text(r2[i], value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    
    # Add labels, title and legend
    ax.set_xlabel('Evaluation Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison: FCM vs GK-FCM', fontsize=14)
    ax.set_xticks([r + barWidth/2 for r in range(len(metric_names))])
    ax.set_xticklabels(metric_names)
    ax.legend()
    
    # Add a note if any metrics are better when lower (like DB Index)
    for metric in metric_names:
        if metric.lower() in ['db index', 'davies-bouldin', 'db', 'davies bouldin']:
            plt.figtext(0.5, 0.01, '* Lower values are better for Davies-Bouldin Index', 
                       ha='center', fontsize=10, fontstyle='italic')
            break
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_algorithm_metrics_comparison(metric1, metric2, algorithms=['K-means', 'FCM'], 
                       inertia1=None, inertia2=None, 
                       filename="algorithm_metrics_comparison.png",
                       title="Algorithm-Specific Metrics Comparison"):
    """
    Plot comparison of algorithm-specific evaluation metrics
    
    Parameters:
    -----------
    metric1 : float
        Evaluation metric for first algorithm
    metric2 : float
        Evaluation metric for second algorithm
    algorithms : list, default=['K-means', 'FCM']
        Names of algorithms to compare
    inertia1, inertia2 : float, optional
        Optional additional metrics to display
    filename : str, default="algorithm_metrics_comparison.png"
        Output filename
    title : str, default="Algorithm-Specific Metrics Comparison"
        Plot title
    """
    if len(algorithms) != 2:
        raise ValueError("This function only supports comparing two algorithms")
    
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    
    # Create bars for each algorithm's metric
    bars = plt.bar([0, 1], [metric1, metric2], width=bar_width, 
                  color=['skyblue', 'lightgreen'], 
                  edgecolor='black', linewidth=1.5)
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Set labels and title
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks([0, 1], algorithms, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add notes about interpretation
    plt.figtext(0.5, 0.01, 
               f"Note: These are algorithm-specific metrics and should not be directly compared.\n"
               f"{algorithms[0]} uses standard Euclidean inertia while {algorithms[1]} uses its own objective function.",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add additional metrics in a text box if provided
    if inertia1 is not None and inertia2 is not None:
        textstr = (f"{algorithms[0]} Additional Metric: {inertia1:.4f}\n"
                  f"{algorithms[1]} Additional Metric: {inertia2:.4f}")
        plt.figtext(0.2, 0.85, textstr, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def compare_all_fuzzy(data_2d, labels_dict, centroids_dict, title="All Fuzzy Clustering Algorithms Comparison", 
                   filename="all_fuzzy_comparison.png", scaler=None):
    """
    Create side-by-side comparison of all fuzzy clustering algorithms
    
    Parameters:
    -----------
    data_2d : array-like
        2D dataset 
    labels_dict : dict
        Dictionary of cluster labels for each algorithm
        Example: {'fcm': fcm_labels, 'gkfcm': gkfcm_labels, 'kfcm': kfcm_labels, 'mkfcm': mkfcm_labels}
    centroids_dict : dict
        Dictionary of centroids for each algorithm
        Example: {'fcm': fcm_centroids, 'gkfcm': gkfcm_centroids, 'kfcm': kfcm_centroids, 'mkfcm': mkfcm_centroids}
    title : str, default="All Fuzzy Clustering Algorithms Comparison"
        Title for the plot
    filename : str, default="all_fuzzy_comparison.png"
        Filename to save the plot
    scaler : object, default=None
        Scaler object to inverse transform data and centroids for visualization
    """
    # Check if required data is provided
    required_algorithms = ['fcm', 'gkfcm', 'kfcm', 'mkfcm']
    for algo in required_algorithms:
        if algo not in labels_dict or algo not in centroids_dict:
            raise ValueError(f"Missing data for algorithm {algo}")
    
    # Setup plot with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Define colors and markers
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    markers = ['o', 's', '^', 'D', 'x', '*', '+', '<']
    
    # Get axis labels from original data
    if scaler is not None:
        # If scaler is provided, reverse transform to get original feature names
        x_label = "Annual Income (k$)"
        y_label = "Spending Score (1-100)"
    else:
        x_label = "Feature 1"
        y_label = "Feature 2"
    
    # Get maximum number of clusters across all algorithms
    max_clusters = max([len(np.unique(labels)) for labels in labels_dict.values()])
    
    # Create color map with enough colors for all clusters
    cmap = plt.cm.get_cmap('tab10', max_clusters)
    
    # Plot each algorithm
    for i, (algo_name, pretty_name) in enumerate(zip(
        required_algorithms, 
        ['FCM', 'GK-FCM', 'KFCM', 'MKFCM']
    )):
        ax = axes[i]
        labels = labels_dict[algo_name]
        centroids = centroids_dict[algo_name]
        
        # Get unique clusters
        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters)
        
        # Transform data back to original space if scaler is provided
        if scaler is not None:
            data_original = scaler.inverse_transform(data_2d)
            centroids_original = scaler.inverse_transform(centroids)
        else:
            data_original = data_2d
            centroids_original = centroids
        
        # Plot data points colored by cluster
        for j, cluster in enumerate(unique_clusters):
            cluster_points = data_original[labels == cluster]
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=[cmap(j)],
                marker=markers[j % len(markers)],
                edgecolor='k',
                s=50,
                alpha=0.7,
                label=f'Cluster {cluster+1}'
            )
        
        # Plot centroids
        ax.scatter(
            centroids_original[:, 0],
            centroids_original[:, 1],
            c=range(n_clusters),
            cmap=cmap,
            marker='X',
            edgecolor='k',
            s=200,
            linewidth=2
        )
        
        # Add centroid labels
        for j, centroid in enumerate(centroids_original):
            ax.annotate(
                f'C{j+1}',
                xy=(centroid[0], centroid[1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold'
            )
        
        # Set labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{pretty_name} Clustering", fontsize=14, fontweight='bold')
        
        # Add legend outside plot
        if i == 1:  # Only add legend to the first plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add main title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Save figure
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    
def compare_fuzzy_metrics(metrics_dict, algorithms=['FCM', 'GK-FCM', 'KFCM', 'MKFCM'], 
                          metric_name='Silhouette Score', higher_better=True,
                          filename='fuzzy_metrics_comparison.png'):
    """
    Create bar chart comparing metrics across fuzzy clustering algorithms
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metric values for each algorithm
        Example: {'fcm': 0.5, 'gkfcm': 0.6, 'kfcm': 0.55, 'mkfcm': 0.65}
    algorithms : list, default=['FCM', 'GK-FCM', 'KFCM', 'MKFCM']
        List of algorithm names for display
    metric_name : str, default='Silhouette Score'
        Name of the metric being compared
    higher_better : bool, default=True
        Whether higher values are better for this metric
    filename : str, default='fuzzy_metrics_comparison.png'
        Filename to save the plot
    """
    if len(metrics_dict) != len(algorithms):
        raise ValueError("Number of metrics must match number of algorithms")
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    values = list(metrics_dict.values())
    bars = plt.bar(algorithms, values, color=['skyblue', 'lightgreen', 'salmon', 'plum'])
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add horizontal line for average
    avg = np.mean(values)
    plt.axhline(y=avg, color='gray', linestyle='--', alpha=0.7,
               label=f'Average: {avg:.4f}')
    
    # Find best algorithm
    if higher_better:
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    
    # Highlight best algorithm
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(1.5)
    
    # Add 'Best' annotation to the best algorithm
    plt.annotate(
        'Best',
        xy=(best_idx, values[best_idx]),
        xytext=(0, 20),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
        ha='center',
        fontweight='bold'
    )
    
    # Set labels and title
    plt.xlabel('Clustering Algorithm', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Comparison of {metric_name} Across Fuzzy Clustering Algorithms', 
              fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def compare_fuzzy_metrics_with_error_bars(metrics_dict, std_dict, algorithms=None, 
                                         metric_name='Silhouette Score', higher_better=True,
                                         filename='fuzzy_metrics_comparison_with_error.png'):
    """
    Create bar chart comparing metrics across algorithms with error bars showing standard deviation
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of average metric values for each algorithm
    std_dict : dict
        Dictionary of standard deviation values for each algorithm
    algorithms : list or None, default=None
        List of algorithm names for display (if None, use metrics_dict.keys())
    metric_name : str, default='Silhouette Score'
        Name of the metric being compared
    higher_better : bool, default=True
        Whether higher values are better for this metric
    filename : str, default='fuzzy_metrics_comparison_with_error.png'
        Filename to save the plot
    """
    if algorithms is None:
        algorithms = list(metrics_dict.keys())
    
    if len(metrics_dict) != len(algorithms) or len(std_dict) != len(algorithms):
        raise ValueError("Number of metrics and std values must match number of algorithms")
    
    # Setup plot
    plt.figure(figsize=(10, 6))
    
    # Create bar chart with error bars
    values = [metrics_dict[algo] for algo in algorithms]
    errors = [std_dict[algo] for algo in algorithms]
    
    bars = plt.bar(algorithms, values, yerr=errors, capsize=8,
                  color=['skyblue', 'lightgreen', 'salmon', 'plum'],
                  ecolor='black', alpha=0.8)
    
    # Add value labels above bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + errors[i] + 0.01,
            f'{height:.4f} ± {errors[i]:.4f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            rotation=0 if len(algorithms) < 6 else 45
        )
    
    # Add horizontal line for average
    avg = np.mean(values)
    plt.axhline(y=avg, color='gray', linestyle='--', alpha=0.7,
               label=f'Average: {avg:.4f}')
    
    # Find best algorithm
    if higher_better:
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)
    
    # Highlight best algorithm
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(1.5)
    
    # Add 'Best' annotation to the best algorithm
    plt.annotate(
        'Best',
        xy=(best_idx, values[best_idx]),
        xytext=(0, 20 + errors[best_idx]),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
        ha='center',
        fontweight='bold'
    )
    
    # Set labels and title
    plt.xlabel('Clustering Algorithm', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Comparison of {metric_name} Across Clustering Algorithms', 
              fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_kernel_sigma_comparison(data, results_dict, algorithm_name, scaler=None, 
                               title="Effect of σ² on Clustering", filename="kernel_sigma_comparison.png"):
    """
    Create visualization to compare clustering results with different sigma_squared values
    
    Parameters:
    -----------
    data : array-like of shape (n_samples, 2)
        2D dataset
    results_dict : list of dicts
        List of dictionaries containing results for each sigma value
    algorithm_name : str
        Name of the algorithm ('KFCM' or 'MKFCM')
    scaler : object, default=None
        Scaler used to normalize data
    title : str
        Plot title
    filename : str
        Filename for the saved plot
    """
    # Number of sigma values to compare
    n_sigmas = len(results_dict)
    
    # Create a grid of subplots
    fig, axes = plt.subplots(1, n_sigmas, figsize=(5*n_sigmas, 5))
    if n_sigmas == 1:
        axes = [axes]  # Make it iterable if only one sigma value
    
    # If scaler is provided, inverse transform the data
    if scaler:
        data = scaler.inverse_transform(data)
    
    # Loop through each sigma result
    for i, result in enumerate(results_dict):
        sigma = result['sigma_squared']
        labels = result['labels']
        centroids = result['centroids']
        silhouette = result['silhouette']
        
        # If scaler is provided, inverse transform centroids
        if scaler:
            centroids = scaler.inverse_transform(centroids)
        
        # Number of clusters
        n_clusters = len(centroids)
        
        # Create a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        
        # Plot the data points for each cluster
        for j in range(n_clusters):
            cluster_points = data[labels == j]
            axes[i].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          color=colors[j], alpha=0.7, s=50)
        
        # Plot centroids
        axes[i].scatter(centroids[:, 0], centroids[:, 1], 
                      marker='x', s=150, linewidths=2, color='black')
        
        # Set title and labels
        axes[i].set_title(f"{algorithm_name} with σ²={sigma}\nSilhouette: {silhouette:.3f}")
        axes[i].set_xlabel("Annual Income (k$)")
        axes[i].set_ylabel("Spending Score (1-100)")
        axes[i].grid(True, alpha=0.3)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    
    # Save the figure
    plt.savefig(filename)
    plt.close()

def plot_sigma_parameter_study(sigma_values, metric_values, std_values, 
                             metric_name='Silhouette Score', algorithm_name='KFCM',
                             filename='sigma_parameter_study.png'):
    """
    Plot the effect of different sigma_squared values on a given metric with error bars
    
    Parameters:
    -----------
    sigma_values : list
        List of sigma_squared values used
    metric_values : list
        List of average metric values for each sigma value
    std_values : list
        List of standard deviation values for each sigma value
    metric_name : str, default='Silhouette Score'
        Name of the metric being plotted
    algorithm_name : str, default='KFCM'
        Name of the algorithm used
    filename : str, default='sigma_parameter_study.png'
        Filename to save the plot
    """
    # Setup plot
    plt.figure(figsize=(10, 6))
    
    # Create line plot with error bars
    plt.errorbar(
        sigma_values, 
        metric_values, 
        yerr=std_values, 
        fmt='o-',
        capsize=5,
        elinewidth=2,
        linewidth=2,
        markersize=8,
        color='royalblue',
        ecolor='black'
    )
    
    # Add value labels near points
    for i, (x, y, std) in enumerate(zip(sigma_values, metric_values, std_values)):
        plt.annotate(
            f'{y:.4f} ± {std:.4f}',
            xy=(x, y),
            xytext=(0, 10),
            textcoords='offset points',
            ha='center',
            fontsize=9,
            fontweight='bold'
        )
    
    # Find optimal sigma value (assuming higher metric is better)
    best_idx = np.argmax(metric_values)
    optimal_sigma = sigma_values[best_idx]
    
    # Highlight optimal sigma
    plt.plot(
        optimal_sigma, 
        metric_values[best_idx], 
        'o', 
        ms=12, 
        color='gold', 
        mec='black',
        mew=1.5,
        label=f'Optimal σ² = {optimal_sigma}'
    )
    
    # Set labels and title
    plt.xlabel('σ² (Kernel Width Parameter)', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Effect of σ² Parameter on {metric_name} for {algorithm_name}', 
              fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Set x-axis to log scale if values span multiple orders of magnitude
    if max(sigma_values) / min(sigma_values) > 10:
        plt.xscale('log')
        plt.xticks(sigma_values, [str(s) for s in sigma_values])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
