# kmeans_clustering.py: Implements K-means clustering
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

class KMeansClustering:
    def __init__(self, n_clusters=4, init_method='k-means++', random_state=42, max_iter=300):
        """
        Initialize KMeans clustering
        
        Parameters:
        -----------
        n_clusters : int, default=4
            Number of clusters
        init_method : str, default='k-means++'
            Initialization method ('k-means++' or 'random')
        random_state : int, default=42
            Random state for reproducibility
        max_iter : int, default=300
            Maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init_method = init_method  # 'k-means++' or 'random'
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            random_state=random_state,
            max_iter=max_iter,
            n_init=10,
            verbose=0
        )

    def fit(self, data):
        """
        Fit the KMeans model to data
        
        Parameters:
        -----------
        data : array-like
            Training data
            
        Returns:
        --------
        labels : array
            Cluster labels
        """
        # Store initial data for reference
        self._data = data
        
        # Fit the model
        self.model.fit(data)
        
        return self.model.labels_

    def predict(self, data):
        """Predict cluster labels for new data"""
        return self.model.predict(data)

    def compute_fitness(self, data, labels=None):
        """
        Compute fitness (inertia) of the clustering
        
        Parameters:
        -----------
        data : array-like
            Input data
        labels : array-like, optional
            Cluster labels (if None, use model labels)
            
        Returns:
        --------
        inertia : float
            Within-cluster sum of squares
        """
        # If no labels provided, use model's labels
        if labels is None:
            return float(self.model.inertia_)
        
        # Calculate WCSS manually
        wcss = 0
        for i in range(self.n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                wcss += np.sum(np.square(np.linalg.norm(cluster_points - centroid, axis=1)))
        
        return float(wcss)

    def get_centroids(self):
        """Return centroids for visualization"""
        return self.model.cluster_centers_
    
    def get_fitness_history(self):
        """
        Return fitness history for convergence analysis
        
        Returns:
        --------
        None : scikit-learn's KMeans implementation does not expose iteration-by-iteration 
        convergence data, so this method returns None.
        """
        return None
