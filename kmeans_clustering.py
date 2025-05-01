from sklearn.cluster import KMeans
import numpy as np

class KMeansClustering:
    def __init__(self, n_clusters=5, init_method='k-means++', random_state=None, max_iter=500):
        """
        Initialize KMeans clustering
        
        Parameters:
        -----------
        n_clusters : int, default=4
            Number of clusters
        init_method : str, default='k-means++'
            Initialization method ('k-means++' or 'random')
        random_state : int, default=None
            Random state for reproducibility (None for random each run)
        max_iter : int, default=500
            Maximum number of iterations (increased for better convergence)
        """
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            random_state=random_state,
            max_iter=max_iter,
            n_init=20,  # Increased from 10 to 20 for better initialization
            verbose=0
        )
        self.memberships = None
        self.centroids = None
        self.labels = None
        self.iterations = 0

    def fit(self, data, K=None):
        """
        Fit the KMeans model to data
        
        Parameters:
        -----------
        data : array-like
            Training data
        K : array-like, optional
            Kernel matrix (ignored for K-Means)
            
        Returns:
        --------
        memberships : array
            Binary membership matrix (1 for assigned cluster, 0 otherwise)
        centroids : array
            Cluster centroids
        labels : array
            Cluster labels
        iterations : int
            Number of iterations
        """
        self._data = data
        self.model.fit(data)
        
        self.labels = self.model.labels_
        self.centroids = self.model.cluster_centers_
        self.iterations = self.model.n_iter_
        
        # Generate binary memberships for K-Means
        self.memberships = np.zeros((data.shape[0], self.n_clusters))
        for i in range(data.shape[0]):
            self.memberships[i, self.labels[i]] = 1.0
        
        return self.memberships, self.centroids, self.labels, self.iterations

    def predict(self, data, K=None):
        """Predict cluster labels for new data"""
        return self.model.predict(data)

    def get_centroids(self):
        """Return centroids for visualization"""
        return self.model.cluster_centers_