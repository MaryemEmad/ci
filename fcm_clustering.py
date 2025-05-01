import numpy as np
from sklearn.cluster import KMeans

class FuzzyCMeansClustering:
    def __init__(self, n_clusters=5, m=2.0, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.memberships = None
        self.labels = None
        self.iterations = 0

    def initialize_memberships(self, n_samples):
        """Initialize random fuzzy memberships with Dirichlet distribution."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        memberships = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)
        return np.clip(memberships, np.finfo(float).eps, 1.0)

    def update_centroids(self, X, memberships):
        """Update centroids based on membership values."""
        memberships_pow = np.power(memberships, self.m)
        numerator = np.dot(memberships_pow.T, X)
        denominator = np.sum(memberships_pow, axis=0)[:, np.newaxis]
        denominator = np.maximum(denominator, np.finfo(float).eps)
        return numerator / denominator

    def update_memberships(self, X, centroids):
        """Update membership matrix."""
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        distances = np.maximum(distances, np.finfo(float).eps)
        distances_pow = distances ** (-2.0 / (self.m - 1))
        new_memberships = distances_pow / np.sum(distances_pow, axis=1, keepdims=True)
        return np.clip(new_memberships, np.finfo(float).eps, 1.0)

    def fit(self, X, K=None):
        """Fit the FCM model to data X."""
        n_samples = X.shape[0]
        
        # Initialize centroids with k-means++
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_
        
        self.memberships = self.initialize_memberships(n_samples)
        self.iterations = 0
        
        for i in range(self.max_iter):
            old_memberships = self.memberships.copy()
            self.centroids = self.update_centroids(X, self.memberships)
            self.memberships = self.update_memberships(X, self.centroids)
            
            diff = np.linalg.norm(self.memberships - old_memberships)
            if diff <= self.tol:
                break
            
            self.iterations += 1
        
        self.labels = np.argmax(self.memberships, axis=1)
        return self.memberships, self.centroids, self.labels, self.iterations

    def predict(self, X, K=None):
        """Predict cluster labels for new data."""
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        
        memberships = self.update_memberships(X, self.centroids)
        return np.argmax(memberships, axis=1)