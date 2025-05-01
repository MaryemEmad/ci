import numpy as np
from scipy.linalg import pinv
from sklearn.cluster import KMeans

class GKFuzzyCMeansClustering:
    def __init__(self, n_clusters=5, m=2.0, max_iter=100, tol=1e-4, random_state=None, min_det_value=1e-2):
        """Initialize Gustafson-Kessel Fuzzy C-Means clustering."""
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.min_det_value = min_det_value  # Increased for numerical stability
        self.centroids = None
        self.memberships = None
        self.labels = None
        self.iterations = 0
        self.covariance_matrices = None
        self.norm_matrices = None
        self.inv_covariance_matrices = None

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

    def update_covariance_matrices(self, X, centroids, memberships):
        """Update covariance matrices for each cluster with stronger regularization."""
        n_samples, n_features = X.shape
        covariance_matrices = np.zeros((self.n_clusters, n_features, n_features))
        memberships_pow = np.power(memberships, self.m)
        
        for i in range(self.n_clusters):
            diff = X - centroids[i]
            weights = memberships_pow[:, i].reshape(-1, 1)
            cov = np.dot((diff * weights).T, diff) / max(np.sum(weights), np.finfo(float).eps)
            # Add stronger regularization to ensure numerical stability
            cov += 1e-1 * np.eye(n_features)  # Increased from 1e-2 to 1e-1
            det = np.linalg.det(cov)
            if det < self.min_det_value:
                cov += self.min_det_value * np.eye(n_features)
            covariance_matrices[i] = cov
        
        return covariance_matrices

    def calculate_norm_matrices(self, covariance_matrices, n_features):
        """Calculate the norm-inducing matrices for each cluster with regularization."""
        norm_matrices = np.zeros_like(covariance_matrices)
        
        for i in range(self.n_clusters):
            det_cov = max(np.linalg.det(covariance_matrices[i]), self.min_det_value)
            scaling_factor = np.power(det_cov + 1e-3, 1.0 / n_features)  # Add small regularization to det_cov
            cov_inv = pinv(covariance_matrices[i])
            norm_matrices[i] = scaling_factor * cov_inv
        
        return norm_matrices

    def update_memberships(self, X, centroids):
        """Update membership matrix using Gustafson-Kessel FCM formula with regularization."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for i in range(self.n_clusters):
            diff = X - centroids[i]
            distances[:, i] = np.sqrt(np.einsum('ij,jk,ik->i', diff, self.inv_covariance_matrices[i], diff))
        
        # Add small regularization to distances to avoid numerical issues
        distances = np.maximum(distances + 1e-5, np.finfo(float).eps)  # Increased from 1e-6 to 1e-5
        distances_pow = distances ** (-2.0 / (self.m - 1))
        new_memberships = distances_pow / np.sum(distances_pow, axis=1, keepdims=True)
        return np.clip(new_memberships, np.finfo(float).eps, 1.0)

    def fit(self, X, K=None):
        """Fit the GK-FCM model to data X."""
        n_samples, n_features = X.shape
        
        # Initialize centroids with k-means++
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state, n_init=20)  # Increased n_init to 20
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_
        
        self.memberships = self.initialize_memberships(n_samples)
        self.covariance_matrices = np.zeros((self.n_clusters, n_features, n_features))
        self.inv_covariance_matrices = np.zeros((self.n_clusters, n_features, n_features))
        self.iterations = 0
        
        for i in range(self.max_iter):
            old_memberships = self.memberships.copy()
            
            self.centroids = self.update_centroids(X, self.memberships)
            self.covariance_matrices = self.update_covariance_matrices(X, self.centroids, self.memberships)
            
            for j in range(self.n_clusters):
                self.inv_covariance_matrices[j] = pinv(self.covariance_matrices[j])
            
            self.norm_matrices = self.calculate_norm_matrices(self.covariance_matrices, n_features)
            self.memberships = self.update_memberships(X, self.centroids)
            
            diff = np.linalg.norm(self.memberships - old_memberships)
            if diff <= self.tol:
                break
            
            self.iterations += 1
        
        self.labels = np.argmax(self.memberships, axis=1)
        return self.memberships, self.centroids, self.labels, self.iterations

    def predict(self, X, K=None):
        """Predict cluster labels for new data."""
        if self.centroids is None or self.norm_matrices is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        
        memberships = self.update_memberships(X, self.centroids)
        return np.argmax(memberships, axis=1)