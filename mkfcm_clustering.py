import numpy as np

class ModifiedKernelFuzzyCMeansClustering:
    def __init__(self, n_clusters=5, m=2.0, max_iter=100, tol=1e-4, random_state=None, sigma_squared=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.sigma_squared = sigma_squared
        self.centroids = None
        self.memberships = None
        self.labels = None
        self.iterations = 0

    def compute_sigma_squared(self, data):
        """Compute sigma_squared dynamically based on data distribution."""
        n = min(100, data.shape[0])
        distances = []
        indices = np.random.choice(data.shape[0], size=n, replace=False)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((data[indices[i]] - data[indices[j]]) ** 2)
                distances.append(dist)
        return np.mean(distances)

    def gaussian_kernel(self, x, y):
        """Compute Gaussian kernel: K(x, y) = exp(-||x-y||^2 / sigma^2)"""
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        x_squared = np.sum(x**2, axis=1, keepdims=True)
        y_squared = np.sum(y**2, axis=1).reshape(1, -1)
        dist_squared = x_squared + y_squared - 2 * np.dot(x, y.T)
        return np.exp(-dist_squared / self.sigma_squared)

    def initialize_memberships(self, n_samples):
        """Initialize random fuzzy memberships with Dirichlet distribution."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        memberships = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)
        return np.clip(memberships, np.finfo(float).eps, 1.0)

    def calculate_kernel_distances(self, X, memberships, K=None):
        """Calculate kernel distances in feature space."""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        memberships_pow = np.power(memberships, self.m)
        
        if K is None:
            for i in range(self.n_clusters):
                kernel_values = self.gaussian_kernel(X, self.centroids[i] if self.centroids is not None else np.mean(X, axis=0)).flatten()
                distances[:, i] = 1.0 - kernel_values
        else:
            for i in range(self.n_clusters):
                u_i = memberships_pow[:, i]
                denominator = max(np.sum(u_i), np.finfo(float).eps)
                term2 = -2.0 * np.dot(K, u_i) / denominator
                weighted_kernel = u_i.reshape(-1, 1) * K * u_i.reshape(1, -1)
                term3 = np.sum(weighted_kernel) / (denominator**2)
                distances[:, i] = 1.0 + term2 + term3
        
        # Apply modification: weight distances by membership variance
        membership_variance = np.var(memberships, axis=0)
        distances *= (1 + membership_variance[np.newaxis, :])
        return np.maximum(distances, np.finfo(float).eps)

    def update_memberships(self, X, memberships, K=None):
        """Update membership matrix using kernel distances."""
        distances = self.calculate_kernel_distances(X, memberships, K)
        distances_pow = distances ** (-1.0 / (self.m - 1))
        new_memberships = distances_pow / np.sum(distances_pow, axis=1, keepdims=True)
        return np.clip(new_memberships, np.finfo(float).eps, 1.0)

    def calculate_input_centroids(self, X, memberships):
        """Approximate centroids in input space."""
        memberships_pow = np.power(memberships, self.m)
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for i in range(self.n_clusters):
            weights = memberships_pow[:, i].reshape(-1, 1)
            centroids[i] = np.sum(X * weights, axis=0) / max(np.sum(weights), np.finfo(float).eps)
        
        return centroids

    def fit(self, X, K=None):
        """Fit the MKFCM model to data X."""
        n_samples = X.shape[0]
        
        if self.sigma_squared is None:
            self.sigma_squared = self.compute_sigma_squared(X)
        
        self.memberships = self.initialize_memberships(n_samples)
        self.iterations = 0
        
        for i in range(self.max_iter):
            old_memberships = self.memberships.copy()
            self.memberships = self.update_memberships(X, self.memberships, K)
            
            diff = np.linalg.norm(self.memberships - old_memberships)
            if diff <= self.tol:
                break
            
            self.iterations += 1
        
        self.centroids = self.calculate_input_centroids(X, self.memberships)
        self.labels = np.argmax(self.memberships, axis=1)
        return self.memberships, self.centroids, self.labels, self.iterations

    def predict(self, X, K=None):
        """Predict cluster labels for new data."""
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        
        memberships = self.update_memberships(X, self.memberships, K)
        return np.argmax(memberships, axis=1)