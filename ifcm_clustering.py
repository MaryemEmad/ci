import numpy as np

class IFCMClustering:
    def __init__(self, n_clusters=5, m=2.0, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers = None
        self.U = None
        self.V = None
        self.R = None

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def fit(self, X, K=None): 
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        C = self.n_clusters

        # Initialize U randomly and normalize it
        U = np.random.rand(n_samples, C)
        U = U / np.sum(U, axis=1, keepdims=True)

        # Initialize V and R
        V = np.random.rand(n_samples, C)
        V = V / np.sum(V, axis=1, keepdims=True)
        R = 1 - (U + V)

        for iteration in range(self.max_iter):
            # Save old U for convergence check
            U_old = U.copy()

            # Update cluster centers based on U
            centers = np.zeros((C, n_features))
            for c in range(C):
                numerator = np.sum((U[:, c] ** self.m)[:, np.newaxis] * X, axis=0)
                denominator = np.sum(U[:, c] ** self.m)
                centers[c] = numerator / denominator

            # Update membership matrices
            dist = np.zeros((n_samples, C))
            for c in range(C):
                for i in range(n_samples):
                    dist[i, c] = self._euclidean_distance(X[i], centers[c])

            # Avoid division by zero
            dist = np.fmax(dist, np.finfo(np.float64).eps)

            U_new = np.zeros((n_samples, C))
            for i in range(n_samples):
                for c in range(C):
                    sum_term = np.sum((dist[i, c] / dist[i, :]) ** (2 / (self.m - 1)))
                    U_new[i, c] = 1 / sum_term

            # Update V and R
            V_new = 1 - U_new
            R_new = 1 - (U_new + V_new)

            # Normalize U, V, R
            total = U_new + V_new + R_new
            U_new = U_new / total
            V_new = V_new / total
            R_new = R_new / total

            # Check for convergence
            if np.linalg.norm(U_new - U_old) < self.tol:
                break

            U, V, R = U_new, V_new, R_new

        self.U = U
        self.V = V
        self.R = R
        self.centers = centers

        # Compute labels (highest membership)
        labels = np.argmax(U, axis=1)

        return U, centers, labels, iteration + 1

    def predict(self, X):
        dist = np.zeros((X.shape[0], self.n_clusters))
        for c in range(self.n_clusters):
            for i in range(X.shape[0]):
                dist[i, c] = self._euclidean_distance(X[i], self.centers[c])
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        U_new = np.zeros((X.shape[0], self.n_clusters))
        for i in range(X.shape[0]):
            for c in range(self.n_clusters):
                sum_term = np.sum((dist[i, c] / dist[i, :]) ** (2 / (self.m - 1)))
                U_new[i, c] = 1 / sum_term
        return np.argmax(U_new, axis=1)