import numpy as np

class RseKFCMClustering:
    def __init__(self, n_clusters, m=2.0, sample_size=50, sigma=None, max_iter=100, epsilon=1e-3, random_state=None):
        self.n_clusters = n_clusters
        self.m = m  # Fuzzifier
        self.sample_size = sample_size  # Number of samples (n_s)
        self.sigma = sigma  # Will compute dynamically if None
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.centroids = None
        self.memberships = None
        self.labels = None
        self.iterations = 0
        self.sample_indices = None
        self.P = None

    def compute_sigma(self, data):
        """Compute sigma dynamically based on data distribution."""
        n = min(100, data.shape[0])  # Use a subset for efficiency
        distances = []
        indices = np.random.choice(data.shape[0], size=n, replace=False)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((data[indices[i]] - data[indices[j]]) ** 2)
                distances.append(dist)
        return np.sqrt(np.mean(distances))  # sigma = sqrt(mean distance)

    def rbf_kernel(self, x1, x2):
        """Compute RBF kernel between two vectors."""
        dist = np.sum((x1 - x2) ** 2)
        return np.exp(-dist / (2 * (self.sigma ** 2)))

    def wKFCM(self, K, weights):
        """Weighted KFCM implementation."""
        n = K.shape[0]
        # Improved initialization of U
        U = np.ones((n, self.n_clusters)) / self.n_clusters
        U += np.random.normal(0, 0.1, U.shape)  # Add small noise
        U = np.maximum(U, 1e-10)  # Ensure positive values
        U /= np.sum(U, axis=1, keepdims=True)

        for iteration in range(self.max_iter):
            U_old = U.copy()
            for j in range(self.n_clusters):
                u_j = U[:, j] ** self.m
                w_u_j = weights * u_j
                norm_w_u_j = np.sum(w_u_j)
                if norm_w_u_j == 0:
                    continue
                term1 = np.dot(w_u_j.T, np.dot(K, w_u_j)) / (norm_w_u_j ** 2)
                for i in range(n):
                    term2 = K[i, i]
                    term3 = 2 * np.dot(w_u_j.T, K[:, i]) / norm_w_u_j
                    d_k = term1 + term2 - term3
                    d_k = max(d_k, 1e-10)  # Ensure d_k is positive
                    denominator = 0
                    for k in range(self.n_clusters):
                        u_k = U[:, k] ** self.m
                        w_u_k = weights * u_k
                        norm_w_u_k = np.sum(w_u_k)
                        if norm_w_u_k == 0:
                            continue
                        term_k = term1 + K[i, i] - 2 * np.dot(w_u_k.T, K[:, i]) / norm_w_u_k
                        term_k = max(term_k, 1e-10)  # Ensure term_k is positive
                        denominator += (d_k / term_k) ** (1 / (self.m - 1))
                    U[i, j] = 1 / denominator if denominator != 0 else 0

            if np.max(np.abs(U - U_old)) < self.epsilon:
                self.iterations = iteration + 1
                break
        else:
            self.iterations = self.max_iter

        P = np.zeros(self.n_clusters, dtype=int)
        for j in range(self.n_clusters):
            u_j = U[:, j] ** self.m
            w_u_j = weights * u_j
            norm_w_u_j = np.sum(w_u_j)
            if norm_w_u_j == 0:
                continue
            term1 = np.dot(w_u_j.T, np.dot(K, w_u_j)) / (norm_w_u_j ** 2)
            distances = [term1 + K[i, i] - 2 * np.dot(w_u_j.T, K[:, i]) / norm_w_u_j for i in range(n)]
            P[j] = np.argmin(distances)

        return U, P

    def fit(self, data, K=None):
        """Fit rseKFCM model using provided kernel matrix."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.sigma is None:
            self.sigma = self.compute_sigma(data)

        n_samples = data.shape[0]
        self.sample_indices = np.random.choice(n_samples, min(self.sample_size, n_samples), replace=False)
        X_s = data[self.sample_indices]
        
        # Use provided kernel matrix if available, otherwise compute for sample
        if K is None:
            K_s = np.zeros((len(self.sample_indices), len(self.sample_indices)))
            for i, idx_i in enumerate(self.sample_indices):
                for j, idx_j in enumerate(self.sample_indices):
                    K_s[i, j] = self.rbf_kernel(data[idx_i], data[idx_j])
        else:
            K_s = K[self.sample_indices][:, self.sample_indices]

        weights = np.ones(len(self.sample_indices))
        U_s, self.P = self.wKFCM(K_s, weights)
        
        self.memberships = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for j in range(self.n_clusters):
                prototype_idx = self.sample_indices[self.P[j]]
                if K is None:
                    d_k = self.rbf_kernel(data[i], data[i]) + self.rbf_kernel(data[prototype_idx], data[prototype_idx]) - 2 * self.rbf_kernel(data[i], data[prototype_idx])
                else:
                    d_k = K[i, i] + K[prototype_idx, prototype_idx] - 2 * K[i, prototype_idx]
                d_k = max(d_k, 1e-10)
                denominator = 0
                for k in range(self.n_clusters):
                    proto_k = self.sample_indices[self.P[k]]
                    if K is None:
                        term_k = self.rbf_kernel(data[i], data[i]) + self.rbf_kernel(data[proto_k], data[proto_k]) - 2 * self.rbf_kernel(data[i], data[proto_k])
                    else:
                        term_k = K[i, i] + K[proto_k, proto_k] - 2 * K[i, proto_k]
                    term_k = max(term_k, 1e-10)
                    denominator += (d_k / term_k) ** (1 / (self.m - 1))
                self.memberships[i, j] = 1 / denominator if denominator != 0 else 0

        self.labels = np.argmax(self.memberships, axis=1)
        self.centroids = np.zeros((self.n_clusters, data.shape[1]))
        for j in range(self.n_clusters):
            weights = self.memberships[:, j] ** self.m
            if np.sum(weights) > 0:
                self.centroids[j] = np.sum(data * weights[:, None], axis=0) / np.sum(weights)

        return self.memberships, self.centroids, self.labels, self.iterations

    def predict(self, X, K=None):
        """Predict cluster labels for new data."""
        if self.memberships is None or self.P is None:
            raise ValueError("Model not fitted. Call 'fit' first.")

        n_samples = X.shape[0]
        U_new = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for j in range(self.n_clusters):
                prototype_idx = self.sample_indices[self.P[j]]
                if K is None:
                    d_k = self.rbf_kernel(X[i], X[i]) + self.rbf_kernel(X[prototype_idx], X[prototype_idx]) - 2 * self.rbf_kernel(X[i], X[prototype_idx])
                else:
                    d_k = K[i, i] + K[prototype_idx, prototype_idx] - 2 * K[i, prototype_idx]
                d_k = max(d_k, 1e-10)
                denominator = 0
                for k in range(self.n_clusters):
                    proto_k = self.sample_indices[self.P[k]]
                    if K is None:
                        term_k = self.rbf_kernel(X[i], X[i]) + self.rbf_kernel(X[proto_k], X[proto_k]) - 2 * self.rbf_kernel(X[i], X[proto_k])
                    else:
                        term_k = K[i, i] + K[proto_k, proto_k] - 2 * K[i, proto_k]
                    term_k = max(term_k, 1e-10)
                    denominator += (d_k / term_k) ** (1 / (self.m - 1))
                U_new[i, j] = 1 / denominator if denominator != 0 else 0

        labels = np.argmax(U_new, axis=1)
        return labels