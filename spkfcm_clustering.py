import numpy as np

class SpKFCMClustering:
    def __init__(self, n_clusters, m=2.0, n_chunks=2, sigma=None, max_iter=100, epsilon=1e-3, random_state=None):
        self.n_clusters = n_clusters
        self.m = m
        self.n_chunks = n_chunks
        self.sigma = sigma
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        self.centroids = None
        self.memberships = None
        self.labels = None
        self.iterations = 0
        self.prototypes = None

    def compute_sigma(self, data):
        """Compute sigma dynamically based on data distribution."""
        n = min(100, data.shape[0])
        distances = []
        indices = np.random.choice(data.shape[0], size=n, replace=False)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((data[indices[i]] - data[indices[j]]) ** 2)
                distances.append(dist)
        return np.sqrt(np.mean(distances))

    def rbf_kernel(self, x1, x2):
        """Compute RBF kernel between two vectors."""
        dist = np.sum((x1 - x2) ** 2)
        return np.exp(-dist / (2 * (self.sigma ** 2)))

    def wKFCM(self, K, weights):
        """Weighted KFCM implementation."""
        n = K.shape[0]
        U = np.ones((n, self.n_clusters)) / self.n_clusters
        U += np.random.normal(0, 0.1, U.shape)
        U = np.maximum(U, 1e-10)
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
                    d_k = max(d_k, 1e-10)
                    denominator = 0
                    for k in range(self.n_clusters):
                        u_k = U[:, k] ** self.m
                        w_u_k = weights * u_k
                        norm_w_u_k = np.sum(w_u_k)
                        if norm_w_u_k == 0:
                            continue
                        term_k = term1 + K[i, i] - 2 * np.dot(w_u_k.T, K[:, i]) / norm_w_u_k
                        term_k = max(term_k, 1e-10)
                        denominator += (d_k / term_k) ** (1 / (self.m - 1))
                    U[i, j] = 1 / denominator if denominator != 0 else 0

            if np.max(np.abs(U - U_old)) < self.epsilon:
                self.iterations += iteration + 1
                break
        else:
            self.iterations += self.max_iter

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
        """Fit spKFCM model using provided kernel matrix."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.sigma is None:
            self.sigma = self.compute_sigma(data)

        n_samples = data.shape[0]
        chunk_size = max(n_samples // self.n_chunks, 1)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        chunks = [indices[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]

        self.iterations = 0
        self.prototypes = chunks[0]
        chunk_data = data[chunks[0]]
        if K is None:
            K_chunk = np.zeros((len(chunks[0]), len(chunks[0])))
            for i, idx_i in enumerate(chunks[0]):
                for j, idx_j in enumerate(chunks[0]):
                    K_chunk[i, j] = self.rbf_kernel(data[idx_i], data[idx_j])
        else:
            K_chunk = K[chunks[0]][:, chunks[0]]
        weights = np.ones(len(chunks[0]))
        U, P = self.wKFCM(K_chunk, weights)
        prototype_weights = np.sum(U, axis=0)
        self.prototypes = chunks[0][P]

        for l in range(1, len(chunks)):
            chunk_indices = chunks[l]
            combined_indices = np.concatenate((self.prototypes, chunk_indices))
            if K is None:
                combined_data = np.vstack((data[self.prototypes], data[chunk_indices]))
                K_combined = np.zeros((len(combined_indices), len(combined_indices)))
                for i, idx_i in enumerate(combined_indices):
                    for j, idx_j in enumerate(combined_indices):
                        K_combined[i, j] = self.rbf_kernel(data[idx_i], data[idx_j])
            else:
                K_combined = K[combined_indices][:, combined_indices]
            weights = np.concatenate((prototype_weights, np.ones(len(chunk_indices))))
            U, P = self.wKFCM(K_combined, weights)
            prototype_weights = np.sum(U[:len(self.prototypes)], axis=0)
            self.prototypes = combined_indices[P]

        self.memberships = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for j in range(self.n_clusters):
                prototype_idx = self.prototypes[j]
                if K is None:
                    d_k = self.rbf_kernel(data[i], data[i]) + self.rbf_kernel(data[prototype_idx], data[prototype_idx]) - 2 * self.rbf_kernel(data[i], data[prototype_idx])
                else:
                    d_k = K[i, i] + K[prototype_idx, prototype_idx] - 2 * K[i, prototype_idx]
                d_k = max(d_k, 1e-10)
                denominator = 0
                for k in range(self.n_clusters):
                    proto_k = self.prototypes[k]
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
        if self.memberships is None or self.prototypes is None:
            raise ValueError("Model not fitted. Call 'fit' first.")

        n_samples = X.shape[0]
        U_new = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for j in range(self.n_clusters):
                prototype_idx = self.prototypes[j]
                if K is None:
                    d_k = self.rbf_kernel(X[i], X[i]) + self.rbf_kernel(X[prototype_idx], X[prototype_idx]) - 2 * self.rbf_kernel(X[i], X[prototype_idx])
                else:
                    d_k = K[i, i] + K[prototype_idx, prototype_idx] - 2 * K[i, prototype_idx]
                d_k = max(d_k, 1e-10)
                denominator = 0
                for k in range(self.n_clusters):
                    proto_k = self.prototypes[k]
                    if K is None:
                        term_k = self.rbf_kernel(X[i], X[i]) + self.rbf_kernel(X[proto_k], X[proto_k]) - 2 * self.rbf_kernel(X[i], X[proto_k])
                    else:
                        term_k = K[i, i] + K[proto_k, proto_k] - 2 * K[i, proto_k]
                    term_k = max(term_k, 1e-10)
                    denominator += (d_k / term_k) ** (1 / (self.m - 1))
                U_new[i, j] = 1 / denominator if denominator != 0 else 0

        labels = np.argmax(U_new, axis=1)
        return labels