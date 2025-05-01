import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

class OKFCMClustering:
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
        self.P = None

    def compute_sigma(self, data):
        """Compute sigma dynamically based on data distribution."""
        n = min(100, data.shape[0])
        distances = []
        indices = np.random.choice(data.shape[0], size=n, replace=False)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((data[indices[i]] - data[indices[j]]) ** 2)
                distances.append(dist)
        sigma = np.sqrt(np.mean(distances)) if distances else 1.0
        return sigma * 1.5  # Increase sigma for better kernel spread

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
                P[j] = np.random.choice(n)
                continue
            term1 = np.dot(w_u_j.T, np.dot(K, w_u_j)) / (norm_w_u_j ** 2)
            distances = [term1 + K[i, i] - 2 * np.dot(w_u_j.T, K[:, i]) / norm_w_u_j for i in range(n)]
            P[j] = np.argmin(distances)

        return U, P

    def fit(self, data, K=None):
        """Fit oKFCM model using provided kernel matrix."""
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
        all_prototypes = []
        all_weights = []
        for l in range(len(chunks)):
            chunk_indices = chunks[l]
            if K is None:
                chunk_data = data[chunk_indices]
                K_chunk = np.zeros((len(chunk_indices), len(chunk_indices)))
                for i, idx_i in enumerate(chunk_indices):
                    for j, idx_j in enumerate(chunk_indices):
                        K_chunk[i, j] = self.rbf_kernel(data[idx_i], data[idx_j])
            else:
                K_chunk = K[chunk_indices][:, chunk_indices]
            weights = np.ones(len(chunk_indices))
            U_l, P_l = self.wKFCM(K_chunk, weights)
            prototypes = chunk_indices[P_l]
            w_l = np.sum(U_l, axis=0)
            all_prototypes.extend(prototypes)
            all_weights.extend(w_l)

        if K is None:
            prototype_data = data[all_prototypes]
            K_final = np.zeros((len(all_prototypes), len(all_prototypes)))
            for i, idx_i in enumerate(all_prototypes):
                for j, idx_j in enumerate(all_prototypes):
                    K_final[i, j] = self.rbf_kernel(data[idx_i], data[idx_j])
        else:
            K_final = K[all_prototypes][:, all_prototypes]
        weights = np.array(all_weights)
        U_final, P_final = self.wKFCM(K_final, weights)
        self.P = np.array(all_prototypes)[P_final]

        self.memberships = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for j in range(self.n_clusters):
                prototype_idx = self.P[j]
                if K is None:
                    d_k = self.rbf_kernel(data[i], data[i]) + self.rbf_kernel(data[prototype_idx], data[prototype_idx]) - 2 * self.rbf_kernel(data[i], data[prototype_idx])
                else:
                    d_k = K[i, i] + K[prototype_idx, prototype_idx] - 2 * K[i, prototype_idx]
                d_k = max(d_k, 1e-10)
                denominator = 0
                for k in range(self.n_clusters):
                    proto_k = self.P[k]
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

        # Check for too-close centroids
        dist_matrix = euclidean_distances(self.centroids, self.centroids)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix)
        if min_dist < 1e-2:  # Threshold to detect too-close centroids
            print(f"Warning: Centroids too close (min_dist={min_dist}). Reinitializing with KMeans...")
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state)
            kmeans.fit(data)
            self.centroids = kmeans.cluster_centers_
            # Recompute memberships based on new centroids
            self.memberships = np.zeros((n_samples, self.n_clusters))
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    d_k = np.sum((data[i] - self.centroids[j]) ** 2)
                    d_k = max(d_k, 1e-10)
                    denominator = 0
                    for k in range(self.n_clusters):
                        term_k = np.sum((data[i] - self.centroids[k]) ** 2)
                        term_k = max(term_k, 1e-10)
                        denominator += (d_k / term_k) ** (1 / (self.m - 1))
                    self.memberships[i, j] = 1 / denominator if denominator != 0 else 0
            self.labels = np.argmax(self.memberships, axis=1)

        return self.memberships, self.centroids, self.labels, self.iterations

    def predict(self, X, K=None):
        """Predict cluster labels for new data."""
        if self.memberships is None or self.P is None:
            raise ValueError("Model not fitted. Call 'fit' first.")

        n_samples = X.shape[0]
        U_new = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for j in range(self.n_clusters):
                prototype_idx = self.P[j]
                if K is None:
                    d_k = self.rbf_kernel(X[i], X[i]) + self.rbf_kernel(X[prototype_idx], X[prototype_idx]) - 2 * self.rbf_kernel(X[i], X[prototype_idx])
                else:
                    d_k = K[i, i] + K[prototype_idx, prototype_idx] - 2 * K[i, prototype_idx]
                d_k = max(d_k, 1e-10)
                denominator = 0
                for k in range(self.n_clusters):
                    proto_k = self.P[k]
                    if K is None:
                        term_k = self.rbf_kernel(X[i], X[i]) + self.rbf_kernel(X[proto_k], X[proto_k]) - 2 * self.rbf_kernel(X[i], X[proto_k])
                    else:
                        term_k = K[i, i] + K[proto_k, proto_k] - 2 * K[i, proto_k]
                    term_k = max(term_k, 1e-10)
                    denominator += (d_k / term_k) ** (1 / (self.m - 1))
                U_new[i, j] = 1 / denominator if denominator != 0 else 0

        labels = np.argmax(U_new, axis=1)
        return labels