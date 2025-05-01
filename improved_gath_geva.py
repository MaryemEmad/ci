import numpy as np
from scipy.spatial.distance import mahalanobis

class ImprovedGathGeva:
    def __init__(self, n_clusters=5, max_iter=100, m=2, epsilon=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.epsilon = epsilon
        self.random_state = random_state
        self.centers = None
        self.covariances = None
        self.membership = None
        self.pi = None

    def _initialize_centers(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centers = X[indices]

    def _initialize_membership(self, X):
        n_samples = X.shape[0]
       
        self.membership = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)

    def _calculate_covariances(self, X):
        n_features = X.shape[1]
        self.covariances = np.zeros((self.n_clusters, n_features, n_features))

        for i in range(self.n_clusters):
            diff = X - self.centers[i]
            weighted_diff = (self.membership[:, i][:, np.newaxis] ** self.m) * diff
            cov = (weighted_diff.T @ diff) / np.sum(self.membership[:, i] ** self.m)

            # Adding small value to ensure positive definiteness
            cov += np.eye(n_features) * 1e-6
            self.covariances[i] = cov

    def _calculate_pi(self):
        self.pi = np.mean(self.membership, axis=0)

    def _calculate_distances(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))

        for i in range(self.n_clusters):
            try:
                inv_cov = np.linalg.inv(self.covariances[i])
                det_cov = np.linalg.det(self.covariances[i])
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(self.covariances[i])
                det_cov = 1e-6

            diff = X - self.centers[i]
            mahalanobis_dist = np.array([mahalanobis(x, np.zeros(len(x)), inv_cov) for x in diff])

            # Capping the mahalanobis distance to prevent overflow
            mahalanobis_dist = np.clip(mahalanobis_dist, None, 100)

            # Ensure that exponential doesn't overflow
            distances[:, i] = (np.sqrt(det_cov) / self.pi[i]) * np.exp(np.clip(0.5 * mahalanobis_dist ** 2, None, 700))

        return distances

    def fit(self, X):
        n_samples, n_features = X.shape
        self._initialize_centers(X)
        self._initialize_membership(X)

        for iteration in range(self.max_iter):
            old_membership = self.membership.copy()

            self._calculate_covariances(X)
            self._calculate_pi()

            distances = self._calculate_distances(X)

            distances = np.fmax(distances, np.finfo(np.float64).eps)
            exponent = 2 / (self.m - 1)
            membership = np.zeros((n_samples, self.n_clusters))

            for i in range(self.n_clusters):
                ratio = distances[:, i, np.newaxis] / distances
                membership[:, i] = 1 / np.sum(ratio ** exponent, axis=1)

            self.membership = membership

            for i in range(self.n_clusters):
                weighted_sum = np.sum((self.membership[:, i] ** self.m)[:, np.newaxis] * X, axis=0)
                sum_weights = np.sum(self.membership[:, i] ** self.m)
                self.centers[i] = weighted_sum / sum_weights

            if np.linalg.norm(self.membership - old_membership) < self.epsilon:
                print(f"Converged at iteration {iteration}")
                break

        return self

    def predict(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)

    def calculate_mml(self, X):
        n, d = X.shape
        k = self.n_clusters
        m = self.m
        U = self.membership
        pi = self.pi
        covariances = self.covariances

        likelihood = 0
        for i in range(k):
            try:
                cov = covariances[i]
                inv_cov = np.linalg.inv(cov)
                det_cov = np.linalg.det(cov)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov)
                det_cov = 1e-6

            diff = X - self.centers[i]
            mahal = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
            likelihood += np.sum((U[:, i] ** m) * (np.log(pi[i]) - 0.5 * (np.log(det_cov) + mahal)))

        penalty = 0.5 * k * d * np.log(n)
        mml = -likelihood + penalty
        return mml
