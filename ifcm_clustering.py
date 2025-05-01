import numpy as np

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Intuitionistic Fuzzy C-Means (IFCM) Algorithm
def intuitionistic_fuzzy_c_means(X, C, m=2, max_iter=100, tol=1e-5):
    n_samples, n_features = X.shape

    # Initialize U randomly and normalize it
    U = np.random.rand(n_samples, C)
    U = U / np.sum(U, axis=1, keepdims=True)

    # Initialize V and R accordingly
    V = np.random.rand(n_samples, C)
    V = V / np.sum(V, axis=1, keepdims=True)

    R = 1 - (U + V)

    for iteration in range(max_iter):
        #cluster centers based on U  
        centers = np.zeros((C, n_features))
        for c in range(C):
            numerator = np.sum((U[:, c] ** m)[:, np.newaxis] * X, axis=0)
            denominator = np.sum(U[:, c] ** m)
            centers[c] = numerator / denominator

        # membership matrices
        dist = np.zeros((n_samples, C))
        for c in range(C):
            for i in range(n_samples):
                dist[i, c] = euclidean_distance(X[i], centers[c])

        # Avoid division by zero
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        U_new = np.zeros((n_samples, C))
        for i in range(n_samples):
            for c in range(C):
                sum_term = np.sum((dist[i, c] / dist[i, :]) ** (2 / (m - 1)))
                U_new[i, c] = 1 / sum_term

        # Update V and R accordingly
        V_new = 1 - U_new
        R_new = 1 - (U_new + V_new)

        # Normalize U, V, R  
        total = U_new + V_new + R_new
        U_new = U_new / total
        V_new = V_new / total
        R_new = R_new / total

        # Check for convergence
        if np.linalg.norm(U_new - U) < tol:
            break

        U, V, R = U_new, V_new, R_new

    return U, V, R, centers
