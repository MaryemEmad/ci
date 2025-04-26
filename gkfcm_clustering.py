# gkfcm_clustering.py: Implements Gustafson-Kessel Fuzzy C-Means clustering
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy import linalg

class GKFuzzyCMeansClustering:
    def __init__(self, n_clusters=4, m=2.0, max_iter=100, tol=1e-4, random_state=None, min_det_value=1e-10):
        """
        Initialize Gustafson-Kessel Fuzzy C-Means algorithm
        
        Parameters:
        -----------
        n_clusters : int, default=4
            Number of clusters to create
        m : float, default=2.0
            Fuzziness coefficient (m > 1)
        max_iter : int, default=100
            Maximum number of iterations
        tol : float, default=1e-4
            Tolerance for convergence
        random_state : int, default=None
            Random state for reproducibility
        min_det_value : float, default=1e-10
            Minimum covariance determinant value to prevent singularity
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.membership = None
        self.n_iter_ = 0
        self.inertia_ = 0
        self.random_state = random_state
        self.fitness_history = []  # Track fitness values over iterations
        self.min_det_value = min_det_value  # Minimum covariance determinant to prevent singularity
        self.covariance_matrices = None  # Will store covariance matrices for all clusters
        self.norm_matrices = None  # Will store norm-inducing matrices for all clusters
        self.inv_covariance_matrices = None  # Will store inverse covariance matrices for all clusters
        
    def initialize_membership(self, n_samples):
        """Initialize random fuzzy memberships"""
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        # Generate random memberships
        membership = np.random.rand(n_samples, self.n_clusters)
        # Normalize the memberships to sum to 1 for each sample
        membership = membership / np.sum(membership, axis=1)[:, np.newaxis]
        return membership
        
    def update_centroids(self, X, membership):
        """Update centroids based on membership values"""
        # Raise membership to power m
        membership_pow = np.power(membership, self.m)
        # Calculate centroids as weighted mean
        numerator = np.dot(membership_pow.T, X)
        denominator = np.sum(membership_pow, axis=0)[:, np.newaxis]
        # Avoid division by zero
        denominator = np.maximum(denominator, np.finfo(float).eps)
        centroids = numerator / denominator
        return centroids
    
    def update_covariance_matrices(self, X, centroids, membership):
        """
        Update covariance matrices for each cluster
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data
        centroids : array-like of shape (n_clusters, n_features)
            Cluster centroids
        membership : array-like of shape (n_samples, n_clusters)
            Fuzzy membership matrix
            
        Returns:
        --------
        covariance_matrices : array-like of shape (n_clusters, n_features, n_features)
            Updated covariance matrices for each cluster
        """
        n_samples, n_features = X.shape
        covariance_matrices = np.zeros((self.n_clusters, n_features, n_features))
        
        for i in range(self.n_clusters):
            # Weighted membership values
            weights = np.power(membership[:, i], self.m).reshape(-1, 1)
            
            # Calculate weighted mean (centroid)
            centroid = centroids[i].reshape(1, -1)
            
            # Calculate weighted covariance matrix using vectorized operations
            diff = X - centroid  # shape: (n_samples, n_features)
            
            # Calculate weighted outer products efficiently using np.einsum
            # This replaces the loop in the original implementation
            weighted_diff = diff * np.sqrt(weights)  # shape: (n_samples, n_features)
            weighted_sum = np.einsum('ki,kj->ij', weighted_diff, weighted_diff)
            
            # Normalize by sum of weights
            sum_weights = np.sum(weights) + 1e-10  # Avoid division by zero
            cov_matrix = weighted_sum / sum_weights
            
            # Add small identity matrix for numerical stability
            cov_matrix += 1e-6 * np.eye(n_features)
            
            # Store covariance matrix
            covariance_matrices[i] = cov_matrix
        
        return covariance_matrices
    
    def calculate_norm_matrices(self, covariance_matrices, n_features):
        """Calculate the norm-inducing matrices for each cluster"""
        norm_matrices = np.zeros_like(covariance_matrices)
        
        for i in range(self.n_clusters):
            # Calculate determinant of covariance matrix
            det_cov = max(np.linalg.det(covariance_matrices[i]), self.min_det_value)
            
            # Cluster volume parameter (rho_i), set to 1.0 as per paper
            rho_i = 1.0
            
            # Calculate the scaling factor: (rho_i * det(C_i))^(1/d)
            scaling_factor = np.power(rho_i * det_cov, 1.0 / n_features)
            
            # Calculate inverse of covariance matrix
            # Use linalg.pinv for numerical stability when inverting
            cov_inv = linalg.pinv(covariance_matrices[i])
            
            # Calculate norm-inducing matrix: (rho_i * det(C_i))^(1/d) * C_i^(-1)
            norm_matrices[i] = scaling_factor * cov_inv
            
        return norm_matrices
    
    def mahalanobis_distance(self, X, centroid, inv_cov_matrix):
        """
        Calculate Mahalanobis distance between points in X and centroid
        using the inverse covariance matrix
        """
        # Vectorized implementation for speed
        diff = X - centroid  # shape: (n_samples, n_features)
        
        # Calculate Mahalanobis distance efficiently
        # dist = sqrt((x - μ)^T Σ^-1 (x - μ))
        # For each sample: dist[i] = sqrt(diff[i] @ inv_cov_matrix @ diff[i].T)
        
        # Method 1: Compute for all samples at once using a dot product approach
        # tmp = diff @ inv_cov_matrix  # shape: (n_samples, n_features)
        # dist = np.sqrt(np.sum(tmp * diff, axis=1))
        
        # Method 2: Compute for all samples using einsum (more efficient)
        dist = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov_matrix, diff))
        
        return dist

    def update_membership(self, X, centroids):
        """
        Update membership matrix using Gustafson-Kessel FCM formula:
        u_{ik} = 1 / sum_{j=1}^c ( d_A(x_k, v_i) / d_A(x_k, v_j) )^(2/(m-1))
        
        where d_A is the Mahalanobis distance using cluster-specific covariance matrices
        """
        n_samples = X.shape[0]
        new_membership = np.zeros((n_samples, self.n_clusters))
        
        # Calculate distances between all data points and all centroids
        distances = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = self.mahalanobis_distance(
                X, centroids[i], self.inv_covariance_matrices[i]
            )
        
        # Small constant to avoid division by zero
        epsilon = 1e-10
        
        # Power factor for membership calculation
        power = 2.0 / (self.m - 1)
        
        # Handle special case for points that are exactly at centroids
        for k in range(n_samples):
            centroid_matches = np.where(distances[k] < epsilon)[0]
            if len(centroid_matches) > 0:
                # Assign full membership to the closest centroid
                membership_row = np.zeros(self.n_clusters)
                membership_row[centroid_matches[0]] = 1.0
                new_membership[k] = membership_row
                continue
                
            # Calculate membership values using vectorized operations
            dist_k = np.maximum(distances[k], epsilon)  # distances for point k to all centroids
            
            for i in range(self.n_clusters):
                # Calculate the ratios for point k to all clusters
                ratios = dist_k[i] / dist_k
                denominators = np.sum(np.power(ratios, power))
                
                if denominators > epsilon:
                    new_membership[k, i] = 1.0 / denominators
                else:
                    # If denominator is too small, assign full membership to this cluster
                    new_membership[k, i] = 1.0
                    
            # Normalize to ensure sum equals 1
            row_sum = np.sum(new_membership[k])
            if row_sum > epsilon:
                new_membership[k] /= row_sum
        
        return new_membership
    
    def _calculate_inertia(self, X, labels=None):
        """
        Calculate the objective function value (inertia)
        
        Parameters:
        -----------
        X : array-like
            Input data
        labels : array-like, optional
            Cluster labels (if None, use fuzzy assignments)
            
        Returns:
        --------
        inertia : float
            Objective function value (sum of weighted distances)
        """
        n_samples = X.shape[0]
        
        if labels is None:
            # Fuzzy inertia using membership values
            inertia = 0.0
            membership_pow = np.power(self.membership, self.m)
            
            # Calculate distances using Mahalanobis distance
            distances = np.zeros((n_samples, self.n_clusters))
            for i in range(self.n_clusters):
                distances[:, i] = self.mahalanobis_distance(X, self.centroids[i], self.inv_covariance_matrices[i])
            
            # Calculate inertia: sum_i sum_k u_ik^m * d_ik^2
            for i in range(self.n_clusters):
                inertia += np.sum(membership_pow[:, i] * distances[:, i])
            
            return inertia
        else:
            # Crisp inertia using hard assignments
            inertia = 0.0
            
            for i in range(self.n_clusters):
                # Get points assigned to cluster i
                cluster_points = X[labels == i]
                
                if len(cluster_points) > 0:
                    # Calculate difference vectors: (x_k - v_i)
                    diff = cluster_points - self.centroids[i]
                    
                    # Calculate Mahalanobis distances
                    A_i = self.norm_matrices[i]
                    
                    for k in range(len(cluster_points)):
                        diff_k = diff[k].reshape(-1, 1)
                        inertia += np.dot(np.dot(diff_k.T, A_i), diff_k).item()
            
            return inertia
    
    def fit(self, X):
        """
        Fit the GK-FCM model to data X
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        labels : array of shape (n_samples,)
            Crisp cluster labels based on highest membership
        """
        n_samples, n_features = X.shape
        
        # Initialize random membership matrix
        self.membership = self.initialize_membership(n_samples)
        
        # Initialize arrays for covariance and inverse covariance matrices
        self.covariance_matrices = np.zeros((self.n_clusters, n_features, n_features))
        self.inv_covariance_matrices = np.zeros((self.n_clusters, n_features, n_features))
        
        # Reset fitness history
        self.fitness_history = []
        
        # Main GK-FCM loop
        for i in range(self.max_iter):
            old_membership = self.membership.copy()
            
            # Update centroids based on memberships
            self.centroids = self.update_centroids(X, self.membership)
            
            # Update covariance matrices
            self.covariance_matrices = self.update_covariance_matrices(X, self.centroids, self.membership)
            
            # Calculate inverse covariance matrices
            for j in range(self.n_clusters):
                try:
                    self.inv_covariance_matrices[j] = np.linalg.inv(self.covariance_matrices[j])
                except np.linalg.LinAlgError:
                    # If matrix is singular, use pseudoinverse
                    self.inv_covariance_matrices[j] = np.linalg.pinv(self.covariance_matrices[j])
            
            # Calculate norm-inducing matrices
            self.norm_matrices = self.calculate_norm_matrices(self.covariance_matrices, n_features)
            
            # Update memberships based on new centroids and norm matrices
            self.membership = self.update_membership(X, self.centroids)
            
            # Calculate and store fitness for this iteration
            current_fitness = self._calculate_inertia(X)
            self.fitness_history.append(current_fitness)
            
            # Check for convergence
            diff = np.linalg.norm(self.membership - old_membership)
            if diff <= self.tol:
                break
                
        self.n_iter_ = i + 1
        
        # Calculate final inertia
        labels = self.get_labels()
        self.inertia_ = self._calculate_inertia(X)
        self.crisp_inertia_ = self._calculate_inertia(X, labels)
        
        # Return crisp labels based on highest membership
        return labels
    
    def get_labels(self):
        """Get crisp cluster labels from fuzzy memberships"""
        if self.membership is None:
            return None
        return np.argmax(self.membership, axis=1)
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        if self.centroids is None or self.norm_matrices is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        # Calculate memberships for new data using Mahalanobis distances
        memberships = self.update_membership(X, self.centroids)
        return np.argmax(memberships, axis=1)
    
    def compute_fitness(self, data, labels=None):
        """Compute fitness (inertia) of the clustering"""
        return float(self.inertia_)
    
    def get_crisp_inertia(self):
        """Get crisp inertia (comparable to K-means)"""
        return float(self.crisp_inertia_)
    
    def get_centroids(self):
        """Return centroids for visualization"""
        return self.centroids
    
    def get_covariance_matrices(self):
        """Return covariance matrices for analysis"""
        return self.covariance_matrices
    
    def get_norm_matrices(self):
        """Return norm-inducing matrices for analysis"""
        return self.norm_matrices
    
    def get_fitness_history(self):
        """Return fitness history for convergence analysis"""
        return self.fitness_history 
