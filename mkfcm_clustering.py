# mkfcm_clustering.py: Implements Modified Kernel Fuzzy C-Means clustering (prototypes in feature space)
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

class ModifiedKernelFuzzyCMeansClustering:
    def __init__(self, n_clusters=4, m=2.0, max_iter=100, tol=1e-4, random_state=None, sigma_squared=10.0):
        """
        Initialize Modified Kernel Fuzzy C-Means algorithm (prototypes in feature space)
        
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
        sigma_squared : float, default=10.0
            Kernel parameter for Gaussian kernel: exp(-||x-y||^2 / sigma_squared)
            Higher values create smoother decision boundaries (recommended: 10.0)
            Note: Algorithm is sensitive to this parameter and may need tuning
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.input_centroids = None  # Approximated centroids in input space
        self.membership = None
        self.n_iter_ = 0
        self.inertia_ = 0
        self.random_state = random_state
        self.fitness_history = []  # Track fitness values over iterations
        self.sigma_squared = sigma_squared  # Kernel parameter
        
    def gaussian_kernel(self, x, y):
        """Compute Gaussian kernel: K(x, y) = exp(-||x-y||^2 / sigma^2)"""
        # x and y can be vectors or matrices
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
            
        # Calculate squared Euclidean distance efficiently
        if x.shape[0] == 1 and y.shape[0] == 1:
            # Single point to single point
            dist_squared = np.sum((x - y) ** 2)
        elif x.shape[0] == 1:
            # Single point to multiple points
            dist_squared = np.sum((y - x) ** 2, axis=1)
        elif y.shape[0] == 1:
            # Multiple points to single point
            dist_squared = np.sum((x - y) ** 2, axis=1)
        else:
            # Multiple points to multiple points
            dist_squared = cdist(x, y, 'sqeuclidean')
            
        # Calculate kernel value: K(x, y) = exp(-||x-y||^2 / sigma^2)
        return np.exp(-dist_squared / self.sigma_squared)
    
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
    
    def calculate_kernel_distances(self, X, membership):
        """
        Calculate kernel distances from data points to prototypes in feature space:
        
        d_{ik}^2 = K(x_k, x_k) - 2 * sum_j(u_{ij}^m * K(x_k, x_j)) / sum_j(u_{ij}^m) + 
                  sum_j(sum_l(u_{ij}^m * u_{il}^m * K(x_j, x_l))) / (sum_j(u_{ij}^m))^2
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        # Precompute kernel matrix for all data points (use vectorized computation)
        # For Gaussian kernel, we can compute it directly
        # Calculate squared pairwise distances efficiently
        X_squared = np.sum(X**2, axis=1, keepdims=True)
        dists_squared = X_squared + X_squared.T - 2 * np.dot(X, X.T)
        # Apply Gaussian kernel
        kernel_matrix = np.exp(-dists_squared / self.sigma_squared)
                
        # Raise membership to power m
        membership_pow = np.power(membership, self.m)
        
        # Small constant to avoid division by zero
        epsilon = 1e-10
        
        # For each cluster prototype
        for i in range(self.n_clusters):
            # Get membership values for current cluster
            u_i = membership_pow[:, i]
            
            # Compute denominator for second term: sum_j(u_{ij}^m)
            denominator2 = np.sum(u_i)
            denominator2 = max(denominator2, epsilon)  # Avoid division by zero
            
            # Compute second term for all data points at once
            # -2 * sum_j(u_{ij}^m * K(x_k, x_j)) / sum_j(u_{ij}^m)
            term2 = -2.0 * np.dot(kernel_matrix, u_i) / denominator2
            
            # Compute third term (same for all data points in the cluster)
            # sum_j(sum_l(u_{ij}^m * u_{il}^m * K(x_j, x_l))) / (sum_j(u_{ij}^m))^2
            weighted_kernel = u_i.reshape(-1, 1) * kernel_matrix * u_i.reshape(1, -1)
            term3 = np.sum(weighted_kernel) / (denominator2**2)
            
            # Combine terms for all data points
            # Term 1 is 1.0 for Gaussian kernel
            distances[:, i] = 1.0 + term2 + term3
            
            # Ensure distances are non-negative
            distances[:, i] = np.maximum(distances[:, i], 0.0)
        
        return distances
    
    def update_membership(self, X, membership):
        """
        Update membership matrix using distances in kernel space:
        u_{ik} = 1 / sum_j((d_{ik}^2 / d_{jk}^2)^(1/(m-1)))
        """
        n_samples = X.shape[0]
        new_membership = np.zeros((n_samples, self.n_clusters))
        
        # Calculate kernel distances
        distances = self.calculate_kernel_distances(X, membership)
        
        # Small constant to avoid division by zero
        epsilon = 1e-10
        
        # For each data point
        for k in range(n_samples):
            # Special case: if a point is exactly at a prototype (zero distance)
            zero_distances = np.where(distances[k] < epsilon)[0]
            if len(zero_distances) > 0:
                # Assign full membership to the closest prototype
                membership_row = np.zeros(self.n_clusters)
                membership_row[zero_distances[0]] = 1.0
                new_membership[k] = membership_row
                continue
            
            # Regular case: compute membership according to formula
            power = 1.0 / (self.m - 1)
            for i in range(self.n_clusters):
                denominator = 0.0
                for j in range(self.n_clusters):
                    # Handle potential numerical issues
                    dist_i = max(distances[k, i], epsilon)
                    dist_j = max(distances[k, j], epsilon)
                    ratio = dist_i / dist_j
                    denominator += np.power(ratio, power)
                
                # Set membership value
                if denominator > epsilon:
                    new_membership[k, i] = 1.0 / denominator
                else:
                    # If denominator is too small, assign full membership to this cluster
                    new_membership[k, i] = 1.0
            
            # Normalize to ensure sum equals 1
            row_sum = np.sum(new_membership[k])
            if row_sum > epsilon:
                new_membership[k] /= row_sum
        
        return new_membership
    
    def calculate_input_centroids(self, X, membership):
        """
        Approximate centroids in input space for visualization and prediction.
        Uses iterative formula:
        v_i = sum_k(u_{ik}^m * exp(-||x_k - v_i||^2/sigma^2) * x_k) / sum_k(u_{ik}^m * exp(-||x_k - v_i||^2/sigma^2))
        """
        n_samples, n_features = X.shape
        input_centroids = np.zeros((self.n_clusters, n_features))
        
        # Raise membership to power m
        membership_pow = np.power(membership, self.m)
        
        # For each cluster
        for i in range(self.n_clusters):
            # Initialize centroid as weighted average of data points
            weights = membership_pow[:, i].reshape(-1, 1)
            initial_centroid = np.sum(X * weights, axis=0) / np.sum(weights)
            
            # Initialize variables for iterative update
            old_centroid = initial_centroid
            new_centroid = np.zeros_like(old_centroid)
            
            # Initialize convergence variables
            converged = False
            centroid_iter = 0
            max_centroid_iter = 100  # Maximum iterations for centroid update
            
            # Iteratively update until convergence
            while not converged and centroid_iter < max_centroid_iter:
                # Calculate kernel values between data points and current centroid
                kernel_values = self.gaussian_kernel(X, old_centroid)
                
                # Calculate weighted kernel sum: sum(u_{ik}^m * K(x_k, v_i) * x_k)
                weighted_kernel = membership_pow[:, i] * kernel_values
                numerator = np.sum(weighted_kernel.reshape(-1, 1) * X, axis=0)
                
                # Calculate normalization factor: sum(u_{ik}^m * K(x_k, v_i))
                denominator = np.sum(weighted_kernel)
                
                # Avoid division by zero
                if denominator > np.finfo(float).eps:
                    new_centroid = numerator / denominator
                else:
                    # If denominator is too small, use weighted average of data points
                    new_centroid = initial_centroid
                
                # Check for convergence
                centroid_diff = np.linalg.norm(new_centroid - old_centroid)
                if centroid_diff < self.tol:
                    converged = True
                
                # Update for next iteration
                old_centroid = new_centroid
                centroid_iter += 1
            
            # Store the final centroid approximation
            input_centroids[i] = new_centroid
        
        return input_centroids
    
    def _calculate_inertia(self, X, membership=None):
        """
        Calculate the objective function value (inertia):
        J = sum_{i=1}^c sum_{k=1}^N u_{ik}^m d_{ik}^2
        
        where d_{ik}^2 is the kernel distance in feature space.
        """
        if membership is None:
            membership = self.membership
            
        # Calculate kernel distances
        distances = self.calculate_kernel_distances(X, membership)
        
        # Raise membership to power m
        membership_pow = np.power(membership, self.m)
        
        # Calculate objective function: sum(u_{ik}^m * d_{ik}^2)
        inertia = np.sum(membership_pow * distances)
        
        return inertia
    
    def fit(self, X):
        """
        Fit the MKFCM model to data X
        
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
        
        # Reset fitness history
        self.fitness_history = []
        
        # Main MKFCM loop
        for i in range(self.max_iter):
            old_membership = self.membership.copy()
            
            # Update memberships based on kernel distances
            self.membership = self.update_membership(X, self.membership)
            
            # Calculate and store fitness for this iteration
            current_fitness = self._calculate_inertia(X, self.membership)
            self.fitness_history.append(current_fitness)
            
            # Check for convergence
            diff = np.linalg.norm(self.membership - old_membership)
            if diff <= self.tol:
                break
                
        self.n_iter_ = i + 1
        
        # Calculate final inertia
        self.inertia_ = self._calculate_inertia(X, self.membership)
        
        # Approximate centroids in input space for visualization
        self.input_centroids = self.calculate_input_centroids(X, self.membership)
        
        # Return crisp labels based on highest membership
        return self.get_labels()
    
    def get_labels(self):
        """Get crisp cluster labels from fuzzy memberships"""
        if self.membership is None:
            return None
        return np.argmax(self.membership, axis=1)
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        For MKFCM, this requires precomputed membership from training data.
        We approximate this by computing kernel distances to input centroids.
        """
        if self.input_centroids is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        # Calculate kernel-based distances to input centroids
        for i in range(self.n_clusters):
            kernel_values = self.gaussian_kernel(X, self.input_centroids[i])
            # For Gaussian kernel, K(x, x) = K(v, v) = 1, so:
            # d^2 = 2 - 2K(x, v)
            distances[:, i] = 2.0 - 2.0 * kernel_values
            
        # Ensure non-negative distances
        distances = np.maximum(distances, np.finfo(float).eps)
        
        # Calculate memberships from distances
        memberships = np.zeros((n_samples, self.n_clusters))
        epsilon = 1e-10
        
        for k in range(n_samples):
            # Special case: if a point has zero distance to a centroid
            zero_distances = np.where(distances[k] < epsilon)[0]
            if len(zero_distances) > 0:
                # Assign full membership to the closest centroid
                memberships[k, zero_distances[0]] = 1.0
                continue
            
            # Regular case: compute membership using FCM formula
            power = 1.0 / (self.m - 1)
            for i in range(self.n_clusters):
                denominator = np.sum(np.power(distances[k, i] / (distances[k, :] + epsilon), power))
                if denominator > epsilon:
                    memberships[k, i] = 1.0 / denominator
                else:
                    memberships[k, i] = 1.0
                    
            # Normalize to ensure sum equals 1
            row_sum = np.sum(memberships[k])
            if row_sum > epsilon:
                memberships[k] /= row_sum
        
        # Return crisp labels
        return np.argmax(memberships, axis=1)
    
    def get_centroids(self):
        """Return approximated centroids in input space for visualization"""
        return self.input_centroids
    
    def get_membership(self):
        """Return membership matrix"""
        return self.membership
    
    def get_fitness_history(self):
        """Return fitness history for convergence analysis"""
        return self.fitness_history 
