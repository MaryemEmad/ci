# kfcm_clustering.py: Implements Kernel Fuzzy C-Means clustering (prototypes in input space)
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

class KernelFuzzyCMeansClustering:
    def __init__(self, n_clusters=4, m=2.0, max_iter=100, tol=1e-4, random_state=None, sigma_squared=10.0):
        """
        Initialize Kernel Fuzzy C-Means algorithm (prototypes in input space)
        
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
        self.centroids = None
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
            
        # Calculate squared Euclidean distance efficiently using broadcasting
        if x.shape[0] == 1 and y.shape[0] > 1:
            # Single point to multiple points
            dist_squared = np.sum((y - x)**2, axis=1)
        elif y.shape[0] == 1 and x.shape[0] > 1:
            # Multiple points to single point
            dist_squared = np.sum((x - y)**2, axis=1)
        else:
            # Multiple points to multiple points or single point to single point
            x_squared = np.sum(x**2, axis=1, keepdims=True)
            y_squared = np.sum(y**2, axis=1).reshape(1, -1)
            dist_squared = x_squared + y_squared - 2 * np.dot(x, y.T)
            
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
        
    def update_centroids(self, X, membership):
        """
        Update centroids based on membership values using KFCM formula:
        v_i = sum_{k=1}^N u_{ik}^m K(x_k, v_i) x_k / sum_{k=1}^N u_{ik}^m K(x_k, v_i)
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Raise membership to power m
        membership_pow = np.power(membership, self.m)
        
        # For each cluster
        for i in range(self.n_clusters):
            # Start with previous centroid estimate or centroid of points with highest membership
            if self.centroids is not None:
                prev_centroid = self.centroids[i]
            else:
                # Get initial centroid as weighted average of data points
                weights = membership_pow[:, i].reshape(-1, 1)
                prev_centroid = np.sum(X * weights, axis=0) / np.sum(weights)
            
            # Initialize variables for iterative update
            old_centroid = prev_centroid
            new_centroid = np.zeros_like(old_centroid)
            
            # Initialize convergence variables
            converged = False
            centroid_iter = 0
            max_centroid_iter = 20  # Maximum iterations for centroid update
            
            # Iteratively update until convergence
            while not converged and centroid_iter < max_centroid_iter:
                # Calculate kernel values between data points and current centroid
                kernel_values = self.gaussian_kernel(X, old_centroid)
                
                # Calculate weighted sum: sum(u_{ik}^m * K(x_k, v_i) * x_k)
                weighted_kernel = membership_pow[:, i] * kernel_values
                numerator = np.sum(weighted_kernel.reshape(-1, 1) * X, axis=0)
                
                # Calculate normalization factor: sum(u_{ik}^m * K(x_k, v_i))
                denominator = np.sum(weighted_kernel)
                
                # Avoid division by zero
                if denominator > np.finfo(float).eps:
                    new_centroid = numerator / denominator
                else:
                    # If denominator is too small, use weighted average of data points
                    weights = membership_pow[:, i].reshape(-1, 1)
                    new_centroid = np.sum(X * weights, axis=0) / np.sum(weights)
                
                # Check for convergence
                centroid_diff = np.linalg.norm(new_centroid - old_centroid)
                if centroid_diff < self.tol:
                    converged = True
                
                # Update for next iteration
                old_centroid = new_centroid
                centroid_iter += 1
            
            # Store the final centroid
            centroids[i] = new_centroid
        
        return centroids
    
    def update_membership(self, X, centroids):
        """
        Update membership matrix using KFCM formula:
        u_{ik} = 1 / sum_{j=1}^c ( (1 - K(x_k, v_i)) / (1 - K(x_k, v_j)) )^(1/(m-1))
        
        For Gaussian kernel, this simplifies to:
        u_{ik} = 1 / sum_{j=1}^c ( (1 - exp(-||x_k - v_i||^2/sigma^2)) / (1 - exp(-||x_k - v_j||^2/sigma^2)) )^(1/(m-1))
        """
        n_samples = X.shape[0]
        new_membership = np.zeros((n_samples, self.n_clusters))
        
        # Calculate kernel values between all data points and all centroids
        kernel_matrix = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            kernel_matrix[:, i] = self.gaussian_kernel(X, centroids[i])
        
        # Calculate 1 - K(x_k, v_i) for all data points and centroids
        distance_matrix = 1.0 - kernel_matrix
        
        # Small constant to avoid division by zero
        epsilon = 1e-10
        
        # Power factor for membership calculation
        power = 1.0 / (self.m - 1)
        
        # For each data point
        for k in range(n_samples):
            # Special case: if a point is exactly at a centroid
            # (kernel value is approximately 1)
            centroid_matches = np.where(kernel_matrix[k] > 1.0 - epsilon)[0]
            if len(centroid_matches) > 0:
                # Assign full membership to the closest centroid
                membership_row = np.zeros(self.n_clusters)
                membership_row[centroid_matches[0]] = 1.0
                new_membership[k] = membership_row
                continue
            
            # Calculate membership denominator for each cluster
            # We compute this in a vectorized way for efficiency
            dist_i = np.maximum(distance_matrix[k], epsilon)
            denominators = np.zeros(self.n_clusters)
            
            for i in range(self.n_clusters):
                # Calculate sum_{j=1}^c ( dist_i / dist_j )^(1/(m-1))
                ratios = dist_i[i] / dist_i  # ratio to all clusters
                denominators[i] = np.sum(np.power(ratios, power))
            
            # Calculate membership values
            for i in range(self.n_clusters):
                if denominators[i] > epsilon:
                    new_membership[k, i] = 1.0 / denominators[i]
                else:
                    # If denominator is too small, assign full membership to this cluster
                    new_membership[k, i] = 1.0
            
            # Normalize to ensure sum equals 1
            row_sum = np.sum(new_membership[k])
            if row_sum > epsilon:
                new_membership[k] /= row_sum
        
        return new_membership
    
    def _calculate_inertia(self, X):
        """
        Calculate kernel-based objective function value (inertia):
        Q = sum_{i=1}^c sum_{k=1}^N u_{ik}^m ||Phi(x_k) - Phi(v_i)||^2
        
        For Gaussian kernel:
        ||Phi(x_k) - Phi(v_i)||^2 = K(x_k, x_k) + K(v_i, v_i) - 2 K(x_k, v_i)
                                    = 2 - 2 K(x_k, v_i)
        
        So inertia becomes:
        Q = sum_{i=1}^c sum_{k=1}^N u_{ik}^m (2 - 2 K(x_k, v_i))
          = 2 sum_{i=1}^c sum_{k=1}^N u_{ik}^m (1 - K(x_k, v_i))
        """
        n_samples = X.shape[0]
        
        # Calculate kernel values between all data points and all centroids
        kernel_matrix = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            kernel_matrix[:, i] = self.gaussian_kernel(X, self.centroids[i])
        
        # Raise membership to power m
        membership_pow = np.power(self.membership, self.m)
        
        # Calculate inertia: 2 * sum(u_{ik}^m * (1 - K(x_k, v_i)))
        inertia = 2.0 * np.sum(membership_pow * (1.0 - kernel_matrix))
        
        return inertia
    
    def get_kernel_distances(self, X, centroids=None):
        """
        Calculate kernel-based distances between data points and centroids:
        d_{ik}^2 = ||Phi(x_k) - Phi(v_i)||^2 = K(x_k, x_k) + K(v_i, v_i) - 2 K(x_k, v_i)
        
        For Gaussian kernel, K(x_k, x_k) = K(v_i, v_i) = 1, so:
        d_{ik}^2 = 2 - 2 K(x_k, v_i)
        """
        if centroids is None:
            centroids = self.centroids
            
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        # Calculate kernel values and distances
        for i in range(self.n_clusters):
            # Kernel values: K(x_k, v_i)
            kernel_values = self.gaussian_kernel(X, centroids[i])
            
            # Kernel distances: 2 - 2 K(x_k, v_i)
            distances[:, i] = 2.0 - 2.0 * kernel_values
        
        return distances
    
    def fit(self, X):
        """
        Fit the KFCM model to data X
        
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
        
        # Main KFCM loop
        for i in range(self.max_iter):
            old_membership = self.membership.copy()
            
            # Update centroids based on memberships
            self.centroids = self.update_centroids(X, self.membership)
            
            # Update memberships based on new centroids
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
        self.inertia_ = self._calculate_inertia(X)
        
        # Return crisp labels based on highest membership
        return self.get_labels()
    
    def get_labels(self):
        """Get crisp cluster labels from fuzzy memberships"""
        if self.membership is None:
            return None
        return np.argmax(self.membership, axis=1)
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        # Calculate kernel-based distances
        distances = self.get_kernel_distances(X)
        
        # Small constant to avoid division by zero
        epsilon = 1e-10
        
        # Calculate memberships based on distances
        n_samples = X.shape[0]
        memberships = np.zeros((n_samples, self.n_clusters))
        
        for k in range(n_samples):
            # Special case: if a point has zero distance to a centroid
            zero_distances = np.where(distances[k] < epsilon)[0]
            if len(zero_distances) > 0:
                # Assign full membership to the closest centroid
                memberships[k, zero_distances[0]] = 1.0
                continue
            
            # Regular case: compute membership using standard FCM formula
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
        """Return centroids for visualization"""
        return self.centroids
    
    def get_membership(self):
        """Return membership matrix"""
        return self.membership
    
    def get_fitness_history(self):
        """Return fitness history for convergence analysis"""
        return self.fitness_history 
