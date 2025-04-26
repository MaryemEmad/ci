# fcm_clustering.py: Implements Fuzzy C-Means clustering
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

class FuzzyCMeansClustering:
    def __init__(self, n_clusters=4, m=2.0, max_iter=100, tol=1e-4, random_state=None):
        """
        Initialize Fuzzy C-Means algorithm
        
        Parameters:
        -----------
        n_clusters : int, default=4
            Number of clusters to create
        m : float, default=2
            Fuzziness coefficient (m > 1)
        max_iter : int, default=100
            Maximum number of iterations
        tol : float, default=1e-4
            Tolerance for convergence
        random_state : int, default=None
            Random state for reproducibility
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
        
    def update_membership(self, X, centroids):
        """Update membership matrix"""
        # Calculate distances between data points and centroids
        distances = cdist(X, centroids, metric='euclidean')
        
        # Handle division by zero
        distances = np.maximum(distances, np.finfo(np.float64).eps)
        
        # Handle case where a point is exactly at a centroid
        for i in range(len(X)):
            zero_distances = np.where(distances[i] == 0)[0]
            if len(zero_distances) > 0:
                membership_row = np.zeros(self.n_clusters)
                membership_row[zero_distances[0]] = 1.0
                distances[i, :] = np.ones(self.n_clusters)  # Prevent division issues
                return distances, membership_row
        
        # Calculate new memberships using FCM formula
        power = -2 / (self.m - 1)
        tmp = np.power(distances, power)
        
        # Normalize membership coefficients
        denominator = np.sum(tmp, axis=1)[:, np.newaxis]
        # Avoid division by zero
        denominator = np.maximum(denominator, np.finfo(float).eps)
        membership = tmp / denominator
        
        return membership
    
    def _calculate_inertia(self, X, labels=None):
        """
        Calculate Within-Cluster Sum of Squares (WCSS)
        
        Parameters:
        -----------
        X : array-like
            Input data
        labels : array-like, optional
            Cluster labels (if None, use fuzzy assignments)
            
        Returns:
        --------
        wcss : float
            Within-Cluster Sum of Squares
        """
        if labels is None:
            # Use fuzzy inertia
            membership_pow = np.power(self.membership, self.m)
            distances = np.square(cdist(X, self.centroids, metric='euclidean'))
            return np.sum(membership_pow * distances)
        else:
            # Use crisp inertia (comparable to K-means)
            wcss = 0
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    wcss += np.sum(np.square(cdist(cluster_points, 
                                               [self.centroids[i]], 
                                               metric='euclidean')))
            return wcss
    
    def fit(self, X):
        """
        Fit the FCM model to data X
        
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
        
        # Main FCM loop
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
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        # Calculate distances and memberships for new data
        membership = self.update_membership(X, self.centroids)
        return np.argmax(membership, axis=1)
    
    def compute_fitness(self, data, labels=None):
        """Compute fitness (inertia) of the clustering"""
        return float(self.inertia_)
    
    def get_crisp_inertia(self):
        """Get crisp inertia (comparable to K-means)"""
        return float(self.crisp_inertia_)
    
    def get_centroids(self):
        """Return centroids for visualization"""
        return self.centroids
    
    def get_fitness_history(self):
        """Return fitness history for convergence analysis"""
        return self.fitness_history
