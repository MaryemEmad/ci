# kmeans_clustering.py: Implements K-means clustering
from sklearn.cluster import KMeans
import numpy as np

class KMeansClustering:
    def __init__(self, n_clusters=4, init_method='k-means++'):
        self.n_clusters = n_clusters
        self.init_method = init_method  # 'k-means++' or 'random'
        self.model = KMeans(n_clusters=n_clusters, init=init_method, random_state=42)

    def fit(self, data):
        self.model.fit(data)
        return self.model.labels_

    def compute_fitness(self, data, labels):
        # Fitness: Within-cluster sum of squares (WCSS, to minimize)
        return float(self.model.inertia_)

    def get_centroids(self):
        # Return centroids for visualization
        return self.model.cluster_centers_
