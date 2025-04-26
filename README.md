# Clustering Algorithms Comparison

This project implements and compares several clustering algorithms:
1. **K-means** - The classic hard clustering algorithm
2. **Fuzzy C-Means (FCM)** - The classic fuzzy clustering algorithm
3. **Gustafson-Kessel FCM (GK-FCM)** - An extension of FCM that can detect clusters of different shapes
4. **Kernel Fuzzy C-Means (KFCM)** - Uses kernel tricks to handle non-linearly separable clusters
5. **Modified Kernel Fuzzy C-Means (MKFCM)** - An enhanced version of KFCM

## Features

- Implementation of K-means, FCM, GK-FCM, KFCM, and MKFCM algorithms
- Comprehensive parameter study capabilities (fuzzy coefficient m, kernel parameter σ²)
- Statistical analysis with multiple runs and error quantification
- Performance metrics calculation (Silhouette Score, Inertia, Execution Time)
- Visualization tools for clustering results and performance comparison
- Example scripts demonstrating algorithm usage

## Key Experiments

The project includes several parameter studies and algorithm comparisons:

1. **K-means vs FCM**: Compares basic hard and fuzzy clustering approaches
2. **K-means Initialization**: Compares random vs. k-means++ initialization
3. **FCM m-value**: Studies the effect of the fuzziness coefficient on clustering
4. **FCM vs GK-FCM**: Compares standard FCM against Gustafson-Kessel variant 
5. **Fuzzy Algorithm Comparison**: Compares all fuzzy clustering approaches
6. **Kernel σ² Parameter Study**: Analyzes the impact of kernel width parameter on KFCM and MKFCM

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clustering-comparison.git
cd clustering-comparison

# Install dependencies
pip install -r requirements.txt
```

## Running the Project

The main script `run.py` performs all experiments and generates visualizations:

```bash
# Run all experiments and generate visualizations
python run.py
```

## Basic Algorithm Usage

```python
# K-means
from kmeans_clustering import KMeansClustering
kmeans = KMeansClustering(n_clusters=3, init_method='k-means++')
kmeans_labels = kmeans.fit(X)

# Fuzzy C-Means
from fcm_clustering import FuzzyCMeansClustering
fcm = FuzzyCMeansClustering(n_clusters=3, m=2.0)
fcm_labels = fcm.fit(X)

# Gustafson-Kessel FCM
from gkfcm_clustering import GKFuzzyCMeansClustering
gkfcm = GKFuzzyCMeansClustering(n_clusters=3, m=2.0)
gkfcm_labels = gkfcm.fit(X)

# Kernel FCM
from kfcm_clustering import KernelFuzzyCMeansClustering
kfcm = KernelFuzzyCMeansClustering(n_clusters=3, m=2.0, sigma_squared=10.0)
kfcm_labels = kfcm.fit(X)

# Modified Kernel FCM
from mkfcm_clustering import ModifiedKernelFuzzyCMeansClustering
mkfcm = ModifiedKernelFuzzyCMeansClustering(n_clusters=3, m=2.0, sigma_squared=10.0)
mkfcm_labels = mkfcm.fit(X)
```

## Performance Metrics

The project uses the following metrics for comparison:

- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters (higher is better)
- **Inertia**: Sum of squared distances to cluster centers (lower is better)
- **Execution Time**: Processing time efficiency (lower is better)

## Statistical Analysis

All experiments are run multiple times with different random seeds to ensure statistical significance. Results include:
- Mean and standard deviation for all metrics
- Visualizations with error bars
- Parameter study trend analysis

## Important Notes

- **Convergence Curves**: Scikit-learn's KMeans implementation doesn't expose the actual iteration-by-iteration convergence data, so convergence curve plots only include fuzzy algorithms with real convergence history.

## When to Use Which Algorithm

- **K-means**: Fast, efficient for spherical clusters of similar size
- **Fuzzy C-Means (FCM)**: When objects might belong to multiple clusters with different degrees
- **Gustafson-Kessel FCM (GK-FCM)**: For detecting clusters of different geometric shapes (ellipsoidal clusters)
- **Kernel FCM (KFCM)**: For handling non-linearly separable clusters in the original space
- **Modified Kernel FCM (MKFCM)**: For improved handling of non-linearly separable clusters with better convergence properties

## Parameter Tuning

- **m value (1.1-3.0)**: Controls the fuzziness of membership assignments
  - Lower values (closer to 1) create more crisp boundaries
  - Higher values create softer boundaries with more overlap
  
- **σ² value (0.1-100.0)**: Controls the width of the Gaussian kernel
  - Lower values make the algorithm more sensitive to local structures
  - Higher values create smoother decision boundaries


