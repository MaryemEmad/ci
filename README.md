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

## Examples

The `examples` directory contains sample scripts demonstrating how to use these algorithms with different datasets:

- `compare_algorithms.py`: Compares both algorithms on blob-shaped and moon-shaped datasets

## Code Structure and Development Guide

### Project Structure

- **Algorithm Implementations**:
  - `kmeans_clustering.py`: K-means implementation (wrapper around scikit-learn)
  - `fcm_clustering.py`: Fuzzy C-Means implementation
  - `gkfcm_clustering.py`: Gustafson-Kessel FCM implementation
  - `kfcm_clustering.py`: Kernel FCM implementation
  - `mkfcm_clustering.py`: Modified Kernel FCM implementation

- **Core components**:
  - `data_loader.py`: Loads and preprocesses the Mall Customers dataset
  - `run.py`: Main script that runs all experiments and generates visualizations
  - `visualization.py`: Functions for creating plots and visualizations




## Detailed File-by-File Guide

For teammates completely new to this codebase, here's a detailed breakdown of each file:

### Core Files

#### `run.py`
The main execution script that orchestrates all experiments and visualizations.
- **Key Functions**:
  - `main()`: Entry point that runs all experiments sequentially
  - `run_kmeans_fcm_comparison()`: Compares K-means and FCM algorithms
  - `run_fcm_m_comparison()`: Studies the effect of the fuzzy coefficient m
  - `run_kmeans_init_comparison()`: Compares K-means initialization methods
  - `run_fcm_gkfcm_comparison()`: Compares FCM and GK-FCM algorithms
  - `run_all_fuzzy_comparison()`: Compares all fuzzy algorithms
  - `run_kernel_sigma_comparison()`: Studies the effect of the kernel width parameter σ²
- **Outputs**: Creates all visualization files in the root directory
- **How to Use**: Simply run `python run.py` to execute all experiments

#### `data_loader.py`
Loads and preprocesses the Mall Customers dataset.
- **Key Functions**:
  - `load_mall_customers_data()`: Loads data, creates 2D and 3D feature sets, and performs normalization
- **Returns**: Original dataframe, 2D features, 3D features, and scalers for inverse transformation
- **Note**: Expects "Mall_Customers.csv" file in the project directory

#### `visualization.py`
Contains all visualization functions for creating plots and charts.
- **Key Functions**:
  - `plot_clusters_2d()` / `plot_clusters_3d()`: Create cluster visualizations
  - `compare_kmeans_fcm()`: Side-by-side comparison of two algorithms
  - `plot_fcm_m_comparison()`: Visualize the effect of the m parameter
  - `plot_convergence_curves()`: Show algorithm convergence over iterations
  - `compare_fuzzy_metrics()`: Compare metrics across algorithms with bar charts
  - `compare_fuzzy_metrics_with_error_bars()`: Enhanced comparison with statistical error bars
  - `plot_sigma_parameter_study()`: Analyze the effect of σ² parameter with trend lines
- **Dependencies**: Requires matplotlib and numpy

### Algorithm Implementations

#### `kmeans_clustering.py`
K-means implementation (wrapper around scikit-learn).
- **Key Methods**:
  - `__init__()`: Initialize with n_clusters, init_method, etc.
  - `fit(data)`: Run clustering and return labels
  - `predict(data)`: Predict cluster for new data
  - `compute_fitness(data)`: Calculate inertia (within-cluster sum of squares)
  - `get_centroids()`: Return final cluster centers
- **Note**: No real convergence history is available from scikit-learn's implementation

#### `fcm_clustering.py`
Fuzzy C-Means implementation.
- **Key Methods**:
  - `fit(data)`: Run FCM algorithm with membership updates
  - `update_centroids()`: Calculate new centroids based on fuzzy memberships
  - `update_membership()`: Update membership matrix
  - `_calculate_inertia()`: Calculate fuzzy inertia objective function
  - `get_fitness_history()`: Return convergence history for visualization
- **Parameters**: n_clusters, m (fuzziness coefficient), max_iter, tolerance

#### `gkfcm_clustering.py`
Gustafson-Kessel FCM that can detect non-spherical clusters.
- **Key Methods**: 
  - Similar to FCM but adds:
  - `update_covariance_matrices()`: Calculate cluster-specific covariance
  - `calculate_mahalanobis_distance()`: Distance measure for elliptical clusters
  - `calculate_norm_matrices()`: Normalization factors for covariance
- **Difference from FCM**: Uses Mahalanobis distance instead of Euclidean distance

#### `kfcm_clustering.py`
Kernel Fuzzy C-Means for non-linearly separable clusters.
- **Key Methods**:
  - `gaussian_kernel()`: Implements Gaussian kernel function
  - `fit(data)`: Runs the kernelized version of FCM
  - `update_centroids_kernel()`: Updates centroids in kernel space
- **Parameters**: Adds sigma_squared (kernel width parameter)

#### `mkfcm_clustering.py`
Modified Kernel FCM with improved convergence.
- **Key Methods**:
  - Similar to KFCM but with modified objective function
  - Different update rules for centroids and memberships
- **Difference from KFCM**: Centroids are computed in feature space first

### Data and Examples

#### `Mall_Customers.csv`
The main dataset used for all experiments.
- **Features**:
  - "CustomerID"
  - "Gender"
  - "Age"
  - "Annual Income (k$)"
  - "Spending Score (1-100)"
- **Used Features for Clustering**:
  - 2D: "Annual Income" and "Spending Score"
  - 3D: "Age", "Annual Income", and "Spending Score"

### Code Relationships and Flow

1. `run.py` begins execution and calls `data_loader.py` to get the dataset
2. It then runs each experiment function, which:
   - Initializes algorithm instances from the appropriate clustering files
   - Runs multiple iterations with different random seeds
   - Collects metrics and results
3. For each experiment, it calls visualization functions from `visualization.py`
4. All results are saved as image files in the project directory



