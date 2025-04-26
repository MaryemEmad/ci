This project implements and compares Fuzzy C-Means (FCM) clustering with K-means for customer segmentation using the Mall Customers dataset, with a focus on analyzing the impact of FCM's fuzziness parameter (m) and K-means initialization methods.

## Features

- **Comprehensive Algorithms**:
  - Fuzzy C-Means with adjustable fuzziness parameter (m)
  - K-means with both k-means++ and random initialization

- **Extensive Analysis**:
  - Performance comparison of FCM and K-means
  - Impact of FCM's fuzziness parameter (m) values (1.1, 1.5, 2.0, 2.5, 3.0)
  - Comparison of K-means initialization methods
  - Statistical significance with 30 runs per algorithm
  - Reproducible experiments with stored random seeds

- **Robust Metrics**:
  - Silhouette score for clustering quality
  - Within-Cluster Sum of Squares (WCSS) for compactness
  - Execution time and convergence analysis

- **Rich Visualizations**:
  - 2D/3D scatter plots of clustering results
  - Elbow plots for determining optimal number of clusters
  - Side-by-side comparisons of K-means and FCM
  - FCM m-value comparison plots
  - K-means initialization comparison plots
  - Convergence curves

- **Interactive UI**:
  - GUI for running algorithms with adjustable parameters
  - Real-time visualization of results
  - Easy comparison between different algorithms and configurations
