# Customer Segmentation Dashboard

This repository contains a comprehensive project for customer segmentation using various clustering algorithms, visualized through a user-friendly dashboard built with Tkinter. The project leverages machine learning techniques to segment customers based on the "Mall_Customers.csv" dataset and provides interactive visualizations of the results.

## Project Overview

The goal of this project is to analyze customer data and segment it into distinct groups using different clustering algorithms. The dashboard allows users to:
- Explore clustering results in 2D and 3D visualizations.
- Compare performance metrics such as Silhouette Score, WCSS, Davies-Bouldin Index, and execution time.
- View heatmaps and comparisons between algorithms (e.g., K-Means vs. FCM).

The project is implemented using Python and relies on libraries such as NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, and Tkinter.

## Features

- **Clustering Algorithms**: Includes rseKFCM, spKFCM, oKFCM, FCM, KFCM, MKFCM, GK-FCM, K-Means, ImprovedGathGeva, and IFCM.
- **Visualizations**: 2D and 3D scatter plots, heatmaps, and algorithm comparisons.
- **Interactive Dashboard**: Built with Tkinter, allowing users to select algorithms and view results dynamically.
- **Performance Metrics**: Displays Silhouette Score, WCSS, Davies-Bouldin Index, and execution time in a table.

## Requirements

To run this project, ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Pillow
- Tkinter (usually included with Python)

Install the required packages using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn pillow
```

## Project Structure

- **`Mall_Customers.csv`**: The dataset containing customer information (Age, Annual Income, Spending Score).
- **`run_results_m_<m>_n_clusters_<n>/`**: Directories storing precomputed clustering results (labels, centroids, memberships) for each combination of `m` and `n_clusters`.
- **`results/`**: Directory containing generated visualization images (e.g., 2D/3D plots, heatmaps).
- **`summary_results.csv`**: File with performance metrics for each algorithm across all experiments.
- **`ui.py`**: Main script for the Tkinter dashboard that loads and displays precomputed visualizations.
- **`visualization.py`**: Script to generate clustering visualizations (used to create images in `results`).
- **`experiment_runner.py`**: Script to run clustering experiments, evaluate performance, and generate results.

## Clustering Algorithms

The project implements and evaluates the following 10 clustering algorithms:

1. **rseKFCM (Random Sampling Enhanced Kernel Fuzzy C-Means)**:
   - A kernel-based fuzzy clustering algorithm that uses random sampling to improve scalability.
2. **spKFCM (Spatial Kernel Fuzzy C-Means)**:
   - A kernel-based fuzzy clustering algorithm that incorporates spatial information.
3. **oKFCM (Online Kernel Fuzzy C-Means)**:
   - A kernel-based fuzzy clustering algorithm designed for online data processing.
4. **FCM (Fuzzy C-Means)**:
   - A traditional fuzzy clustering algorithm that assigns membership probabilities to data points.
5. **KFCM (Kernel Fuzzy C-Means)**:
   - An extension of FCM that uses kernel functions to handle non-linear data distributions.
6. **MKFCM (Modified Kernel Fuzzy C-Means)**:
   - A modified version of KFCM with improved convergence and performance.
7. **GK-FCM (Gustafson-Kessel Fuzzy C-Means)**:
   - A fuzzy clustering algorithm that uses adaptive distance norms to capture cluster shapes.
8. **K-Means**:
   - A classic hard clustering algorithm that partitions data into `k` clusters.
9. **ImprovedGathGeva**:
   - An improved version of the Gath-Geva algorithm, enhancing fuzzy clustering with better initialization.
10. **IFCM (Improved Fuzzy C-Means)**:
    - An enhanced version of FCM with optimizations for better clustering accuracy.

Each algorithm is implemented in its respective Python file (e.g., `rsekfcm_clustering.py`, `fcm_clustering.py`), and their performance is evaluated using multiple metrics.

## Experiments

The experiments are conducted using the `experiment_runner.py` script, which performs the following:

### Setup
- **Dataset**: The "Mall_Customers.csv" dataset is used, with 2D (Annual Income, Spending Score) and 3D (Age, Annual Income, Spending Score) features extracted.
- **Preprocessing**: Data is standardized using `StandardScaler` from Scikit-learn.
- **Parameters**:
  - Fuzziness parameter (`m`): Tested with values `[1.5, 2.0, 2.5]`.
  - Number of clusters (`n_clusters`): Tested with values `[3, 5]`.
  - Number of runs: 30 runs per algorithm to ensure robust evaluation.

### Evaluation Metrics
For each algorithm, the following metrics are computed:
- **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters (higher is better).
- **Within-Cluster Sum of Squares (WCSS)**: Measures the compactness of clusters (lower is better).
- **Davies-Bouldin Index**: Measures the average similarity ratio of each cluster with its most similar cluster (lower is better).
- **Partition Coefficient**: Measures the fuzziness of the clustering (higher is better, used for fuzzy algorithms).
- **Xie-Beni Index**: Measures the ratio of intra-cluster distance to inter-cluster separation (lower is better, used for fuzzy algorithms).
- **Execution Time**: Measures the time taken for each run.

### Process
- For each combination of `m` and `n_clusters`, the script:
  1. Initializes the 10 algorithms with the specified parameters.
  2. Runs each algorithm 30 times with different random seeds to account for variability.
  3. Computes the evaluation metrics for both 2D and 3D data.
  4. Saves the best run (based on Silhouette Score) for each algorithm, including labels, centroids, and memberships, in a directory named `run_results_m_<m>_n_clusters_<n>`.
  5. Saves detailed run results for each algorithm in a CSV file (e.g., `rseKFCM_run_results.csv`).
- A summary of average metrics across all runs is saved in `summary_results.csv`.

### Results
- The results are used to generate visualizations in `visualization.py`, which are then displayed in the dashboard via `ui.py`.
- Example metrics for `m=2.0`, `n_clusters=5` (as shown in the dashboard screenshot):
  - rseKFCM: Avg Silhouette (2D) = 0.5447, Avg WCSS (2D) = 52.1462, Avg Time = 0.4108s
  - K-Means: Avg Silhouette (2D) = 0.5539, Avg WCSS (2D) = 65.5664, Avg Time = 0.0416s
  - ImprovedGathGeva: Avg Silhouette (2D) = 0.5530, Avg WCSS (2D) = 134.8649, Avg Time = 0.3896s

## Setup and Usage

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone <repository-url>
cd customer-segmentation-dashboard
```

### 2. Prepare the Data and Results
- Ensure `Mall_Customers.csv` is in the project directory.
- If the `run_results_m_<m>_n_clusters_<n>/` directories and `summary_results.csv` are not present, run `experiment_runner.py` to generate them:
  ```python
  import pandas as pd
  from sklearn.preprocessing import StandardScaler
  from experiment_runner import run_experiments

  data = pd.read_csv('Mall_Customers.csv')
  data_2d = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
  data_3d = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

  scaler = StandardScaler()
  data_2d = scaler.fit_transform(data_2d)
  data_3d = scaler.fit_transform(data_3d)

  m_values = [1.5, 2.0, 2.5]
  n_clusters_values = [3, 5]

  for m in m_values:
      for n_clusters in n_clusters_values:
          # Define algorithms
          from rsekfcm_clustering import RseKFCMClustering
          from spkfcm_clustering import SpKFCMClustering
          from okfcm_clustering import OKFCMClustering
          from fcm_clustering import FuzzyCMeansClustering
          from kfcm_clustering import KernelFuzzyCMeansClustering
          from mkfcm_clustering import ModifiedKernelFuzzyCMeansClustering
          from gkfcm_clustering import GKFuzzyCMeansClustering
          from kmeans_clustering import KMeansClustering
          from improved_gath_geva import ImprovedGathGeva
          from ifcm_clustering import IFCMClustering

          algorithms = {
              'rseKFCM': RseKFCMClustering(n_clusters=n_clusters, m=m),
              'spKFCM': SpKFCMClustering(n_clusters=n_clusters, m=m),
              'oKFCM': OKFCMClustering(n_clusters=n_clusters, m=m),
              'FCM': FuzzyCMeansClustering(n_clusters=n_clusters, m=m),
              'KFCM': KernelFuzzyCMeansClustering(n_clusters=n_clusters, m=m),
              'MKFCM': ModifiedKernelFuzzyCMeansClustering(n_clusters=n_clusters, m=m),
              'GK-FCM': GKFuzzyCMeansClustering(n_clusters=n_clusters, m=m),
              'K-Means': KMeansClustering(n_clusters=n_clusters),
              'ImprovedGathGeva': ImprovedGathGeva(n_clusters=n_clusters, m=m),
              'IFCM': IFCMClustering(n_clusters=n_clusters, m=m)
          }

          results_dir = f"run_results_m_{m}_n_clusters_{n_clusters}"
          run_experiments(data_2d, data_3d, algorithms, n_runs=30, m=m, results_dir=results_dir)
  ```
- Run `visualization.py` to generate images in the `results/` directory if not already present:
  ```python
  from visualization import run_all_visualizations
  import pandas as pd
  from sklearn.preprocessing import StandardScaler

  data = pd.read_csv('Mall_Customers.csv')
  data_2d = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
  data_3d = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

  scaler_2d = StandardScaler()
  scaler_3d = StandardScaler()
  data_2d = scaler_2d.fit_transform(data_2d)
  data_3d = scaler_3d.fit_transform(data_3d)

  run_all_visualizations(data_2d, data_3d, scaler_2d, scaler_3d, m_values=[2.0], n_clusters_values=[5])
  ```

### 3. Run the Dashboard
Execute the dashboard script:
```bash
python ui.py
```
- The dashboard will open, displaying a table of performance metrics and buttons to view 2D/3D plots, heatmaps, and comparisons.
- Click on an algorithm button (e.g., "Show spKFCM") to load its precomputed visualizations from the `results/` directory.

## Contributions

The project welcomes contributions in the following areas:

### Potential Enhancements
1. **Algorithm Improvements**:
   - Optimize the existing clustering algorithms for better performance (e.g., reduce execution time).
   - Add new clustering algorithms to compare with the current ones.
2. **Visualization Enhancements**:
   - Add interactive visualizations using libraries like Plotly for better user experience.
   - Include more comparison plots (e.g., effect of `m` on all algorithms in a single plot).
3. **UI Improvements**:
   - Enhance the Tkinter dashboard with more features (e.g., filters for `m` and `n_clusters`).
   - Migrate the dashboard to a web-based interface using frameworks like Flask or Dash.
4. **Evaluation Metrics**:
   - Add more clustering evaluation metrics (e.g., Calinski-Harabasz Index).
   - Implement cross-validation techniques to validate clustering results.
5. **Scalability**:
   - Optimize the code to handle larger datasets efficiently.
   - Add support for parallel processing to speed up experiments.

### How to Contribute
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m "Description of changes"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source community for providing powerful libraries like Matplotlib, Seaborn, and Scikit-learn.
- Special thanks to the contributors of the "Mall_Customers.csv" dataset.

## Contact

For questions or feedback, please open an issue on the GitHub repository or contact the maintainer.