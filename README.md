# Clustering-Based Customer Segmentation

![UI Screenshot](https://github.com/MaryemEmad/ci/blob/main/ui.png?raw=true)

This repository presents a customer segmentation project using various clustering algorithms, developed as part of a course on Evolutionary Algorithms & Computing (CI/EC). The goal is to apply intelligent clustering techniques to segment customers based on behavioral and demographic features using the "Mall_Customers.csv" dataset.

---

## 🔍 Project Idea

The aim of this project is to apply CI/EC algorithms to perform intelligent customer segmentation. The project explores multiple clustering techniques, including kernel-based fuzzy methods and classic algorithms such as K-Means. The interactive GUI allows users to compare clustering performance visually and statistically.

---

## 🎯 Objectives

- Apply multiple CI-based clustering algorithms (10 algorithms implemented).
- Compare clustering results using performance metrics such as Silhouette Score, WCSS, Davies-Bouldin Index, and execution time.
- Build an interactive Tkinter dashboard for visualization.
- Evaluate algorithms across different cluster numbers and parameters.

---

## 📁 Dataset

- **Name**: Mall_Customers.csv  
- **Features Used**:
  - 2D: Annual Income, Spending Score  
  - 3D: Age, Annual Income, Spending Score  
- **Source**: Public dataset  
- **Preprocessing**: Standardized using `StandardScaler` from Scikit-learn

---

## 🧠 Implemented Algorithms

1. **rseKFCM** – Random Sampling Enhanced Kernel Fuzzy C-Means  
2. **spKFCM** – Spatial Kernel Fuzzy C-Means  
3. **oKFCM** – Online Kernel Fuzzy C-Means  
4. **FCM** – Fuzzy C-Means  
5. **KFCM** – Kernel Fuzzy C-Means  
6. **MKFCM** – Modified Kernel Fuzzy C-Means  
7. **GK-FCM** – Gustafson-Kessel Fuzzy C-Means  
8. **K-Means**  
9. **ImprovedGathGeva**  
10. **IFCM** – Improved Fuzzy C-Means  

Each algorithm is implemented in its own module and is tested across various cluster numbers and fuzzification levels.

---

## ⚙️ CI Algorithm Components

For each CI algorithm, the following components are clearly defined and implemented:

- **Representation**: Cluster centers and membership degrees
- **Initialization**: Random, heuristic-based, or kernel-initialized
- **Evaluation Function**: Based on clustering metrics
- **Selection**: N/A (applied per fuzzy algorithm structure)
- **Variation Operators**: Kernel mapping, membership updates
- **Termination Condition**: Max iterations or convergence threshold
- **Constraint Handling**: N/A (unconstrained problem)
- **Diversity Preservation**: Different initialization and kernel strategies

---

## 📊 Performance Metrics

- **Silhouette Score**
- **WCSS** (Within-Cluster Sum of Squares)
- **Davies-Bouldin Index**
- **Execution Time (in seconds)**

Each result is recorded in the file `summary_results.csv`.

---

## 🧪 Experiments

The experiments were run using the `experiment_runner.py` script:
- Runs each algorithm with varying `m` values (fuzzification) and cluster counts
- Stores results (centroids, memberships, plots) under `run_results_m_<m>_n_clusters_<n>/`
- Visualizations are generated and saved under `results/`

---

## 🖼️ Visualizations

- 2D & 3D scatter plots  
- Heatmaps comparing algorithms  
- Cluster membership surfaces

![Summary Screenshot](https://github.com/MaryemEmad/ci/blob/main/summary.png?raw=true)

---

## 🖥️ GUI Dashboard

Implemented in `ui.py` using Tkinter:
- Select algorithm, cluster count, and `m` parameter
- View visual clustering results and metrics
- Compare algorithms side by side

---

## 📦 Requirements

Install required packages using:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn pillow



