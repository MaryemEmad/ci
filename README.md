# Customer Segmentation using Computational Intelligence (CI) Algorithms

This project implements and compares various Computational Intelligence (CI) algorithms for customer segmentation, including:
- K-means (baseline, fully implemented - DO NOT MODIFY)
- Genetic Algorithm (GA)
- Ant Colony Optimization (ACO)
- Artificial Bee Colony (ABC)
- ACO+K-means (hybrid approach)
- Differential Evolution (DE)

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Two options to run the program:

### Option 1: Run with K-means only (ready to use)
```bash
python run.py
```
This will:
- Load the Mall Customers dataset
- Generate an elbow plot to help determine optimal cluster count
- Launch a user interface where you can click "Run K-means" to execute the algorithm
- Show both 2D and 3D visualizations after running

### Option 2: Run with all algorithms
```bash
python main.py
```
This will:
- Load the Mall Customers dataset
- Generate elbow plots
- Launch a user interface with buttons for all algorithms
- NOTE: You need to implement the CI algorithms first (see Implementation Guide below)

## Implementation Guide for New Contributors

### Important Note
- The K-means implementation (`kmeans_clustering.py`) is complete and should NOT be modified
- You only need to implement the CI algorithms in their respective files
- K-means serves as the baseline for comparison with your CI implementations

### Step 1: Understanding the Code Structure

Each algorithm has its own file:
- `ga_clustering.py` - Genetic Algorithm
- `aco_clustering.py` - Ant Colony Optimization
- `abc_clustering.py` - Artificial Bee Colony
- `acokmeans_clustering.py` - ACO+K-means hybrid
- `de_clustering.py` - Differential Evolution

### Step 2: Understanding the Interface Requirements

Each algorithm class MUST implement:

1. `__init__(self, n_clusters=4, ...)` - Constructor with at least:
   - `n_clusters` parameter to set number of clusters
   - Algorithm-specific parameters
   - `random_state` parameter for reproducibility
   - Must initialize `self.model` attribute with an object that has `n_iter_` attribute

2. `fit(data)` - Main method that:
   - Takes a dataset as input
   - Applies the clustering algorithm
   - Returns cluster labels (integers from 0 to n_clusters-1)
   - Updates `self.centroids` with final cluster centers

3. `get_centroids()` - Returns the cluster centers

### Step 3: Implementing Your Algorithm

1. Replace the placeholder K-means code with your algorithm implementation in each CI file
2. Each file already contains detailed comments about what your algorithm should do
3. Example workflow for GA:
   ```python
   def fit(self, data):
       # Set random seed
       if self.random_state is not None:
           np.random.seed(self.random_state)
       
       # 1. Initialize population of solutions
       # 2. For each generation:
       #    a. Evaluate fitness of solutions
       #    b. Select parents
       #    c. Apply crossover and mutation
       #    d. Create new generation
       # 3. Find best solution (cluster assignments)
       # 4. Calculate centroids
       
       # Store results
       self.centroids = ...  # calculated centroids
       return labels  # cluster assignments for each data point
   ```

### Step 4: How the UI Works

When a user clicks an algorithm button in the UI:
1. The algorithm's `fit()` method is called on 2D data
2. Silhouette score, execution time, and iterations are calculated
3. A 2D visualization is generated
4. The algorithm is run again on 3D data
5. A 3D visualization is generated
6. All metrics and file paths are displayed in the UI

## Project Structure Explained

- `main.py`: Entry point that initializes all algorithms and launches the UI
- `run.py`: Simplified entry point that only initializes K-means
- `data_loader.py`: Handles loading and preprocessing the Mall Customers dataset
- `visualization.py`: Creates 2D and 3D visualizations of clustering results
- `ui.py`: Implements the interactive user interface using Tkinter
- Algorithm files:
  - `kmeans_clustering.py`: K-means baseline (fully implemented, DO NOT MODIFY)
  - CI algorithm files (to be implemented by you):
    - Each file currently contains placeholder implementations that call K-means
    - You need to replace this placeholder code with your actual algorithm implementation
    - Keep the same interface (constructor, fit, get_centroids)

## Algorithm Parameters

When implementing your algorithms, consider these parameters:

1. **Genetic Algorithm (GA)**:
   - Population size, crossover rate, mutation rate
   - Selection method (tournament, roulette wheel)
   - Encoding scheme for cluster assignments

2. **Ant Colony Optimization (ACO)**:
   - Number of ants, pheromone importance, heuristic importance
   - Pheromone evaporation rate
   - Pheromone update strategy

3. **Artificial Bee Colony (ABC)**:
   - Colony size, limit for abandonment
   - Employed, onlooker, and scout bee behaviors

4. **ACO+K-means**:
   - ACO parameters (ants, pheromone rates)
   - How/when to apply K-means (initialization, refinement)
   - Balance between exploration and exploitation

5. **Differential Evolution (DE)**:
   - Population size, differential weight (F)
   - Crossover probability (CR)
   - Mutation strategy

## Implementation Details for ACO+K-means Hybrid

For the ACO+K-means hybrid approach, consider these implementation strategies:
1. **Sequential Approach**: Use ACO to find initial centroids, then refine with K-means
2. **Iterative Approach**: Alternate between ACO and K-means steps
3. **Enhanced Approach**: Use K-means to improve solutions found by ants during ACO iterations

The hybrid should leverage the global search capability of ACO with the fast convergence of K-means.

## Performance Evaluation

Your implementation will be evaluated on:
1. Clustering quality (silhouette score)
2. Convergence speed (number of iterations)
3. Computational efficiency (execution time)

## Dataset Details

The Mall Customers dataset includes:
- CustomerID: Unique identifier
- Gender: Male/Female
- Age: Customer's age
- Annual Income (k$): Yearly income in thousands of dollars
- Spending Score (1-100): Score based on customer spending behavior and purchasing data
