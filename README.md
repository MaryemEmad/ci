# ci

main.py: Coordinates the workflow (loads data, runs experiments, launches UI).

data_loader.py: Loads and preprocesses the Mall Customers dataset (standardizes Annual Income and Spending Score).

kmeans_clustering.py: Implements K-means as our baseline.

ga_clustering.py: Implements the Genetic Algorithm (GA) for clustering .

aco_clustering.py: Implements Ant Colony Optimization (ACO) for clustering .

abc_clustering.py: Implements Artificial Bee Colony (ABC) for clustering .

fa_clustering.py: Implements Firefly Algorithm (FA) for clustering.

de_clustering.py: Implements Differential Evolution (DE) for clustering.

experiment_runner.py: Tests each algorithm by running it several times for each setting (e.g., different mutation rates for GA). It measures performance (clustering quality, speed, and time taken) and saves random seeds so we can rerun the tests and get the same results.

visualization.py: Generates scatter plots for clustering results.

ui.py: Implements a Tkinter-based UI to run algorithms and display results.
