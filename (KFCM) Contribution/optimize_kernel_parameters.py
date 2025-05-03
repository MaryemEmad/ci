#!/usr/bin/env python
# optimize_kernel_parameters.py: Example script for kernel parameter optimization

import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_mall_customers_data
from parameter_optimization import (
    run_parameter_optimization, 
    compare_optimization_methods,
    run_multi_trial_comparison
)
from kfcm_clustering import KernelFuzzyCMeansClustering
from mkfcm_clustering import ModifiedKernelFuzzyCMeansClustering
from sklearn.metrics import silhouette_score

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and preprocess data
    print("Loading Mall Customers dataset...")
    df, features_2d, features_3d, scaler_2d, scaler_3d = load_mall_customers_data()
    
    # Set optimization parameters
    n_clusters = 4
    pop_size = 20
    n_generations = 30
    
    # Choose which optimization process to run
    # Options: 'kfcm_only', 'mkfcm_only', 'compare_methods', 'multi_trial_comparison'
    optimization_mode = 'multi_trial_comparison'  # Change this to switch between modes
    
    if optimization_mode == 'kfcm_only':
        # Optimize KFCM parameters using GA only
        print("\nOptimizing KFCM parameters...")
        kfcm_results = run_parameter_optimization(
            data=features_2d,
            algorithm='kfcm',
            n_clusters=n_clusters,
            pop_size=pop_size,
            n_generations=n_generations,
            random_state=42,
            method='ga'  # Using Genetic Algorithm
        )
        
        # Get the best parameters
        best_params = kfcm_results['best_params']
        best_sigma = best_params['sigma_squared']
        best_m = best_params['m']
        best_score = kfcm_results['best_fitness']
        
        print(f"\nBest KFCM Parameters: σ²={best_sigma:.4f}, m={best_m:.4f}")
        print(f"Best Silhouette Score: {best_score:.4f}")
        
        # Compare with default parameters
        print("\nComparing with default parameters (σ²=10.0, m=2.0)...")
        default_results = run_parameter_optimization(
            data=features_2d,
            algorithm='kfcm',
            n_clusters=n_clusters,
            pop_size=1,  # Just to run once with default parameters
            n_generations=1,
            random_state=42,
            method='ga'
        )
        default_score = default_results['best_fitness']
        improvement = (best_score - default_score) / default_score * 100
        
        print(f"Default KFCM Silhouette Score: {default_score:.4f}")
        print(f"Improvement: {improvement:.2f}%")
        
        # Plot convergence
        print("\nPlotting optimization convergence...")
        
        # Visualize clustering results
        print("\nVisualizing clustering results...")
        
        print("\nOptimization complete! Results saved to:")
        print("  - kfcm_ga_convergence.png")
        print("  - kfcm_default_vs_optimized.png")
        
    elif optimization_mode == 'mkfcm_only':
        # Optimize MKFCM parameters using GA only
        print("\nOptimizing MKFCM parameters...")
        mkfcm_results = run_parameter_optimization(
            data=features_2d,
            algorithm='mkfcm',
            n_clusters=n_clusters,
            pop_size=pop_size,
            n_generations=n_generations,
            random_state=42,
            method='ga'  # Using Genetic Algorithm
        )
        
        # Get the best parameters
        best_params = mkfcm_results['best_params']
        best_sigma = best_params['sigma_squared']
        best_m = best_params['m']
        best_score = mkfcm_results['best_fitness']
        
        print(f"\nBest MKFCM Parameters: σ²={best_sigma:.4f}, m={best_m:.4f}")
        print(f"Best Silhouette Score: {best_score:.4f}")
        
        # Compare with default parameters
        print("\nComparing with default parameters (σ²=10.0, m=2.0)...")
        default_results = run_parameter_optimization(
            data=features_2d,
            algorithm='mkfcm',
            n_clusters=n_clusters,
            pop_size=1,  # Just to run once with default parameters
            n_generations=1,
            random_state=42,
            method='ga'
        )
        default_score = default_results['best_fitness']
        improvement = (best_score - default_score) / default_score * 100
        
        print(f"Default MKFCM Silhouette Score: {default_score:.4f}")
        print(f"Improvement: {improvement:.2f}%")
        
        # Plot convergence
        print("\nPlotting optimization convergence...")
        
        # Visualize clustering results
        print("\nVisualizing clustering results...")
        
        print("\nOptimization complete! Results saved to:")
        print("  - mkfcm_ga_convergence.png")
        print("  - mkfcm_default_vs_optimized.png")
        
    elif optimization_mode == 'compare_methods':
        # Compare GA and DE for KFCM optimization
        print("\nComparing GA and DE for KFCM parameter optimization...")
        
        # Reduced generations for comparison to make it faster
        comparison_generations = 15
        
        comparison_results = compare_optimization_methods(
            data=features_2d,
            algorithm='kfcm',
            n_clusters=n_clusters,
            pop_size=pop_size,
            n_generations=comparison_generations,
            random_state=42
        )
        
        # Extract results
        ga_results = comparison_results['ga_results']
        de_results = comparison_results['de_results']
        winner = comparison_results['winner']
        improvement = comparison_results['improvement']
        
        print("\nSummary of Comparison:")
        print(f"Total optimization time: {comparison_results['total_time']:.2f}s")
        print(f"GA best parameters: σ²={ga_results['best_params']['sigma_squared']:.4f}, m={ga_results['best_params']['m']:.4f}")
        print(f"DE best parameters: σ²={de_results['best_params']['sigma_squared']:.4f}, m={de_results['best_params']['m']:.4f}")
        print(f"GA best silhouette score: {ga_results['best_fitness']:.4f}")
        print(f"DE best silhouette score: {de_results['best_fitness']:.4f}")
        print(f"Winner: {winner} with {improvement:.2f}% improvement")
        
        print("\nOptimization comparison complete! Results saved to:")
        print("  - kfcm_ga_convergence.png")
        print("  - kfcm_de_convergence.png")
        print("  - kfcm_optimization_comparison.png")
        
    elif optimization_mode == 'multi_trial_comparison':
        # Run multiple trials to compare GA and DE more robustly
        print("\nRunning multi-trial comparison of GA and DE...")
        
        # Use fewer generations per trial to make the overall process faster
        n_trials = 3  # Number of trials with different random seeds
        comparison_generations = 10  # Reduced for faster execution
        
        # Run multi-trial comparison
        multi_trial_results = run_multi_trial_comparison(
            data=features_2d,
            algorithm='kfcm',
            n_clusters=n_clusters,
            pop_size=pop_size,
            n_generations=comparison_generations,
            n_trials=n_trials,
            base_seed=42
        )
        
        # Results are printed within the function and a plot is generated
        
        print("\nMulti-trial comparison complete! Results saved to:")
        print(f"  - kfcm_multi_trial_comparison.png")
        
        # Apply the best average parameters to see the final clustering result
        overall_winner = multi_trial_results['overall_winner']
        best_avg_params = (multi_trial_results['avg_ga_params'] 
                          if overall_winner == "Genetic Algorithm" 
                          else multi_trial_results['avg_de_params'])
        
        print(f"\nApplying {overall_winner} average best parameters:")
        print(f"σ²={best_avg_params['sigma_squared']:.4f}, m={best_avg_params['m']:.4f}")
        
        # Create and fit a model with the average best parameters
        kfcm = KernelFuzzyCMeansClustering(
            n_clusters=n_clusters,
            m=best_avg_params['m'],
            sigma_squared=best_avg_params['sigma_squared']
        )
        
        labels = kfcm.fit(features_2d)
        silhouette = silhouette_score(features_2d, labels)
        print(f"Final silhouette score with average parameters: {silhouette:.4f}")
    
    else:
        print(f"Unknown optimization mode: {optimization_mode}")
        print("Available modes: 'kfcm_only', 'mkfcm_only', 'compare_methods', 'multi_trial_comparison'")

if __name__ == "__main__":
    main() 