# ui.py: Implements the UI using Tkinter
import tkinter as tk
from tkinter import messagebox
from visualization import plot_clusters_2d, plot_clusters_3d
import numpy as np
import time
from sklearn.metrics import silhouette_score

def start_ui(data_2d, algorithms, data_3d=None, results=None):
    root = tk.Tk()
    root.title("Customer Segmentation App")
    root.geometry("500x550")

    tk.Label(root, text="Customer Segmentation", font=("Arial", 16)).pack(pady=10)
    
    # Information panel (optional)
    info_frame = tk.Frame(root)
    info_frame.pack(pady=5, fill="x")
    
    info_label = tk.Label(
        info_frame, 
        text="Press a button to run a clustering algorithm",
        anchor="w",
        font=("Arial", 10)
    )
    info_label.pack(fill="x", padx=20)
    
    # Result display area
    result_frame = tk.Frame(root)
    result_frame.pack(pady=10, fill="x")
    
    result_text = tk.StringVar()
    result_text.set("Select an algorithm to see results")
    
    result_label = tk.Label(
        result_frame,
        textvariable=result_text,
        justify="left",
        anchor="w",
        font=("Courier", 10),
        bg="#f0f0f0",
        relief="sunken",
        padx=10,
        pady=10
    )
    result_label.pack(fill="x", padx=20, ipady=10)

    def run_and_show(algo_name):
        # Update UI to show processing
        result_text.set(f"Running {algo_name}...")
        root.update()
        
        # Get the algorithm
        algo = algorithms[algo_name]
        
        # Run the algorithm on 2D data and measure the time
        start_time = time.time()
        labels_2d = algo.fit(data_2d)
        end_time = time.time()
        execution_time_2d = end_time - start_time
        
        # Get centroids and calculate metrics for 2D
        centroids_2d = algo.get_centroids() if hasattr(algo, 'get_centroids') else None
        
        if labels_2d is not None and centroids_2d is not None:
            # Calculate silhouette score for 2D
            score_2d = silhouette_score(data_2d, labels_2d)
            
            # Get iterations if available
            iterations = getattr(algo.model, 'n_iter_', 0) if hasattr(algo, 'model') else 0
            
            # Generate 2D visualization
            plot_clusters_2d(data_2d, labels_2d, centroids_2d, algo_name)
            
            # Run on 3D data if available
            labels_3d = None
            centroids_3d = None
            score_3d = None
            execution_time_3d = 0
            
            if data_3d is not None:
                # Update UI
                result_text.set(f"Running {algo_name} on 3D data...")
                root.update()
                
                # Run on 3D data
                start_time = time.time()
                labels_3d = algo.fit(data_3d)
                end_time = time.time()
                execution_time_3d = end_time - start_time
                
                centroids_3d = algo.get_centroids() if hasattr(algo, 'get_centroids') else None
                
                if labels_3d is not None and centroids_3d is not None:
                    # Calculate silhouette score for 3D
                    score_3d = silhouette_score(data_3d, labels_3d)
                    
                    # Generate 3D visualization
                    plot_clusters_3d(data_3d, labels_3d, centroids_3d, algo_name)
            
            # Display results
            result_info = f"Algorithm: {algo_name}\n"
            result_info += f"2D Silhouette Score: {score_2d:.4f}\n"
            result_info += f"2D Execution Time: {execution_time_2d:.4f}s\n"
            
            if score_3d is not None:
                result_info += f"3D Silhouette Score: {score_3d:.4f}\n"
                result_info += f"3D Execution Time: {execution_time_3d:.4f}s\n"
                
            result_info += f"Iterations: {iterations}\n\n"
            result_info += f"Visualizations saved as:\n"
            result_info += f"- {algo_name}_clusters_2d.png"
            
            if data_3d is not None:
                result_info += f"\n- {algo_name}_clusters_3d.png"
                
            result_text.set(result_info)
            
            # Show message box
            msg = f"Clustering completed!\n"
            msg += f"2D Silhouette Score: {score_2d:.4f}\n"
            
            if score_3d is not None:
                msg += f"3D Silhouette Score: {score_3d:.4f}\n"
                
            msg += f"\nCheck visualization files in project folder"
            
            messagebox.showinfo(algo_name, msg)
        else:
            result_text.set(f"Error running {algo_name}")
            messagebox.showerror(algo_name, "Error: Clustering failed or centroids unavailable.")

    # Frame for buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    # Create buttons in a grid layout
    row, col = 0, 0
    for idx, algo_name in enumerate(algorithms.keys()):
        tk.Button(
            button_frame, 
            text=f"Run {algo_name}", 
            command=lambda name=algo_name: run_and_show(name),
            width=15,
            height=2
        ).grid(row=row, column=col, padx=5, pady=5)
        
        # Update grid position
        col += 1
        if col > 1:  # 2 buttons per row
            col = 0
            row += 1

    tk.Button(root, text="Exit", command=root.quit, width=15).pack(pady=20)

    root.mainloop()
