
# ui.py: Implements the UI using Tkinter
import tkinter as tk
from tkinter import messagebox
from visualization import plot_clusters_2d, plot_clusters_3d

def start_ui(data, algorithms, results):
    root = tk.Tk()
    root.title("Customer Segmentation App")
    root.geometry("400x400")

    tk.Label(root, text="Customer Segmentation", font=("Arial", 16)).pack(pady=10)

    def run_and_show(algo_name):
        algo = algorithms[algo_name]
        labels = algo.fit(data)
        centroids = algo.get_centroids() if hasattr(algo, 'get_centroids') else None
        if labels is not None and centroids is not None:
            plot_clusters_2d(data, labels, centroids, algo_name)
            avg_score = results[algo_name]["best_score"][0]  # Example metric
            avg_time = results[algo_name]["time"][0]  # Example metric
            messagebox.showinfo(algo_name, f"Clustering completed!\nAvg Silhouette Score: {avg_score:.2f}\nAvg Time: {avg_time:.2f}s\nCheck {algo_name}_clusters_2d.png and {algo_name}_clusters_3d.png for visualizations")
        else:
            messagebox.showerror(algo_name, "Error: Clustering failed or centroids unavailable.")

    for algo_name in algorithms.keys():
        tk.Button(root, text=f"Run {algo_name}", command=lambda name=algo_name: run_and_show(name)).pack(pady=5)

    tk.Button(root, text="Exit", command=root.quit).pack(pady=10)

    root.mainloop()
