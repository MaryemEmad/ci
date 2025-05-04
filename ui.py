import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import os
from sklearn.metrics import silhouette_score
import pandas as pd
from PIL import Image, ImageTk

class CustomerSegmentationApp:
    def __init__(self, root, data_2d, data_3d, scaler_2d, scaler_3d):
        self.root = root
        self.data_2d = data_2d
        self.data_3d = data_3d
        self.scaler_2d = scaler_2d
        self.scaler_3d = scaler_3d
        self.root.title("Customer Segmentation Dashboard")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f5f5f5")

        # Define the directory where existing plots are stored
        self.plot_dir = "results"

        # Load summary results
        try:
            self.summary_df = pd.read_csv('summary_results.csv')
            self.summary_df = self.summary_df[(self.summary_df['m'] == 2.0) & (self.summary_df['n_clusters'] == 5)]
            self.summary_df = self.summary_df.set_index('Algorithm')
        except FileNotFoundError:
            messagebox.showerror("Error", "summary_results.csv not found. Please run experiment_runner.py first.")
            self.root.quit()
            return

        # Define algorithms
        self.results_dir = "run_results_m_2.0_n_clusters_5"
        self.algorithms = [
            'rseKFCM', 'spKFCM', 'oKFCM', 'FCM', 'KFCM', 'MKFCM',
            'GK-FCM', 'K-Means', 'ImprovedGathGeva', 'IFCM'
        ]
        self.available_algorithms = [
            algo for algo in self.algorithms
            if os.path.exists(os.path.join(self.results_dir, f"{algo}_labels_2d.npy"))
        ]

        # Keep track of PhotoImage objects
        self.photos = []

        self.setup_ui()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root, text="Customer Segmentation Dashboard (m=2.0, n_clusters=5)",
            font=("Arial", 18, "bold"), bg="#f5f5f5", fg="#333333"
        )
        title_label.pack(pady=10)

        # Main frame with two sections: Left (Controls) and Right (Images)
        main_frame = tk.Frame(self.root, bg="#f5f5f5")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left frame for controls and table
        left_frame = tk.Frame(main_frame, bg="#ffffff", relief="raised", bd=2)
        left_frame.pack(side="left", fill="both", expand=False, padx=(0, 10))

        # Right frame for image display
        self.image_frame = tk.Frame(main_frame, bg="#ffffff", relief="raised", bd=2)
        self.image_frame.pack(side="right", fill="both", expand=True)

        # Algorithm buttons frame
        button_frame = tk.LabelFrame(
            left_frame, text="Select Algorithm", font=("Arial", 12, "bold"),
            bg="#ffffff", fg="#333333", padx=10, pady=10
        )
        button_frame.pack(fill="x", padx=5, pady=5)

        # Create buttons in a grid layout
        row, col = 0, 0
        for algo_name in self.available_algorithms:
            tk.Button(
                button_frame, text=f"Show {algo_name}",
                command=lambda name=algo_name: self.show_results(name),
                width=15, height=2, font=("Arial", 10),
                bg="#4CAF50", fg="white", activebackground="#45a049"
            ).grid(row=row, column=col, padx=5, pady=5)
            col += 1
            if col > 1:  # 2 buttons per row
                col = 0
                row += 1

        # Additional buttons for heatmap and comparison
        extra_button_frame = tk.Frame(left_frame, bg="#ffffff")
        extra_button_frame.pack(fill="x", padx=5, pady=5)

        tk.Button(
            extra_button_frame, text="Show Heatmap",
            command=self.show_heatmap, width=15, height=2,
            font=("Arial", 10), bg="#2196F3", fg="white",
            activebackground="#1e88e5"
        ).pack(side="left", padx=5)

        tk.Button(
            extra_button_frame, text="Compare K-Means vs FCM",
            command=self.compare_kmeans_fcm, width=25, height=2,
            font=("Arial", 10), bg="#FF9800", fg="white",
            activebackground="#f57c00"
        ).pack(side="left", padx=5)

        # Metrics table
        table_frame = tk.LabelFrame(
            left_frame, text="Performance Metrics", font=("Arial", 12, "bold"),
            bg="#ffffff", fg="#333333", padx=10, pady=10
        )
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Create Treeview for table
        columns = ("Algorithm", "2D Silhouette", "2D WCSS", "2D DB", "Time", "3D Silhouette")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=80, anchor="center")
        self.tree.pack(fill="both", expand=True)

        # Populate table
        for algo in self.available_algorithms:
            try:
                labels_2d = np.load(os.path.join(self.results_dir, f"{algo}_labels_2d.npy"))
                labels_3d = np.load(os.path.join(self.results_dir, f"{algo}_labels_3d.npy"))
                score_2d = silhouette_score(self.scaler_2d.inverse_transform(self.data_2d), labels_2d) if len(np.unique(labels_2d)) > 1 else np.nan
                score_3d = silhouette_score(self.scaler_3d.inverse_transform(self.data_3d), labels_3d) if len(np.unique(labels_3d)) > 1 else np.nan
                metrics = self.summary_df.loc[algo] if algo in self.summary_df.index else {}
                wcss_2d = metrics.get('Avg_WCSS_2D', np.nan)
                db_2d = metrics.get('Avg_Davies_Bouldin_2D', np.nan)
                time_2d = metrics.get('Avg_Time', np.nan)
                self.tree.insert("", "end", values=(
                    algo, f"{score_2d:.4f}", f"{wcss_2d:.4f}",
                    f"{db_2d:.4f}", f"{time_2d:.4f}", f"{score_3d:.4f}"
                ))
            except FileNotFoundError:
                continue

        # Image display labels with placeholders
        self.image_label_2d = tk.Label(self.image_frame, bg="#ffffff", text="2D Plot will appear here", font=("Arial", 10))
        self.image_label_2d.pack(pady=5, fill="both", expand=True)
        self.image_label_3d = tk.Label(self.image_frame, bg="#ffffff", text="3D Plot will appear here", font=("Arial", 10))
        self.image_label_3d.pack(pady=5, fill="both", expand=True)

        # Exit button
        tk.Button(
            self.root, text="Exit", command=self.root.quit, width=15,
            font=("Arial", 10), bg="#f44336", fg="white",
            activebackground="#d32f2f"
        ).pack(pady=10)

    def show_results(self, algo_name):
        try:
            # Load precomputed results for silhouette scores
            labels_2d = np.load(os.path.join(self.results_dir, f"{algo_name}_labels_2d.npy"))
            labels_3d = np.load(os.path.join(self.results_dir, f"{algo_name}_labels_3d.npy"))

            # Calculate Silhouette Scores
            score_2d = silhouette_score(self.scaler_2d.inverse_transform(self.data_2d), labels_2d) if len(np.unique(labels_2d)) > 1 else np.nan
            score_3d = silhouette_score(self.scaler_3d.inverse_transform(self.data_3d), labels_3d) if len(np.unique(labels_3d)) > 1 else np.nan

            # Load 2D plot directly from the results directory
            plot_2d_path = os.path.join(self.plot_dir, f"{algo_name}_clusters_2d_m_2.0_n_clusters_5.png")
            if os.path.exists(plot_2d_path):
                img_2d = Image.open(plot_2d_path)
                img_2d = img_2d.resize((400, 300), Image.Resampling.LANCZOS)
                photo_2d = ImageTk.PhotoImage(img_2d)
                self.photos.append(photo_2d)
                self.image_label_2d.configure(image=photo_2d, text="")
                self.root.update()
            else:
                self.image_label_2d.configure(text=f"2D Plot not found for {algo_name}")

            # Load 3D plot directly from the results directory
            plot_3d_path = os.path.join(self.plot_dir, f"{algo_name}_clusters_3d_m_2.0_n_clusters_5.png")
            if os.path.exists(plot_3d_path):
                img_3d = Image.open(plot_3d_path)
                img_3d = img_3d.resize((400, 300), Image.Resampling.LANCZOS)
                photo_3d = ImageTk.PhotoImage(img_3d)
                self.photos.append(photo_3d)
                self.image_label_3d.configure(image=photo_3d, text="")
                self.root.update()
            else:
                self.image_label_3d.configure(text=f"3D Plot not found for {algo_name}")

            # Show message box
            messagebox.showinfo(algo_name, f"{algo_name} Results Loaded!\n2D Silhouette Score: {score_2d:.4f}\n3D Silhouette Score: {score_3d:.4f}")

        except FileNotFoundError as e:
            messagebox.showerror("Error", f"File not found: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display plots: {str(e)}")

    def show_heatmap(self):
        try:
            # Load heatmap directly from the results directory
            heatmap_path = os.path.join(self.plot_dir, "metrics_heatmap_2d_m_2.0_n_clusters_5.png")
            if os.path.exists(heatmap_path):
                img = Image.open(heatmap_path)
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.photos.append(photo)
                self.image_label_2d.configure(image=photo, text="")
                self.image_label_3d.configure(image="", text="3D Plot will appear here")
                self.root.update()
            else:
                self.image_label_2d.configure(text="Heatmap not found")

            messagebox.showinfo("Heatmap", "Heatmap loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load heatmap: {str(e)}")

    def compare_kmeans_fcm(self):
        try:
            # Load comparison plot directly from the results directory
            compare_path = os.path.join(self.plot_dir, "comparison_kmeans_fcm_2d_m_2.0_n_clusters_5.png")
            if os.path.exists(compare_path):
                img = Image.open(compare_path)
                img = img.resize((400, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.photos.append(photo)
                self.image_label_2d.configure(image=photo, text="")
                self.image_label_3d.configure(image="", text="3D Plot will appear here")
                self.root.update()
            else:
                self.image_label_2d.configure(text="Comparison plot not found")

            messagebox.showinfo("Comparison", "K-Means vs FCM comparison loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load comparison: {str(e)}")

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('Mall_Customers.csv')
    data_2d = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
    data_3d = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

    from sklearn.preprocessing import StandardScaler
    scaler_2d = StandardScaler()
    scaler_3d = StandardScaler()
    data_2d = scaler_2d.fit_transform(data_2d)
    data_3d = scaler_3d.fit_transform(data_3d)

    # Start UI
    root = tk.Tk()
    app = CustomerSegmentationApp(root, data_2d, data_3d, scaler_2d, scaler_3d)
    root.mainloop()