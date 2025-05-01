
# data_loader.py: Loads and preprocesses the Mall Customers dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_mall_customers_data():
    """
    Loads and preprocesses the Mall Customers dataset for clustering.
    
    Returns:
    --------
    df : pandas.DataFrame
        Original dataset with all columns.
    features_2d : numpy.ndarray
        Normalized 2D features (Annual Income, Spending Score).
    features_3d : numpy.ndarray
        Normalized 3D features (Age, Annual Income, Spending Score).
    scaler_2d : MinMaxScaler
        Scaler for 2D features (for inverse transform if needed).
    scaler_3d : MinMaxScaler
        Scaler for 3D features (for inverse transform if needed).
    """
    # Load the dataset
    # Note: Download the dataset from https://www.kaggle.com/datasets/mosesmoncy/mall-customerscsv/data
    # and place it in your project directory
    df = pd.read_csv("Mall_Customers.csv")

    # Select features for clustering
    # 2D features for scatter plots (as per project requirements)
    features_2d = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    # 3D features for additional analysis (including Age)
    features_3d = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values

    # Normalize the features using MinMaxScaler
    scaler_2d = MinMaxScaler()
    features_2d = scaler_2d.fit_transform(features_2d)

    scaler_3d = MinMaxScaler()
    features_3d = scaler_3d.fit_transform(features_3d)

    return df, features_2d, features_3d, scaler_2d, scaler_3d
