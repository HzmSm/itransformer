import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

def process_pca():
    # 1. Load Data
    data_path = './data/part_5two_1min.csv'
    output_path = './data/part_5two_1min_pca.csv'
    scaler_path = './data/target_scaler.pkl'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Identify columns
    date_col = 'datetime' if 'datetime' in df.columns else 'date'
    targets = ['MM256', 'MM264']
    
    # Features are all columns except date and targets
    feature_cols = [c for c in df.columns if c not in [date_col] + targets]
    
    # 2. Split Data (Train/Val/Test)
    n = len(df)
    num_train = int(n * 0.7)
    train_slice = slice(0, num_train)
    
    # 3. Process Features (PCA)
    print("Processing features with PCA...")
    x = df[feature_cols].values
    scaler_feat = StandardScaler()
    scaler_feat.fit(x[train_slice])
    x_scaled = scaler_feat.transform(x)
    
    pca = PCA(n_components=0.95)
    pca.fit(x_scaled[train_slice])
    x_pca = pca.transform(x_scaled)
    
    n_components = x_pca.shape[1]
    print(f"PCA components: {n_components}")
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 4. Process Targets (Standard Scaling) - CRITICAL FIX
    print("Processing targets with StandardScaler...")
    y = df[targets].values
    scaler_target = StandardScaler()
    scaler_target.fit(y[train_slice]) # Fit on train only
    y_scaled = scaler_target.transform(y)
    
    # Save scaler for later inverse transform
    joblib.dump(scaler_target, scaler_path)
    print(f"Saved target scaler to {scaler_path}")
    
    # 5. Create New DataFrame
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(x_pca, columns=pc_cols)
    df_targets = pd.DataFrame(y_scaled, columns=targets)
    
    # Add date back
    df_final = pd.concat([df[[date_col]], df_pca, df_targets], axis=1)
    
    # 6. Save Data
    print(f"Saving processed data to {output_path}...")
    df_final.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    process_pca()
