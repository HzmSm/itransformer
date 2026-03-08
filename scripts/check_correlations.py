
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import os

def analyze_correlations():
    data_path = '/root/autodl-tmp/iTransformer-main/data/part_5two_1min.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Drop datetime for correlation
    if 'datetime' in df.columns:
        df = df.drop(columns=['datetime'])
    elif 'date' in df.columns:
        df = df.drop(columns=['date'])
        
    targets = ['MM256', 'MM264']
    features = [c for c in df.columns if c not in targets]
    
    print("\nCorrelation with Targets (Pearson):")
    corr_matrix = df.corr()
    
    for t in targets:
        print(f"\nTop 5 features correlated with {t}:")
        print(corr_matrix[t].drop(targets).abs().sort_values(ascending=False).head(5))
        
    # Check lagged correlation (does feature X at t-1 predict Target at t?)
    print("\nLag-1 Correlation (Feature[t-1] vs Target[t]):")
    df_shifted = df.shift(1)
    for t in targets:
        corrs = {}
        for f in features:
            # Correlation between Feature(t-1) and Target(t)
            c = df[t].corr(df_shifted[f])
            corrs[f] = c
        
        s_corrs = pd.Series(corrs).abs().sort_values(ascending=False).head(5)
        print(f"\nTop 5 Lag-1 features for {t}:")
        print(s_corrs)

if __name__ == "__main__":
    analyze_correlations()
