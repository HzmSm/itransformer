import pandas as pd
import numpy as np
import os

def check_data(root_path, data_path, target_cols):
    full_path = os.path.join(root_path, data_path)
    print(f"Checking file: {full_path}")
    
    try:
        df = pd.read_csv(full_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 1. Time column check
    date_col = None
    for cand in ['date', 'datetime', 'time', 'timestamp']:
        if cand in df.columns:
            date_col = cand
            break
            
    if date_col:
        print(f"\n[Time Column] Found '{date_col}'")
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            is_monotonic = df[date_col].is_monotonic_increasing
            print(f"  Is monotonic increasing? {is_monotonic}")
            if not is_monotonic:
                print("  WARNING: Time column is NOT sorted!")
            
            duplicates = df[date_col].duplicated().sum()
            print(f"  Duplicate timestamps: {duplicates}")
            if duplicates > 0:
                print("  WARNING: Found duplicate timestamps!")
                
            diffs = df[date_col].diff().dropna()
            print(f"  Time diff stats:\n{diffs.describe()}")
        except Exception as e:
            print(f"  Error parsing time column: {e}")
    else:
        print("\n[Time Column] WARNING: No standard time column found (date/datetime/time/timestamp)")

    # 2. Target columns check
    targets = [t.strip() for t in target_cols.split(',')]
    print(f"\n[Target Columns] {targets}")
    
    for t in targets:
        if t not in df.columns:
            print(f"  ERROR: Target '{t}' not found in CSV!")
            continue
            
        col_data = df[t]
        n_nan = col_data.isna().sum()
        n_inf = np.isinf(col_data).sum()
        n_zeros = (col_data == 0).sum()
        
        print(f"  Column '{t}':")
        print(f"    NaNs: {n_nan} ({n_nan/len(df)*100:.2f}%)")
        print(f"    Infs: {n_inf}")
        print(f"    Zeros: {n_zeros} ({n_zeros/len(df)*100:.2f}%)")
        print(f"    Mean: {col_data.mean():.4f}, Std: {col_data.std():.4f}")
        print(f"    Min: {col_data.min():.4f}, Max: {col_data.max():.4f}")
        
        if col_data.std() == 0:
            print("    WARNING: Constant column (std=0)!")
            
    # 3. General check for all features
    print("\n[All Features Check]")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        if c in targets: continue
        if df[c].std() == 0:
            print(f"  WARNING: Feature '{c}' is constant (std=0). Consider removing.")
        if df[c].isna().sum() > 0:
            print(f"  WARNING: Feature '{c}' has {df[c].isna().sum()} NaNs.")

if __name__ == "__main__":
    # Adjust these paths to match your run command
    ROOT_PATH = "./data/"
    DATA_PATH = "part_8two_jitter.csv"
    TARGETS = "MM264,MM256"
    
    check_data(ROOT_PATH, DATA_PATH, TARGETS)
