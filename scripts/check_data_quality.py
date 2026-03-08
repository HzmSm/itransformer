
import pandas as pd
import numpy as np
import os

def process_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Parse datetime
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        # Try to find a date column
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            df['datetime'] = pd.to_datetime(df[date_cols[0]])
        else:
            print("Error: No datetime column found.")
            return

    # Sort by time
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Original shape: {df.shape}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # 1. Check for duplicates
    duplicates = df.duplicated(subset=['datetime']).sum()
    print(f"Duplicate timestamps found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates(subset=['datetime'], keep='first')
        print("Duplicates dropped.")

    # 2. Handle Missing Values
    missing = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing[missing > 0])
    
    # Forward fill then backward fill for small gaps
    df = df.ffill().bfill()
    
    # 3. Physical Constraints & Outlier Handling based on meanings
    # Columns: 
    # Wind: AN311, AN422, AN423 (Should be >= 0)
    # Temp: TP1711, TC862 (Likely > 0 in mine, definitely > -273.15)
    # Humidity: RH1712 (0-100)
    # Pressure: BA1713, P_864 (Should be > 0)
    # Gas: MM264, MM256, CM861 (Should be >= 0)
    # Diff Pressure: CR863 (Could be negative? Usually pressure diff is magnitude or directional. Assuming >=0 for now or check distribution)
    # Flow: WM868 (>= 0)
    # Current: AMP..., DMP... (>= 0)
    # Speed: V (>= 0)

    non_negative_cols = [
        'AN311', 'AN422', 'AN423', 
        'RH1712', 'BA1713', 'P_864', 
        'MM264', 'MM256', 'CM861', 
        'WM868', 
        'AMP1_IR', 'AMP2_IR', 'DMP3_IR', 'DMP4_IR', 'AMP5_IR', 
        'V'
    ]
    
    print("\nChecking for negative values in strictly positive columns...")
    for col in non_negative_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"  {col}: {neg_count} negative values. Clipping to 0.")
                df[col] = df[col].clip(lower=0)

    # Humidity check > 100
    if 'RH1712' in df.columns:
        high_rh = (df['RH1712'] > 100).sum()
        if high_rh > 0:
            print(f"  RH1712: {high_rh} values > 100. Clipping to 100.")
            df['RH1712'] = df['RH1712'].clip(upper=100)

    # 4. Statistical Outlier Detection (Z-score > 5 or similar extreme)
    # Just reporting for now, replacing with NaN and interpolating might be better for extreme spikes
    # But for sensor data, sometimes spikes are real. 
    # Let's look for "impossible" spikes. 
    # For simplicity in this script, we will just report basic stats to see if max is crazy.
    
    print("\nBasic Statistics after basic cleaning:")
    print(df.describe().T[['min', 'max', 'mean', '50%']])

    # 5. Resampling
    # The data is likely 1-second data. For forecasting, 1-minute or 10-minute is usually better.
    # It reduces noise (jitter) and makes the horizon (96 steps) cover a meaningful period.
    # 96 seconds is too short for "Long Term Forecasting". 96 minutes is better.
    
    df.set_index('datetime', inplace=True)
    
    # Resample to 1 minute mean
    df_1min = df.resample('1T').mean()
    
    # Resample to 10 minute mean
    df_10min = df.resample('10T').mean()
    
    # Handle NaNs introduced by resampling (if any gaps)
    df_1min = df_1min.interpolate(method='time')
    df_10min = df_10min.interpolate(method='time')
    
    print(f"\nResampled to 1 min shape: {df_1min.shape}")
    print(f"Resampled to 10 min shape: {df_10min.shape}")

    # Save processed files
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    path_1min = os.path.join(output_dir, f"{base_name}_1min.csv")
    path_10min = os.path.join(output_dir, f"{base_name}_10min.csv")
    
    df_1min.reset_index().to_csv(path_1min, index=False)
    df_10min.reset_index().to_csv(path_10min, index=False)
    
    print(f"\nSaved processed data to:\n  {path_1min}\n  {path_10min}")

if __name__ == "__main__":
    input_csv = "/root/autodl-tmp/iTransformer-main/data/part_5two.csv"
    output_csv = "/root/autodl-tmp/iTransformer-main/data/part_5two_processed.csv" # Placeholder
    process_data(input_csv, output_csv)
